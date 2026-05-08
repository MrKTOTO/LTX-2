import logging
import time
import copy
from collections.abc import Iterator
from contextlib import ExitStack, nullcontext
from dataclasses import replace

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.modality_tiling import VideoModalityTilingHelper
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tiling import DimensionTilingConfig, TileCountConfig
from ltx_core.tools import VideoLatentTools
from ltx_core.types import Audio, LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
    detect_params,
)
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    create_noised_state,
    get_device,
    modality_from_latent_state,
    post_process_latent,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.progress import progress
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        prompt_encoder_device: torch.device | None = None,
        prompt_encoder_staging_device: torch.device | None = None,
        stage_2_device: torch.device | None = None,
        model_parallel_block_devices: list[torch.device] | None = None,
        decoder_devices: list[torch.device] | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
        offload_mode: OffloadMode = OffloadMode.NONE,
        prompt_encoder_offload_mode: OffloadMode | None = None,
    ):
        self.device = device or get_device()
        self.prompt_encoder_device = prompt_encoder_device or self.device
        self.stage_2_device = stage_2_device or self.device
        self.model_parallel_block_devices = model_parallel_block_devices
        self.decoder_devices = decoder_devices or [self.device]
        self.dtype = torch.bfloat16
        prompt_offload_mode = prompt_encoder_offload_mode or offload_mode
        self.distilled_checkpoint_path = distilled_checkpoint_path
        self.loras = tuple(loras)
        self.quantization = quantization
        self.registry = registry
        self.torch_compile = torch_compile
        self.offload_mode = offload_mode

        logger.info("[LTX distilled] Primary device: %s", self.device)
        logger.info("[LTX distilled] Prompt encoder device: %s", self.prompt_encoder_device)
        logger.info("[LTX distilled] Stage 2 device: %s", self.stage_2_device)
        logger.info(
            "[LTX distilled] Diffusion block devices: %s",
            ",".join(str(device) for device in model_parallel_block_devices)
            if model_parallel_block_devices
            else self.device,
        )
        logger.info(
            "[LTX distilled] Video decoder devices: %s",
            ",".join(str(device) for device in self.decoder_devices),
        )
        logger.info("[LTX distilled] Prompt encoder staging device: %s", prompt_encoder_staging_device or "cpu")
        logger.info("[LTX distilled] Diffusion offload: %s", offload_mode.value)
        logger.info("[LTX distilled] Prompt encoder offload: %s", prompt_offload_mode.value)

        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path,
            gemma_root,
            self.dtype,
            self.prompt_encoder_device,
            embeddings_processor_device=self.device,
            registry=registry,
            offload_mode=prompt_offload_mode,
            text_encoder_staging_device=prompt_encoder_staging_device,
        )
        self.image_conditioner = ImageConditioner(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.stage = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
            offload_mode=offload_mode,
            block_devices=model_parallel_block_devices,
        )
        self.upsampler = VideoUpsampler(
            distilled_checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
            decoder_devices=self.decoder_devices,
        )
        self.audio_decoder = AudioDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)

    def run_stage_1_only(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        enhance_prompt: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        max_batch_size: int = 1,
        generate_audio: bool = False,
        tiled_denoising: bool = False,
        tiled_latent_frames: int = 16,
        tiled_latent_overlap: int = 4,
        tiled_devices: list[torch.device] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run only Stage 1 of the pipeline (low-res denoising).
        
        Returns:
            Tuple of (video_latent, video_context) for Stage 2 processing
        """
        assert_resolution(height=height, width=width, is_two_stage=True)
        logger.info(
            "[LTX distilled] Stage 1 only: %sx%s, frames=%s, fps=%s",
            width,
            height,
            num_frames,
            frame_rate,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        logger.info("[LTX distilled] Encoding prompt")
        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding
        if video_context.device != self.device:
            logger.info("[LTX distilled] Moving prompt video context to %s", self.device)
            video_context = video_context.to(self.device)
        
        # Explicitly cleanup prompt encoder device after encoding to free GPU1 for Stage 1
        from ltx_pipelines.utils.helpers import cleanup_device_memory
        cleanup_device_memory(self.prompt_encoder_device)
        logger.info("[LTX distilled] Cleaned up prompt encoder device %s", self.prompt_encoder_device)

        # Stage 1: Initial low resolution video generation
        logger.info("[LTX distilled] Stage 1 conditioning")
        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_1_w, stage_1_h = width // 2, height // 2
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        logger.info("[LTX distilled] Stage 1 denoising")
        if tiled_denoising and not generate_audio:
            video_state, audio_state = self._run_tiled_video_stage(
                stage=self.stage,
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=stage_1_w,
                height=stage_1_h,
                frames=num_frames,
                fps=frame_rate,
                video_context=video_context,
                conditionings=stage_1_conditionings,
                tiled_latent_frames=tiled_latent_frames,
                tiled_latent_overlap=tiled_latent_overlap,
                tiled_devices=tiled_devices,
            )
        else:
            video_state, audio_state = self.stage(
                denoiser=SimpleDenoiser(video_context, audio_context if generate_audio else None),
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=stage_1_w,
                height=stage_1_h,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
                audio=ModalitySpec(context=audio_context) if generate_audio else None,
                max_batch_size=max_batch_size,
            )

        return video_state.latent, video_context

    def run_stage_2_only(
        self,
        stage_1_latent: torch.Tensor,
        video_context: torch.Tensor,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        max_batch_size: int = 1,
        generate_audio: bool = False,
        tiled_denoising: bool = False,
        tiled_latent_frames: int = 16,
        tiled_latent_overlap: int = 4,
        tiled_devices: list[torch.device] | None = None,
        stage_2_tiled_threshold_frames: int = 0,
        stage_2_tiled_latent_frames: int = 64,
        stage_2_tiled_latent_overlap: int = 16,
    ) -> Iterator[torch.Tensor]:
        """
        Run only Stage 2 of the pipeline (upsampling + high-res denoising + decode).
        
        Args:
            stage_1_latent: Output latent from Stage 1
            video_context: Video context from prompt encoding
            
        Returns:
            Iterator of decoded video frames
        """
        logger.info(
            "[LTX distilled] Stage 2 only: %sx%s, frames=%s",
            width,
            height,
            num_frames,
        )

        dtype = torch.bfloat16
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Stage 2: Upsample and refine
        logger.info("[LTX distilled] Stage 2 latent upsampling")
        upscaled_video_latent = self.upsampler(stage_1_latent[:1])
        del stage_1_latent
        self._empty_cuda_cache(self.device, self.stage_2_device)

        logger.info("[LTX distilled] Stage 2 conditioning")
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )
        stage_2_device = self.stage_2_device
        stage_2_stage = self.stage if stage_2_device == self.device else self._make_diffusion_stage(stage_2_device)
        stage_2_noiser = GaussianNoiser(torch.Generator(device=stage_2_device).manual_seed(seed + 1))
        stage_2_video_context = video_context
        stage_2_initial_latent = upscaled_video_latent
        
        if stage_2_device != self.device:
            logger.info("[LTX distilled] Moving Stage 2 denoising tensors to %s", stage_2_device)
            stage_2_sigmas = stage_2_sigmas.to(stage_2_device)
            stage_2_video_context = video_context.to(stage_2_device)
            stage_2_initial_latent = upscaled_video_latent.to(stage_2_device)
            stage_2_conditionings = self._move_conditionings(stage_2_conditionings, stage_2_device)
            del upscaled_video_latent
            self._empty_cuda_cache(self.device, stage_2_device)

        logger.info("[LTX distilled] Stage 2 denoising")
        if tiled_denoising and not generate_audio:
            video_state, audio_state = self._run_tiled_video_stage(
                stage=stage_2_stage,
                sigmas=stage_2_sigmas,
                noiser=stage_2_noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video_context=stage_2_video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=stage_2_initial_latent,
                tiled_latent_frames=tiled_latent_frames,
                tiled_latent_overlap=tiled_latent_overlap,
                tiled_devices=tiled_devices,
            )
        elif (
            not generate_audio
            and stage_2_tiled_threshold_frames > 0
            and num_frames > stage_2_tiled_threshold_frames
        ):
            logger.info(
                "[LTX distilled] Stage 2-only tiled denoising enabled for %s frames",
                num_frames,
            )
            video_state, audio_state = self._run_tiled_video_stage(
                stage=stage_2_stage,
                sigmas=stage_2_sigmas,
                noiser=stage_2_noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video_context=stage_2_video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=stage_2_initial_latent,
                tiled_latent_frames=stage_2_tiled_latent_frames,
                tiled_latent_overlap=stage_2_tiled_latent_overlap,
                tiled_devices=[stage_2_device],
            )
        else:
            video_state, audio_state = stage_2_stage(
                denoiser=SimpleDenoiser(stage_2_video_context, None),
                sigmas=stage_2_sigmas,
                noiser=stage_2_noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=stage_2_video_context,
                    conditionings=stage_2_conditionings,
                    noise_scale=stage_2_sigmas[0].item(),
                    initial_latent=stage_2_initial_latent,
                ),
                audio=None,
                max_batch_size=max_batch_size,
            )
        
        if video_state is not None and video_state.latent.device != self.device:
            logger.info("[LTX distilled] Moving Stage 2 video latent back to %s for decoding", self.device)
            video_state = replace(video_state, latent=video_state.latent.to(self.device))

        logger.info("[LTX distilled] Creating video decoder iterator")
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        return decoded_video

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        max_batch_size: int = 1,
        generate_audio: bool = True,
        tiled_denoising: bool = False,
        tiled_latent_frames: int = 16,
        tiled_latent_overlap: int = 4,
        tiled_devices: list[torch.device] | None = None,
        stage_2_tiled_threshold_frames: int = 0,
        stage_2_tiled_latent_frames: int = 64,
        stage_2_tiled_latent_overlap: int = 16,
    ) -> tuple[Iterator[torch.Tensor], Audio | None]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        started_at = time.monotonic()
        logger.info(
            "[LTX distilled] Starting pipeline: %sx%s, frames=%s, fps=%s, images=%s",
            width,
            height,
            num_frames,
            frame_rate,
            len(images),
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        logger.info("[LTX distilled] Encoding prompt")
        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding
        if video_context.device != self.device:
            logger.info("[LTX distilled] Moving prompt video context to %s", self.device)
            video_context = video_context.to(self.device)
        if generate_audio and audio_context.device != self.device:
            logger.info("[LTX distilled] Moving prompt audio context to %s", self.device)
            audio_context = audio_context.to(self.device)
        if not generate_audio:
            logger.info("[LTX distilled] Audio latent generation disabled")
            audio_context = None

        # Stage 1: Initial low resolution video generation.
        logger.info("[LTX distilled] Stage 1 conditioning")
        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_1_w, stage_1_h = width // 2, height // 2
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        logger.info("[LTX distilled] Stage 1 denoising")
        if tiled_denoising and not generate_audio:
            video_state, audio_state = self._run_tiled_video_stage(
                stage=self.stage,
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=stage_1_w,
                height=stage_1_h,
                frames=num_frames,
                fps=frame_rate,
                video_context=video_context,
                conditionings=stage_1_conditionings,
                tiled_latent_frames=tiled_latent_frames,
                tiled_latent_overlap=tiled_latent_overlap,
                tiled_devices=tiled_devices,
            )
        else:
            video_state, audio_state = self.stage(
                denoiser=SimpleDenoiser(video_context, audio_context),
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=stage_1_w,
                height=stage_1_h,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
                audio=ModalitySpec(context=audio_context) if generate_audio else None,
                max_batch_size=max_batch_size,
            )

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        logger.info("[LTX distilled] Stage 2 latent upsampling")
        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        del video_state, audio_state
        self._empty_cuda_cache(self.device, self.stage_2_device)

        logger.info("[LTX distilled] Stage 2 conditioning")
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )
        stage_2_device = self.stage_2_device
        stage_2_stage = self.stage if stage_2_device == self.device else self._make_diffusion_stage(stage_2_device)
        stage_2_noiser = noiser
        stage_2_video_context = video_context
        stage_2_initial_latent = upscaled_video_latent
        if stage_2_device != self.device:
            logger.info("[LTX distilled] Moving Stage 2 denoising tensors to %s", stage_2_device)
            stage_2_sigmas = stage_2_sigmas.to(stage_2_device)
            stage_2_video_context = video_context.to(stage_2_device)
            stage_2_initial_latent = upscaled_video_latent.to(stage_2_device)
            stage_2_conditionings = self._move_conditionings(stage_2_conditionings, stage_2_device)
            stage_2_noiser = GaussianNoiser(torch.Generator(device=stage_2_device).manual_seed(seed + 1))
            del upscaled_video_latent
            self._empty_cuda_cache(self.device, stage_2_device)

        logger.info("[LTX distilled] Stage 2 denoising")
        if tiled_denoising and not generate_audio:
            video_state, audio_state = self._run_tiled_video_stage(
                stage=stage_2_stage,
                sigmas=stage_2_sigmas,
                noiser=stage_2_noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video_context=stage_2_video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=stage_2_initial_latent,
                tiled_latent_frames=tiled_latent_frames,
                tiled_latent_overlap=tiled_latent_overlap,
                tiled_devices=tiled_devices,
            )
        elif (
            not generate_audio
            and stage_2_tiled_threshold_frames > 0
            and num_frames > stage_2_tiled_threshold_frames
        ):
            logger.info(
                "[LTX distilled] Stage 2-only tiled denoising enabled for %s frames "
                "(threshold=%s, latent_tile=%s, overlap=%s)",
                num_frames,
                stage_2_tiled_threshold_frames,
                stage_2_tiled_latent_frames,
                stage_2_tiled_latent_overlap,
            )
            video_state, audio_state = self._run_tiled_video_stage(
                stage=stage_2_stage,
                sigmas=stage_2_sigmas,
                noiser=stage_2_noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video_context=stage_2_video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=stage_2_initial_latent,
                tiled_latent_frames=stage_2_tiled_latent_frames,
                tiled_latent_overlap=stage_2_tiled_latent_overlap,
                tiled_devices=[stage_2_device],
            )
        else:
            video_state, audio_state = stage_2_stage(
                denoiser=SimpleDenoiser(stage_2_video_context, None),
                sigmas=stage_2_sigmas,
                noiser=stage_2_noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=stage_2_video_context,
                    conditionings=stage_2_conditionings,
                    noise_scale=stage_2_sigmas[0].item(),
                    initial_latent=stage_2_initial_latent,
                ),
                audio=None,
                max_batch_size=max_batch_size,
            )
        if video_state is not None and video_state.latent.device != self.device:
            logger.info("[LTX distilled] Moving Stage 2 video latent back to %s for decoding", self.device)
            video_state = replace(video_state, latent=video_state.latent.to(self.device))

        logger.info("[LTX distilled] Creating video decoder iterator")
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        if generate_audio and audio_state is not None:
            logger.info("[LTX distilled] Decoding audio")
            decoded_audio = self.audio_decoder(audio_state.latent)
        else:
            decoded_audio = None
        logger.info("[LTX distilled] Pipeline tensors ready in %.1fs", time.monotonic() - started_at)
        return decoded_video, decoded_audio

    def _make_diffusion_stage(self, device: torch.device) -> DiffusionStage:
        return DiffusionStage(
            self.distilled_checkpoint_path,
            self.dtype,
            device,
            loras=self.loras,
            quantization=self.quantization,
            registry=self.registry,
            torch_compile=self.torch_compile,
            offload_mode=self.offload_mode,
            block_devices=self.model_parallel_block_devices,
        )

    @staticmethod
    def _empty_cuda_cache(*devices: torch.device) -> None:
        for device in devices:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    @staticmethod
    def _move_conditionings(conditionings: list, device: torch.device) -> list:
        moved = []
        for conditioning in conditionings:
            clone = copy.copy(conditioning)
            for attr in ("latent", "keyframes"):
                tensor = getattr(clone, attr, None)
                if isinstance(tensor, torch.Tensor) and tensor.device != device:
                    setattr(clone, attr, tensor.detach().to(device))
            moved.append(clone)
        return moved

    @staticmethod
    def _move_modality(modality, device: torch.device):
        def move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return tensor.to(device) if tensor is not None and tensor.device != device else tensor

        return replace(
            modality,
            latent=move(modality.latent),
            sigma=move(modality.sigma),
            timesteps=move(modality.timesteps),
            positions=move(modality.positions),
            context=move(modality.context),
            context_mask=move(modality.context_mask),
            attention_mask=move(modality.attention_mask),
        )

    @staticmethod
    def _copy_to_device_without_peer(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        if tensor.device == device:
            return tensor
        if tensor.device.type == "cuda" and device.type == "cuda":
            return tensor.detach().to("cpu").to(device)
        return tensor.detach().to(device)

    def _run_tiled_video_stage(
        self,
        stage: DiffusionStage,
        sigmas: torch.Tensor,
        noiser: GaussianNoiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video_context: torch.Tensor,
        conditionings: list,
        tiled_latent_frames: int,
        tiled_latent_overlap: int,
        tiled_devices: list[torch.device] | None,
        noise_scale: float = 1.0,
        initial_latent: torch.Tensor | None = None,
    ) -> tuple[LatentState, None]:
        pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
        latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
        video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), latent_shape, fps)
        stage_device = getattr(stage, "_device", self.device)
        video_state = create_noised_state(
            tools=video_tools,
            conditionings=conditionings,
            noiser=noiser,
            dtype=self.dtype,
            device=stage_device,
            noise_scale=noise_scale,
            initial_latent=initial_latent,
        )

        tile_frames = max(1, int(tiled_latent_frames))
        tile_overlap = max(0, min(int(tiled_latent_overlap), tile_frames - 1))
        frame_tiling = DimensionTilingConfig.from_tile_size(latent_shape.frames, tile_frames, tile_overlap)
        tiling = TileCountConfig(frames=frame_tiling)
        helper = VideoModalityTilingHelper(tiling, video_tools)

        devices = tiled_devices or [stage_device]
        devices = [device for device in devices if device.type == "cuda" or device == stage_device]
        if not devices:
            devices = [stage_device]
        if stage_device not in devices:
            devices.insert(0, stage_device)
        if self.offload_mode == OffloadMode.DISK and len(devices) > 1:
            logger.warning(
                "[LTX distilled] Multiple tiled devices are disabled with disk offload. "
                "Windows/PyTorch can crash when several block-streaming contexts read the "
                "same disk cache in one process; using %s for diffusion tiles.",
                stage_device,
            )
            devices = [stage_device]

        logger.info(
            "[LTX distilled] Tiled video denoising: latent=%s, tile_frames=%s, overlap=%s, tiles=%s, devices=%s",
            tuple(latent_shape),
            tile_frames,
            tile_overlap,
            len(helper.tiles),
            ",".join(str(device) for device in devices),
        )

        stages_by_device: dict[torch.device, DiffusionStage] = {stage_device: stage}
        for device in devices:
            if device != stage_device:
                stages_by_device[device] = self._make_diffusion_stage(device)

        stepper = EulerDiffusionStep()
        sigmas = sigmas.to(dtype=torch.float32, device=stage_device)
        output_device = stage_device

        def run_tile(transformer, tile_modality, device: torch.device) -> torch.Tensor:
            device_context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), device_context:
                moved_modality = self._move_modality(tile_modality, device)
                denoised, _ = transformer(video=moved_modality, audio=None, perturbations=None)
                return denoised

        with ExitStack() as stack:
            transformers = {
                device: stack.enter_context(stages_by_device[device].model_context(video_tools=video_tools))
                for device in devices
            }
            for step_idx, _ in enumerate(progress(sigmas[:-1])):
                full_modality = modality_from_latent_state(video_state, video_context, sigmas[step_idx])
                blend_output = torch.zeros_like(video_state.latent)
                tile_results = []
                for tile_idx, tile in enumerate(helper.tiles):
                    device = devices[tile_idx % len(devices)]
                    tile_modality, ctx = helper.tile_modality(full_modality, tile, normalize_positions=True)
                    tile_denoised = run_tile(transformers[device], tile_modality, device)
                    tile_results.append((tile_denoised, tile, ctx))

                for tile_denoised, tile, ctx in tile_results:
                    tile_denoised = self._copy_to_device_without_peer(tile_denoised, output_device)
                    blend_output = helper.blend(tile_denoised, tile, ctx, blend_output)

                denoised = post_process_latent(blend_output, video_state.denoise_mask, video_state.clean_latent)
                video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised, sigmas, step_idx))

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        return video_state, None


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
    logging.getLogger().setLevel(logging.INFO)
    started_at = time.monotonic()
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else None
    prompt_encoder_device = torch.device(args.prompt_encoder_device) if args.prompt_encoder_device else None
    stage_2_device = torch.device(args.stage_2_device) if args.stage_2_device else None
    model_parallel_block_devices = None
    if args.model_parallel_block_devices:
        model_parallel_block_devices = [
            torch.device(part.strip())
            for part in args.model_parallel_block_devices.split(",")
            if part.strip()
        ]
    decoder_devices = None
    if args.decoder_devices:
        decoder_devices = [torch.device(part.strip()) for part in args.decoder_devices.split(",") if part.strip()]
    prompt_encoder_staging_device = (
        torch.device(args.prompt_encoder_staging_device) if args.prompt_encoder_staging_device else None
    )
    tiled_devices = None
    if args.tiled_devices:
        tiled_devices = [torch.device(part.strip()) for part in args.tiled_devices.split(",") if part.strip()]
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        device=device,
        prompt_encoder_device=prompt_encoder_device,
        prompt_encoder_staging_device=prompt_encoder_staging_device,
        stage_2_device=stage_2_device,
        model_parallel_block_devices=model_parallel_block_devices,
        decoder_devices=decoder_devices,
        quantization=args.quantization,
        torch_compile=args.compile,
        offload_mode=args.offload_mode,
        prompt_encoder_offload_mode=args.prompt_encoder_offload_mode,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    logger.info("[LTX distilled] Video chunks to decode/encode: %s", video_chunks_number)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        max_batch_size=args.max_batch_size,
        generate_audio=not args.disable_audio,
        tiled_denoising=args.tiled_denoising,
        tiled_latent_frames=args.tiled_latent_frames,
        tiled_latent_overlap=args.tiled_latent_overlap,
        tiled_devices=tiled_devices,
        stage_2_tiled_threshold_frames=args.stage_2_tiled_threshold_frames,
        stage_2_tiled_latent_frames=args.stage_2_tiled_latent_frames,
        stage_2_tiled_latent_overlap=args.stage_2_tiled_latent_overlap,
    )

    logger.info("[LTX distilled] Encoding output video to %s", args.output_path)
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )
    logger.info("[LTX distilled] Finished in %.1fs", time.monotonic() - started_at)


if __name__ == "__main__":
    main()
