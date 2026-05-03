import logging
import time
from collections.abc import Iterator

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
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
    get_device,
)
from ltx_pipelines.utils.media_io import encode_video
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
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
        offload_mode: OffloadMode = OffloadMode.NONE,
        prompt_encoder_offload_mode: OffloadMode | None = None,
    ):
        self.device = device or get_device()
        self.prompt_encoder_device = prompt_encoder_device or self.device
        self.dtype = torch.bfloat16
        prompt_offload_mode = prompt_encoder_offload_mode or offload_mode

        logger.info("[LTX distilled] Primary device: %s", self.device)
        logger.info("[LTX distilled] Prompt encoder device: %s", self.prompt_encoder_device)
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
        )
        self.upsampler = VideoUpsampler(
            distilled_checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)

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
    ) -> tuple[Iterator[torch.Tensor], Audio]:
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
        if audio_context.device != self.device:
            logger.info("[LTX distilled] Moving prompt audio context to %s", self.device)
            audio_context = audio_context.to(self.device)

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
        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context),
            max_batch_size=max_batch_size,
        )

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        logger.info("[LTX distilled] Stage 2 latent upsampling")
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

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

        logger.info("[LTX distilled] Stage 2 denoising")
        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
            max_batch_size=max_batch_size,
        )

        logger.info("[LTX distilled] Creating video decoder iterator")
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        logger.info("[LTX distilled] Decoding audio")
        decoded_audio = self.audio_decoder(audio_state.latent)
        logger.info("[LTX distilled] Pipeline tensors ready in %.1fs", time.monotonic() - started_at)
        return decoded_video, decoded_audio


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
    prompt_encoder_staging_device = (
        torch.device(args.prompt_encoder_staging_device) if args.prompt_encoder_staging_device else None
    )
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        device=device,
        prompt_encoder_device=prompt_encoder_device,
        prompt_encoder_staging_device=prompt_encoder_staging_device,
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
