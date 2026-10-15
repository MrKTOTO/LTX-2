"""Scene-batched denoising: one pass of the transformer weights per step for many clips.

With block streaming on an 8 GB card the distilled transformer is limited by how fast its
weights reach the GPU, not by compute: Stage 1 takes about as long for a 5 s clip as for a
17 s one. Generating a video's scenes one at a time therefore pays the full weight traffic
once per scene. Here each block is applied to every scene before the next block is
fetched, so the weights travel once per step for the whole batch.

Every scene is still computed on its own (batch size 1, its own token count) with the same
operations ``X0Model.forward`` runs, and is stepped by the same sampler rule
``DiffusionStage.__call__`` would use for it (plain or ancestral Euler, with the scene's
own noise generator), so each clip matches what the per-clip pipeline produces. Between
blocks a scene's hidden state is parked: on the GPU while it fits a budget, otherwise in
host RAM. RoPE tables are rebuilt on every visit instead of being kept for all scenes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial

import torch

from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep, EulerDiffusionStep
from ltx_core.components.noisers import Noiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape
from ltx_core.utils import to_denoised
from ltx_pipelines.utils.blocks import DiffusionStage, _build_state
from ltx_pipelines.utils.helpers import modality_from_latent_state, post_process_latent
from ltx_pipelines.utils.samplers import _get_plain_noise, _ModalityStep, _step_state
from ltx_pipelines.utils.types import DenoisedLatentResult, ModalitySpec, VideoAudio

logger = logging.getLogger(__name__)

SceneModalities = tuple[Modality | None, Modality | None]


def _pin_host_state() -> bool:
    # Pageable by default, like the RAM weight source: registering GBs of host memory
    # with CUDA fails on Windows long before RAM runs out.
    return os.environ.get("LTX_SCENE_STATE_PIN", "").strip().lower() not in ("", "0", "false")


def _resident_block_window(velocity_model: torch.nn.Module, num_blocks: int) -> int:
    window = getattr(velocity_model, "resident_block_window", None)
    if callable(window):
        return max(1, min(num_blocks, int(window())))
    # Not streamed: every block is already resident.
    return num_blocks


@dataclass
class _ParkedScene:
    """One scene's transformer arguments between blocks, without RoPE tables."""

    video: TransformerArgs | None
    audio: TransformerArgs | None
    on_gpu: bool

    @staticmethod
    def _strip(args: TransformerArgs | None) -> TransformerArgs | None:
        if args is None:
            return None
        return replace(args, positional_embeddings=None, cross_positional_embeddings=None)

    @staticmethod
    def _to_host(args: TransformerArgs | None, host_x: torch.Tensor | None) -> TransformerArgs | None:
        if args is None:
            return None
        if host_x is None:
            host_x = torch.empty(args.x.shape, dtype=args.x.dtype, device="cpu", pin_memory=_pin_host_state())
        host_x.copy_(args.x)
        return replace(args, x=host_x)

    @classmethod
    def park(
        cls,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        on_gpu: bool,
        previous: _ParkedScene | None = None,
    ) -> _ParkedScene:
        video, audio = cls._strip(video), cls._strip(audio)
        if not on_gpu:
            # Reuse the host buffers from the previous visit: allocating them per block
            # would dominate the transfer time.
            video = cls._to_host(video, previous.video.x if previous and previous.video else None)
            audio = cls._to_host(audio, previous.audio.x if previous and previous.audio else None)
        return cls(video=video, audio=audio, on_gpu=on_gpu)

    def unpark(
        self, device: torch.device, scene: SceneModalities, video_pre: object, audio_pre: object
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """The scene's args on ``device`` with RoPE tables rebuilt exactly as ``prepare`` builds them."""
        video, audio = scene

        def restore(
            args: TransformerArgs | None, pre: object, modality: Modality, cross: Modality | None
        ) -> TransformerArgs | None:
            if args is None:
                return None
            x = args.x if args.x.device == device else args.x.to(device, non_blocking=True)
            pe, cross_pe = pre.positional_embeddings(modality, cross)
            return replace(args, x=x, positional_embeddings=pe, cross_positional_embeddings=cross_pe)

        return restore(self.video, video_pre, video, audio), restore(self.audio, audio_pre, audio, video)


def _memory_note(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    return (
        f" (GPU peak allocated {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GB, "
        f"reserved {torch.cuda.memory_reserved(device) / 2**30:.2f} GB)"
    )


def _x_bytes(args: TransformerArgs | None) -> int:
    return 0 if args is None else args.x.numel() * args.x.element_size()


def scene_batched_denoise(
    transformer: torch.nn.Module,
    scenes: Sequence[SceneModalities],
    device: torch.device,
    gpu_state_budget_bytes: int = 0,
) -> list[tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Denoised ``(video, audio)`` for every scene, as ``X0Model(video, audio, None)`` returns.
    ``transformer`` is an ``X0Model`` around an ``LTXModel`` or a ``BlockStreamingWrapper``.
    Hidden states of the scenes that fit ``gpu_state_budget_bytes`` stay on the GPU between
    blocks; the rest wait in host RAM.
    """
    vm = transformer.velocity_model
    blocks = vm.transformer_blocks
    window = _resident_block_window(vm, len(blocks))
    video_pre = getattr(vm, "video_args_preprocessor", None)
    audio_pre = getattr(vm, "audio_args_preprocessor", None)
    block_input_processor = vm.block_input_processor

    parked: list[_ParkedScene | None] = []
    budget_left = int(gpu_state_budget_bytes)
    perturbations = None
    for video, audio in scenes:
        video_args = video_pre.prepare(video, audio) if video is not None else None
        audio_args = audio_pre.prepare(audio, video) if audio is not None else None
        if perturbations is None:
            # Same as LTXModel.forward with perturbations=None: perturb nothing. Every scene
            # has batch size 1, so one config serves them all.
            ref = (video_args or audio_args).x
            perturbations = BatchedPerturbationConfig.empty(ref.shape[0], vm.num_blocks, ref.device, ref.dtype)
        size = _x_bytes(video_args) + _x_bytes(audio_args)
        on_gpu = size <= budget_left
        if on_gpu:
            budget_left -= size
        parked.append(_ParkedScene.park(video_args, audio_args, on_gpu))
        del video_args, audio_args

    for start in range(0, len(blocks), window):
        for idx, scene in enumerate(scenes):
            video_args, audio_args = parked[idx].unpark(device, scene, video_pre, audio_pre)
            for block_idx in range(start, min(start + window, len(blocks))):
                if video_args is not None:
                    video_args = block_input_processor(
                        video_args,
                        perturbations,
                        block_idx,
                        self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                        cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
                    )
                if audio_args is not None:
                    audio_args = block_input_processor(
                        audio_args,
                        perturbations,
                        block_idx,
                        self_attn_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                        cross_attn_type=PerturbationType.SKIP_V2A_CROSS_ATTN,
                    )
                video_args, audio_args = blocks[block_idx](video=video_args, audio=audio_args)
            parked[idx] = _ParkedScene.park(video_args, audio_args, parked[idx].on_gpu, previous=parked[idx])
            del video_args, audio_args
            if not parked[idx].on_gpu and device.type == "cuda":
                # Clips differ in token count, so the caching allocator's blocks for one rarely
                # fit the next and the reserve keeps growing; past the card's VRAM the WDDM
                # driver spills into shared memory instead of failing (a Stage 2 step that took
                # minutes never finished). Parking to host already waited for this visit, so
                # handing the cache back here costs little.
                torch.cuda.empty_cache()

    results = []
    for idx, (video, audio) in enumerate(scenes):
        video_args, audio_args = parked[idx].video, parked[idx].audio
        vx = (
            vm._process_output(
                vm.scale_shift_table, vm.norm_out, vm.proj_out, video_args.x.to(device), video_args.embedded_timestep
            )
            if video_args is not None
            else None
        )
        ax = (
            vm._process_output(
                vm.audio_scale_shift_table,
                vm.audio_norm_out,
                vm.audio_proj_out,
                audio_args.x.to(device),
                audio_args.embedded_timestep,
            )
            if audio_args is not None
            else None
        )
        parked[idx] = None  # release this clip's parked state before the next one
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        results.append((denoised_video, denoised_audio))
        del video_args, audio_args, vx, ax
    return results


# ---------------------------------------------------------------------------
# Per-scene sampler rules
# ---------------------------------------------------------------------------


class EulerSceneSampler:
    """The step ``euler_denoising_loop`` takes."""

    def __init__(self) -> None:
        self._stepper = EulerDiffusionStep()

    def step(
        self,
        states: VideoAudio[LatentState],
        denoised: tuple[torch.Tensor | None, torch.Tensor | None],
        sigmas: torch.Tensor,
        step_idx: int,
    ) -> VideoAudio[LatentState]:
        video_state, audio_state = states
        return VideoAudio(
            _step_state(video_state, denoised[0], self._stepper, sigmas, step_idx),
            _step_state(audio_state, denoised[1], self._stepper, sigmas, step_idx),
        )


class AncestralSceneSampler:
    """The step ``euler_ancestral_denoising_loop`` takes, with this scene's own noise generator.
    ``noise_seed`` is what the per-clip pipeline would pass for this clip and pass
    (``seed + ANCESTRAL_*_NOISE_SEED_OFFSET``); noise is drawn video first, then audio.
    """

    def __init__(self, noise_seed: int, model_dtype: torch.dtype, eta: float = 1.0, s_noise: float = 1.0) -> None:
        self._noise_seed = noise_seed
        self._model_dtype = model_dtype
        self._eta = eta
        self._draw_noise = eta > 0
        stepper = EulerAncestralDiffusionStep()
        self._make_step_fn = lambda _result, _modality: partial(stepper.step, eta=eta, s_noise=s_noise)
        self._generator: torch.Generator | None = None

    def step(
        self,
        states: VideoAudio[LatentState],
        denoised: tuple[torch.Tensor | None, torch.Tensor | None],
        sigmas: torch.Tensor,
        step_idx: int,
    ) -> VideoAudio[LatentState]:
        video_state, audio_state = states
        current = {"video": video_state, "audio": audio_state}
        if self._generator is None:
            present = current["video"] if current["video"] is not None else current["audio"]
            self._generator = torch.Generator(device=present.latent.device).manual_seed(self._noise_seed)
        pending = [
            _ModalityStep.from_modality_result(
                modality, state, DenoisedLatentResult.result_or_none(denoised=result), self._make_step_fn
            )
            for modality, state, result in (
                ("video", current["video"], denoised[0]),
                ("audio", current["audio"], denoised[1]),
            )
            if state is not None
        ]
        terminal_step = sigmas[step_idx + 1] == 0
        for item in pending:
            if item.denoised is None:
                continue
            if terminal_step:
                current[item.modality] = replace(item.state, latent=item.denoised.to(self._model_dtype))
                continue
            noise = _get_plain_noise(item.state.latent, self._generator) if self._draw_noise else None
            x_next = item.step_fn(
                sample=item.state.latent.float(),
                denoised_sample=item.denoised,
                sigmas=sigmas,
                step_index=step_idx,
                noise=noise,
            )
            if self._draw_noise:
                x_next = post_process_latent(x_next, item.state.denoise_mask, item.state.clean_latent)
            current[item.modality] = replace(item.state, latent=x_next.to(self._model_dtype))
        return VideoAudio(current["video"], current["audio"])


SceneSampler = EulerSceneSampler | AncestralSceneSampler


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


@dataclass
class SceneStageInput:
    """One clip's inputs to a scene-batched stage: what ``DiffusionStage.__call__`` takes for it."""

    modalities: VideoAudio[ModalitySpec]
    noiser: Noiser
    sampler: SceneSampler


def scene_batched_stage(
    stage: DiffusionStage,
    scenes: Sequence[SceneStageInput],
    sigmas: torch.Tensor,
    gpu_state_budget_bytes: int = 0,
) -> list[VideoAudio[LatentState]]:
    """``stage(SimpleDenoiser(...), sigmas, noiser, modalities, loop=...)`` for every clip, with one build.
    Returns, per clip, the unpatchified states with cleared conditionings, as the stage does.
    """
    dtype, device = stage.dtype, stage.device
    for scene in scenes:
        if scene.modalities.video is not None:
            stage._assert_supports_conditionings(scene.modalities.video.conditionings)

    tools = []
    for scene in scenes:
        video, audio = scene.modalities
        video_tools = (
            VideoLatentTools(
                VideoLatentPatchifier(patch_size=1),
                VideoLatentShape.from_torch_shape(video.latent.shape),
                video.conditioning_fps,
                scale_factors=stage.video_scale_factors,
            )
            if video is not None
            else None
        )
        audio_tools = (
            AudioLatentTools(AudioPatchifier(patch_size=1), AudioLatentShape.from_torch_shape(audio.latent.shape))
            if audio is not None
            else None
        )
        tools.append((video_tools, audio_tools))

    logger.info("[LTX scene batch] Building transformer for %s clips (%s steps)", len(scenes), len(sigmas) - 1)
    with stage.model_context() as transformer:
        states = []
        for scene, (video_tools, audio_tools) in zip(scenes, tools, strict=True):
            video, audio = scene.modalities
            video_state = _build_state(video, video_tools, scene.noiser, dtype, device) if video is not None else None
            audio_state = _build_state(audio, audio_tools, scene.noiser, dtype, device) if audio is not None else None
            states.append(VideoAudio(video_state, audio_state))

        for step_idx in range(len(sigmas) - 1):
            sigma = sigmas[step_idx]
            step_inputs = []
            for scene, (video_state, audio_state) in zip(scenes, states, strict=True):
                video, audio = scene.modalities
                step_inputs.append(
                    (
                        modality_from_latent_state(
                            video_state, video.context, sigma.expand(video_state.latent.shape[0])
                        )
                        if video_state is not None
                        else None,
                        modality_from_latent_state(
                            audio_state, audio.context, sigma.expand(audio_state.latent.shape[0])
                        )
                        if audio_state is not None
                        else None,
                    )
                )
            denoised = scene_batched_denoise(transformer, step_inputs, device, gpu_state_budget_bytes)
            del step_inputs
            states = [
                scene.sampler.step(state, result, sigmas, step_idx)
                for scene, state, result in zip(scenes, states, denoised, strict=True)
            ]
            del denoised
            logger.info(
                "[LTX scene batch] step %s/%s done for %s clips%s",
                step_idx + 1,
                len(sigmas) - 1,
                len(states),
                _memory_note(device),
            )
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    results = []
    for (video_state, audio_state), (video_tools, audio_tools) in zip(states, tools, strict=True):
        results.append(
            VideoAudio(
                video_tools.unpatchify(video_tools.clear_conditioning(video_state))
                if video_state is not None
                else None,
                audio_tools.unpatchify(audio_tools.clear_conditioning(audio_state))
                if audio_state is not None
                else None,
            )
        )
    return results
