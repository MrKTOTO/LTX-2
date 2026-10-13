"""Scene-batched denoising: one pass of the transformer weights per step for many clips.

With block streaming on an 8 GB card the distilled transformer is limited by how fast
its weights reach the GPU, not by compute: Stage 1 takes about as long for a 5 s clip
as for a 17 s one. Generating a video's scenes one at a time therefore pays the full
weight traffic once per scene. Here each block is applied to every scene before the
next block is fetched, so the weights travel once per step for the whole batch.

Every scene is still computed on its own (batch size 1, its own token count) with the
same operations ``X0Model.forward`` runs, so each clip matches what the sequential
pipeline produces. Between blocks a scene's hidden state is parked: on the GPU while it
fits a budget, otherwise in host RAM. RoPE tables are rebuilt on every visit instead of
being kept for all scenes, because at Stage 2 they would cost hundreds of MB per clip
and rebuilding them takes milliseconds.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace

import torch

from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    wants_lazy_rope,
)
from ltx_core.types import LatentState
from ltx_core.utils import to_denoised
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.utils.samplers import _step_state

logger = logging.getLogger(__name__)

SceneModalities = tuple[Modality | None, Modality | None]
SceneStates = tuple[LatentState | None, LatentState | None]
SceneContexts = tuple[torch.Tensor | None, torch.Tensor | None]


def _pin_host_state() -> bool:
    # Pageable by default, like RamWeightSource: registering GBs of host memory with
    # CUDA fails on Windows long before RAM runs out.
    return os.environ.get("LTX_SCENE_STATE_PIN", "").strip() not in ("", "0", "false")


def _resident_block_window(transformer: torch.nn.Module, num_blocks: int) -> int:
    window = getattr(transformer, "resident_block_window", None)
    if callable(window):
        return max(1, min(num_blocks, int(window())))
    # Not streamed: every block is already resident.
    return num_blocks


def _rope(
    preprocessor: object,
    modality: Modality,
    cross_modality: Modality | None,
) -> tuple[object, object | None]:
    """Rebuild the RoPE tables ``preprocessor.prepare`` computes for *modality*."""
    simple = getattr(preprocessor, "simple_preprocessor", preprocessor)
    dtype = modality.latent.dtype
    pe = simple._prepare_positional_embeddings(
        positions=modality.positions,
        inner_dim=simple.inner_dim,
        max_pos=simple.max_pos,
        use_middle_indices_grid=simple.use_middle_indices_grid,
        num_attention_heads=simple.num_attention_heads,
        x_dtype=dtype,
        lazy=wants_lazy_rope(modality.positions.shape[2]),
    )
    cross_pe = None
    if isinstance(preprocessor, MultiModalTransformerArgsPreprocessor) and cross_modality is not None:
        cross_pe = simple._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=preprocessor.audio_cross_attention_dim,
            max_pos=[preprocessor.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=simple.num_attention_heads,
            x_dtype=dtype,
        )
    # The preprocessor caches tables by tensor address; across many scenes that
    # cache would keep every scene's tables alive at once.
    simple._positional_embeddings_cache.clear()
    return pe, cross_pe


def _forget_rope(preprocessor: object | None) -> None:
    if preprocessor is None:
        return
    simple = getattr(preprocessor, "simple_preprocessor", preprocessor)
    simple._positional_embeddings_cache.clear()


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
            # Reuse the host buffers from the previous visit: allocating them per
            # block would dominate the transfer time.
            video = cls._to_host(video, previous.video.x if previous and previous.video else None)
            audio = cls._to_host(audio, previous.audio.x if previous and previous.audio else None)
        return cls(video=video, audio=audio, on_gpu=on_gpu)

    def x_on(self, device: torch.device) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        def move(args: TransformerArgs | None) -> TransformerArgs | None:
            if args is None or args.x.device == device:
                return args
            return replace(args, x=args.x.to(device, non_blocking=True))

        return move(self.video), move(self.audio)


def _x_bytes(args: TransformerArgs | None) -> int:
    return 0 if args is None else args.x.numel() * args.x.element_size()


def scene_batched_denoise(
    transformer: torch.nn.Module,
    scenes: list[SceneModalities],
    device: torch.device,
    gpu_state_budget_bytes: int = 0,
) -> list[tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Denoised ``(video, audio)`` for every scene, as ``X0Model(video, audio)`` returns.

    ``transformer`` is an ``X0Model`` or a ``BlockStreamingWrapper`` around one.
    Hidden states of the scenes that fit ``gpu_state_budget_bytes`` stay on the GPU
    between blocks; the rest wait in host RAM.
    """
    vm = transformer.velocity_model
    blocks = vm.transformer_blocks
    window = _resident_block_window(transformer, len(blocks))
    video_pre = getattr(vm, "video_args_preprocessor", None)
    audio_pre = getattr(vm, "audio_args_preprocessor", None)

    parked: list[_ParkedScene | None] = []
    budget_left = int(gpu_state_budget_bytes)
    for video, audio in scenes:
        video_args = video_pre.prepare(video, audio) if video is not None else None
        audio_args = audio_pre.prepare(audio, video) if audio is not None else None
        _forget_rope(video_pre)
        _forget_rope(audio_pre)
        size = _x_bytes(video_args) + _x_bytes(audio_args)
        on_gpu = size <= budget_left
        if on_gpu:
            budget_left -= size
        parked.append(_ParkedScene.park(video_args, audio_args, on_gpu))
        del video_args, audio_args

    for start in range(0, len(blocks), window):
        run = blocks[start : start + window]
        for idx, (video, audio) in enumerate(scenes):
            video_args, audio_args = parked[idx].x_on(device)
            if video_args is not None:
                pe, cross_pe = _rope(video_pre, video, audio)
                video_args = replace(video_args, positional_embeddings=pe, cross_positional_embeddings=cross_pe)
            if audio_args is not None:
                pe, cross_pe = _rope(audio_pre, audio, video)
                audio_args = replace(audio_args, positional_embeddings=pe, cross_positional_embeddings=cross_pe)
            for block in run:
                video_args, audio_args = block(video=video_args, audio=audio_args, perturbations=None)
            parked[idx] = _ParkedScene.park(video_args, audio_args, parked[idx].on_gpu, previous=parked[idx])
            del video_args, audio_args

    results = []
    for idx, (video, audio) in enumerate(scenes):
        video_args, audio_args = parked[idx].x_on(device)
        vx = (
            vm._process_output(
                vm.scale_shift_table, vm.norm_out, vm.proj_out, video_args.x, video_args.embedded_timestep
            )
            if video_args is not None
            else None
        )
        ax = (
            vm._process_output(
                vm.audio_scale_shift_table,
                vm.audio_norm_out,
                vm.audio_proj_out,
                audio_args.x,
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


def scene_batched_euler_loop(
    transformer: torch.nn.Module,
    sigmas: torch.Tensor,
    states: list[SceneStates],
    contexts: list[SceneContexts],
    stepper: DiffusionStepProtocol,
    device: torch.device,
    gpu_state_budget_bytes: int = 0,
) -> list[SceneStates]:
    """``euler_denoising_loop`` with ``SimpleDenoiser`` for several independent scenes."""
    if len(states) != len(contexts):
        raise ValueError(f"Got {len(states)} scene states but {len(contexts)} contexts")
    for step_idx in range(len(sigmas) - 1):
        sigma = sigmas[step_idx]
        scenes = [
            (
                modality_from_latent_state(video_state, video_ctx, sigma) if video_state is not None else None,
                modality_from_latent_state(audio_state, audio_ctx, sigma) if audio_state is not None else None,
            )
            for (video_state, audio_state), (video_ctx, audio_ctx) in zip(states, contexts, strict=True)
        ]
        denoised = scene_batched_denoise(transformer, scenes, device, gpu_state_budget_bytes)
        states = [
            (
                _step_state(video_state, denoised_video, stepper, sigmas, step_idx),
                _step_state(audio_state, denoised_audio, stepper, sigmas, step_idx),
            )
            for (video_state, audio_state), (denoised_video, denoised_audio) in zip(states, denoised, strict=True)
        ]
        del scenes, denoised
        logger.info("[LTX scene batch] step %s/%s done for %s scenes", step_idx + 1, len(sigmas) - 1, len(states))
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    return states
