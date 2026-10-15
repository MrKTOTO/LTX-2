"""Scene-batched denoising must match the per-clip path exactly."""

import pytest
import torch
from _tiny_model import (
    AUDIO_HEAD_DIM,
    AUDIO_HEADS,
    CHANNELS,
    CONFIGS,
    VIDEO_HEAD_DIM,
    VIDEO_HEADS,
    audio_modality,
    tiny_model,
    video_modality,
)

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.model.transformer.model import LTXModel
from ltx_pipelines.distilled import ancestral_sampler_kwargs
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.scene_batch import (
    AncestralSceneSampler,
    EulerSceneSampler,
    SceneStageInput,
    scene_batched_denoise,
    scene_batched_stage,
)
from ltx_pipelines.utils.types import ModalitySpec, VideoAudio

CPU = torch.device("cpu")


class _Windowed(torch.nn.Module):
    """Stands in for BlockStreamingWrapper: same model, limited resident block window."""

    def __init__(self, model: LTXModel, window: int) -> None:
        super().__init__()
        self.inner = model
        self.window = window

    def resident_block_window(self) -> int:
        return self.window

    def __getattr__(self, name: str) -> object:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


class _WindowedX0(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, window: int) -> None:
        super().__init__()
        self.velocity_model = _Windowed(model.velocity_model, window)


def _scenes() -> list:
    return [
        (video_modality(frames=2, sigma=0.9, seed=1), audio_modality(tokens=7, sigma=0.9, seed=2)),
        (video_modality(frames=3, sigma=0.5, seed=3), audio_modality(tokens=11, sigma=0.5, seed=4)),
        (video_modality(frames=4, sigma=0.7, seed=5), None),
    ]


@pytest.mark.parametrize("chunk", ["", "5"], ids=["plain_block", "chunked_block"])
@pytest.mark.parametrize("window", [1, 2, 3])
@pytest.mark.parametrize("budget", [0, 1 << 30], ids=["host_parked", "gpu_parked"])
def test_batched_denoise_matches_per_scene_calls(
    chunk: str, window: int, budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LTX_BLOCK_CHUNK_TOKENS", chunk)
    model = tiny_model(**CONFIGS["ltx25"])
    scenes = _scenes()
    with torch.inference_mode():
        expected = [model(video=video, audio=audio, perturbations=None) for video, audio in scenes]
        actual = scene_batched_denoise(_WindowedX0(model, window), scenes, CPU, budget)

    assert len(actual) == len(expected)
    for (exp_video, exp_audio), (got_video, got_audio) in zip(expected, actual, strict=True):
        torch.testing.assert_close(got_video, exp_video, rtol=0, atol=0)
        if exp_audio is None:
            assert got_audio is None
        else:
            torch.testing.assert_close(got_audio, exp_audio, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Whole stage vs DiffusionStage.__call__
# ---------------------------------------------------------------------------


class _TinyBuilder:
    """The parts of the builder protocol DiffusionStage touches, building the tiny model."""

    checkpoint = "tiny"
    model_sd_ops = None
    module_ops = ()
    loras = ()
    keeps_gpu_resident_weights = False

    def with_module_ops(self, _ops: object) -> "_TinyBuilder":
        return self

    def with_sd_ops(self, _ops: object) -> "_TinyBuilder":
        return self

    def with_loras(self, _loras: object) -> "_TinyBuilder":
        return self

    def model_config(self) -> dict:
        return {"transformer": {"use_keyframes_abs_pos_embedding": True}}

    def build(self, device: torch.device | None = None, **_kwargs: object) -> LTXModel:
        return tiny_model(**CONFIGS["ltx25"]).velocity_model.to(device or CPU)


SIGMAS = torch.tensor([1.0, 0.725, 0.4219, 0.0])


def _clip_specs(
    frames: int, audio_frames: int, seed: int, with_audio: bool, noise_scale: float = 1.0
) -> VideoAudio[ModalitySpec]:
    gen = torch.Generator().manual_seed(seed)
    video_latent = torch.randn(1, CHANNELS, frames, 2, 3, generator=gen)
    video_context = torch.randn(1, 5, VIDEO_HEADS * VIDEO_HEAD_DIM, generator=gen)
    first_frame = torch.randn(1, CHANNELS, 1, 2, 3, generator=gen)
    video = ModalitySpec(
        latent=video_latent,
        conditioning_fps=24.0,
        context=video_context,
        conditionings=[VideoConditionByLatentIndex(latent=first_frame, strength=1.0, latent_idx=0)],
        noise_scale=noise_scale,
    )
    audio = None
    if with_audio:
        audio = ModalitySpec(
            latent=torch.randn(1, CHANNELS, audio_frames, 1, generator=gen),
            conditioning_fps=24.0,
            context=torch.randn(1, 4, AUDIO_HEADS * AUDIO_HEAD_DIM, generator=gen),
            noise_scale=noise_scale,
        )
    return VideoAudio(video, audio)


@pytest.mark.parametrize("sampler", ["euler", "ancestral"])
@pytest.mark.parametrize("chunk", ["", "5"], ids=["plain_block", "chunked_block"])
def test_batched_stage_matches_per_clip_stage(sampler: str, chunk: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LTX_BLOCK_CHUNK_TOKENS", chunk)
    stage = DiffusionStage(_TinyBuilder(), torch.float32, CPU)
    clips = [
        (_clip_specs(frames=2, audio_frames=7, seed=21, with_audio=True), 101),
        (_clip_specs(frames=4, audio_frames=11, seed=22, with_audio=True, noise_scale=0.725), 102),
        (_clip_specs(frames=3, audio_frames=0, seed=23, with_audio=False), 103),
    ]

    def make_sampler(seed: int) -> EulerSceneSampler | AncestralSceneSampler:
        if sampler == "euler":
            return EulerSceneSampler()
        return AncestralSceneSampler(noise_seed=seed + 10000, model_dtype=torch.float32)

    with torch.inference_mode():
        expected = []
        for modalities, seed in clips:
            loop_kwargs = {} if sampler == "euler" else ancestral_sampler_kwargs(seed, torch.float32, 10000)
            expected.append(
                stage(
                    denoiser=SimpleDenoiser(
                        modalities.video.context, modalities.audio.context if modalities.audio else None
                    ),
                    sigmas=SIGMAS,
                    noiser=GaussianNoiser(generator=torch.Generator().manual_seed(seed)),
                    modalities=modalities,
                    **loop_kwargs,
                )
            )
        actual = scene_batched_stage(
            stage,
            [
                SceneStageInput(
                    modalities=modalities,
                    noiser=GaussianNoiser(generator=torch.Generator().manual_seed(seed)),
                    sampler=make_sampler(seed),
                )
                for modalities, seed in clips
            ],
            SIGMAS,
        )

    for (exp_video, exp_audio), (got_video, got_audio) in zip(expected, actual, strict=True):
        torch.testing.assert_close(got_video.latent, exp_video.latent, rtol=0, atol=0)
        if exp_audio is None:
            assert got_audio is None
        else:
            torch.testing.assert_close(got_audio.latent, exp_audio.latent, rtol=0, atol=0)


def test_ancestral_sampler_draws_from_its_own_generator() -> None:
    """Two clips with different seeds get different noise even with identical inputs."""
    stage = DiffusionStage(_TinyBuilder(), torch.float32, CPU)
    modalities = _clip_specs(frames=2, audio_frames=5, seed=31, with_audio=False)
    with torch.inference_mode():
        first, second = scene_batched_stage(
            stage,
            [
                SceneStageInput(
                    modalities=modalities,
                    noiser=GaussianNoiser(generator=torch.Generator().manual_seed(5)),
                    sampler=AncestralSceneSampler(noise_seed=seed, model_dtype=torch.float32),
                )
                for seed in (1, 2)
            ],
            SIGMAS,
        )
    assert not torch.equal(first.video.latent, second.video.latent)
