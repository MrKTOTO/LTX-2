"""Scene-batched denoising must match per-scene X0Model calls exactly."""

import pytest
import torch

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_pipelines.utils.scene_batch import scene_batched_denoise

VIDEO_HEADS, VIDEO_HEAD_DIM = 2, 16
AUDIO_HEADS, AUDIO_HEAD_DIM = 2, 8
CHANNELS = 8


def _model() -> X0Model:
    torch.manual_seed(0)
    model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=VIDEO_HEADS,
        attention_head_dim=VIDEO_HEAD_DIM,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        num_layers=3,
        cross_attention_dim=VIDEO_HEADS * VIDEO_HEAD_DIM,
        audio_num_attention_heads=AUDIO_HEADS,
        audio_attention_head_dim=AUDIO_HEAD_DIM,
        audio_in_channels=CHANNELS,
        audio_out_channels=CHANNELS,
        audio_cross_attention_dim=AUDIO_HEADS * AUDIO_HEAD_DIM,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=0.05)
    return X0Model(model).eval()


def _video(frames: int, sigma: float, seed: int) -> Modality:
    gen = torch.Generator().manual_seed(seed)
    h, w = 2, 3
    tokens = frames * h * w
    grid = torch.stack(
        torch.meshgrid(torch.arange(frames), torch.arange(h), torch.arange(w), indexing="ij"), dim=0
    ).reshape(3, -1).float()
    positions = torch.stack([grid, grid + 1], dim=-1).unsqueeze(0)  # (1, 3, T, 2)
    # (B, T, 1) like timesteps_from_mask(denoise_mask, sigma) in the pipeline.
    timesteps = torch.full((1, tokens, 1), sigma)
    timesteps[:, : h * w] = 0.0  # a conditioned first frame, like an image keyframe
    return Modality(
        latent=torch.randn(1, tokens, CHANNELS, generator=gen),
        sigma=torch.tensor([sigma]),
        timesteps=timesteps,
        positions=positions,
        context=torch.randn(1, 5, VIDEO_HEADS * VIDEO_HEAD_DIM, generator=gen),
    )


def _audio(tokens: int, sigma: float, seed: int) -> Modality:
    gen = torch.Generator().manual_seed(seed)
    grid = torch.arange(tokens).float().reshape(1, 1, -1)
    return Modality(
        latent=torch.randn(1, tokens, CHANNELS, generator=gen),
        sigma=torch.tensor([sigma]),
        timesteps=torch.full((1, tokens, 1), sigma),
        positions=torch.stack([grid, grid + 1], dim=-1),  # (1, 1, T, 2)
        context=torch.randn(1, 4, AUDIO_HEADS * AUDIO_HEAD_DIM, generator=gen),
    )


class _Windowed(torch.nn.Module):
    """Stands in for BlockStreamingWrapper: same model, limited resident block window."""

    def __init__(self, model: X0Model, window: int) -> None:
        super().__init__()
        self.inner = model
        self.window = window

    @property
    def velocity_model(self) -> LTXModel:
        return self.inner.velocity_model

    def resident_block_window(self) -> int:
        return self.window


@pytest.mark.parametrize("window", [1, 2, 3])
@pytest.mark.parametrize("budget", [0, 1 << 30])
def test_batched_matches_per_scene_calls(window: int, budget: int) -> None:
    model = _model()
    scenes = [
        (_video(frames=2, sigma=0.9, seed=1), _audio(tokens=7, sigma=0.9, seed=2)),
        (_video(frames=3, sigma=0.5, seed=3), _audio(tokens=11, sigma=0.5, seed=4)),
        (_video(frames=4, sigma=0.7, seed=5), None),  # Stage 2 shape: video only
    ]
    with torch.inference_mode():
        expected = [model(video=video, audio=audio, perturbations=None) for video, audio in scenes]
        actual = scene_batched_denoise(_Windowed(model, window), scenes, torch.device("cpu"), budget)

    assert len(actual) == len(expected)
    for (exp_video, exp_audio), (got_video, got_audio) in zip(expected, actual, strict=True):
        torch.testing.assert_close(got_video, exp_video, rtol=0, atol=0)
        if exp_audio is None:
            assert got_audio is None
        else:
            torch.testing.assert_close(got_audio, exp_audio, rtol=0, atol=0)


def test_plain_model_runs_all_blocks_per_visit() -> None:
    model = _model()
    scenes = [(_video(frames=2, sigma=0.3, seed=7), _audio(tokens=5, sigma=0.3, seed=8))]
    with torch.inference_mode():
        expected = model(video=scenes[0][0], audio=scenes[0][1], perturbations=None)
        (actual,) = scene_batched_denoise(model, scenes, torch.device("cpu"))
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
