"""The token-chunked transformer block (LTX_BLOCK_CHUNK_TOKENS) must match the unchunked one."""

import pytest
import torch

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.model.transformer.rope import (
    LazyRope,
    LTXRopeType,
    generate_freq_grid_np,
    precompute_freqs_cis,
    rope_rows,
)
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_pipelines.utils.scene_batch import scene_batched_denoise

VIDEO_HEADS, VIDEO_HEAD_DIM = 2, 16
AUDIO_HEADS, AUDIO_HEAD_DIM = 2, 8
CHANNELS = 8
CHUNK = "5"  # does not divide the token counts below, so the last chunk is ragged

# Upstream defaults, and the LTX-2.3 options (split RoPE, gated attention, AdaLN on
# the text cross-attention) that the production checkpoint enables.
CONFIGS = {
    "default": {},
    "ltx23": {
        "rope_type": LTXRopeType.SPLIT,
        "double_precision_rope": True,
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
    },
}


def _model(**options: object) -> X0Model:
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
        **options,
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


@pytest.mark.parametrize("rope_type", [LTXRopeType.INTERLEAVED, LTXRopeType.SPLIT])
def test_lazy_rope_rows_equal_full_table_slices(rope_type: LTXRopeType) -> None:
    positions = _video(frames=4, sigma=0.5, seed=0).positions
    lazy = LazyRope(
        positions=positions,
        build=lambda pos: precompute_freqs_cis(
            pos,
            dim=VIDEO_HEADS * VIDEO_HEAD_DIM,
            out_dtype=torch.float32,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=VIDEO_HEADS,
            rope_type=rope_type,
            freq_grid_generator=generate_freq_grid_np,
        ),
    )
    full = lazy.full()
    for start, end in [(0, 5), (5, 10), (20, 24)]:
        for got, expected in zip(lazy.rows(start, end), rope_rows(full, start, end), strict=True):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("with_audio", [True, False], ids=["stage1_av", "stage2_video"])
def test_chunked_block_matches_unchunked(config: str, with_audio: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(**CONFIGS[config])
    video = _video(frames=4, sigma=0.7, seed=1)
    audio = _audio(tokens=11, sigma=0.7, seed=2) if with_audio else None
    chunked_calls = []
    original = BasicAVTransformerBlock._chunked_video_attention

    def spy(self, *args, **kwargs):
        chunked_calls.append(self.idx)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BasicAVTransformerBlock, "_chunked_video_attention", spy)
    with torch.inference_mode():
        monkeypatch.delenv("LTX_BLOCK_CHUNK_TOKENS", raising=False)
        expected = model(video=video, audio=audio, perturbations=None)
        assert chunked_calls == []
        monkeypatch.setenv("LTX_BLOCK_CHUNK_TOKENS", CHUNK)
        actual = model(video=video, audio=audio, perturbations=None)
        assert chunked_calls == [0, 1, 2]

    # Projections run on chunks instead of the whole sequence, so BLAS may round
    # differently; every op is per token, so nothing beyond rounding may change.
    torch.testing.assert_close(actual[0], expected[0])
    if with_audio:
        torch.testing.assert_close(actual[1], expected[1])
    else:
        assert actual[1] is None


@pytest.mark.parametrize("config", CONFIGS)
def test_scene_batch_with_chunking_matches_chunked_model(config: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LTX_BLOCK_CHUNK_TOKENS", CHUNK)
    model = _model(**CONFIGS[config])
    scenes = [
        (_video(frames=3, sigma=0.5, seed=3), _audio(tokens=11, sigma=0.5, seed=4)),
        (_video(frames=4, sigma=0.7, seed=5), None),
    ]
    with torch.inference_mode():
        expected = [model(video=video, audio=audio, perturbations=None) for video, audio in scenes]
        actual = scene_batched_denoise(model, scenes, torch.device("cpu"))

    for (exp_video, exp_audio), (got_video, got_audio) in zip(expected, actual, strict=True):
        torch.testing.assert_close(got_video, exp_video, rtol=0, atol=0)
        if exp_audio is None:
            assert got_audio is None
        else:
            torch.testing.assert_close(got_audio, exp_audio, rtol=0, atol=0)
