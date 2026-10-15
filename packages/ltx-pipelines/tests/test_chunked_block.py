"""The memory-saving transformer paths must compute what the plain ones do.

- token-chunked block (``LTX_BLOCK_CHUNK_TOKENS``), including the chunked audio-video
  cross-attention it uses whenever audio is present;
- AdaLN values stored per distinct timestep (``LTX_COMPRESS_TIMESTEPS``);
- lazily built RoPE rows.
"""

import pytest
import torch
from _tiny_model import CONFIGS, VIDEO_HEAD_DIM, VIDEO_HEADS, audio_modality, tiny_model, video_modality

from ltx_core.model.transformer.rope import (
    LazyRope,
    LTXRopeType,
    generate_freq_grid_np,
    precompute_freqs_cis,
    rope_rows,
)
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

CHUNK = "5"  # does not divide the token counts below, so the last chunk is ragged


@pytest.mark.parametrize("rope_type", [LTXRopeType.INTERLEAVED, LTXRopeType.SPLIT])
def test_lazy_rope_rows_equal_full_table_slices(rope_type: LTXRopeType) -> None:
    positions = video_modality(frames=4, sigma=0.5, seed=0).positions
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
@pytest.mark.parametrize("with_audio", [True, False], ids=["audio_video", "video_only"])
def test_chunked_block_matches_unchunked(config: str, with_audio: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tiny_model(**CONFIGS[config])
    video = video_modality(frames=4, sigma=0.7, seed=1)
    audio = audio_modality(tokens=11, sigma=0.7, seed=2) if with_audio else None
    chunked_calls = []
    original = BasicAVTransformerBlock._chunked_video_attention

    def spy(self: BasicAVTransformerBlock, *args: object, **kwargs: object) -> None:
        chunked_calls.append(id(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BasicAVTransformerBlock, "_chunked_video_attention", spy)
    with torch.inference_mode():
        monkeypatch.delenv("LTX_BLOCK_CHUNK_TOKENS", raising=False)
        expected = model(video=video, audio=audio, perturbations=None)
        assert chunked_calls == []
        monkeypatch.setenv("LTX_BLOCK_CHUNK_TOKENS", CHUNK)
        actual = model(video=video, audio=audio, perturbations=None)
        assert len(chunked_calls) == 3

    # Projections run on chunks instead of the whole sequence, so BLAS may round
    # differently; every op is per token, so nothing beyond rounding may change.
    torch.testing.assert_close(actual[0], expected[0])
    if with_audio:
        torch.testing.assert_close(actual[1], expected[1])
    else:
        assert actual[1] is None


@pytest.mark.parametrize("config", CONFIGS)
def test_compressed_timesteps_match_dense(config: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tiny_model(**CONFIGS[config])
    video = video_modality(frames=3, sigma=0.6, seed=3)
    audio = audio_modality(tokens=9, sigma=0.6, seed=4)
    with torch.inference_mode():
        monkeypatch.setenv("LTX_COMPRESS_TIMESTEPS", "0")
        expected = model(video=video, audio=audio, perturbations=None)
        monkeypatch.setenv("LTX_COMPRESS_TIMESTEPS", "1")
        actual = model(video=video, audio=audio, perturbations=None)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_block_does_not_touch_inputs_under_autograd() -> None:
    model = tiny_model(**CONFIGS["ltx25"])
    video = video_modality(frames=2, sigma=0.5, seed=5)
    latent = video.latent.clone()
    model(video=video, audio=None, perturbations=None)
    torch.testing.assert_close(video.latent, latent, rtol=0, atol=0)
