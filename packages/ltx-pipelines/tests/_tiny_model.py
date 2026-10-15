"""A tiny LTX audio-video model and inputs shaped like the pipeline's, for equivalence tests."""

import torch

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.model.transformer.rope import LTXRopeType

VIDEO_HEADS, VIDEO_HEAD_DIM = 2, 16
AUDIO_HEADS, AUDIO_HEAD_DIM = 2, 8
CHANNELS = 8

# Upstream defaults, and the options the LTX-2.5 distilled checkpoint enables (split RoPE,
# float64 RoPE grid, gated attention, AdaLN on the text cross-attention, no MLP bias, the
# keyframe absolute-position embedding).
CONFIGS = {
    "default": {},
    "ltx25": {
        "rope_type": LTXRopeType.SPLIT,
        "double_precision_rope": True,
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "ff_bias": False,
        "audio_ff_bias": False,
        "use_keyframes_abs_pos_embedding": True,
    },
}


def tiny_model(**options: object) -> X0Model:
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


def video_modality(frames: int, sigma: float, seed: int) -> Modality:
    gen = torch.Generator().manual_seed(seed)
    h, w = 2, 3
    tokens = frames * h * w
    grid = (
        torch.stack(torch.meshgrid(torch.arange(frames), torch.arange(h), torch.arange(w), indexing="ij"), dim=0)
        .reshape(3, -1)
        .float()
    )
    positions = torch.stack([grid, grid + 1], dim=-1).unsqueeze(0)  # (1, 3, T, 2)
    # (B, T, 1) like timesteps_from_mask(denoise_mask, sigma) in the pipeline.
    timesteps = torch.full((1, tokens, 1), sigma)
    timesteps[:, : h * w] = 0.0  # a conditioned first frame, like an image keyframe
    keyframes_mask = torch.zeros((1, tokens, 1))
    keyframes_mask[:, : h * w] = 1.0
    return Modality(
        latent=torch.randn(1, tokens, CHANNELS, generator=gen),
        sigma=torch.tensor([sigma]),
        timesteps=timesteps,
        positions=positions,
        context=torch.randn(1, 5, VIDEO_HEADS * VIDEO_HEAD_DIM, generator=gen),
        keyframes_mask=keyframes_mask,
    )


def audio_modality(tokens: int, sigma: float, seed: int) -> Modality:
    gen = torch.Generator().manual_seed(seed)
    grid = torch.arange(tokens).float().reshape(1, 1, -1)
    return Modality(
        latent=torch.randn(1, tokens, CHANNELS, generator=gen),
        sigma=torch.tensor([sigma]),
        timesteps=torch.full((1, tokens, 1), sigma),
        positions=torch.stack([grid, grid + 1], dim=-1),  # (1, 1, T, 2)
        context=torch.randn(1, 4, AUDIO_HEADS * AUDIO_HEAD_DIM, generator=gen),
    )
