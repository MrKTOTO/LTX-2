import math

import torch
from einops import rearrange
from torch import nn

# ---------------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------------


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Normalize and flatten multi-layer hidden states, respecting padding.
    Performs per-batch, per-layer normalization using masked mean and range,
    then concatenates across the layer dimension.
    Args:
        encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
        sequence_lengths: Number of valid (non-padded) tokens per batch item.
        padding_side: Whether padding is on "left" or "right".
    Returns:
        Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
        with padded positions zeroed out.
    """
    b, t, d, l = encoded_text.shape  # noqa: E741
    device = encoded_text.device

    token_indices = torch.arange(t, device=device)[None, :]

    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = t - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")

    eps = 1e-6

    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    normed = 8 * (encoded_text - mean) / (range_ + eps)
    normed = normed.reshape(b, t, -1)

    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    normed = normed.masked_fill(~mask_flattened, 0.0)

    return normed


def norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMSNorm normalization for V2 models.
    Args:
        encoded_text: [B, T, D, L]
        attention_mask: [B, T] binary mask
    Returns:
        [B, T, D*L] normalized tensor with padding zeroed out.
    """
    B, T, D, L = encoded_text.shape  # noqa: N806
    rms_denom = torch.linalg.vector_norm(encoded_text, ord=2, dim=2, keepdim=True)
    rms_denom = rms_denom * (D**-0.5)
    normed = encoded_text / (rms_denom + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
    normed.masked_fill_(~mask_3d, 0)
    return normed


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    """Rescale normalization: x * sqrt(target_dim / source_dim)."""
    return x * math.sqrt(target_dim / source_dim)


def _rescale_norm_(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    """In-place variant used in low-VRAM inference paths."""
    return x.mul_(math.sqrt(target_dim / source_dim))


# ---------------------------------------------------------------------------
# Feature extractor variants
# ---------------------------------------------------------------------------


class FeatureExtractorV1(nn.Module):
    """19B: per-segment norm -> aggregate_embed -> 3840"""

    def __init__(self, aggregate_embed: nn.Module, is_av: bool = False):
        super().__init__()
        self.aggregate_embed = aggregate_embed
        self.is_av = is_av

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = "left"
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        dtype = encoded.dtype
        sequence_lengths = attention_mask.sum(dim=-1)
        normed = _norm_and_concat_padded_batch(encoded, sequence_lengths, padding_side)
        features = self.aggregate_embed(normed.to(dtype))
        if self.is_av:
            return features, features
        return features, None


class FeatureExtractorV2(nn.Module):
    """22B: per-token RMS norm → rescale → dual aggregate embeds"""

    def __init__(
        self,
        video_aggregate_embed: nn.Linear,
        embedding_dim: int,
        audio_aggregate_embed: nn.Linear | None = None,
    ):
        super().__init__()
        self.video_aggregate_embed = video_aggregate_embed
        self.audio_aggregate_embed = audio_aggregate_embed
        self.embedding_dim = embedding_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        normed = norm_and_concat_per_token_rms(encoded, attention_mask)
        normed = normed.to(encoded.dtype)
        v_dim = self.video_aggregate_embed.out_features
        _rescale_norm_(normed, v_dim, self.embedding_dim)
        video = self.video_aggregate_embed(normed)
        audio = None
        if self.audio_aggregate_embed is not None:
            a_dim = self.audio_aggregate_embed.out_features
            if a_dim != v_dim:
                normed.mul_(math.sqrt(a_dim / v_dim))
            audio = self.audio_aggregate_embed(normed)
        return video, audio
