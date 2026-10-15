"""The rewritten (in-place) transformer block must compute what the upstream block computed.

``_upstream_forward`` is ``BasicAVTransformerBlock.forward`` as released upstream (LTX-2
``be6d6d0``), kept verbatim apart from dedenting. It allocates a fresh tensor for every
residual update; the fork's block adds into the residual in place and frees AdaLN values as
soon as they are consumed. Both run with dense timesteps and full RoPE tables here.
"""

from dataclasses import replace

import pytest
import torch
from _tiny_model import CONFIGS, audio_modality, tiny_model, video_modality

from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm


def _upstream_forward(  # noqa: PLR0915
    self: BasicAVTransformerBlock,
    video: TransformerArgs | None,
    audio: TransformerArgs | None,
) -> tuple[TransformerArgs | None, TransformerArgs | None]:
    if video is None and audio is None:
        raise ValueError("At least one of video or audio must be provided")

    vx = video.x if video is not None else None
    ax = audio.x if audio is not None else None

    run_vx = video is not None and video.enabled and vx.numel() > 0
    run_ax = audio is not None and audio.enabled and ax.numel() > 0

    run_a2v = run_vx and (audio is not None and ax.numel() > 0)
    run_v2a = run_ax and (video is not None and vx.numel() > 0)

    if run_vx:
        vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
            self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
        )
        norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
        del vshift_msa, vscale_msa

        vx_msa_out = self.attn1(
            norm_vx,
            pe=video.positional_embeddings,
            mask=video.self_attention_mask,
            perturbation_mask=video.self_attn_perturbation_mask,
            all_perturbed=video.self_attn_all_perturbed,
        )
        vx = vx + vx_msa_out * vgate_msa
        vx_normed = rms_norm(vx, eps=self.norm_eps)
        del vgate_msa, norm_vx, vx_msa_out
        vx = vx + self._apply_text_cross_attention(
            vx_normed,
            video.context,
            self.attn2,
            self.scale_shift_table,
            getattr(self, "prompt_scale_shift_table", None),
            video.timesteps,
            video.prompt_timestep,
            video.context_mask,
            cross_attention_adaln=self.cross_attention_adaln,
        )
        del vx_normed

    if run_ax:
        ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
        )

        norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
        del ashift_msa, ascale_msa
        ax_msa_out = self.audio_attn1(
            norm_ax,
            pe=audio.positional_embeddings,
            mask=audio.self_attention_mask,
            perturbation_mask=audio.self_attn_perturbation_mask,
            all_perturbed=audio.self_attn_all_perturbed,
        )
        ax = ax + ax_msa_out * agate_msa
        ax_normed = rms_norm(ax, eps=self.norm_eps)
        del agate_msa, norm_ax, ax_msa_out
        ax = ax + self._apply_text_cross_attention(
            ax_normed,
            audio.context,
            self.audio_attn2,
            self.audio_scale_shift_table,
            getattr(self, "audio_prompt_scale_shift_table", None),
            audio.timesteps,
            audio.prompt_timestep,
            audio.context_mask,
            cross_attention_adaln=self.cross_attention_adaln,
        )
        del ax_normed

    # Audio - Video cross attention.
    if run_a2v or run_v2a:
        # Snapshot vx/ax before A2V mutates vx; V2A's video keys/values must
        # use the pre-A2V state so direction order doesn't bias the result.
        vx_pre_av = vx
        ax_pre_av = ax
        if run_a2v and not video.cross_attn_skip_all:
            scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
                slice(0, 2),
            )
            a2v_vx_scaled = rms_norm(vx_pre_av, eps=self.norm_eps) * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
            del scale_ca_video_a2v, shift_ca_video_a2v

            scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
                slice(0, 2),
            )
            a2v_ax_scaled = rms_norm(ax_pre_av, eps=self.norm_eps) * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
            del scale_ca_audio_a2v, shift_ca_audio_a2v
            vx = vx + (
                self.audio_to_video_attn(
                    a2v_vx_scaled,
                    context=a2v_ax_scaled,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                )
                * gate_out_a2v
                * video.cross_attn_perturbation_mask
            )
            del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled

        if run_v2a and not audio.cross_attn_skip_all:
            scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
                slice(2, 4),
            )
            v2a_ax_scaled = rms_norm(ax_pre_av, eps=self.norm_eps) * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
            del scale_ca_audio_v2a, shift_ca_audio_v2a
            scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
                slice(2, 4),
            )
            v2a_vx_scaled = rms_norm(vx_pre_av, eps=self.norm_eps) * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
            del scale_ca_video_v2a, shift_ca_video_v2a
            ax = ax + (
                self.video_to_audio_attn(
                    v2a_ax_scaled,
                    context=v2a_vx_scaled,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                )
                * gate_out_v2a
                * audio.cross_attn_perturbation_mask
            )
            del gate_out_v2a, v2a_vx_scaled, v2a_ax_scaled
        del vx_pre_av, ax_pre_av

    if run_vx:
        vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
            self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
        )
        vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
        vx = vx + self.ff(vx_scaled) * vgate_mlp

        del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

    if run_ax:
        ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
        )
        ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
        ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

    return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("with_audio", [True, False], ids=["audio_video", "video_only"])
def test_block_matches_upstream(config: str, with_audio: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LTX_BLOCK_CHUNK_TOKENS", raising=False)
    monkeypatch.setenv("LTX_COMPRESS_TIMESTEPS", "0")
    model = tiny_model(**CONFIGS[config])
    video = video_modality(frames=4, sigma=0.7, seed=11)
    audio = audio_modality(tokens=11, sigma=0.7, seed=12) if with_audio else None
    with torch.inference_mode():
        actual = model(video=video, audio=audio, perturbations=None)
        monkeypatch.setattr(BasicAVTransformerBlock, "forward", _upstream_forward)
        expected = model(video=video, audio=audio, perturbations=None)

    torch.testing.assert_close(actual[0], expected[0])
    if with_audio:
        torch.testing.assert_close(actual[1], expected[1])
