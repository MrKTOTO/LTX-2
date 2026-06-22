from dataclasses import dataclass, replace

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.adaln import adaln_embedding_coefficient
from ltx_core.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import CompressedTimestep, TransformerArgs
from ltx_core.utils import rms_norm


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )

        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor | CompressedTimestep, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        if isinstance(timestep, CompressedTimestep):
            values = timestep.values.view(timestep.values.shape[0], num_ada_params, -1)[:, indices, :]
            table = scale_shift_table[indices].to(device=values.device, dtype=values.dtype)
            return tuple(
                (values[:, value_idx, :] + table[value_idx]).index_select(0, timestep.indices).view(
                    timestep.batch_size,
                    timestep.token_count,
                    -1,
                )
                for value_idx in range(values.shape[1])
            )

        selected = timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        table = scale_shift_table[indices].to(device=timestep.device, dtype=timestep.dtype)
        return tuple(selected[:, :, value_idx, :] + table[value_idx] for value_idx in range(selected.shape[2]))

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, scale_shift_indices
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
        (gate,) = (t.squeeze(2) for t in gate_ada_values)

        return scale, shift, gate

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation."""
        if cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
                self.norm_eps,
            )
        return attn(rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask)

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        batch_size = (video or audio).x.shape[0]

        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            # Modulate first with only shift/scale live, then allocate the gate
            # right before it is consumed. This keeps at most 4xN*dim activations
            # resident here instead of 6xN*dim (shift+scale+gate plus the `1+scale`
            # temporary), which is the peak that OOMs long single-pass scenes.
            vshift_msa, vscale_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 2)
            )
            norm_vx = rms_norm(vx, eps=self.norm_eps)
            vscale_msa.add_(1)
            norm_vx.mul_(vscale_msa)
            norm_vx.add_(vshift_msa)
            del vshift_msa, vscale_msa

            all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
            none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
            v_mask = (
                perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                if not all_perturbed and not none_perturbed
                else None
            )
            (vgate_msa,) = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(2, 3)
            )
            self.attn1.add_to_residual(
                vx,
                norm_vx,
                outer_gate=vgate_msa,
                pe=video.positional_embeddings,
                mask=video.self_attention_mask,
                perturbation_mask=v_mask,
                all_perturbed=all_perturbed,
            )
            del vgate_msa, norm_vx, v_mask
            apply_text_cross_attention_to_residual(
                vx,
                vx,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
                norm_eps=self.norm_eps,
            )

        if run_ax:
            ashift_msa, ascale_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 2)
            )

            norm_ax = rms_norm(ax, eps=self.norm_eps)
            ascale_msa.add_(1)
            norm_ax.mul_(ascale_msa)
            norm_ax.add_(ashift_msa)
            del ashift_msa, ascale_msa
            all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx)
            none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx)
            a_mask = (
                perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                if not all_perturbed and not none_perturbed
                else None
            )
            (agate_msa,) = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(2, 3)
            )
            self.audio_attn1.add_to_residual(
                ax,
                norm_ax,
                outer_gate=agate_msa,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
                perturbation_mask=a_mask,
                all_perturbed=all_perturbed,
            )
            del agate_msa, norm_ax, a_mask
            apply_text_cross_attention_to_residual(
                ax,
                ax,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
                norm_eps=self.norm_eps,
            )

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            if run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx):
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                scale_ca_video_a2v.add_(1)
                vx_scaled = vx_norm3 * scale_ca_video_a2v
                vx_scaled.add_(shift_ca_video_a2v)
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                scale_ca_audio_a2v.add_(1)
                ax_scaled = ax_norm3 * scale_ca_audio_a2v
                ax_scaled.add_(shift_ca_audio_a2v)
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * a2v_mask
                )
                del gate_out_a2v, a2v_mask, vx_scaled, ax_scaled

            if run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx):
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                scale_ca_audio_v2a.add_(1)
                ax_scaled = ax_norm3 * scale_ca_audio_v2a
                ax_scaled.add_(shift_ca_audio_v2a)
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                scale_ca_video_v2a.add_(1)
                vx_scaled = vx_norm3 * scale_ca_video_v2a
                vx_scaled.add_(shift_ca_video_v2a)
                del scale_ca_video_v2a, shift_ca_video_v2a
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                    * v2a_mask
                )
                del gate_out_v2a, v2a_mask, ax_scaled, vx_scaled

            del vx_norm3, ax_norm3

        if run_vx:
            vshift_mlp, vscale_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 5)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps)
            vscale_mlp.add_(1)
            vx_scaled.mul_(vscale_mlp)
            vx_scaled.add_(vshift_mlp)
            del vshift_mlp, vscale_mlp
            (vgate_mlp,) = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(5, 6)
            )
            self.ff.add_to_residual(vx, vx_scaled, gate=vgate_mlp)

            del vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 5)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps)
            ascale_mlp.add_(1)
            ax_scaled.mul_(ascale_mlp)
            ax_scaled.add_(ashift_mlp)
            del ashift_mlp, ascale_mlp
            (agate_mlp,) = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(5, 6)
            )
            self.audio_ff.add_to_residual(ax, ax_scaled, gate=agate_mlp)

            del agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor | CompressedTimestep,
    context_mask: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    batch_size = x.shape[0]
    prompt_timestep = expand_timestep(prompt_timestep)
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = rms_norm(x, eps=norm_eps)
    q_scale.add_(1)
    attn_input.mul_(q_scale)
    attn_input.add_(q_shift)
    encoder_hidden_states = context * (1 + scale_kv)
    encoder_hidden_states.add_(shift_kv)
    out = attn(attn_input, context=encoder_hidden_states, mask=context_mask)
    out.mul_(q_gate)
    return out


def apply_text_cross_attention_to_residual(
    residual: torch.Tensor,
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    scale_shift_table: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor | None,
    timestep: torch.Tensor | CompressedTimestep,
    prompt_timestep: torch.Tensor | CompressedTimestep | None,
    context_mask: torch.Tensor | None = None,
    cross_attention_adaln: bool = False,
    norm_eps: float = 1e-6,
) -> None:
    """Apply text cross-attention and add it into residual without a full output copy."""
    if cross_attention_adaln:
        shift_q, scale_q, gate = BasicAVTransformerBlock.get_ada_values(
            None, scale_shift_table, x.shape[0], timestep, slice(6, 9)
        )
        batch_size = x.shape[0]
        prompt_timestep = expand_timestep(prompt_timestep)
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        attn_input = rms_norm(x, eps=norm_eps)
        scale_q.add_(1)
        attn_input.mul_(scale_q)
        attn_input.add_(shift_q)
        encoder_hidden_states = context * (1 + scale_kv)
        encoder_hidden_states.add_(shift_kv)
        attn.add_to_residual(residual, attn_input, outer_gate=gate, context=encoder_hidden_states, mask=context_mask)
        return

    attn.add_to_residual(residual, rms_norm(x, eps=norm_eps), context=context, mask=context_mask)


def expand_timestep(timestep: torch.Tensor | CompressedTimestep | None) -> torch.Tensor | None:
    if isinstance(timestep, CompressedTimestep):
        return timestep.expand()
    return timestep
