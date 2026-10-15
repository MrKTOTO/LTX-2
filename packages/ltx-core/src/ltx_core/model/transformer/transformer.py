from dataclasses import dataclass, replace

import torch

from ltx_core.model.transformer.adaln import adaln_embedding_coefficient
from ltx_core.model.transformer.attention import Attention, AttentionCallable, AttentionOps
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.rope import LazyRope, LTXRopeType, rope_rows
from ltx_core.model.transformer.transformer_args import (
    CompressedTimestep,
    TransformerArgs,
    block_chunk_tokens,
    timestep_rows,
)
from ltx_core.utils import rms_norm


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False
    ff_bias: bool = True


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        attention_ops: AttentionOps | None = None,
    ):
        super().__init__()

        if attention_ops is None:
            attention_ops = AttentionOps()
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim, bias=video.ff_bias)
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
                ops=attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim, bias=audio.ff_bias)
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
                ops=attention_ops,
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
                ops=attention_ops,
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
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,  # noqa: ARG002 -- kept for the upstream signature
        timestep: torch.Tensor | CompressedTimestep,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        """AdaLN values ``indices`` of every token, one ``(B, tokens or 1, dim)`` tensor each.
        Separate tensors (not views of one stacked tensor) so callers can free a value as
        soon as it is consumed.
        """
        return self._ada_rows(scale_shift_table, timestep, indices, 0, timestep.shape[1])

    @staticmethod
    def _ada_rows(
        scale_shift_table: torch.Tensor,
        timestep: torch.Tensor | CompressedTimestep,
        indices: slice,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, ...]:
        """:meth:`get_ada_values` for tokens ``start:end`` only."""
        rows = timestep_rows(timestep, scale_shift_table.shape[0], indices, start, end)
        table = scale_shift_table[indices].to(device=rows.device, dtype=rows.dtype)
        return tuple(rows[:, :, value_idx, :] + table[value_idx] for value_idx in range(rows.shape[2]))

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor | CompressedTimestep,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
        rows: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale, shift and gate of an audio-video cross-attention (``rows``: tokens ``start:end`` only)."""
        start, end = rows if rows is not None else (0, scale_shift_timestep.shape[1])
        scale, shift = self._ada_rows(
            scale_shift_table[:num_scale_shift_values, :], scale_shift_timestep, scale_shift_indices, start, end
        )
        (gate,) = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )
        return scale, shift, gate

    def _modulated(
        self,
        x: torch.Tensor,
        scale_shift_table: torch.Tensor,
        timestep: torch.Tensor | CompressedTimestep,
        indices: slice,
        start: int = 0,
        end: int | None = None,
    ) -> torch.Tensor:
        """``rms_norm(x[start:end]) * (1 + scale) + shift`` with ``indices`` = (shift, scale), in place."""
        end = x.shape[1] if end is None else end
        shift, scale = self._ada_rows(scale_shift_table, timestep, indices, start, end)
        out = rms_norm(x[:, start:end], eps=self.norm_eps)
        scale.add_(1)
        out.mul_(scale)
        out.add_(shift)
        return out

    @staticmethod
    def _modulate_(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """``x * (1 + scale) + shift`` into ``x``."""
        scale.add_(1)
        x.mul_(scale)
        return x.add_(shift)

    def _prompt_kv_modulation(
        self, x: torch.Tensor, prompt_scale_shift_table: torch.Tensor, prompt_timestep: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shift and scale of the text context under cross-attention AdaLN (see :func:`apply_cross_attention_adaln`)."""
        kv_modulation = prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        if prompt_timestep is not None:
            kv_modulation = kv_modulation + prompt_timestep.reshape(x.shape[0], prompt_timestep.shape[1], 2, -1)
        shift_kv, scale_kv = kv_modulation.unbind(dim=2)
        return shift_kv, scale_kv

    def _add_text_cross_attention(
        self,
        x: torch.Tensor,
        x_normed: torch.Tensor,
        context: torch.Tensor,
        attn: Attention,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor | CompressedTimestep,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> None:
        """In-place ``x += text cross-attention(x_normed)`` (``x_normed`` is consumed)."""
        if not self.cross_attention_adaln:
            attn.add_to_residual(x, x_normed, context=context, mask=context_mask)
            return
        shift_q, scale_q = self._ada_rows(scale_shift_table, timestep, slice(6, 8), 0, timestep.shape[1])
        attn_input = self._modulate_(x_normed, scale_q, shift_q)
        del shift_q, scale_q
        shift_kv, scale_kv = self._prompt_kv_modulation(x, prompt_scale_shift_table, prompt_timestep)
        encoder_hidden_states = context * (1 + scale_kv) + shift_kv
        (gate,) = self._ada_rows(scale_shift_table, timestep, slice(8, 9), 0, timestep.shape[1])
        attn.add_to_residual(x, attn_input, outer_gate=gate, context=encoder_hidden_states, mask=context_mask)

    def _apply_text_cross_attention(
        self,
        x_normed: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation.
        ``x_normed`` is the RMS-normalized post-self-attention output (produced by
        the PostSA composite); this method does not normalize again.
        """
        if cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x_normed.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x_normed,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
            )
        return attn(x_normed, context=context, mask=context_mask)

    def _self_and_text_attention(
        self,
        x: torch.Tensor,
        args: TransformerArgs,
        attn1: Attention,
        attn2: Attention,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
    ) -> None:
        """Self-attention then text cross-attention, added into ``x`` in place.
        Shift/scale are live only while the attention input is built; the gate is
        allocated right before it is consumed. That keeps ~4 instead of ~6 hidden-state
        sizes resident, which is the peak that OOMs long single-pass scenes.
        """
        norm_x = self._modulated(x, scale_shift_table, args.timesteps, slice(0, 2))
        (gate,) = self.get_ada_values(scale_shift_table, x.shape[0], args.timesteps, slice(2, 3))
        attn1.add_to_residual(
            x,
            norm_x,
            outer_gate=gate,
            pe=args.positional_embeddings,
            mask=args.self_attention_mask,
            perturbation_mask=args.self_attn_perturbation_mask,
            all_perturbed=args.self_attn_all_perturbed,
        )
        del norm_x, gate
        self._add_text_cross_attention(
            x,
            rms_norm(x, eps=self.norm_eps),
            args.context,
            attn2,
            scale_shift_table,
            prompt_scale_shift_table,
            args.timesteps,
            args.prompt_timestep,
            args.context_mask,
        )

    def _feed_forward(
        self, x: torch.Tensor, args: TransformerArgs, ff: FeedForward, scale_shift_table: torch.Tensor
    ) -> None:
        x_scaled = self._modulated(x, scale_shift_table, args.timesteps, slice(3, 5))
        (gate,) = self.get_ada_values(scale_shift_table, x.shape[0], args.timesteps, slice(5, 6))
        ff.add_to_residual(x, x_scaled, gate=gate)

    # ------------------------------------------------------------------
    # Token-chunked video path (LTX_BLOCK_CHUNK_TOKENS)
    # ------------------------------------------------------------------

    def _chunked_video_attention(self, vx: torch.Tensor, video: TransformerArgs, chunk: int, with_ff: bool) -> None:
        """Video self-attention, text cross-attention (and the MLP) over token chunks, in place.
        Only the residual stream and the self-attention keys/values span the whole
        sequence; every other activation is ``chunk`` tokens wide, so the peak drops from
        ~8 to ~3 hidden-state sizes. Keys/values come from the block input, so a chunk
        may update its residual rows as soon as its queries are answered. Attention rows,
        norms, AdaLN and the MLP are per token, so the result matches the unchunked path.
        """
        batch_size, tokens = vx.shape[0], vx.shape[1]
        spans = [(start, min(start + chunk, tokens)) for start in range(0, tokens, chunk)]
        pe = video.positional_embeddings
        self_mask = video.self_attention_mask
        table = self.scale_shift_table

        keys = vx.new_empty((batch_size, tokens, self.attn1.heads * self.attn1.dim_head))
        values = torch.empty_like(keys)
        for start, end in spans:
            attn_input = self._modulated(vx, table, video.timesteps, slice(0, 2), start, end)
            k, v = self.attn1.project_kv(attn_input, rope_rows(pe, start, end))
            keys[:, start:end].copy_(k)
            values[:, start:end].copy_(v)
            del attn_input, k, v

        context = video.context
        if self.cross_attention_adaln:
            shift_kv, scale_kv = self._prompt_kv_modulation(vx, self.prompt_scale_shift_table, video.prompt_timestep)
            context = context * (1 + scale_kv) + shift_kv
            del shift_kv, scale_kv
        context_k, context_v = self.attn2.project_kv(context)
        del context

        for start, end in spans:
            rows = vx[:, start:end]
            attn_input = self._modulated(vx, table, video.timesteps, slice(0, 2), start, end)
            (gate,) = self._ada_rows(table, video.timesteps, slice(2, 3), start, end)
            self.attn1.add_chunk_to_residual(
                rows,
                attn_input,
                keys,
                values,
                outer_gate=gate,
                mask=None if self_mask is None else self_mask[:, :, start:end],
                pe=rope_rows(pe, start, end),
            )
            del attn_input, gate

            if self.cross_attention_adaln:
                attn_input = self._modulated(vx, table, video.timesteps, slice(6, 8), start, end)
                (gate,) = self._ada_rows(table, video.timesteps, slice(8, 9), start, end)
            else:
                attn_input, gate = rms_norm(rows, eps=self.norm_eps), None
            self.attn2.add_chunk_to_residual(
                rows, attn_input, context_k, context_v, outer_gate=gate, mask=video.context_mask
            )
            del attn_input, gate

            if with_ff:
                self._feed_forward_rows(vx, video, start, end)
        del keys, values, context_k, context_v

    def _feed_forward_rows(self, vx: torch.Tensor, video: TransformerArgs, start: int, end: int) -> None:
        vx_scaled = self._modulated(vx, self.scale_shift_table, video.timesteps, slice(3, 5), start, end)
        (gate,) = self._ada_rows(self.scale_shift_table, video.timesteps, slice(5, 6), start, end)
        self.ff.add_to_residual(vx[:, start:end], vx_scaled, gate=gate)
        del vx_scaled, gate

    def _chunked_audio_video_attention(
        self,
        vx: torch.Tensor,
        ax: torch.Tensor,
        video: TransformerArgs,
        audio: TransformerArgs,
        chunk: int,
        run_a2v: bool,
        run_v2a: bool,
    ) -> None:
        """Audio-to-video and video-to-audio cross-attention with the video side in token chunks.
        Both directions read the pre-cross-attention state, as the unchunked path does:
        V2A's video keys/values are projected chunk by chunk before A2V touches ``vx``,
        and its (audio-sized) output is added to ``ax`` only after A2V read the audio.
        A2V's queries are video rows, so each chunk updates its own rows in place.
        """
        batch_size, tokens = vx.shape[0], vx.shape[1]
        spans = [(start, min(start + chunk, tokens)) for start in range(0, tokens, chunk)]
        video_cross_pe = video.cross_positional_embeddings
        audio_cross_pe = audio.cross_positional_embeddings

        v2a_out = None
        if run_v2a:
            attn = self.video_to_audio_attn
            keys = vx.new_empty((batch_size, tokens, attn.heads * attn.dim_head))
            values = torch.empty_like(keys)
            for start, end in spans:
                scale, shift, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    batch_size,
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                    rows=(start, end),
                )
                rows = self._modulate_(rms_norm(vx[:, start:end], eps=self.norm_eps), scale, shift)
                k, v = attn.project_kv(rows, rope_rows(video_cross_pe, start, end))
                keys[:, start:end].copy_(k)
                values[:, start:end].copy_(v)
                del scale, shift, rows, k, v
            scale, shift, gate = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                batch_size,
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
                slice(2, 4),
            )
            v2a_ax_scaled = self._modulate_(rms_norm(ax, eps=self.norm_eps), scale, shift)
            gate = gate * audio.cross_attn_perturbation_mask
            v2a_out = torch.zeros_like(ax)
            attn.add_chunk_to_residual(v2a_out, v2a_ax_scaled, keys, values, outer_gate=gate, pe=audio_cross_pe)
            del keys, values, scale, shift, gate, v2a_ax_scaled

        if run_a2v:
            attn = self.audio_to_video_attn
            scale, shift, _ = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                batch_size,
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
                slice(0, 2),
            )
            a2v_ax_scaled = self._modulate_(rms_norm(ax, eps=self.norm_eps), scale, shift)
            del scale, shift
            keys, values = attn.project_kv(a2v_ax_scaled, audio_cross_pe)
            del a2v_ax_scaled
            for start, end in spans:
                scale, shift, gate = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    batch_size,
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                    rows=(start, end),
                )
                rows = self._modulate_(rms_norm(vx[:, start:end], eps=self.norm_eps), scale, shift)
                gate = gate * video.cross_attn_perturbation_mask
                attn.add_chunk_to_residual(
                    vx[:, start:end], rows, keys, values, outer_gate=gate, pe=rope_rows(video_cross_pe, start, end)
                )
                del scale, shift, gate, rows
            del keys, values

        if v2a_out is not None:
            ax.add_(v2a_out)

    def forward(  # noqa: PLR0912, PLR0915
        self,
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

        # The residual streams are updated in place: a copy would keep one more hidden state
        # alive, and the caller replaces its args with the returned ones anyway. Autograd
        # needs the input intact, so training works on a copy.
        if torch.is_grad_enabled():
            vx = vx.clone() if vx is not None else None
            ax = ax.clone() if ax is not None else None

        chunk = block_chunk_tokens()
        chunked_vx = (
            run_vx
            and 0 < chunk < vx.shape[1]
            and video.self_attn_perturbation_mask is None
            and not video.self_attn_all_perturbed
        )
        do_a2v = run_a2v and not video.cross_attn_skip_all
        do_v2a = run_v2a and not audio.cross_attn_skip_all
        # The MLP may join the chunk loop unless audio-video cross-attention sits in between.
        chunked_ff_done = chunked_vx and not (do_a2v or do_v2a)

        if chunked_vx:
            self._chunked_video_attention(vx, video, chunk, with_ff=chunked_ff_done)
        elif run_vx:
            self._self_and_text_attention(
                vx,
                video,
                self.attn1,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
            )

        if run_ax:
            self._self_and_text_attention(
                ax,
                audio,
                self.audio_attn1,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
            )

        # Audio - Video cross attention.
        if chunked_vx and (do_a2v or do_v2a):
            self._chunked_audio_video_attention(vx, ax, video, audio, chunk, do_a2v, do_v2a)
        elif run_a2v or run_v2a:
            # Snapshot vx/ax before A2V mutates vx; V2A's video keys/values must
            # use the pre-A2V state so direction order doesn't bias the result.
            vx_pre_av = vx
            ax_pre_av = ax
            if do_a2v:
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
                        pe=_full_rope(video.cross_positional_embeddings),
                        k_pe=_full_rope(audio.cross_positional_embeddings),
                    )
                    * gate_out_a2v
                    * video.cross_attn_perturbation_mask
                )
                del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled

            if do_v2a:
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
                        pe=_full_rope(audio.cross_positional_embeddings),
                        k_pe=_full_rope(video.cross_positional_embeddings),
                    )
                    * gate_out_v2a
                    * audio.cross_attn_perturbation_mask
                )
                del gate_out_v2a, v2a_vx_scaled, v2a_ax_scaled
            del vx_pre_av, ax_pre_av

        if chunked_vx:
            if not chunked_ff_done:
                for start in range(0, vx.shape[1], chunk):
                    self._feed_forward_rows(vx, video, start, min(start + chunk, vx.shape[1]))
        elif run_vx:
            self._feed_forward(vx, video, self.ff, self.scale_shift_table)

        if run_ax:
            self._feed_forward(ax, audio, self.audio_ff, self.audio_scale_shift_table)

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def _full_rope(
    freqs_cis: tuple[torch.Tensor, torch.Tensor] | LazyRope | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    return freqs_cis.full() if isinstance(freqs_cis, LazyRope) else freqs_cis


def apply_cross_attention_adaln(
    x_normed: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor | None,
    context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply query/key AdaLN modulation then cross-attention.
    ``x_normed`` is already RMS-normalized by the PostSA composite; this only
    applies the affine (scale/shift) modulation, so normalization is not repeated.
    """
    batch_size = x_normed.shape[0]
    # K/V modulation. With the prompt-side AdaLN MLP disabled (use_prompt_adaln_single=False),
    # prompt_timestep is None and only the static per-block table applies, so K/V are
    # timestep-independent and cacheable across denoising/AR steps. Otherwise the timestep-
    # conditioned MLP output is added on top.
    kv_modulation = prompt_scale_shift_table[None, None].to(device=x_normed.device, dtype=x_normed.dtype)
    if prompt_timestep is not None:
        kv_modulation = kv_modulation + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    shift_kv, scale_kv = kv_modulation.unbind(dim=2)
    attn_input = x_normed * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate
