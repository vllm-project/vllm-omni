# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native SolarWM-H3 specialization of the packed MiniMax-H3 transformer."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D, all_to_all_5D
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
)

from .attention import SolarAttentionContext, prepare_solar_rotaries, solar_rollout_attention
from .layout import H3PackedLayout, build_row_timesteps, patchify_video, unpatchify_video

_SOLAR_ATTENTION: ContextVar[SolarAttentionContext | None] = ContextVar("solarwm_h3_attention", default=None)


class SolarWMH3Attention(MiniMaxH3Attention):
    """Keep the native projection parameters and replace only attention math."""

    def __init__(self, native: MiniMaxH3Attention, layer: int) -> None:
        nn.Module.__init__(self)
        # Reusing these modules preserves every base-checkpoint parameter path
        # and avoids temporarily constructing a second copy of this large DiT.
        self.qkv_proj = native.qkv_proj
        self.out_proj = native.out_proj
        self.q_norm = native.q_norm
        self.k_norm = native.k_norm
        self.rope = native.rope
        self.attention = native.attention
        self.to_gate_compress = None
        self.vdn = None
        self.total_num_heads = native.total_num_heads
        self.num_heads = native.num_heads
        self.num_kv_heads = native.num_kv_heads
        self.head_dim = native.head_dim
        self.rot_dim = native.rot_dim
        self._prefix = native._prefix
        self.layer = layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_table: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int | None = None,
        num_requests: int = 1,
        sp_seq_lens: list[int] | None = None,
        video_layout: Any = None,
        vsa_prefix_segments: tuple[int, ...] = (),
        vdn_window: Any = None,
    ) -> torch.Tensor:
        del rope_table, cu_seqlens, max_seqlen, sp_seq_lens, video_layout, vsa_prefix_segments
        if num_requests != 1 or vdn_window is not None:
            raise ValueError("SolarWM attention supports one rollout and dense attention")
        control = _SOLAR_ATTENTION.get()
        if control is None:
            raise RuntimeError("SolarWM native attention requires a SolarAttentionContext")
        qkv, _ = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        query = self.q_norm(query.reshape(1, x.shape[0], self.num_heads, self.head_dim))
        key = self.k_norm(key.reshape(1, x.shape[0], self.num_kv_heads, self.head_dim))
        value = value.reshape(1, x.shape[0], self.num_kv_heads, self.head_dim)

        strategy = self.attention._get_active_parallel_strategy()
        ulysses = bool(strategy.enabled)
        group = None
        if ulysses:
            if strategy.name != "ulysses":
                raise NotImplementedError("SolarWM raw KV attention supports pure Ulysses sequence parallelism")
            group = get_sp_group().ulysses_group
            exchanged = all_to_all_5D(torch.stack((query, key, value), dim=2), scatter_idx=3, gather_idx=1, group=group)
            query, key, value = exchanged.unbind(2)
        global_rows = query.shape[1]
        if packed_total is not None and global_rows != packed_total:
            raise RuntimeError("SolarWM Ulysses exchange did not restore the full packed sequence")
        valid_rows = int(control.layout.sequence_length)
        if global_rows < valid_rows:
            raise ValueError("SolarWM packed sequence is shorter than its attention layout")
        output = solar_rollout_attention(
            query[:, :valid_rows], key[:, :valid_rows], value[:, :valid_rows], control=control, layer=self.layer
        )
        if global_rows > valid_rows:
            # Padding is a separate document, and never enters the world KV.
            output = torch.cat((output, torch.zeros_like(query[:, valid_rows:])), dim=1)
        if ulysses:
            output = SeqAllToAll4D.apply(group, output, 1, 2, False)
        output, _ = self.out_proj(output.reshape(x.shape[0], q_size))
        return output


class SolarWMH3Transformer(MiniMaxH3DiTModel):
    """H3 checkpoint-compatible modules with Solar's W6 causal execution.

    The inherited embedding, token refiner, AdaLN, MLP and output layers remain
    native vLLM-Omni modules. The scoped control never becomes module state and
    is cleared on both successful and failed forwards.
    """

    _cache_dit_adapter_config = None

    def __init__(self, od_config: Any, quant_config: Any = None, *, diffusers_weights: bool | None = None) -> None:
        if quant_config is not None:
            raise NotImplementedError("SolarWM-H3 currently requires unquantized native weights")
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("SolarWM-H3 currently supports tensor_parallel_size=1")
        parallel = od_config.parallel_config
        for field in ("ring_degree", "allgather_degree", "cfg_parallel_size", "pipeline_parallel_size"):
            if int(getattr(parallel, field, 1) or 1) != 1:
                raise NotImplementedError(f"SolarWM-H3 does not support {field}>1")
        if getattr(parallel, "ulysses_mode", "strict") != "strict":
            raise NotImplementedError("SolarWM-H3 requires strict Ulysses sequence parallelism")
        if bool(getattr(parallel, "ulysses_a2a_permute", False)):
            raise NotImplementedError("SolarWM-H3 does not support ulysses_a2a_permute")
        if bool(getattr(parallel, "use_hsdp", False)):
            raise NotImplementedError("SolarWM-H3 does not support HSDP")
        super().__init__(od_config, quant_config, diffusers_weights=diffusers_weights)
        if self.arch.attention_head_dim != 128 or self.arch.rope_inv_freq_len != 16:
            raise ValueError("SolarWM-H3 requires 128-channel heads and 96-channel native MM-RoPE")
        if tuple(self.arch.patch_size) != (1, 2, 2):
            raise ValueError("SolarWM-H3 requires the released (1,2,2) video patch geometry")
        for layer, block in enumerate(self.blocks):
            block.attn = SolarWMH3Attention(block.attn, layer)

    def predict_video(
        self,
        prompt: torch.Tensor,
        anchor_rows: torch.Tensor,
        current_latents: torch.Tensor,
        audio_rows: torch.Tensor,
        layout: H3PackedLayout,
        video_t: float | torch.Tensor,
        audio_t: float | torch.Tensor,
        context: SolarAttentionContext,
        *,
        keyframe_timestep: float = 0.999,
    ) -> torch.Tensor:
        """Return data-ward latent velocity for one packed five-frame chunk.

        Input video latents are ``[1,24,5,H,W]``; the anchor is already
        patchified into ``[1,anchor_rows,96]`` and audio into
        ``[1,audio_rows,32]``. Text features remain separate from packed video
        rows even when their Qwen modality tags identify image tokens.
        """
        if current_latents.ndim != 5 or current_latents.shape[:3] != (1, self.arch.latents_dim, 5):
            raise ValueError("SolarWM prediction requires [1,24,5,H,W] current latents")
        if current_latents.shape[-2:] != (layout.latent_height, layout.latent_width):
            raise ValueError("SolarWM current latent resolution differs from its packed layout")
        if context.layout is not layout:
            raise ValueError("SolarWM prediction and attention must share one packed layout")
        device = current_latents.device
        video_rows = torch.cat((anchor_rows, patchify_video(current_latents, layout.patch_size)), dim=1)
        width = self.arch.latents_dim * 4
        if video_rows.shape != (1, int(layout.video_indices.numel()), width):
            raise ValueError("SolarWM anchor/video rows differ from the packed layout")
        if audio_rows.shape != (1, int(layout.audio_indices.numel()), self.arch.audio_latents_dim):
            raise ValueError("SolarWM audio rows differ from the packed layout")
        if prompt.ndim == 3 and prompt.shape[0] == 1:
            prompt = prompt[0]
        if prompt.shape != (int(layout.text_indices.numel()), self.arch.text_dim):
            raise ValueError("SolarWM text embeddings differ from the packed layout")
        total = int(layout.sequence_length)
        # Strict Ulysses shards equal rows. Alignment rows form a separate
        # document and never become a condition, current token or cached key.
        ulysses_size = int(getattr(self.parallel_config, "ulysses_degree", 1) or 1)
        packed_rows = ((total + ulysses_size - 1) // ulysses_size) * ulysses_size
        x = current_latents.new_zeros((1, packed_rows, width))
        x.index_copy_(1, layout.video_indices.to(device), video_rows.to(device))
        audio = audio_rows.new_zeros((1, packed_rows, self.arch.audio_latents_dim))
        audio.index_copy_(1, layout.audio_indices.to(device), audio_rows)
        timesteps, inverse = build_row_timesteps(
            layout, video_t, audio_t, text_timestep=1.0, condition_video_timestep=keyframe_timestep
        )
        if packed_rows > total:
            row_times = torch.cat((timesteps[inverse], timesteps.new_ones(packed_rows - total)))
            timesteps, inverse = torch.unique(row_times, sorted=True, return_inverse=True)
        token_tags = torch.full((packed_rows,), -1, dtype=torch.long, device=device)
        token_tags[:total] = layout.token_tags.to(device)
        position_ids = torch.zeros((1, packed_rows, 3), dtype=layout.position_ids.dtype, device=device)
        position_ids[0, :total] = layout.position_ids.to(device)
        cu_seqlens = torch.tensor(
            [0, total] if total == packed_rows else [0, total, packed_rows], dtype=torch.int32, device=device
        )
        text_rows = int(layout.text_indices.numel())
        psp = {"cu_seqlens_q": cu_seqlens, "max_seqlen_q": total, "num_requests": 1}
        refiner_psp = {
            "cu_seqlens_q": torch.tensor([0, text_rows], dtype=torch.int32, device=device),
            "max_seqlen_q": text_rows,
        }
        output, _ = self(
            solar_context=context,
            x=x,
            audio_x=audio,
            prompt_embeds=prompt,
            img_position_ids=position_ids,
            unique_timesteps=timesteps,
            inverse_indices=inverse,
            token_tags=token_tags,
            img_pos_info={"position_ids": layout.video_indices},
            audio_pos_info={"position_ids": layout.audio_indices},
            text_pos_info={"position_ids": layout.text_indices},
            img_pos_for_infer_output_info={"position_ids": layout.noisy_video_indices},
            update_mask=torch.ones(int(layout.noisy_video_indices.numel()), device=device),
            packed_seq_params=psp,
            refiner_packed_seq_params=refiner_psp,
        )
        return unpatchify_video(output.unsqueeze(0), 5, layout.latent_height, layout.latent_width, layout.patch_size)

    def forward(self, *, solar_context: SolarAttentionContext, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(solar_context, SolarAttentionContext):
            raise TypeError("SolarWM-H3 forward requires SolarAttentionContext")
        if _SOLAR_ATTENTION.get() is not None:
            raise RuntimeError("SolarWM-H3 attention contexts cannot be nested")
        context = prepare_solar_rotaries(solar_context, self.rope)
        token = _SOLAR_ATTENTION.set(context)
        try:
            return super().forward(**kwargs)
        finally:
            _SOLAR_ATTENTION.reset(token)
