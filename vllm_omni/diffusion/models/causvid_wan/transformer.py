# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-local Wan DiT: cell tick = ``embed`` → ``block`` → ``unembed``.

Layer split uses stage group ``G`` via ``layer_pp_rank`` / ``layer_pp_world``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import paged_write_attn

logger = init_logger(__name__)

_WAN21_1_3B_CONFIG: dict[str, Any] = {
    "patch_size": [1, 2, 2],
    "num_attention_heads": 12,
    "attention_head_dim": 128,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "ffn_dim": 8960,
    "num_layers": 30,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "rope_max_seq_len": 1024,
}


def stage_layer_range(num_layers: int, group: int, groups: int) -> tuple[int, int]:
    """Match Wan ``layer_pp_*`` / vLLM ``get_pp_indices`` boundaries."""
    from vllm.distributed.utils import get_pp_indices

    if groups < 1:
        raise ValueError("groups must be positive")
    if not 0 <= group < groups:
        raise ValueError(f"group {group} out of range for {groups}")
    return get_pp_indices(num_layers, group, groups)


def _wan_self_attn_with_kv(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    kv_ctx: Any | None,
) -> torch.Tensor:
    qkv, _ = attn.to_qkv(hidden_states)
    q_size = attn.num_heads * attn.head_dim
    kv_size = attn.num_kv_heads * attn.head_dim
    query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
    query = attn.norm_q(query)
    key = attn.norm_k(key)
    query = query.unflatten(2, (attn.num_heads, attn.head_dim))
    key = key.unflatten(2, (attn.num_kv_heads, attn.head_dim))
    value = value.unflatten(2, (attn.num_kv_heads, attn.head_dim))
    if rotary_emb is not None:
        freqs_cos, freqs_sin = rotary_emb
        query = attn.rotary_embedding(query, freqs_cos, freqs_sin)
        key = attn.rotary_embedding(key, freqs_cos, freqs_sin)
    scale = 1.0 / (attn.head_dim**0.5)
    if kv_ctx is not None:
        inputs = kv_ctx.to_layer_inputs() if hasattr(kv_ctx, "to_layer_inputs") else kv_ctx
        if query.shape[0] == 1:
            out = paged_write_attn(inputs, query[0], key[0], value[0], None, None, scale)
            hidden_states = out.unsqueeze(0).flatten(2, 3).type_as(query)
        else:
            outs = [
                paged_write_attn(inputs, query[i], key[i], value[i], None, None, scale) for i in range(query.shape[0])
            ]
            hidden_states = torch.stack(outs, dim=0).flatten(2, 3).type_as(query)
    else:
        hidden_states = attn.attn(query, key, value, None)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
    return attn.dropout(attn.to_out(hidden_states))


class StageWanTransformer(nn.Module):
    """Wan DiT with stage-local G and paged Latest-KV on self-attention."""

    def __init__(
        self,
        *,
        num_layers: int = 30,
        dim: int = 1536,
        num_heads: int = 12,
        ffn_dim: int = 8960,
        in_channels: int = 16,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        layer_groups: int = 1,
        pp_rank: int = 0,
        transformer_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.layer_groups = max(1, int(layer_groups))
        self.pp_rank = int(pp_rank)
        self.group = 0 if self.layer_groups == 1 else self.pp_rank % self.layer_groups
        self.is_stage_first = self.group == 0
        self.is_stage_last = self.group == self.layer_groups - 1
        self.patch_size = tuple(patch_size)
        self.in_features = in_channels * math.prod(self.patch_size)
        self._wan: nn.Module | None = None
        self._cell_kv_ctx: Any | None = None
        self._text: torch.Tensor | None = None
        self._embeddings: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._rotary: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._init_wan(
            transformer_config=transformer_config,
            fallback_geo={
                "num_layers": num_layers,
                "dim": dim,
                "num_heads": num_heads,
                "ffn_dim": ffn_dim,
                "in_channels": in_channels,
                "patch_size": self.patch_size,
            },
        )

    def _init_wan(self, *, transformer_config: dict[str, Any] | None, fallback_geo: dict[str, Any]) -> None:
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import create_transformer_from_config

        cfg = dict(_WAN21_1_3B_CONFIG)
        if transformer_config:
            cfg.update({k: v for k, v in transformer_config.items() if v is not None})
        else:
            patch = fallback_geo["patch_size"]
            heads = int(fallback_geo["num_heads"])
            dim = int(fallback_geo["dim"])
            cfg.update(
                {
                    "patch_size": list(patch),
                    "num_attention_heads": heads,
                    "attention_head_dim": dim // heads,
                    "in_channels": int(fallback_geo["in_channels"]),
                    "out_channels": int(fallback_geo["in_channels"]),
                    "ffn_dim": int(fallback_geo["ffn_dim"]),
                    "num_layers": int(fallback_geo["num_layers"]),
                }
            )
        wan = create_transformer_from_config(cfg, layer_pp_rank=self.group, layer_pp_world=self.layer_groups)
        self._wan = wan
        self.num_layers = int(cfg["num_layers"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.head_dim = int(cfg["attention_head_dim"])
        self.dim = self.num_heads * self.head_dim
        self.start_layer = int(wan.start_layer)
        self.end_layer = int(wan.end_layer)
        self.patch_size = tuple(cfg["patch_size"])
        self.in_features = int(cfg["in_channels"]) * math.prod(self.patch_size)
        self.schedule_patch_embed = None
        self._install_cell_attn_hooks()
        logger.info(
            "StageWanTransformer: local_layers=[%d, %d) G=%d group=%d",
            self.start_layer,
            self.end_layer,
            self.layer_groups,
            self.group,
        )

    def _install_cell_attn_hooks(self) -> None:
        wan = self._wan
        if wan is None:
            return
        owner = self
        for idx in range(self.start_layer, self.end_layer):
            attn = wan.blocks[idx].attn1
            if getattr(attn, "_omni_cell_kv_hook", False):
                continue
            orig = attn.forward

            def _make_forward(attn_mod: nn.Module, orig_fn: Any):
                def _forward(
                    hs: torch.Tensor,
                    rotary_emb_arg: tuple[torch.Tensor, torch.Tensor] | None = None,
                    attn_metadata: Any = None,
                ) -> torch.Tensor:
                    del attn_metadata
                    ctx = owner._cell_kv_ctx
                    if ctx is None:
                        return orig_fn(hs, rotary_emb_arg, None)
                    return _wan_self_attn_with_kv(attn_mod, hs, rotary_emb_arg, ctx)

                return _forward

            attn.forward = _make_forward(attn, orig)  # type: ignore[method-assign]
            attn._omni_cell_kv_hook = True  # type: ignore[attr-defined]
            attn._omni_cell_kv_orig = orig  # type: ignore[attr-defined]

    @property
    def local_num_layers(self) -> int:
        return max(0, self.end_layer - self.start_layer)

    @property
    def wan(self) -> nn.Module | None:
        return self._wan

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str] | None:
        if self._wan is None:
            return None

        def _strip() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                key = name
                for prefix in ("transformer.", "model.transformer.", "dit."):
                    if key.startswith(prefix):
                        key = key[len(prefix) :]
                        break
                if key.startswith("_wan."):
                    key = key[len("_wan.") :]
                yield key, tensor

        self._wan.load_weights(_strip())
        return None

    def seq_len_for_latent(self, latent_shape: tuple[int, ...]) -> int:
        if len(latent_shape) != 5:
            raise ValueError(f"expected B,C,T,H,W; got {latent_shape}")
        _, _, t, h, w = latent_shape
        pt, ph, pw = self.patch_size
        if t % pt or h % ph or w % pw:
            raise ValueError(f"latent {latent_shape} not aligned to patch {self.patch_size}")
        return (t // pt) * (h // ph) * (w // pw)

    def prepare_cell(self, encoder_hidden_states: torch.Tensor, **_kwargs) -> None:
        """Bind text; clear per-step embedding / per-chunk RoPE caches."""
        self._text = encoder_hidden_states
        self._embeddings = {}
        self._rotary = {}

    def embed(self, latent: torch.Tensor) -> torch.Tensor:
        assert self._wan is not None
        hidden = self._wan.patch_embedding(latent)
        return hidden.flatten(2).transpose(1, 2)

    def _embedding(
        self, step: int, timestep: torch.Tensor, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._wan is not None and self._text is not None
        hit = self._embeddings.get(step)
        if hit is not None:
            return hit
        ts = timestep.to(device=device, dtype=torch.float32)
        if ts.ndim == 0 or (ts.ndim == 1 and ts.shape[0] == 1 and batch_size > 1):
            ts = ts.expand(batch_size)
        temb, timestep_proj, enc, _ = self._wan.condition_embedder(ts, self._text, None, timestep_seq_len=None)
        timestep_proj = self._wan.timestep_proj_prepare(timestep_proj, None)
        out = (temb.to(dtype=dtype), timestep_proj.to(dtype=dtype), enc.to(dtype=dtype))
        self._embeddings[step] = out
        return out

    def _rope_with_temporal_offset(
        self, latent: torch.Tensor, *, start_frame: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wan 3D RoPE with absolute temporal positions (CausVid ``current_start``)."""
        assert self._wan is not None
        rope = self._wan.rope
        _, _, num_frames, height, width = latent.shape
        p_t, p_h, p_w = rope.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        if start_frame % p_t:
            raise ValueError(f"start_frame={start_frame} must align to temporal patch {p_t}")
        t0 = start_frame // p_t
        if t0 + ppf > rope.max_seq_len:
            raise ValueError(f"temporal RoPE range [{t0}, {t0 + ppf}) exceeds rope_max_seq_len={rope.max_seq_len}")

        split_sizes = [
            rope.attention_head_dim - 2 * (rope.attention_head_dim // 3),
            rope.attention_head_dim // 3,
            rope.attention_head_dim // 3,
        ]
        freqs_cos = rope.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = rope.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][t0 : t0 + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_f = freqs_sin[0][t0 : t0 + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos_out = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin_out = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        return freqs_cos_out.to(latent.device), freqs_sin_out.to(latent.device)

    def _rotary_emb(self, chunk: int, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._wan is not None
        hit = self._rotary.get(chunk)
        if hit is not None:
            return hit
        # Absolute temporal offset so history KV and the current chunk do not share
        # the same frame indices (official CausVid ``current_start``).
        start_frame = int(chunk) * int(latent.shape[2])
        freqs_cos, freqs_sin = self._rope_with_temporal_offset(latent, start_frame=start_frame)
        rotary = (freqs_cos[..., 0::2].to(latent.dtype), freqs_sin[..., 1::2].to(latent.dtype))
        self._rotary = {chunk: rotary}
        return rotary

    def block(
        self,
        index: int,
        hidden: torch.Tensor,
        *,
        chunk: int,
        step: int,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        kv_ctx: Any | None = None,
    ) -> torch.Tensor:
        assert self._wan is not None
        if not self.start_layer <= index < self.end_layer:
            raise ValueError(f"block {index} not in [{self.start_layer}, {self.end_layer})")
        _temb, timestep_proj, enc = self._embedding(
            step, timestep, batch_size=latent.shape[0], device=latent.device, dtype=latent.dtype
        )
        rotary_emb = self._rotary_emb(chunk, latent)
        self._cell_kv_ctx = kv_ctx
        try:
            return self._wan.blocks[index](hidden, enc, timestep_proj, rotary_emb, None, None, False)
        finally:
            self._cell_kv_ctx = None

    def unembed(
        self,
        hidden: torch.Tensor,
        *,
        chunk: int,
        step: int,
        latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del chunk
        assert self._wan is not None
        wan = self._wan
        batch_size, _c, num_frames, height, width = latent.shape
        p_t, p_h, p_w = self.patch_size
        post_t, post_h, post_w = num_frames // p_t, height // p_h, width // p_w
        temb, _, _ = self._embedding(step, timestep, batch_size=batch_size, device=latent.device, dtype=latent.dtype)
        shift, scale = wan.output_scale_shift_prepare(temb)
        shift, scale = shift.to(hidden.device), scale.to(hidden.device)
        if shift.ndim == 2:
            shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
        hidden = wan.norm_out(hidden, scale, shift).type_as(hidden)
        hidden = wan.proj_out(hidden)
        hidden = hidden.reshape(batch_size, post_t, post_h, post_w, p_t, p_h, p_w, -1)
        return hidden.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def forward_latent_step(
        self,
        latent: torch.Tensor,
        *,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        kv_contexts: list[Any] | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        only_block: int | None = None,
        chunk: int = 0,
        step: int = 0,
    ) -> torch.Tensor | IntermediateTensors:
        """Compose cell APIs (multi-layer or ``only_block``)."""
        if self._text is None:
            self.prepare_cell(encoder_hidden_states)
        layer_ids = [only_block] if only_block is not None else list(range(self.start_layer, self.end_layer))
        first_local = layer_ids[0] == self.start_layer
        last_local = layer_ids[-1] == self.end_layer - 1
        if first_local and self.is_stage_first:
            hidden = self.embed(latent)
        else:
            if intermediate_tensors is None:
                raise RuntimeError('need intermediate_tensors["hidden_states"]')
            hidden = intermediate_tensors["hidden_states"]
        for idx in layer_ids:
            ctx = None
            if kv_contexts is not None:
                ctx = kv_contexts[0] if len(kv_contexts) == 1 else kv_contexts[idx - self.start_layer]
            hidden = self.block(idx, hidden, chunk=chunk, step=step, latent=latent, timestep=timestep, kv_ctx=ctx)
        if not (last_local and self.is_stage_last):
            return IntermediateTensors({"hidden_states": hidden})
        return self.unembed(hidden, chunk=chunk, step=step, latent=latent, timestep=timestep)

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("use embed/block/unembed or forward_latent_step")
