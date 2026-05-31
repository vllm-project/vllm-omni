# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


class Flux2KVLayerCache:
    """Per-layer KV cache for reference image tokens."""

    def __init__(self):
        self.k_ref: torch.Tensor | None = None
        self.v_ref: torch.Tensor | None = None

    def store(self, k_ref: torch.Tensor, v_ref: torch.Tensor) -> None:
        if k_ref is None or v_ref is None:
            raise ValueError("KV cache tensors cannot be None.")
        if k_ref.shape != v_ref.shape:
            raise ValueError(f"KV cache tensors must have identical shapes, got {k_ref.shape} and {v_ref.shape}.")
        if k_ref.dtype != v_ref.dtype:
            raise ValueError(f"KV cache tensors must have identical dtypes, got {k_ref.dtype} and {v_ref.dtype}.")
        self.k_ref = k_ref
        self.v_ref = v_ref

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_ref is None or self.v_ref is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.k_ref, self.v_ref

    def clear(self) -> None:
        self.k_ref = None
        self.v_ref = None


class Flux2KVCache:
    """Container for all layers' reference-token KV caches."""

    def __init__(self, num_double_layers: int, num_single_layers: int):
        self.double_block_caches = [Flux2KVLayerCache() for _ in range(num_double_layers)]
        self.single_block_caches = [Flux2KVLayerCache() for _ in range(num_single_layers)]
        self.num_ref_tokens: int = 0
        self.layout_id: str | None = None

    def get_double(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.double_block_caches[layer_idx]

    def get_single(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.single_block_caches[layer_idx]

    def clear(self) -> None:
        for cache in self.double_block_caches:
            cache.clear()
        for cache in self.single_block_caches:
            cache.clear()
        self.num_ref_tokens = 0
        self.layout_id = None


@dataclass(frozen=True)
class Flux2KVTokenLayout:
    num_txt_tokens: int
    num_ref_tokens_global: int
    total_nontext_tokens_global: int
    local_nontext_tokens: int
    local_ref_tokens: int
    local_img_tokens: int
    sp_rank: int
    sp_world_size: int
    layout_id: str


def build_flux2_kv_token_layout(
    *,
    num_txt_tokens: int,
    num_ref_tokens_global: int,
    total_nontext_tokens_global: int,
    local_nontext_tokens: int,
    sp_rank: int,
    sp_world_size: int,
    ref_tokens_at_end: bool = False,
) -> Flux2KVTokenLayout:
    effective_sp_world_size = max(sp_world_size, 1)
    effective_sp_rank = sp_rank

    # Some runtime paths provide full non-text sequence on each rank while the
    # parallel config still reports SP>1. In that case treat this as full-seq.
    if total_nontext_tokens_global > 0 and local_nontext_tokens >= total_nontext_tokens_global:
        effective_sp_world_size = 1
        effective_sp_rank = 0

    local_ref_tokens = _get_local_ref_token_count(
        local_seq_len=local_nontext_tokens,
        total_seq_len=total_nontext_tokens_global,
        num_ref_tokens=num_ref_tokens_global,
        sp_rank=effective_sp_rank,
        sp_world_size=effective_sp_world_size,
        ref_tokens_at_end=ref_tokens_at_end,
    )
    local_img_tokens = max(local_nontext_tokens - local_ref_tokens, 0)
    layout_id = (
        f"txt={num_txt_tokens}|ref={num_ref_tokens_global}|nontext={total_nontext_tokens_global}|"
        f"local={local_nontext_tokens}|rank={effective_sp_rank}|ws={effective_sp_world_size}|"
        f"ref_end={int(ref_tokens_at_end)}"
    )
    return Flux2KVTokenLayout(
        num_txt_tokens=num_txt_tokens,
        num_ref_tokens_global=num_ref_tokens_global,
        total_nontext_tokens_global=total_nontext_tokens_global,
        local_nontext_tokens=local_nontext_tokens,
        local_ref_tokens=local_ref_tokens,
        local_img_tokens=local_img_tokens,
        sp_rank=effective_sp_rank,
        sp_world_size=effective_sp_world_size,
        layout_id=layout_id,
    )


def _gather_sequence_shards(
    tensor: torch.Tensor,
    *,
    total_tokens: int,
    sp_world_size: int,
    sp_gather_fn: Callable[[torch.Tensor, int], torch.Tensor] | None,
) -> torch.Tensor:
    tensor = tensor.contiguous()
    if sp_world_size <= 1:
        return tensor[:, :total_tokens, ...]
    if sp_gather_fn is None:
        raise ValueError("sp_gather_fn is required when sequence parallel world size is greater than 1.")
    gathered = sp_gather_fn(tensor, 1)
    return gathered[:, :total_tokens, ...]


def _get_local_ref_token_count(
    *,
    local_seq_len: int,
    total_seq_len: int,
    num_ref_tokens: int,
    sp_rank: int,
    sp_world_size: int,
    ref_tokens_at_end: bool = False,
) -> int:
    if sp_world_size <= 1:
        return min(local_seq_len, num_ref_tokens)
    shard_span = local_seq_len
    global_start = sp_rank * shard_span
    global_end = min(global_start + shard_span, total_seq_len)
    if ref_tokens_at_end:
        ref_start = max(total_seq_len - num_ref_tokens, 0)
        ref_end = total_seq_len
        return max(0, min(global_end, ref_end) - max(global_start, ref_start))
    return max(0, min(global_end, num_ref_tokens) - global_start)


def gather_full_reference_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_txt_tokens: int,
    num_ref_tokens: int,
    total_nontext_tokens: int,
    sp_world_size: int,
    sp_gather_fn: Callable[[torch.Tensor, int], torch.Tensor] | None,
    ref_tokens_at_end: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    nontext_key = _gather_sequence_shards(
        key[:, num_txt_tokens:, ...],
        total_tokens=total_nontext_tokens,
        sp_world_size=sp_world_size,
        sp_gather_fn=sp_gather_fn,
    )
    nontext_value = _gather_sequence_shards(
        value[:, num_txt_tokens:, ...],
        total_tokens=total_nontext_tokens,
        sp_world_size=sp_world_size,
        sp_gather_fn=sp_gather_fn,
    )
    if ref_tokens_at_end:
        return nontext_key[:, -num_ref_tokens:, ...], nontext_value[:, -num_ref_tokens:, ...]
    return nontext_key[:, :num_ref_tokens, ...], nontext_value[:, :num_ref_tokens, ...]


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    if query.shape[2] == 0:
        return query.new_empty(query.shape)
    return F.scaled_dot_product_attention(query, key, value)


def flux2_kv_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_txt_tokens: int,
    num_ref_tokens: int,
    kv_cache: Flux2KVLayerCache | None = None,
    total_nontext_tokens: int | None = None,
    sp_rank: int = 0,
    sp_world_size: int = 1,
    sp_gather_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ref_tokens_at_end: bool = False,
) -> torch.Tensor:
    """Causal attention where reference tokens only self-attend.

    Tensor format: (B, L, H, D).
    """
    if sp_world_size > 1:
        if total_nontext_tokens is None:
            raise ValueError("total_nontext_tokens is required when sequence parallel world size is greater than 1.")

        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)

        q_txt = query_t[:, :, :num_txt_tokens]
        k_txt = key_t[:, :, :num_txt_tokens]
        v_txt = value_t[:, :, :num_txt_tokens]

        q_nontext_local = query_t[:, :, num_txt_tokens:]
        k_nontext_full = _gather_sequence_shards(
            key[:, num_txt_tokens:, ...],
            total_tokens=total_nontext_tokens,
            sp_world_size=sp_world_size,
            sp_gather_fn=sp_gather_fn,
        ).transpose(1, 2)
        v_nontext_full = _gather_sequence_shards(
            value[:, num_txt_tokens:, ...],
            total_tokens=total_nontext_tokens,
            sp_world_size=sp_world_size,
            sp_gather_fn=sp_gather_fn,
        ).transpose(1, 2)

        if kv_cache is not None:
            k_ref_t, v_ref_t = (x.transpose(1, 2) for x in kv_cache.get())
            q_img = q_nontext_local
            k_all = torch.cat([k_txt, k_ref_t, k_nontext_full], dim=2)
            v_all = torch.cat([v_txt, v_ref_t, v_nontext_full], dim=2)
            txt_img_out = _scaled_dot_product_attention(torch.cat([q_txt, q_img], dim=2), k_all, v_all)
            return txt_img_out.transpose(1, 2).flatten(2, 3)

        local_ref_tokens = _get_local_ref_token_count(
            local_seq_len=q_nontext_local.shape[2],
            total_seq_len=total_nontext_tokens,
            num_ref_tokens=num_ref_tokens,
            sp_rank=sp_rank,
            sp_world_size=sp_world_size,
            ref_tokens_at_end=ref_tokens_at_end,
        )
        if ref_tokens_at_end:
            local_img_tokens = max(q_nontext_local.shape[2] - local_ref_tokens, 0)
            q_img = q_nontext_local[:, :, :local_img_tokens]
            q_ref = q_nontext_local[:, :, local_img_tokens:]
            global_img_tokens = max(total_nontext_tokens - num_ref_tokens, 0)
            k_img = k_nontext_full[:, :, :global_img_tokens]
            v_img = v_nontext_full[:, :, :global_img_tokens]
            k_ref = k_nontext_full[:, :, global_img_tokens:]
            v_ref = v_nontext_full[:, :, global_img_tokens:]
        else:
            q_ref = q_nontext_local[:, :, :local_ref_tokens]
            q_img = q_nontext_local[:, :, local_ref_tokens:]
            k_ref = k_nontext_full[:, :, :num_ref_tokens]
            v_ref = v_nontext_full[:, :, :num_ref_tokens]
            k_img = k_nontext_full[:, :, num_ref_tokens:]
            v_img = v_nontext_full[:, :, num_ref_tokens:]
        k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
        v_all = torch.cat([v_txt, v_ref, v_img], dim=2)

        txt_img_out = _scaled_dot_product_attention(torch.cat([q_txt, q_img], dim=2), k_all, v_all)
        txt_out = txt_img_out[:, :, :num_txt_tokens]
        img_out = txt_img_out[:, :, num_txt_tokens:]
        ref_out = _scaled_dot_product_attention(q_ref, k_ref, v_ref)

        if ref_tokens_at_end:
            return torch.cat([txt_out, img_out, ref_out], dim=2).transpose(1, 2).flatten(2, 3)
        return torch.cat([txt_out, ref_out, img_out], dim=2).transpose(1, 2).flatten(2, 3)

    if num_ref_tokens == 0 and kv_cache is None:
        B, L, H, D = query.shape
        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(query_t, key_t, value_t)
        return out.transpose(1, 2).flatten(2, 3)

    if kv_cache is not None:
        k_ref, v_ref = kv_cache.get()
        k_ref_t = k_ref.transpose(1, 2)
        v_ref_t = v_ref.transpose(1, 2)

        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)

        k_all = torch.cat([key_t[:, :, :num_txt_tokens], k_ref_t, key_t[:, :, num_txt_tokens:]], dim=2)
        v_all = torch.cat([value_t[:, :, :num_txt_tokens], v_ref_t, value_t[:, :, num_txt_tokens:]], dim=2)

        out = F.scaled_dot_product_attention(query_t, k_all, v_all)
        return out.transpose(1, 2).flatten(2, 3)

    query_t = query.transpose(1, 2)
    key_t = key.transpose(1, 2)
    value_t = value.transpose(1, 2)
    if ref_tokens_at_end:
        nontext_len = query_t.shape[2] - num_txt_tokens
        img_len = max(nontext_len - num_ref_tokens, 0)
        img_end = num_txt_tokens + img_len

        q_txt = query_t[:, :, :num_txt_tokens]
        q_img = query_t[:, :, num_txt_tokens:img_end]
        q_ref = query_t[:, :, img_end:]

        k_txt = key_t[:, :, :num_txt_tokens]
        k_img = key_t[:, :, num_txt_tokens:img_end]
        k_ref = key_t[:, :, img_end:]

        v_txt = value_t[:, :, :num_txt_tokens]
        v_img = value_t[:, :, num_txt_tokens:img_end]
        v_ref = value_t[:, :, img_end:]
    else:
        ref_start = num_txt_tokens
        ref_end = num_txt_tokens + num_ref_tokens

        q_txt = query_t[:, :, :ref_start]
        q_ref = query_t[:, :, ref_start:ref_end]
        q_img = query_t[:, :, ref_end:]

        k_txt = key_t[:, :, :ref_start]
        k_ref = key_t[:, :, ref_start:ref_end]
        k_img = key_t[:, :, ref_end:]

        v_txt = value_t[:, :, :ref_start]
        v_ref = value_t[:, :, ref_start:ref_end]
        v_img = value_t[:, :, ref_end:]

    q_txt_img = torch.cat([q_txt, q_img], dim=2)
    k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
    v_all = torch.cat([v_txt, v_ref, v_img], dim=2)
    attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all)
    attn_txt = attn_txt_img[:, :, :num_txt_tokens]
    attn_img = attn_txt_img[:, :, num_txt_tokens:]

    attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)

    if ref_tokens_at_end:
        out = torch.cat([attn_txt, attn_img, attn_ref], dim=2)
    else:
        out = torch.cat([attn_txt, attn_ref, attn_img], dim=2)
    return out.transpose(1, 2).flatten(2, 3)


def _blend_mod_params(
    img_params: tuple[torch.Tensor, ...],
    ref_params: tuple[torch.Tensor, ...],
    num_ref: int,
    seq_len: int,
    ref_tokens_at_end: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Blend [ref, img] modulation without symbolic slicing.

    Using a fixed-length mask keeps sequence length equal to ``seq_len`` under
    Dynamo symbolic shape tracing.
    """
    blended = []
    for im, rm in zip(img_params, ref_params):
        if im.ndim == 2:
            im = im.unsqueeze(1)
            rm = rm.unsqueeze(1)
        B = im.shape[0]
        im_expanded = im.expand(B, seq_len, -1)
        rm_expanded = rm.expand(B, seq_len, -1)
        if ref_tokens_at_end:
            mask = torch.arange(seq_len, device=im.device).view(1, seq_len, 1) >= (seq_len - num_ref)
        else:
            mask = torch.arange(seq_len, device=im.device).view(1, seq_len, 1) < num_ref
        blended.append(torch.where(mask, rm_expanded, im_expanded))
    return tuple(blended)


def blend_double_block_mods(
    img_mod: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    ref_mod: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    num_ref: int,
    seq_len: int,
    ref_tokens_at_end: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """Blend double-block modulations for [ref, img] layout.

    Args:
        img_mod: Tuple of (shift, scale, gate) modulation tuples for image stream.
            Format: ((shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp))
        ref_mod: Tuple of (shift, scale, gate) modulation tuples for reference tokens.
        num_ref: Number of reference tokens.
        seq_len: Total sequence length (including ref tokens).

    Returns:
        Blended modulation tuple where first num_ref positions use ref_mod.
    """
    blended_sets = []
    for img_set, ref_set in zip(img_mod, ref_mod):
        blended = _blend_mod_params(img_set, ref_set, num_ref, seq_len, ref_tokens_at_end=ref_tokens_at_end)
        blended_sets.append(blended)
    return tuple(blended_sets)


def blend_single_block_mods(
    single_mod: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ref_mod: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_ref: int,
    seq_len: int,
    ref_tokens_at_end: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend single-block modulations for [txt, img, ref] or [txt, ref, img].

    Args:
        single_mod: Tuple of (shift, scale, gate) for image stream.
        ref_mod: Tuple of (shift, scale, gate) for reference tokens.
        num_ref: Number of reference tokens.
        seq_len: Total sequence length.
        ref_tokens_at_end: Whether reference tokens are at the sequence tail.

    Returns:
        Blended modulation tuple aligned with token layout.
    """
    blended = []
    for im, rm in zip(single_mod, ref_mod):
        if im.ndim == 2:
            im = im.unsqueeze(1)
            rm = rm.unsqueeze(1)
        B = im.shape[0]
        im_expanded = im.expand(B, seq_len, -1)
        rm_expanded = rm.expand(B, seq_len, -1)
        idx = torch.arange(seq_len, device=im.device).view(1, seq_len, 1)
        if ref_tokens_at_end:
            ref_mask = idx >= (seq_len - num_ref)
        else:
            ref_mask = idx < num_ref
        blended.append(torch.where(ref_mask, rm_expanded, im_expanded))
    return tuple(blended)
