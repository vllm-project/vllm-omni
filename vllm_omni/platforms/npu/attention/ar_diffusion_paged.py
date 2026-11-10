# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend paged self-attention for AR diffusion."""

import os

import torch


from vllm_omni.experimental.ar_diffusion.kv_cache.config import contiguous_kv_gather_enabled
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import _stage_window


def npu_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    causal: bool = False,
    host_lengths: tuple[int, int] | None = None,
    stage_key: torch.Tensor | None = None,
    stage_value: torch.Tensor | None = None,
    reuse_history: bool = False,
    stage_first_block: int = 0,
) -> torch.Tensor:
    """Gather the visible window and attend, optionally without device readback.

    Host lengths are authoritative only for the singleton context that created
    them. They describe valid tokens, never the padded block-table capacity.
    """
    import torch_npu

    query_flat = query.flatten(0, 1) if query.dim() == 4 else query
    if host_lengths is not None:
        q_len, kv_len = host_lengths
        if (
            query_start_loc.numel() != 2
            or seq_lens.numel() != 1
            or q_len != query_flat.shape[0]
            or q_len <= 0
            or not 0 < kv_len <= max_seq_len
        ):
            raise ValueError("NPU host lengths require a matching singleton query and valid KV length")
        actual_seq_q, actual_seq_kv = [q_len], [kv_len]

    # Preserve the existing whole-block gather, including its padding behavior;
    # only the source of the singleton valid lengths changes in the fast path.
    block_size = key_cache.shape[1]
    num_blocks = (int(max_seq_len) + block_size - 1) // block_size
    block_ids = block_table[0, :num_blocks].long()
    # Host lengths are an explicit opt-in from the LingBot deployment.
    # Keep the process-wide FP8 setting out of other paged attention callers.
    fp8_fia_mode = os.environ.get("VLLM_OMNI_AR_DIFFUSION_FP8_FIA_MODE") if host_lengths is not None else None
    if fp8_fia_mode:
        if fp8_fia_mode != "C8V16_TILING512":
            raise ValueError("FP8 FIA mode must be C8V16_TILING512")
        if causal:
            raise ValueError("Dense FP8 EagleFIA currently requires noncausal block attention")
    staging = (
        contiguous_kv_gather_enabled()
        and stage_key is not None
        and stage_value is not None
        and stage_key.numel() > 0
        and stage_value.numel() > 0
    )
    if staging:
        assert stage_key is not None and stage_value is not None
        capacity = num_blocks * block_size
        if stage_key.shape[0] < capacity or stage_value.shape[0] < capacity:
            raise ValueError("NPU staging buffers must hold the padded visible window")
        if not 0 <= stage_first_block < num_blocks:
            raise ValueError("NPU stage_first_block must address a visible block")
        if (
            fp8_fia_mode == "C8V16_TILING512"
            and os.environ.get("VLLM_OMNI_LINGBOT_FP8_FIA_BNSD_GATHER") == "1"
            and host_lengths is not None
            and key_cache.dtype == value_cache.dtype == torch.bfloat16
            and key_cache.shape == value_cache.shape
            and key_cache.shape[-1] == 128
            and key_cache.is_contiguous()
            and value_cache.is_contiguous()
        ):
            from vllm_omni.platforms.npu.attention.eagle_fp8_fia import eagle_fp8_fia
            from vllm_omni.platforms.npu.attention.head_major_kv_gather import gather_kv_bnsd

            first_block = stage_first_block if reuse_history else 0
            if first_block * block_size >= kv_len:
                raise ValueError("NPU head-major staging suffix must overlap the valid KV window")
            k, v = gather_kv_bnsd(
                key_cache, value_cache, block_ids, kv_len, stage_key, stage_value, first_block=first_block
            )
            return eagle_fp8_fia(query_flat, k, v, scale=softmax_scale, kv_head_major=True).reshape(query.shape)
        staged_k, staged_v = stage_key[:capacity], stage_value[:capacity]
        _stage_window(
            staged_k,
            staged_v,
            key_cache,
            value_cache,
            block_ids,
            num_blocks,
            block_size,
            first_block=stage_first_block if reuse_history else 0,
        )
        gathered_k, gathered_v = staged_k[:max_seq_len], staged_v[:max_seq_len]
    else:
        gathered_k = key_cache.index_select(0, block_ids).flatten(0, 1)[: int(max_seq_len)]
        gathered_v = value_cache.index_select(0, block_ids).flatten(0, 1)[: int(max_seq_len)]
    if host_lengths is None:
        # Keep the legacy gather/readback ordering when the option is off.
        kv_len = int(seq_lens[0].item())
        actual_seq_q = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int32).cpu().tolist()
        actual_seq_kv = seq_lens.to(torch.int32).cpu().tolist()
    if fp8_fia_mode:
        from vllm_omni.platforms.npu.attention.eagle_fp8_fia import eagle_fp8_fia

        return eagle_fp8_fia(query_flat, gathered_k[:kv_len], gathered_v[:kv_len], scale=softmax_scale).reshape(
            query.shape
        )
    out = torch_npu.npu_fusion_attention(
        query_flat.contiguous(),
        gathered_k[:kv_len].contiguous(),
        gathered_v[:kv_len].contiguous(),
        head_num=query_flat.shape[1],
        input_layout="TND",
        actual_seq_qlen=actual_seq_q,
        actual_seq_kvlen=actual_seq_kv,
        scale=float(softmax_scale),
        keep_prob=1.0,
    )[0]
    return out.reshape(query.shape)
