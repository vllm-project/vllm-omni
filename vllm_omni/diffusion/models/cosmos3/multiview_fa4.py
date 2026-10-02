# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapt Cosmos3 multiview sparsity to vLLM's bundled FlashAttention-4.

The CuTe mask reads the host-built truth table over semantic runs. A custom op
keeps FA4's JIT cache outside torch.compile; lazy imports keep CPU hosts usable.
"""

from __future__ import annotations

from functools import cache
from typing import Any, NamedTuple

import torch

from vllm.logger import init_logger

from .multiview_flex_attention import FA4_SPARSE_KV_BLOCK_SIZE, FA4_SPARSE_Q_BLOCK_SIZE, MultiviewBlockSparsity

logger = init_logger(__name__)


class _Fa4Entry(NamedTuple):
    flash_attn_func: Any
    block_sparse_cls: Any
    mask_mod: Any
    vector_mask_mod: Any


def _build_mask_mod(cutlass, cute, fa_utils, *, vec_size: int = 1):
    """Read q word offsets, k run IDs, and the truth table from aux_tensors.

    Offsets include the truth table's row stride so kernels work across layouts.
    FA4 wraps aux indices modulo sequence lengths and masks padded lanes, so
    the token maps must cover those lengths exactly. Scalar callbacks return
    Boolean predicates; SM100/SM110 vector callbacks return packed Uint32 bits.
    """
    if vec_size not in (1, 8, 32):
        raise ValueError(f"Cosmos3 multiview FA4 mask vector size must be 1, 8, or 32, got {vec_size}.")

    @cute.jit
    def multiview_mask_mod(
        batch: Any,
        head: Any,
        m_idx: Any,
        n_idx: Any,
        seqlen_info: Any,
        aux_tensors: Any,
    ) -> Any:
        q_word_base = aux_tensors[0]
        k_group_ids = aux_tensors[1]
        allowed_words = aux_tensors[2]

        # FA4 broadcasts one logical query row across the vector, including
        # when GQA packs multiple heads into the physical query tile.
        base = q_word_base[m_idx[0]]
        if cutlass.const_expr(vec_size == 1):
            result = cute.make_rmem_tensor(n_idx.shape, dtype=cutlass.Boolean)
        else:
            result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
            result[0] = cutlass.Uint32(0)
        for j in cutlass.range_constexpr(cute.size(n_idx.shape)):
            group_k = k_group_ids[n_idx[j]]
            # Run ids are non-negative, so the unsigned read is lossless and lets
            # the divide and modulo reduce to one shift each.
            group_u = cutlass.Uint32(group_k)
            word = allowed_words[base + cutlass.Int32(group_u // cutlass.Uint32(32))]
            shift = group_u % cutlass.Uint32(32)
            keep = fa_utils.shr_u32(cutlass.Uint32(word), shift) & cutlass.Uint32(1)
            if cutlass.const_expr(vec_size == 1):
                result[j] = cutlass.Boolean(keep)
            else:
                result[0] = result[0] | (keep << cutlass.Uint32(j))
        return result.load()

    # FA4 includes this attribute in the compile-cache key as well as using it
    # to select the scalar/vector callback ABI. Never mutate it during a launch.
    multiview_mask_mod.__vec_size__ = vec_size
    return multiview_mask_mod


@cache
def _load_fa4() -> _Fa4Entry:
    """Load bundled FA4 and cache its callbacks; report dependency errors."""
    try:
        import cutlass
        import cutlass.cute as cute

        from vllm.vllm_flash_attn.cute import flash_attn_func
        from vllm.vllm_flash_attn.cute import utils as fa_utils
        from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Cosmos3 multiview backend='fa4' requires vLLM's bundled FlashAttention-4 "
            f"(vllm.vllm_flash_attn.cute, CUDA builds only). Import failed: {exc}"
        ) from exc

    entry = _Fa4Entry(
        flash_attn_func=flash_attn_func,
        block_sparse_cls=BlockSparseTensorsTorch,
        mask_mod=_build_mask_mod(cutlass, cute, fa_utils),
        vector_mask_mod=_build_mask_mod(cutlass, cute, fa_utils, vec_size=32),
    )
    logger.info("Cosmos3 multiview attention using the FlashAttention-4 CuTe backend.")
    return entry


def _validate_sparsity(
    q: torch.Tensor,
    k: torch.Tensor,
    sparsity: MultiviewBlockSparsity,
) -> None:
    """Check the host mask contract; FA4 validates its tensor and kernel inputs."""
    if (sparsity.q_block_size, sparsity.kv_block_size) != (FA4_SPARSE_Q_BLOCK_SIZE, FA4_SPARSE_KV_BLOCK_SIZE):
        raise ValueError(
            "Cosmos3 multiview FA4 requires a "
            f"({FA4_SPARSE_Q_BLOCK_SIZE}, {FA4_SPARSE_KV_BLOCK_SIZE}) sparse block map, got "
            f"({sparsity.q_block_size}, {sparsity.kv_block_size})."
        )
    for name, tensor, token_map, length in (
        ("query", q, sparsity.q_word_base, sparsity.q_len),
        ("key", k, sparsity.k_group_ids, sparsity.kv_len),
    ):
        if tensor.shape[1] != length or token_map.numel() != length:
            raise ValueError(
                f"Cosmos3 multiview FA4 {name} tokens and mask IDs must match the padded length: "
                f"tokens={tensor.shape[1]}, mask_ids={token_map.numel()}, expected={length}."
            )


# Keep Dynamo from tracing FA4's mutable JIT cache and CuTe DSL. The guard
# preserves registration across module re-imports in tests.
if not hasattr(torch.ops.vllm_omni, "cosmos3_multiview_fa4"):

    @torch.library.custom_op("vllm_omni::cosmos3_multiview_fa4", mutates_args=())
    def _cosmos3_multiview_fa4_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        partial_counts: torch.Tensor,
        partial_indices: torch.Tensor,
        full_counts: torch.Tensor,
        full_indices: torch.Tensor,
        q_word_base: torch.Tensor,
        k_group_ids: torch.Tensor,
        allowed_words: torch.Tensor,
        q_block_size: int,
        kv_block_size: int,
    ) -> torch.Tensor:
        """Rebuild FA4's inputs from the tensors/ints allowed in an op schema."""
        entry = _load_fa4()
        # Only SM100/SM110 implement vector callbacks; resolve inside the op.
        mask_mod = (
            entry.vector_mask_mod if torch.cuda.get_device_capability(q.device)[0] in (10, 11) else entry.mask_mod
        )
        # Broadcast the shared mask over batch and heads, preserving GQA packing.
        block_sparse = entry.block_sparse_cls(
            mask_block_cnt=partial_counts[None, None],
            mask_block_idx=partial_indices[None, None],
            full_block_cnt=full_counts[None, None],
            full_block_idx=full_indices[None, None],
            block_size=(q_block_size, kv_block_size),
        )
        out, _ = entry.flash_attn_func(
            q,
            k,
            v,
            mask_mod=mask_mod,
            aux_tensors=[q_word_base, k_group_ids, allowed_words],
            block_sparse_tensors=block_sparse,
        )
        return out.contiguous()

    @_cosmos3_multiview_fa4_op.register_fake
    def _(q, k, v, *args):
        return torch.empty_like(q, memory_format=torch.contiguous_format)


_cosmos3_multiview_fa4_op = torch.ops.vllm_omni.cosmos3_multiview_fa4


def multiview_fa4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: MultiviewBlockSparsity,
) -> torch.Tensor:
    """Attend over ``[B, S, H, D]`` tensors using the host-built multiview mask.

    Keep model-specific shape checks outside the op and CuTe callbacks inside it.
    FA4's default scale is 1/sqrt(head_dim), matching FlexAttention.
    """
    _validate_sparsity(q, k, sparsity)

    return _cosmos3_multiview_fa4_op(
        q,
        k,
        v,
        sparsity.partial_counts,
        sparsity.partial_indices,
        sparsity.full_counts,
        sparsity.full_indices,
        sparsity.q_word_base,
        sparsity.k_group_ids,
        sparsity.allowed_words,
        sparsity.q_block_size,
        sparsity.kv_block_size,
    )
