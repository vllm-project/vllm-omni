# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Select one Ulysses wire protocol for every rank in an H3 denoise forward."""

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils.rocm_h3_head_fmha import prepare_head_fmha
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_ulysses_mode

# Qualified [real document, padding] partitions of the 37,760-row BF16 workload.
QUALIFIED_CUTS = frozenset((37746, 37747, 37748, 37749, 37750, 37751, 37753, 37755, 37757))


def _supports_layer(owner, group) -> bool:
    if owner is None:  # A Cache-DiT wrapper can replace a physical H3 block.
        return False
    layer = owner.attention
    impl, _, _ = layer.effective_attention()
    strategy = layer._get_active_parallel_strategy()
    return (
        type(impl) is FlashAttentionImpl
        and impl._kernel_variant == "aiter"
        and not impl.causal
        and not impl.is_cross_attn
        and not impl.fa_deterministic
        and impl.softmax_scale == 128**-0.5
        and owner.num_heads == owner.num_kv_heads == 56
        and owner.head_dim == 128
        and owner.to_gate_compress is None
        and owner.vsa_sparsity is None
        and owner.vdn is None
        and layer._effective_kv_cache_dtype() in (None, "float")
        and strategy.name == "ulysses"
        and strategy._ulysses_pg is group
        and (strategy._scatter_idx, strategy._gather_idx) == (2, 1)
        and not strategy._use_sync
        and not strategy._ulysses_a2a_permute
    )


@torch.compiler.disable
def prepare_overlap(blocks, hidden: torch.Tensor, bounds: torch.Tensor, cut: int) -> bool:
    """Vote once before the blocks. A missing kernel or invalid layout on any rank selects stock on all ranks."""
    group = get_sp_group().ulysses_group
    if dist.get_world_size(group) != 4 or torch.get_device_module(hidden.device).is_current_stream_capturing():
        return False

    eligible = (
        hidden.dtype == torch.bfloat16
        and hidden.shape[0] == 9440
        and bounds.shape == (3,)
        and bounds.device == hidden.device
        and bounds.is_contiguous()
        and not bounds.is_neg()
        and bounds.dtype == torch.int32
        and not torch.is_grad_enabled()
        and not hidden.requires_grad
        and not torch.are_deterministic_algorithms_enabled()
        and get_ulysses_mode(default="strict") == "strict"
        and all(_supports_layer(getattr(block, "attn", None), group) for block in blocks)
        and prepare_head_fmha(hidden.device) is not None
    )
    # Include the device values in the vote. Shape and max_seqlen alone do not
    # establish the document boundary consumed by the grouped kernel.
    vote = torch.zeros(2, device=hidden.device, dtype=torch.int32)
    if eligible and cut in QUALIFIED_CUTS:
        matches = (bounds == bounds.new_tensor([0, cut, 37760])).all()
        vote = torch.where(matches, bounds.new_tensor([cut, -cut]), vote)
    # MIN yields (smallest cut, -largest cut). A zero vetoes the split.
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    smallest, negative_largest = vote.tolist()
    return smallest > 0 and smallest == -negative_largest


def run_overlap(q, k, v, bounds):
    from vllm_omni.diffusion.attention.parallel.rocm_h3_head_overlap import run_head_overlap

    compute = prepare_head_fmha(q.device)
    assert compute is not None  # All ranks loaded their kernel in prepare_overlap.
    return run_head_overlap(q, k, v, bounds, get_sp_group().ulysses_group, compute)
