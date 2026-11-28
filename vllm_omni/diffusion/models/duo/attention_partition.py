# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full attention partition merge; every KV row contributes to one softmax."""

from vllm.triton_utils import tl, triton


@triton.jit
def merge_partitions(
    partial,
    lse,
    output,
    rows: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    parts: tl.constexpr,
    lse_head: tl.constexpr,
    lse_row: tl.constexpr,
    bp: tl.constexpr,
    bd: tl.constexpr,
    lse_part: tl.constexpr = 0,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    ss = tl.arange(0, bp)
    dd = tl.arange(0, bd)
    part_stride = lse_part if lse_part else rows * lse_row
    log = tl.load(lse + head * lse_head + ss * part_stride + row * lse_row, ss < parts, -float("inf"))
    weight = tl.exp(log - tl.max(log, 0))
    values = tl.load(
        partial + ((ss[:, None] * rows + row) * heads + head) * dim + dd[None, :],
        (ss[:, None] < parts) & (dd[None, :] < dim),
        0,
    ).to(tl.float32)
    result = tl.sum(values * weight[:, None], 0) / tl.sum(weight, 0)
    tl.store(output + (row * heads + head) * dim + dd, result, dd < dim)


def broadcast_partition_attention(q, k, v, cu_k, max_k, scale, parts, q_descale=None, k_descale=None, v_descale=None):
    """Run the installed FA3 op with dense Q batches sharing one storage view.

    KV remains ragged and every row is included. The public varlen wrapper
    assumes a non-null cu_q; the underlying FA3 op supports dense Q directly.
    It returns LSE as [partition, head, row], without a layout copy.
    """
    import torch

    return torch.ops._vllm_fa3_C.fwd(
        q.unsqueeze(0).expand(parts, -1, -1, -1),
        k,
        v,
        None,
        None,
        None,
        None,  # new KV, q_v, output
        None,
        cu_k,
        None,
        None,
        None,  # sequence metadata
        q.shape[0],
        max_k,
        None,
        None,
        None,  # lengths, paging
        None,
        None,
        None,
        q_descale,
        k_descale,
        v_descale,  # rotary, descales
        scale,
        False,
        -1,
        -1,
        0.0,
        True,
        None,
        1,
        None,
        0,
        None,
        1,
        0,
        None,
    )[:2]
