# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Provider-independent input generation and selected-attention reference."""

import math

import torch

from vllm_omni.diffusion.attention.block_selection.abstract import BlockSelection


def make_attention_inputs(
    heads=4, kv_heads=2, q_len=129, kv_len=1089, head_size=128, value_size=None, dtype=torch.bfloat16
):
    generator = torch.Generator(device="cuda").manual_seed(17)
    q = torch.randn(2, q_len, heads, head_size, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(2, kv_len, kv_heads, head_size, device=q.device, dtype=q.dtype, generator=generator)
    v = torch.randn(2, kv_len, kv_heads, value_size or head_size, device=q.device, dtype=q.dtype, generator=generator)
    return q, k, v


def selected_attention_reference(q, k, v, selection: BlockSelection, scale, block_size):
    """FP32 masked SDPA over exactly the active entries of each selection row."""
    block_q, block_kv = block_size
    indices, counts = selection
    active = torch.arange(indices.shape[-1], device=q.device) < counts[..., None]
    # Padding entries are unspecified by the contract; never use them as indices.
    safe_indices = torch.where(active, indices, 0).long()
    mask = torch.zeros(*indices.shape[:-1], math.ceil(k.shape[1] / block_kv), device=q.device, dtype=torch.int32)
    mask.scatter_add_(-1, safe_indices, active.int())
    mask = mask.bool().repeat_interleave(block_q, 2).repeat_interleave(block_kv, 3)
    mask = mask[:, :, : q.shape[1], : k.shape[1]]
    return torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        attn_mask=mask,
        scale=scale,
        enable_gqa=True,
    ).transpose(1, 2)


def pinned_subblock_reference(q, k, scale, sparsity, prefix, block_size):
    """Independent Torch oracle for the pinned SubBlock recipe.

    Recipe: SGLang 704808ed27cef61e210100e7581c115db3ee8401;
    threshold search: kernels.py at 367e3700cfb6a0b03b2fa41a4524febf18ec1f15.
    Includes this integration's native GQA and additive protected-prefix policy.
    Does not call production pooling, scoring, budget or top-k helpers.
    """
    bq, bk = block_size
    nq, nk = math.ceil(q.shape[1] / bq), math.ceil(k.shape[1] / bk)

    def pool(x, blocks, width, factor):
        cells = blocks * (width // 16)
        result = torch.zeros(x.shape[0], x.shape[2], cells, x.shape[3], device=x.device, dtype=x.dtype)
        for cell in range(math.ceil(x.shape[1] / 16)):
            result[:, :, cell] = (x[:, cell * 16 : (cell + 1) * 16].float().mean(1) * factor).to(x.dtype)
        return result.float()

    qp = pool(q, nq, bq, scale * math.log2(math.e))
    kp = pool(k, nk, bk, 1).repeat_interleave(q.shape[2] // k.shape[2], dim=1)
    dots = qp @ kp.transpose(-1, -2)
    dots[..., math.ceil(q.shape[1] / 16) :, :] = -float("inf")
    dots[..., math.ceil(k.shape[1] / 16) :] = -float("inf")
    pairs = dots.reshape(q.shape[0], q.shape[2], nq, bq // 16, nk, bk // 16)
    scores = torch.logsumexp(pairs.permute(0, 1, 2, 4, 3, 5).flatten(-2) * math.log(2), -1)
    protected = math.ceil(prefix / bk)
    candidates = nk - protected
    retained = min(candidates, 8 * math.ceil(math.ceil((1 - sparsity) * candidates) / 8))
    rows = []
    for row in scores.reshape(-1, nk):
        values = row[protected:]
        if retained == candidates:
            chosen = torch.arange(candidates, device=q.device)
        else:
            lo, hi = values.min(), values.max() + 1
            clo, chi = float(candidates), 0.0
            ratio = math.log2(candidates / retained)
            iterations = 16 if ratio <= 2.5 else 24 if ratio <= 3.5 else 32
            for _ in range(iterations):
                fraction = min(max((clo - retained) / (clo - chi if clo - chi > 0.5 else 1), 0.05), 0.95)
                threshold = lo + (hi - lo) * fraction
                count = float((values >= threshold).sum())
                if count >= retained:
                    lo, clo = threshold, count
                else:
                    hi, chi = threshold, count
            chosen = torch.where(values >= lo)[0][:retained]
        rows.append(torch.cat((torch.arange(protected, device=q.device), chosen + protected)))
    indices = torch.stack(rows).reshape(*scores.shape[:-1], protected + retained).int()
    return scores, BlockSelection(
        indices, torch.full(indices.shape[:-1], indices.shape[-1], device=q.device, dtype=torch.int32)
    )
