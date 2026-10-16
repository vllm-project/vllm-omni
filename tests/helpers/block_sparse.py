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


@torch.inference_mode()
def check_dynamic_selected_execution(adapter_cls, implementation, kv_heads, query_lengths=(129, 257, 385)):
    """Check real provider execution with dynamic lengths and mutable selections."""
    from torch._dynamo.testing import CompileCounterWithBackend

    adapter = adapter_cls()
    adapter.prepare(implementation, 128, 4, kv_heads, torch.device("cuda"), (64, 64))

    def execute(q, k, v, selection):
        # Configuration constants must not become symbolic scalar arguments.
        return adapter.execute(q, k, v, selection, 0.125, (64, 64))

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(execute, backend=counter, fullgraph=True, dynamic=True)
    shapes = list(zip(query_lengths, (321, 449, 577)))
    for q_len, kv_len in (*shapes, shapes[0]):
        q, k, v = make_attention_inputs(kv_heads=kv_heads, q_len=q_len, kv_len=kv_len)
        rows = (q.shape[0], q.shape[2], math.ceil(q_len / 64))
        first = (torch.arange(math.prod(rows), device=q.device).reshape(rows) % 2).int()
        counts = first + 1
        last = torch.where(counts == 2, math.ceil(kv_len / 64) - 1, -999).int()
        results = []
        for active_first in (first, 1 - first, first):
            selection = BlockSelection(torch.stack((active_first, last), dim=-1), counts)
            expected = selected_attention_reference(q, k, v, selection, 0.125, (64, 64))
            eager = execute(q, k, v, selection)
            actual = compiled(q, k, v, selection)
            torch.testing.assert_close(actual.float(), expected, atol=0.004, rtol=0.02)
            torch.testing.assert_close(actual, eager, atol=0, rtol=0)
            assert actual.is_contiguous() and actual.dtype == q.dtype and actual.device == q.device
            results.append(actual.clone())
        assert not torch.equal(results[0], results[1])
        torch.testing.assert_close(results[0], results[2], atol=0, rtol=0)
    assert counter.frame_count == 1


@torch.inference_mode()
def check_dynamic_sparse_owners(adapter_cls, implementation, kv_heads, query_lengths=(65, 193)):
    """Exercise regional compilation across owners and changing block counts."""
    from torch._dynamo.testing import CompileCounterWithBackend

    from vllm_omni.diffusion.attention.block_sparse import BlockSparseAttention
    from vllm_omni.diffusion.compile import regionally_compile
    from vllm_omni.diffusion.data import BlockSparseAttentionSpec

    class SparseBlock(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            spec = BlockSparseAttentionSpec(
                name="block_sparse",
                config={"backend": {"require": adapter_cls.provider, "implementation": implementation}},
            )
            # Distinct owner configuration catches dispatch to the wrong instance.
            self.attention = BlockSparseAttention(
                4, kv_heads, 128, 0.125 * (1 + index / 10), False, "BSHD", spec, adapter=adapter_cls()
            )

        def forward(self, q, k, v):
            return self.attention.forward(q, k, v)

    model = torch.nn.Module()
    model._repeated_blocks = ["SparseBlock"]
    # More owners than Dynamo's default per-code recompilation limit.
    model.blocks = torch.nn.ModuleList(SparseBlock(index) for index in range(10))
    counter = CompileCounterWithBackend("inductor")
    regionally_compile(model, backend=counter, fullgraph=True, dynamic=True)
    shapes = list(zip(query_lengths, (1089, 1217)))
    for q_len, kv_len in (*shapes, shapes[0]):
        q, k, v = make_attention_inputs(kv_heads=kv_heads, q_len=q_len, kv_len=kv_len)
        outputs = []
        for block in model.blocks:
            impl = block.attention
            selection = impl.selector.select(q, k, impl.scale, 0)
            expected = selected_attention_reference(q, k, v, selection, impl.scale, (64, 64))
            # First calls prepare each owner/geometry from inside compiled dispatch.
            actual = block(q, k, v)
            torch.testing.assert_close(actual.float(), expected, atol=0.004, rtol=0.02)
            torch.testing.assert_close(actual, impl.forward(q, k, v), atol=0, rtol=0)
            outputs.append(actual.clone())
        assert not torch.equal(outputs[0], outputs[-1])
    assert counter.frame_count == 1
