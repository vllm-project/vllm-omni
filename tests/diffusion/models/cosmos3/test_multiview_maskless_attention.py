# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Maskless backend: exact coverage of the sparse predicate, the FP32 merge, and the production op."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import multiview_maskless_attention as m
from vllm_omni.diffusion.models.cosmos3.multiview_attention import multiview_attention
from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
    PaddedAttentionGeometry,
    _make_pair_allowed,
    build_multiview_flex_metadata,
    padded_multiview_flex_attention,
)
from vllm_omni.diffusion.models.cosmos3.multiview_maskless_plan import (
    META_IDENTITY_K,
    META_IDENTITY_Q,
    META_MAX_SEQLEN_K,
    META_MAX_SEQLEN_Q,
    TENSORS_PER_PASS,
    build_multiview_maskless_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def layout(
    *,
    views: int = 2,
    controls: bool = True,
    lidar: bool = True,
    window: float = 0.4,
    caption_lengths: tuple[int, ...] | None = None,
    backend: str = "maskless",
) -> MultiviewLayout:
    items = [
        MaskItem((views * 5, 1, 2), views, is_control=is_control, seconds_per_frame=0.2)
        for is_control in ((True, False) if controls else (False,))
    ]
    if lidar:
        items += [
            MaskItem((11, 1, 1), 1, view_offset=views, is_control=is_control, is_lidar=True, seconds_per_frame=0.1)
            for is_control in ((True, False) if controls else (False,))
        ]
    if caption_lengths is None:
        caption_lengths = tuple(2 + view for view in range(views))
    return MultiviewLayout(
        tuple(items),
        cross_view_past_window_seconds=window,
        caption_lengths=caption_lengths,
        max_und_tokens=64,
        backend=backend,
    )


def predicate_mask(spec: MultiviewLayout, num_und: int) -> torch.Tensor:
    """The production predicate, dense, over the unpadded ``[UND | GEN]`` key stream."""
    geometry = PaddedAttentionGeometry(spec.gen_tokens, spec.gen_tokens, num_und, num_und)
    metadata = build_multiview_flex_metadata(spec, geometry, "cpu")
    pair_allowed = _make_pair_allowed(
        metadata.query_vectors(), metadata.key_vectors(), spec.cross_view_past_window_seconds
    )
    return pair_allowed(torch.arange(spec.gen_tokens)[:, None], torch.arange(num_und + spec.gen_tokens)[None, :])


def tensors(spec, num_und, *, dtype=torch.float64, heads=4, kv_heads=2, head_dim=8, device="cpu"):
    generator = torch.Generator().manual_seed(4)
    shapes = (
        (spec.gen_tokens, heads),
        (spec.gen_tokens, kv_heads),
        (spec.gen_tokens, kv_heads),
        (num_und, kv_heads),
        (num_und, kv_heads),
    )
    return [torch.randn(1, n, h, head_dim, generator=generator).to(device=device, dtype=dtype) for n, h in shapes]


def dense_reference(spec, q, k, v, k_und, v_und) -> torch.Tensor:
    mask = predicate_mask(spec, k_und.shape[1]).to(q.device)
    ratio = q.shape[2] // k.shape[2]
    keys = torch.cat([k_und, k], 1).double().repeat_interleave(ratio, 2)
    values = torch.cat([v_und, v], 1).double().repeat_interleave(ratio, 2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.double(), keys) / math.sqrt(q.shape[-1])
    scores.masked_fill_(~mask, -float("inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), values)


def prepared_context(spec, num_und, *, kernel="torch", device="cpu") -> MultiviewAttentionContext:
    return m.prepare_maskless_context(
        MultiviewAttentionContext(spec, {}, {}), num_und_tokens=num_und, device=device, kernel=kernel
    )


def merge_oracle(outputs, lse_tensors):
    weights = torch.stack(lse_tensors).double().softmax(0)
    return (torch.stack(outputs).double() * weights[..., None]).sum(0)


# -- Planner --------------------------------------------------------------------------


def _token_pair_counts(spec, plan, num_und) -> torch.Tensor:
    counts = torch.zeros(spec.gen_tokens, num_und + spec.gen_tokens, dtype=torch.int64)
    for attention_pass in plan.passes:
        q_bounds = zip(attention_pass.cu_seqlens_q[:-1].tolist(), attention_pass.cu_seqlens_q[1:].tolist())
        k_bounds = zip(attention_pass.cu_seqlens_k[:-1].tolist(), attention_pass.cu_seqlens_k[1:].tolist())
        key_offset = 0 if attention_pass.keys_from_und else num_und
        for (qa, qb), (ka, kb) in zip(q_bounds, k_bounds, strict=True):
            assert qb > qa and kb > ka, "no empty varlen segment"
            queries = attention_pass.q_index[qa:qb]
            keys = attention_pass.k_index[ka:kb] + key_offset
            counts[queries[:, None], keys[None, :]] += 1
    return counts


@pytest.mark.cpu
@pytest.mark.parametrize("window", [0.0, 0.4, 0.40001])
@pytest.mark.parametrize(
    ("views", "controls", "lidar"),
    [(1, False, False), (1, True, True), (2, True, True), (3, False, True), (5, True, False), (4, False, False)],
)
def test_plan_covers_predicate_exactly_once(window: float, views: int, controls: bool, lidar: bool) -> None:
    spec = layout(views=views, controls=controls, lidar=lidar, window=window)
    num_und = sum(spec.caption_lengths)
    plan = build_multiview_maskless_plan(spec, num_und, "cpu")

    assert torch.equal(_token_pair_counts(spec, plan, num_und), predicate_mask(spec, num_und).to(torch.int64))

    same_view, caption = plan.passes[0], plan.passes[-1]
    assert same_view.name == "same_view" and not same_view.keys_from_und
    assert same_view.q_index.numel() == spec.gen_tokens
    assert caption.name == "caption" and caption.keys_from_und
    groups = views + int(lidar)
    assert plan.num_levels == (0 if groups == 1 else plan.num_levels)
    assert plan.num_levels <= (math.ceil(math.log2(groups)) if groups > 1 else 0)
    assert len(plan.passes) == 2 + plan.num_levels
    for attention_pass in plan.passes:
        assert torch.unique(attention_pass.q_index).numel() == attention_pass.q_index.numel()
        q_lengths = attention_pass.cu_seqlens_q.diff().tolist()
        k_lengths = attention_pass.cu_seqlens_k.diff().tolist()
        assert attention_pass.meta[META_MAX_SEQLEN_Q] == max(q_lengths)
        assert attention_pass.meta[META_MAX_SEQLEN_K] == max(k_lengths)
        assert attention_pass.meta.device.type == "cpu"
        assert attention_pass.cu_seqlens_q.dtype == attention_pass.cu_seqlens_k.dtype == torch.int32
    # The same-view pass needs no gathers when each view's tokens are already
    # contiguous: no control items, or a single camera view (control and target
    # items of one view follow each other in the packed stream).
    identity = (not controls) or views == 1
    assert bool(same_view.meta[META_IDENTITY_Q]) == bool(same_view.meta[META_IDENTITY_K]) == identity
    assert len(plan.flatten()) == TENSORS_PER_PASS * len(plan.passes)


@pytest.mark.cpu
def test_plan_handles_mixed_rates_at_tolerance_boundaries() -> None:
    # Five single-camera items with distinct frame periods straddling the
    # 1e-4 s window tolerance exercise every window edge the predicate has.
    rates = (0.1, 0.2, 0.39995, 0.40005, 0.40015)
    spec = MultiviewLayout(
        tuple(MaskItem((6, 1, 1), 1, view_offset=index, seconds_per_frame=rate) for index, rate in enumerate(rates)),
        cross_view_past_window_seconds=0.4,
        caption_lengths=(1, 2, 3, 2, 1),
        max_und_tokens=64,
        backend="maskless",
    )
    plan = build_multiview_maskless_plan(spec, 9, "cpu")
    assert torch.equal(_token_pair_counts(spec, plan, 9), predicate_mask(spec, 9).to(torch.int64))
    assert 1 <= plan.num_levels <= 3


@pytest.mark.cpu
def test_plan_marks_only_caption_keys_dynamic() -> None:
    spec = layout()
    plan = build_multiview_maskless_plan(spec, sum(spec.caption_lengths), "cpu")
    for attention_pass in plan.passes:
        dynamic = getattr(attention_pass.k_index, "_dynamo_dynamic_indices", set())
        assert dynamic == ({0} if attention_pass.keys_from_und else set())
        assert not getattr(attention_pass.q_index, "_dynamo_dynamic_indices", set())
        assert not getattr(attention_pass.meta, "_dynamo_dynamic_indices", set())


@pytest.mark.cpu
def test_plan_is_cached_per_prompt_length() -> None:
    spec = layout()
    context = MultiviewAttentionContext(spec, {}, {})
    first = m.get_maskless_plan(context, num_und_tokens=5, device="cpu")
    assert m.get_maskless_plan(context, num_und_tokens=5, device="cpu") is first
    other = m.get_maskless_plan(
        replace(context, layout=replace(spec, caption_lengths=(3, 4))), num_und_tokens=7, device="cpu"
    )
    assert other is not first and len(context.mask_cache) == 2


# -- Merge ----------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize("branches", [1, 2, 3, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_merge_matches_fp64_reference(branches: int, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(721)
    outputs = [torch.randn(64, 2, 16, generator=generator).to(dtype) for _ in range(branches)]
    lses = [torch.randn(64, 2, generator=generator) * 3 for _ in range(branches)]
    if branches > 1:
        # Weighted cancellation detects any cast of weights or products below
        # FP32; the other rows cover absent and dominant branches.
        outputs[0][:1], outputs[1][:1] = 1, -1
        lses[0][:1], lses[1][:1] = math.log(0.3), math.log(0.7)
        for branch in range(1, branches):
            outputs[branch][1:15] = 0
            lses[branch][1:8] = float("-inf")
            lses[branch][8:15] = torch.finfo(torch.float32).min
            lses[branch][15:22] = -10000
            lses[branch][22:29] = 10000 if branch == 1 else -10000
        if branches > 2:
            outputs[2][:1] = 0
            lses[2][:1] = torch.finfo(torch.float32).min
    expected = merge_oracle(outputs, lses)
    tolerance = {torch.float32: 1e-6, torch.float16: 1e-3, torch.bfloat16: 1e-2}[dtype]

    merged = m._merge_attention_outputs(outputs, lses)
    assert merged.dtype == dtype and torch.isfinite(merged).all()
    torch.testing.assert_close(merged.double(), expected, atol=tolerance, rtol=tolerance)
    if branches > 1 and dtype != torch.float32:
        torch.testing.assert_close(merged[:1], expected[:1].to(dtype), atol=0, rtol=0)

    # The streaming recurrence the op runs equals the all-branches reference,
    # and its LSE accumulator is the logaddexp over branches.
    acc, acc_lse = outputs[0].float(), lses[0].clone()
    for out, lse in zip(outputs[1:], lses[1:], strict=True):
        acc, acc_lse = m.merge_step(acc, acc_lse, out, lse)
    torch.testing.assert_close(acc.double(), expected, atol=tolerance, rtol=tolerance)
    finite = torch.isfinite(torch.stack(lses)).all(0)
    torch.testing.assert_close(acc_lse[finite], torch.stack(lses).logsumexp(0)[finite], atol=1e-5, rtol=1e-5)


# -- Attention op ---------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize("window", [0.0, 0.4])
@pytest.mark.parametrize(
    ("views", "controls", "lidar"),
    [(1, False, False), (1, True, True), (2, True, True), (3, False, True), (3, True, False)],
)
def test_attention_matches_dense_reference(window: float, views: int, controls: bool, lidar: bool) -> None:
    spec = layout(views=views, controls=controls, lidar=lidar, window=window)
    num_und = sum(spec.caption_lengths)
    qkv = tensors(spec, num_und)
    actual = multiview_attention(*qkv, prepared_context(spec, num_und))
    assert actual.dtype == torch.float64 and actual.shape == qkv[0].shape
    # The passes run in fp64 here, but the production merge accumulates in
    # FP32 by design (as NATTEN does), so agreement is FP32-level.
    torch.testing.assert_close(actual, dense_reference(spec, *qkv), atol=1e-6, rtol=1e-5)


@pytest.mark.cpu
def test_maskless_matches_triton_flex_on_cpu() -> None:
    spec = layout()
    num_und = sum(spec.caption_lengths)
    qkv = tensors(spec, num_und, dtype=torch.float32, head_dim=16)
    maskless = multiview_attention(*qkv, prepared_context(spec, num_und))
    triton = padded_multiview_flex_attention(*qkv, MultiviewAttentionContext(replace(spec, backend="triton"), {}))
    torch.testing.assert_close(maskless, triton, atol=1e-5, rtol=1e-5)


@pytest.mark.cpu
@pytest.mark.parametrize("chunk_size", [3, 7, 64])
def test_chunked_merge_is_exact_for_any_chunk_size(monkeypatch: pytest.MonkeyPatch, chunk_size: int) -> None:
    assert m.MERGE_CHUNK_SIZE == 8192
    monkeypatch.setattr(m, "MERGE_CHUNK_SIZE", chunk_size)
    spec = layout(views=3)
    num_und = sum(spec.caption_lengths)
    qkv = tensors(spec, num_und)
    torch.testing.assert_close(
        multiview_attention(*qkv, prepared_context(spec, num_und)), dense_reference(spec, *qkv), atol=1e-6, rtol=1e-5
    )


@pytest.mark.cpu
def test_batch_two_rejected() -> None:
    spec = layout()
    num_und = sum(spec.caption_lengths)
    qkv = tensors(spec, num_und)
    with pytest.raises(ValueError, match="B == 1"):
        multiview_attention(*(x.expand(2, -1, -1, -1) for x in qkv), prepared_context(spec, num_und))


@pytest.mark.cpu
@pytest.mark.parametrize(("planned", "actual"), [(5, 10), (10, 5)])
def test_mismatched_gen_length_rejected_before_attention(planned: int, actual: int) -> None:
    def unexpected_kernel(*args, **kwargs):
        pytest.fail("GEN length mismatch must be rejected before launching attention")

    m.register_pass_kernel("test-unexpected", lambda: unexpected_kernel)
    spec = MultiviewLayout(
        (MaskItem((planned, 1, 1), 1),), 0.4, caption_lengths=(3,), max_und_tokens=64, backend="maskless"
    )
    context = prepared_context(spec, 3, kernel="test-unexpected")
    qkv = tensors(replace(spec, items=(MaskItem((actual, 1, 1), 1),)), 3)
    with pytest.raises(ValueError, match="GEN length does not match the request plan"):
        multiview_attention(*qkv, context)


@pytest.mark.cpu
def test_unprepared_context_is_rejected() -> None:
    spec = layout()
    qkv = tensors(spec, sum(spec.caption_lengths))
    with pytest.raises(RuntimeError, match="prepare_maskless_context"):
        multiview_attention(*qkv, MultiviewAttentionContext(spec, {}, {}))


@pytest.mark.cpu
@pytest.mark.parametrize(("lengths", "heads", "dim"), [([2**31], 1, 1), ([2**30, 2**30], 1, 1), ([2**20], 16, 128)])
def test_int32_boundaries_without_allocating(lengths: list[int], heads: int, dim: int) -> None:
    with pytest.raises(ValueError, match="int32 indexing"):
        m.validate_indexing(lengths, heads, dim)
    m.validate_indexing([2**31 - 1], 1, 1)


@pytest.mark.cpu
def test_lse_axis_normalization_is_explicit() -> None:
    square = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert torch.equal(m.normalize_varlen_lse(square, 4, 4), square.T)
    with pytest.raises(ValueError, match="Expected FlashAttention LSE"):
        m.normalize_varlen_lse(torch.zeros(1, 4, 4), 4, 4)
    with pytest.raises(ValueError, match="Unknown Cosmos3 maskless pass kernel"):
        m.get_pass_kernel("no-such-kernel")


@pytest.mark.cpu
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_output_preserves_callers_tensor_mode(inference: bool, compiled: bool) -> None:
    spec = layout()
    num_und = sum(spec.caption_lengths)
    qkv = tensors(spec, num_und)
    context = prepared_context(spec, num_und)
    attention = torch.compile(multiview_attention, backend="eager", fullgraph=True) if compiled else multiview_attention
    with torch.inference_mode() if inference else torch.no_grad():
        result = attention(*qkv, context)
        assert result.is_inference() == inference
        if not inference:
            # HSDP/offload callers require outputs with accessible version counters.
            assert isinstance(result._version, int)
        torch.testing.assert_close(result, dense_reference(spec, *qkv), atol=1e-6, rtol=1e-5)


@pytest.mark.cpu
def test_prompt_lengths_do_not_recompile() -> None:
    from torch._dynamo.testing import CompileCounterWithBackend

    torch._dynamo.reset()
    counter = CompileCounterWithBackend("eager")

    # Exercise the dispatch boundary with a small projection as the GEN region.
    def region(q, k, v, k_und, v_und, context):
        return multiview_attention(q + 0.0, k, v, k_und, v_und, context) * 1.0

    compiled = torch.compile(region, backend=counter, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        for length in [*range(2, 13), 4, 8, 2]:
            for branch_length in (length, length + 3):
                spec = layout(caption_lengths=(1, branch_length - 1))
                qkv = tensors(spec, branch_length)
                for tensor in qkv[3:]:
                    torch._dynamo.mark_dynamic(tensor, 1)
                context = prepared_context(spec, branch_length)
                torch.testing.assert_close(compiled(*qkv, context), dense_reference(spec, *qkv), atol=1e-6, rtol=1e-5)
    assert counter.frame_count == 1


@pytest.mark.cpu
def test_ulysses_maskless_trims_padding_and_preserves_batch_one(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import multiview_parallel as parallel

    spec = layout(views=1)
    num_und = sum(spec.caption_lengths)
    q, k, v, k_und, v_und = tensors(spec, num_und)
    cp = 2
    length = (spec.gen_tokens + cp - 1) // cp
    padded = [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, length * cp - x.shape[1])) for x in (q, k, v)]
    for rank in range(cp):
        full = [x[:, :, rank * (x.shape[2] // cp) : (rank + 1) * (x.shape[2] // cp)] for x in padded]
        calls = iter(full)

        def exchange(x, group, scatter, gather):
            assert x.shape[0] == 1
            if scatter == 2:
                return next(calls)
            assert x.shape[1] == length * cp
            return x[:, rank * length : (rank + 1) * length]

        monkeypatch.setattr(parallel, "_all_to_all", exchange)
        local_qkv = [x[:, rank * length : (rank + 1) * length] for x in padded] + [k_und, v_und]
        head_qkv = [x[:, : spec.gen_tokens] for x in full] + [
            k_und[:, :, rank : rank + 1],
            v_und[:, :, rank : rank + 1],
        ]
        context = prepared_context(spec, num_und)
        result = parallel.multiview_ulysses_attention(*local_qkv, context, group=object(), rank=rank, world_size=cp)
        expected = multiview_attention(*head_qkv, context)
        expected = torch.nn.functional.pad(expected, (0, 0, 0, 0, 0, length * cp - spec.gen_tokens))
        torch.testing.assert_close(result, expected[:, rank * length : (rank + 1) * length])


# -- CUDA -----------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and vLLM's bundled FlashAttention")
@pytest.mark.parametrize("fa_version", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_maskless_matches_dense_and_sparse(fa_version: int, dtype: torch.dtype) -> None:
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

    if not is_fa_version_supported(fa_version):
        pytest.skip(f"FlashAttention {fa_version} is unavailable on this device")
    spec = layout(views=3)
    num_und = sum(spec.caption_lengths)
    qkv = tensors(spec, num_und, dtype=dtype, head_dim=128, kv_heads=2, device="cuda")
    tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
    with torch.inference_mode():
        maskless = multiview_attention(*qkv, prepared_context(spec, num_und, kernel=f"fa{fa_version}", device="cuda"))
        torch.testing.assert_close(maskless.double(), dense_reference(spec, *qkv), atol=tolerance, rtol=tolerance)
        triton = padded_multiview_flex_attention(*qkv, MultiviewAttentionContext(replace(spec, backend="triton"), {}))
        torch.testing.assert_close(maskless, triton, atol=tolerance, rtol=tolerance)
