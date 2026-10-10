# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact request-local cache coverage; no checkpoint or optional kernels needed."""

import pytest
import torch

from vllm_omni.diffusion.models.seedvr2 import nadit as n
from vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2 import SeedVR2Pipeline
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture(params=[pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request: pytest.FixtureRequest) -> torch.device:
    if request.param == "cuda" and not (current_omni_platform.is_cuda() and torch.accelerator.is_available()):
        pytest.skip("CUDA required")
    return torch.device(request.param)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("text_len", [0, 3])
def test_rotary_tables_equal_reference_and_share_by_frequency(device, dtype, text_len):
    rope = n.NaMMRotaryEmbedding3d(24).to(device)
    other = n.NaMMRotaryEmbedding3d(24).to(device)
    cache_cls = getattr(n, "SeedVR2RotaryCache", None)
    assert cache_cls is not None, "request-local rotary cache missing"
    cache = cache_cls()
    shapes = ((1, 2, 3), (2, 1, 2))
    tables = cache.get(rope, shapes, text_len, device=device, dtype=torch.float32)
    assert tables is cache.get(other, shapes, text_len, device=device, dtype=torch.float32)
    for freqs, cos, sin in (
        (tables.video_freqs, tables.video_cos, tables.video_sin),
        (tables.text_freqs, tables.text_cos, tables.text_sin),
    ):
        q = torch.randn(2, freqs.shape[0], 32, device=device, dtype=dtype)
        torch.testing.assert_close(n.apply_rotary_emb(freqs, q), n.apply_rotary_emb_cached(cos, sin, q), rtol=0, atol=0)
    torch.testing.assert_close(
        tables.video_freqs,
        rope.window_freqs_batch(list(shapes), text_len, device=device, dtype=torch.float32),
        rtol=0,
        atol=0,
    )


@pytest.mark.cpu
def test_rotary_invalidation_and_bounded_request_ownership():
    rope = n.NaMMRotaryEmbedding3d(24)
    cache_cls = getattr(n, "SeedVR2RotaryCache", None)
    assert cache_cls is not None, "request-local rotary cache missing"
    cache = cache_cls(capacity=2)
    kwargs = dict(device=torch.device("cpu"), dtype=torch.float32)
    shapes = ((1, 2, 3),)
    first = cache.get(rope, shapes, 2, **kwargs)
    with torch.no_grad():
        rope.freqs.mul_(0.5)
    changed = cache.get(rope, shapes, 2, **kwargs)
    assert changed is not first and not torch.equal(changed.video_freqs, first.video_freqs)
    state = {"freqs": rope.freqs * 0.25}
    rope.load_state_dict(state)
    loaded = cache.get(rope, shapes, 2, **kwargs)
    assert loaded is not changed
    fresh_rope = n.NaMMRotaryEmbedding3d(24)
    fresh_rope.load_state_dict(state)
    torch.testing.assert_close(
        loaded.video_freqs,
        fresh_rope.window_freqs_batch(list(shapes), 2, **kwargs),
        rtol=0,
        atol=0,
    )
    cache.get(rope, shapes, 3, **kwargs)
    cache.get(rope, shapes, 4, **kwargs)
    assert len(cache.entries) == 2
    assert cache_cls().get(rope, shapes, 4, **kwargs) is not cache.get(rope, shapes, 4, **kwargs)
    rope.to(torch.float64)
    assert cache.get(rope, shapes, 4, **kwargs) is not loaded


def _context(device, empty=False):
    lengths = [] if empty else [3, 1, 3]
    video_offsets = [0]
    joint_offsets = [0]
    for length in lengths:
        video_offsets.append(video_offsets[-1] + length)
        joint_offsets.append(joint_offsets[-1] + length + 2)
    return n.LocalWindowContext(
        layout_key=None,
        window_shapes=torch.tensor([[1, 1, k] for k in lengths], device=device).reshape(-1, 3),
        video_cu_seqlens=torch.tensor(video_offsets, device=device, dtype=torch.int32),
        joint_cu_seqlens=torch.tensor(joint_offsets, device=device, dtype=torch.int32),
        joint_order=torch.arange(joint_offsets[-1], device=device),
        vid_src=torch.arange(video_offsets[-1], device=device),
        txt_src=torch.arange(2 * len(lengths), device=device),
        text_len=2,
        local_windows=len(lengths),
        global_windows=len(lengths),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_grouped_sdpa_reuses_rows_without_crossing_windows(device, dtype):
    ctx = _context(device)
    q = torch.randn(13, 2, 8, device=device, dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = n.grouped_window_sdpa(q, k, v, ctx, softmax_scale=0.3)
    groups = getattr(ctx, "sdpa_groups", None)
    assert groups is not None, "GPU row plan missing"
    pointers = [rows.data_ptr() for _, _, rows in groups]
    assert torch.equal(out, n.grouped_window_sdpa(q, k, v, ctx, softmax_scale=0.3))
    assert pointers == [rows.data_ptr() for _, _, rows in ctx.sdpa_groups]
    ctx.cache_enabled = False
    torch.testing.assert_close(out, n.grouped_window_sdpa(q, k, v, ctx, softmax_scale=0.3), rtol=0, atol=0)
    # A separate window must never influence its neighbors.
    changed = v.clone()
    changed[5:8] += 20
    ctx.cache_enabled = True
    other = n.grouped_window_sdpa(q, k, changed, ctx, softmax_scale=0.3)
    assert torch.equal(out[:5], other[:5]) and torch.equal(out[8:], other[8:])
    assert _context(device).sdpa_groups is None


def test_empty_rank_does_not_build_cache(device):
    ctx = _context(device, empty=True)
    q = torch.empty(0, 2, 8, device=device)
    assert n.grouped_window_sdpa(q, q, q, ctx, softmax_scale=0.3).shape == q.shape
    assert getattr(ctx, "sdpa_groups", "missing") is None


@pytest.mark.cpu
def test_runtime_uses_cpu_planner_metadata(monkeypatch):
    runtime = n.SeedVR2WindowRuntime((2, 3, 5), text_len=2, window=(1, 2, 2), num_layers=2)
    layout = runtime.layout_for_layer(0)
    context = runtime.context(layout, "cpu")
    assert getattr(context, "cpu_window_shapes", None) == tuple(map(tuple, layout.window_shapes.tolist()))
    assert context.cpu_joint_offsets is not None
    assert context is runtime.context(layout, "cpu")
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_DIT_CACHE", "0")
    uncached = n.SeedVR2WindowRuntime((2, 3, 5), text_len=2, window=(1, 2, 2), num_layers=2)
    assert not uncached.context(uncached.layout_for_layer(0), "cpu").cache_enabled


@pytest.mark.cpu
def test_checkpoint_loader_invalidates_inference_tensor_frequencies():
    # Exercise the real strict loader on the smallest registered module graph;
    # loading a multi-GB checkpoint is unnecessary for this buffer contract.
    with torch.inference_mode():
        pipeline = SeedVR2Pipeline.__new__(SeedVR2Pipeline)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = torch.nn.Module()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.rope = n.NaMMRotaryEmbedding3d(24)
        pipeline.transformer.blocks = torch.nn.ModuleList([block])
        rope = block.attn.rope
        cache = n.SeedVR2RotaryCache()
        kwargs = dict(device=torch.device("cpu"), dtype=torch.float32)
        first = cache.get(rope, ((1, 2, 3),), 2, **kwargs)
        key = "transformer.blocks.0.attn.rope.freqs"
        changed = rope.freqs * 0.5
        assert pipeline.load_weights([(key, changed)]) == {key}
        updated = cache.get(rope, ((1, 2, 3),), 2, **kwargs)
        assert updated is not first
        fresh = n.NaMMRotaryEmbedding3d(24)
        fresh.load_state_dict({"freqs": changed})
        torch.testing.assert_close(
            updated.video_freqs, fresh.window_freqs_batch([(1, 2, 3)], 2, **kwargs), rtol=0, atol=0
        )
