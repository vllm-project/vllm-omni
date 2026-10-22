# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the compiled kernel, without importing vLLM or torch_npu."""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def flex_module(monkeypatch):
    # Load this dependency-light helper directly; each case gets fresh runtime
    # state, and environments without vLLM can still exercise its contracts.
    path = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/attention/backends/utils/npu_flex_attention.py"
    spec = importlib.util.spec_from_file_location("npu_flex_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # CPU tests isolate the vendor package. Registration order is covered below;
    # actual NPU registration/codegen is exercised by test_flex_attn.py.
    npu = ModuleType("torch_npu")
    npu._inductor = ModuleType("torch_npu._inductor")
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "torch_npu._inductor", npu._inductor)
    return module


def _metadata(spans=None, **overrides):
    values = dict(
        full_attn_spans=[[(2, 5), (6, 9)]] if spans is None else [spans],
        attn_mask=None,
        query_ranges=None,
        joint_query=None,
        joint_key=None,
        joint_value=None,
        extra={},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("length,spans", [(299, []), (299, [(16, 299)]), (409, [(8, 136), (144, 272), (280, 409)])])
def test_span_rule_matches_independent_image_ids(flex_module, length, spans):
    ids = torch.full((length,), -1)
    for image_id, (start, end) in enumerate(spans):
        ids[start:end] = image_id
    positions = torch.arange(length)
    same_image = (ids[:, None] >= 0) & (ids[:, None] == ids[None, :])
    expected = (positions[:, None] >= positions[None, :]) | same_image
    mask_mod = flex_module._make_span_mask(tuple(spans), length)
    actual = mask_mod(0, 0, positions[:, None], positions[None, :])
    assert torch.equal(actual, expected)
    # Partial 128-token blocks must not expose padded keys or queries.
    assert not mask_mod(0, 0, torch.tensor(length), torch.tensor(0)).item()
    assert not mask_mod(0, 0, torch.tensor(0), torch.tensor(length)).item()


@pytest.mark.parametrize(
    "case",
    [
        "supported",
        "cpu",
        "cuda",
        "xpu",
        "mps",
        "decode",
        "batch",
        "dtype",
        "mask",
        "ranges",
        "joint",
        "quant",
        "grad",
        "overlap",
    ],
)
def test_supported_contract(flex_module, case):
    device = SimpleNamespace(type="npu")
    query = SimpleNamespace(device=device, ndim=4, shape=(1, 9, 2, 8), dtype=torch.bfloat16)
    key = SimpleNamespace(**vars(query))
    value = SimpleNamespace(**vars(query))
    metadata = _metadata()
    if case in ("cpu", "cuda", "xpu", "mps"):
        query.device = SimpleNamespace(type=case)
    elif case == "decode":
        query.shape = (1, 3, 2, 8)
    elif case == "batch":
        query.shape = key.shape = value.shape = (2, 9, 2, 8)
    elif case == "dtype":
        query.dtype = key.dtype = value.dtype = torch.float32
    elif case == "mask":
        metadata.attn_mask = torch.ones(1, 9, dtype=torch.bool)
    elif case == "ranges":
        metadata.query_ranges = [object()]
    elif case == "joint":
        metadata.joint_key = object()
    elif case == "quant":
        metadata.extra["kv_cache_dtype"] = "fp8"
    elif case == "overlap":
        metadata.full_attn_spans = [[(2, 7), (6, 9)]]
    with torch.set_grad_enabled(case == "grad"):
        assert flex_module._can_use_flex(query, key, value, metadata) is (case == "supported")


@pytest.fixture
def cpu_flex_call(flex_module, monkeypatch):
    from torch.nn.attention import flex_attention as flex_api

    monkeypatch.setattr(flex_module, "_can_use_flex", lambda *args: True)
    calls = SimpleNamespace(masks=[], compiles=[])
    real_create = flex_api.create_block_mask

    def create(*args, **kwargs):
        calls.masks.append(kwargs)
        return real_create(*args, **kwargs)

    def compile_call(fn, **kwargs):
        assert fn is flex_api.flex_attention
        assert kwargs == {"backend": "inductor", "fullgraph": True, "dynamic": True}
        calls.compiles.append(kwargs)

        def evaluate(q, k, v, *, block_mask, scale):
            assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            positions = torch.arange(q.shape[2])
            keep = block_mask.mask_mod(0, 0, positions[:, None], positions[None, :])
            return F.scaled_dot_product_attention(q, k, v, attn_mask=keep, scale=scale)

        return evaluate

    monkeypatch.setattr(flex_api, "create_block_mask", create)
    monkeypatch.setattr(torch, "compile", compile_call)
    return calls


@torch.no_grad()
def test_npu_registration_precedes_mask_creation_and_compile(flex_module, monkeypatch):
    from torch.nn.attention import flex_attention as flex_api

    events = []
    real_import = builtins.__import__
    real_create = flex_api.create_block_mask

    def registered_flex(q, k, v, *, block_mask, scale):
        events.append("run")
        return q

    def initialize_npu(name, *args, **kwargs):
        if name == "torch_npu._inductor":
            events.append("register")
            # The real vendor import patches the public FlexAttention API.
            monkeypatch.setattr(flex_api, "flex_attention", registered_flex)
        return real_import(name, *args, **kwargs)

    def create(*args, **kwargs):
        assert events == ["register"]
        events.append("mask")
        return real_create(*args, **kwargs)

    def compile_call(fn, **kwargs):
        assert fn is registered_flex
        assert events == ["register", "mask"]
        events.append("compile")
        return fn

    monkeypatch.setattr(flex_module, "_can_use_flex", lambda *args: True)
    monkeypatch.setattr(builtins, "__import__", initialize_npu)
    monkeypatch.setattr(flex_api, "create_block_mask", create)
    monkeypatch.setattr(torch, "compile", compile_call)
    q = torch.zeros(1, 9, 2, 8, dtype=torch.bfloat16)
    actual = flex_module.try_npu_flex_attention(q, q, q, _metadata(), 0.5)
    torch.testing.assert_close(actual, q)
    assert events == ["register", "mask", "compile", "run"]


@torch.no_grad()
def test_layout_scale_numerics_and_request_mask_reuse(flex_module, cpu_flex_call):
    generator = torch.Generator().manual_seed(42)
    q, k, v = [torch.randn(1, 9, 2, 8, generator=generator).bfloat16() for _ in range(3)]
    metadata = _metadata()
    scale = 0.37
    ids = torch.tensor([-1, -1, 0, 0, 0, -1, 1, 1, 1])
    positions = torch.arange(9)
    keep = (positions[:, None] >= positions[None, :]) | ((ids[:, None] >= 0) & (ids[:, None] == ids[None, :]))
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=keep, scale=scale
    )
    for _ in range(2):
        actual = flex_module.try_npu_flex_attention(q, k, v, metadata, scale)
        torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=0, atol=0)
        assert actual.shape == q.shape and actual.dtype == q.dtype and actual.is_contiguous()
        # Match Attention._with_kv_cache_dtype's shallow copy between layers.
        metadata = _metadata(extra=dict(metadata.extra))
    assert len(cpu_flex_call.masks) == 1
    assert len(cpu_flex_call.compiles) == 1
    assert cpu_flex_call.masks[0]["BLOCK_SIZE"] == 128
    metadata.full_attn_spans = [[(1, 9)]]
    flex_module.try_npu_flex_attention(q, k, v, metadata, scale)
    assert len(cpu_flex_call.masks) == 2
    assert len(cpu_flex_call.compiles) == 1


@pytest.mark.parametrize(
    "error",
    [
        ImportError("missing Inductor"),
        RuntimeError("no compilable choices"),
        ValueError("unsupported npu"),
        AttributeError("missing API"),
        OSError("missing compiler library"),
    ],
)
def test_compile_errors_propagate_without_disabling_future_requests(flex_module, monkeypatch, cpu_flex_call, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(flex_module, "_compiled_flex", unavailable)
    q = torch.zeros(1, 9, 2, 8, dtype=torch.bfloat16)
    metadata = _metadata()
    with pytest.raises(type(error), match=str(error)):
        flex_module.try_npu_flex_attention(q, q, q, metadata, 0.5)
    monkeypatch.setattr(flex_module, "_compiled_flex", None)
    assert flex_module.try_npu_flex_attention(q, q, q, metadata, 0.5) is not None
    assert len(cpu_flex_call.masks) == 1


def test_oom_is_not_hidden_by_fallback(flex_module, monkeypatch, cpu_flex_call):
    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("out of memory")

    monkeypatch.setattr(flex_module, "_compiled_flex", oom)
    q = torch.zeros(1, 9, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(torch.OutOfMemoryError):
        flex_module.try_npu_flex_attention(q, q, q, _metadata(), 0.5)


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "xpu", "mps"])
@torch.no_grad()
def test_non_npu_input_never_enters_compiler(flex_module, monkeypatch, device_type):
    def unexpected_compiler(*args, **kwargs):
        pytest.fail("non-NPU input must not load or modify the Flex compiler")

    monkeypatch.setitem(sys.modules, "torch_npu._inductor", None)
    monkeypatch.setattr(torch, "compile", unexpected_compiler)
    query = SimpleNamespace(device=SimpleNamespace(type=device_type), ndim=4, shape=(1, 9, 2, 8), dtype=torch.bfloat16)
    metadata = _metadata()
    assert flex_module.try_npu_flex_attention(query, query, query, metadata, 0.5) is None
    assert metadata.extra == {}
    assert flex_module._compiled_flex is None


@torch.no_grad()
def test_dynamic_lengths_and_image_spans_reuse_compiled_graph(flex_module, monkeypatch):
    real_compile = torch.compile
    compiled_graphs = []

    def counting_backend(graph, inputs):
        compiled_graphs.append(graph)
        return graph.forward

    def compile_for_cpu(fn, **kwargs):
        kwargs["backend"] = counting_backend
        return real_compile(fn, **kwargs)

    monkeypatch.setattr(flex_module, "_can_use_flex", lambda *args: True)
    monkeypatch.setattr(torch, "compile", compile_for_cpu)
    generator = torch.Generator().manual_seed(42)
    # More distinct requests than Dynamo's default recompile_limit. Changes in
    # span count/boundaries are tensor data, not new Python mask expressions.
    for index, length in enumerate((257, 299, 385, 409, 512, 677, 781, 896, 913, 1024)):
        spans = [(8 + index, length)] if index % 2 else [(8, 136), (144 + index, length)]
        q, k, v = [torch.randn(1, length, 2, 8, generator=generator) for _ in range(3)]
        output = flex_module.try_npu_flex_attention(q, k, v, _metadata(spans=spans), 0.37)
        ids = torch.full((length,), -1)
        for image_id, (start, end) in enumerate(spans):
            ids[start:end] = image_id
        positions = torch.arange(length)
        keep = (positions[:, None] >= positions[None, :]) | ((ids[:, None] >= 0) & (ids[:, None] == ids[None, :]))
        expected = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=keep, scale=0.37
        ).transpose(1, 2)
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
    assert len(compiled_graphs) == 1
