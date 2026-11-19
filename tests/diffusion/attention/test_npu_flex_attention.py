# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MindIE dependency and request-isolation contracts for the NPU Flex adapter."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, QueryRange

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def adapter():
    path = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/attention/backends/utils/npu_flex_attention.py"
    spec = importlib.util.spec_from_file_location("npu_flex_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def npu_query():
    class NPUTensor(torch.Tensor):
        """Keep CPU storage while exposing the device tag used by adapter checks."""

        @property
        def device(self):
            return SimpleNamespace(type="npu")

    return torch.zeros(1, 9, 2, 128, dtype=torch.bfloat16).as_subclass(NPUTensor)


def test_missing_mindie_is_actionable(adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    with pytest.raises(ImportError, match="MindIE-SD.*FlexAttention"):
        adapter.load_mindie_flex_runtime()


def test_old_mindie_reports_missing_api(adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd", ModuleType("mindiesd"))
    with pytest.raises(ImportError, match="Upgrade.*MindIE-SD"):
        adapter.load_mindie_flex_runtime()


def test_transitive_import_failure_keeps_its_cause(adapter, monkeypatch):
    error = ModuleNotFoundError("missing compiler dependency", name="catlass")

    def broken_import(name):
        raise error

    monkeypatch.setattr(adapter.importlib, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError) as captured:
        adapter.load_mindie_flex_runtime()
    assert captured.value is error


@torch.no_grad()
def test_non_npu_does_not_import_mindie(adapter, monkeypatch):
    def unexpected_import():
        pytest.fail("CPU inputs must not load MindIE")

    monkeypatch.setattr(adapter, "load_mindie_flex_runtime", unexpected_import)
    q = torch.zeros(1, 9, 2, 128, dtype=torch.bfloat16)
    metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]])
    with pytest.raises(ValueError, match="requires NPU inputs"):
        adapter.run_npu_flex_attention(q, q, q, metadata, 0.25)
    assert metadata.extra == {}


@pytest.mark.parametrize("argument", ["query", "key", "value"])
def test_invalid_layout_is_rejected_before_indexing(adapter, npu_query, argument):
    inputs = dict.fromkeys(("query", "key", "value"), npu_query)
    inputs[argument] = npu_query.flatten(1)
    metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]])
    with pytest.raises(ValueError, match="requires BSND inputs"):
        adapter.run_npu_flex_attention(**inputs, metadata=metadata, softmax_scale=0.25)


@pytest.mark.parametrize("batch_size,dtype", [(2, torch.bfloat16), (1, torch.float16)])
def test_adapter_restricts_mindie_to_one_bf16_sample(adapter, npu_query, batch_size, dtype):
    q = npu_query.repeat(batch_size, 1, 1, 1).to(dtype)
    metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]])
    with pytest.raises(ValueError, match="requires one BF16 sample"):
        adapter.run_npu_flex_attention(q, q, q, metadata, 0.25)


def test_adapter_requires_equal_sequence_lengths(adapter, npu_query):
    metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]])
    with pytest.raises(ValueError, match="requires equal Q/K sequence lengths"):
        adapter.run_npu_flex_attention(npu_query, npu_query[:, :-1], npu_query[:, :-1], metadata, 0.25)


@pytest.mark.parametrize("spans", [None, [], [[(2, 9)], [(2, 9)]]])
def test_adapter_requires_single_sample_spans(adapter, npu_query, spans):
    metadata = AttentionMetadata(full_attn_spans=spans)
    with pytest.raises(ValueError, match="requires single-sample block-causal spans"):
        adapter.run_npu_flex_attention(npu_query, npu_query, npu_query, metadata, 0.25)


@pytest.mark.parametrize("field", ["attn_mask", "query_ranges", "joint_key", "joint_value"])
def test_adapter_rejects_metadata_it_cannot_forward(adapter, npu_query, field):
    metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]])
    fields = {
        "attn_mask": torch.ones(1, 1, 9, 9, dtype=torch.bool),
        "query_ranges": (QueryRange(0, 9, 0),),
        "joint_key": npu_query,
        "joint_value": npu_query,
    }
    setattr(metadata, field, fields[field])
    with pytest.raises(ValueError, match="requires local span metadata"):
        adapter.run_npu_flex_attention(npu_query, npu_query, npu_query, metadata, 0.25)


def test_adapter_rejects_attention_quantization(adapter, npu_query):
    metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]], extra={"kv_cache_dtype": "fp8"})
    with pytest.raises(ValueError, match="does not support attention quantization"):
        adapter.run_npu_flex_attention(npu_query, npu_query, npu_query, metadata, 0.25)


def test_outer_compile_is_rejected_before_loading_runtime(adapter, monkeypatch, npu_query):
    def unexpected_import():
        pytest.fail("Outer compilation must be rejected before loading MindIE")

    monkeypatch.setattr(adapter, "load_mindie_flex_runtime", unexpected_import)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]])
    with pytest.raises(ValueError, match="outside an outer torch.compile region"):
        adapter.run_npu_flex_attention(npu_query, npu_query, npu_query, metadata, 0.25)
    assert metadata.extra == {}


@torch.no_grad()
def test_request_mask_is_shared_after_metadata_copy(adapter, monkeypatch, npu_query):
    runtime = ModuleType("mindiesd")
    created = []
    calls = []

    def make_mask(spans, length, *, device, valid_tokens):
        assert spans == ((2, 9),)
        assert length == 9
        mask = object()
        created.append(mask)
        return mask

    runtime.make_block_causal_mask = make_mask
    runtime.check_flex_attention_device = lambda device: None

    def run(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    runtime.flex_attention = run
    monkeypatch.setitem(sys.modules, "mindiesd", runtime)
    shared = {}
    for _ in range(2):
        metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]], extra={"_npu_flex_cache": shared})
        assert adapter.run_npu_flex_attention(npu_query, npu_query, npu_query, metadata, 0.25) is npu_query
    assert len(created) == 1
    assert calls[0]["mask"] is calls[1]["mask"]
    assert calls[0] == {"mask": created[0], "scale": 0.25, "layout": "BSND", "enable_gqa": False}
