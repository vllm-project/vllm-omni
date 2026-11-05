# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MindIE dependency and request-isolation contracts for the NPU Flex adapter."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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


def test_missing_mindie_is_actionable(adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    with pytest.raises(ImportError, match="MindIE-SD.*FlexAttention"):
        adapter.load_mindie_flex_runtime()


def test_old_mindie_reports_missing_api(adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd", ModuleType("mindiesd"))
    with pytest.raises(ImportError, match="Upgrade.*MindIE-SD"):
        adapter.load_mindie_flex_runtime()


def test_transitive_import_failure_keeps_its_cause(adapter, monkeypatch):
    error = ModuleNotFoundError("missing compiler dependency", name="triton")

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
    q = torch.zeros(1, 9, 2, 16, dtype=torch.bfloat16)
    metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]])
    assert adapter.try_npu_flex_attention(q, q, q, metadata, 0.25) is None
    assert metadata.extra == {}


@torch.no_grad()
def test_request_mask_is_shared_after_metadata_copy(adapter, monkeypatch):
    runtime = ModuleType("mindiesd")
    created = []
    calls = []

    class Mask:
        def __init__(self, mask_mod, *, batch_shared, head_shared):
            created.append(self)
            assert batch_shared and head_shared

    runtime.FlexAttentionMask = Mask
    runtime.make_block_causal_mask_mod = lambda *args, **kwargs: object()

    def run(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    runtime.parallel_flex_attention = run
    runtime.flex_attention = run
    monkeypatch.setitem(sys.modules, "mindiesd", runtime)
    monkeypatch.setattr(adapter, "_can_use_flex", lambda *args, **kwargs: True)
    q = torch.zeros(1, 9, 2, 16, dtype=torch.bfloat16)
    shared = {}
    for _ in range(2):
        metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]], extra={"_qwen_image21_flex_cache": shared})
        assert adapter.try_npu_flex_attention(q, q, q, metadata, 0.25) is q
    assert len(created) == 1
    assert calls[0]["mask"] is calls[1]["mask"]
    assert calls[0]["query_positions"] is calls[1]["query_positions"]
    assert calls[0]["ulysses_group"] is None


def test_disjoint_query_ranges_preserve_global_coordinates(adapter):
    ranges = (QueryRange(0, 2, 0), QueryRange(2, 5, 8))
    actual = adapter._query_positions(ranges, 5, torch.device("cpu"))
    torch.testing.assert_close(actual, torch.tensor([0, 1, 8, 9, 10]))


@pytest.mark.parametrize("ranges", [(QueryRange(1, 3, 0),), (QueryRange(0, 2, 0),)])
def test_incomplete_query_mapping_is_rejected(adapter, ranges):
    with pytest.raises(ValueError, match="query_ranges"):
        adapter._query_positions(ranges, 3, torch.device("cpu"))
