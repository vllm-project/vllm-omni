# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MindIE dependency and request-isolation contracts for the NPU Flex adapter."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

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
    assert adapter.try_npu_flex_attention(q, q, q, metadata, 0.25) is None
    assert metadata.extra == {}


@torch.no_grad()
def test_request_mask_is_shared_after_metadata_copy(adapter, monkeypatch):
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
    monkeypatch.setattr(adapter, "_can_use_flex", lambda *args, **kwargs: True)
    q = torch.zeros(1, 9, 2, 128, dtype=torch.bfloat16)
    shared = {}
    for _ in range(2):
        metadata = AttentionMetadata(full_attn_spans=[[(2, 9)]], extra={"_npu_flex_cache": shared})
        assert adapter.try_npu_flex_attention(q, q, q, metadata, 0.25) is q
    assert len(created) == 1
    assert calls[0]["mask"] is calls[1]["mask"]
    assert calls[0] == {"mask": created[0], "scale": 0.25, "layout": "BSND", "enable_gqa": False}
