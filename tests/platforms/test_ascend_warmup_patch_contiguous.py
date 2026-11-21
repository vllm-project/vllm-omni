# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the contiguous-KV patch binding (#10094).

The patch resolves a vllm-ascend module at call time. If that binding fails the
KV cache silently stays strided, which the one-query FIA graph pays for on every
frame -- a per-token slowdown with no startup error. These tests pin the
fail-loud contract without needing the NPU stack: the module only imports
``vllm.logger`` at import time.
"""

from __future__ import annotations

import importlib
import logging
import types

import pytest

from vllm_omni.platforms.npu import ascend_warmup_patch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_V1 = "vllm_ascend.worker.v1.model_runner"
_LEGACY = "vllm_ascend.worker.model_runner_v1"
_V2 = "vllm_ascend.worker.v2.attn_utils"


def test_tries_the_candidates_in_runner_generation_order(monkeypatch, caplog):
    seen: list[str] = []

    def _missing(name, *args, **kwargs):
        seen.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", _missing)
    with caplog.at_level(logging.ERROR):
        ascend_warmup_patch._apply_contiguous_kv_patch()

    assert seen == [_V1, _LEGACY, _V2]


def test_error_when_no_candidate_imports(monkeypatch, caplog):
    def _missing(name, *args, **kwargs):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", _missing)
    with caplog.at_level(logging.ERROR):
        ascend_warmup_patch._apply_contiguous_kv_patch()

    records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert records, "a failed binding must not be silent"
    message = " ".join(r.getMessage() for r in records)
    assert "contiguous KV patch NOT applied" in message
    assert "strided" in message


def test_error_when_the_symbol_is_missing(monkeypatch, caplog):
    fake = types.ModuleType(_V1)

    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: fake)
    with caplog.at_level(logging.ERROR):
        ascend_warmup_patch._apply_contiguous_kv_patch()

    records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert records, "a module without the symbol must not be silent either"
    message = " ".join(r.getMessage() for r in records)
    assert "no vllm-ascend consumer module exposes" in message
    assert "requires_contiguous_pa_kv_cache" in message


def test_applied_patch_stays_idempotent(monkeypatch, caplog):
    fake = types.ModuleType(_V1)
    already = lambda *a, **k: True  # noqa: E731
    already._vllm_omni_contiguous = True
    fake.requires_contiguous_pa_kv_cache = already

    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: fake)
    with caplog.at_level(logging.ERROR):
        ascend_warmup_patch._apply_contiguous_kv_patch()

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_patches_the_v2_consumer_when_only_it_exists(monkeypatch, caplog):
    """nightly-main ships no v1 model runner; the v2 consumer is the target."""
    v2 = types.ModuleType(_V2)

    def original(*args, **kwargs):
        return True

    v2.requires_contiguous_pa_kv_cache = original

    def _import(name, *args, **kwargs):
        if name == _V2:
            return v2
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", _import)
    with caplog.at_level(logging.INFO):
        ascend_warmup_patch._apply_contiguous_kv_patch()

    assert getattr(v2.requires_contiguous_pa_kv_cache, "_vllm_omni_contiguous", False) is True
    assert any("contiguous KV patch applied" in r.getMessage() for r in caplog.records)
