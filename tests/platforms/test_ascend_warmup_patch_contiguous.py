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

_NEW = "vllm_ascend.worker.v1.model_runner"
_OLD = "vllm_ascend.worker.model_runner_v1"


def test_tries_the_new_path_then_the_legacy_one(monkeypatch, caplog):
    seen: list[str] = []

    def _missing(name, *args, **kwargs):
        seen.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", _missing)
    with caplog.at_level(logging.ERROR):
        ascend_warmup_patch._apply_contiguous_kv_patch()

    assert seen == [_NEW, _OLD]


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
    fake = types.ModuleType(_NEW)

    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: fake)
    with caplog.at_level(logging.ERROR):
        ascend_warmup_patch._apply_contiguous_kv_patch()

    records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert records, "a module without the symbol must not be silent either"
    message = " ".join(r.getMessage() for r in records)
    assert "does not export" in message and "requires_contiguous_pa_kv_cache" in message


def test_applied_patch_stays_idempotent(monkeypatch, caplog):
    fake = types.ModuleType(_NEW)
    already = lambda *a, **k: True  # noqa: E731
    already._vllm_omni_contiguous = True
    fake.requires_contiguous_pa_kv_cache = already

    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: fake)
    with caplog.at_level(logging.ERROR):
        ascend_warmup_patch._apply_contiguous_kv_patch()

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
