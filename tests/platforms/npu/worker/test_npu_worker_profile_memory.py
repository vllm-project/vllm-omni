# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NPU worker's per-step allocator walk is DEBUG-only diagnostics.

``NPUWorker.profile_memory`` runs inside every ``execute_model`` call and the
numbers it collects are read by nothing but its own DEBUG log line. These tests
pin the gate that keeps that walk off the serving hot path. vLLM-Ascend is faked
and ``base.py`` is loaded from source, so they run on a plain CPU box like the
rest of ``tests/platforms/npu``.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ASCEND_WORKER_MODULE = "vllm_ascend.worker.worker"


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu" / "worker" / "base.py"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_file():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def worker_base(monkeypatch: pytest.MonkeyPatch):
    """``platforms/npu/worker/base.py``, loaded with vLLM-Ascend faked out.

    Yields the module and the list the fake parent appends to whenever its
    ``profile_memory`` -- the allocator walk this PR is gating -- is reached.
    """
    parent_calls: list[object] = []

    class FakeNPUWorker:
        __module__ = ASCEND_WORKER_MODULE

        def profile_memory(self) -> None:
            parent_calls.append(self)

    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu._310p", is_310p=lambda *a, **k: False)
    _install_fake_module(monkeypatch, ASCEND_WORKER_MODULE, NPUWorker=FakeNPUWorker)

    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "worker" / "base.py"
    spec = importlib.util.spec_from_file_location("omni_npu_worker_base_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    yield module, parent_calls


def test_allocator_walk_is_skipped_at_the_default_log_level(worker_base, caplog):
    module, parent_calls = worker_base
    caplog.set_level(logging.INFO, logger=ASCEND_WORKER_MODULE)

    worker = object.__new__(module.OmniNPUWorkerBase)
    worker.profile_memory()

    assert parent_calls == []


def test_allocator_walk_still_runs_for_debug(worker_base, caplog):
    module, parent_calls = worker_base
    caplog.set_level(logging.DEBUG, logger=ASCEND_WORKER_MODULE)

    worker = object.__new__(module.OmniNPUWorkerBase)
    worker.profile_memory()

    assert len(parent_calls) == 1


def test_the_deciding_level_is_the_vllm_ascend_worker_logger(worker_base, caplog):
    """Turning vLLM-Omni's own logging up must not switch the walk back on.

    The values are logged by the parent, under the parent's logger, so that is
    the only level that can decide whether collecting them is worth anything.
    """
    module, parent_calls = worker_base
    caplog.set_level(logging.DEBUG, logger="vllm_omni")
    caplog.set_level(logging.INFO, logger=ASCEND_WORKER_MODULE)

    worker = object.__new__(module.OmniNPUWorkerBase)
    worker.profile_memory()

    assert parent_calls == []
