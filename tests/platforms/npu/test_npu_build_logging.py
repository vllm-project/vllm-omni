# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The NPU package logs its torch_npu / CANN build identity (#9286, request 2).

torch_npu post-release builds of the same 2.10.0 line differ materially
(post4's with_stack SIGSEGV vs post6's fix), so the running build has to be
visible in each stage's startup log. These tests load the NPU package
``__init__`` from source with a fake ``torch_npu`` and a recording logger,
so the contract is checked on a CPU host without an Ascend device.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_npu_init(monkeypatch: pytest.MonkeyPatch, records: list[str]):
    def fake_info(fmt: str, *args: object) -> None:
        records.append(fmt % args)

    fake_logger = SimpleNamespace(info=fake_info)
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.init_logger = lambda _name: fake_logger  # type: ignore[attr-defined]
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.logger = fake_vllm_logger  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", fake_vllm_logger)

    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "__init__.py"
    return _load_source_module("vllm_omni_test_npu_init_build_log", path)


def _install_fake_torch_npu(monkeypatch: pytest.MonkeyPatch, **attrs):
    module = types.ModuleType("torch_npu")
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return module


def test_logs_torch_npu_build_identity(monkeypatch):
    records: list[str] = []
    npu_init = _load_npu_init(monkeypatch, records)
    _install_fake_torch_npu(
        monkeypatch,
        __version__="2.10.0.post4",
        version=SimpleNamespace(git_version="5dd8ef3f9b375b5ae4a83538d5785754148c3302"),
    )

    npu_init.log_ascend_runtime_build()

    assert len(records) == 1
    assert "torch_npu 2.10.0.post4" in records[0]
    assert "git 5dd8ef3f9b375b5ae4a83538d5785754148c3302" in records[0]


def test_tolerates_missing_git_version(monkeypatch):
    records: list[str] = []
    npu_init = _load_npu_init(monkeypatch, records)
    _install_fake_torch_npu(monkeypatch, __version__="2.10.0.post6")

    npu_init.log_ascend_runtime_build()

    assert len(records) == 1
    assert "torch_npu 2.10.0.post6" in records[0]
    assert "git unknown" in records[0]


def test_no_log_without_torch_npu(monkeypatch):
    records: list[str] = []
    npu_init = _load_npu_init(monkeypatch, records)
    # No torch_npu in sys.modules: the import inside the helper must fail
    # silently instead of raising on CPU-only hosts.
    monkeypatch.setitem(sys.modules, "torch_npu", None)

    npu_init.log_ascend_runtime_build()

    assert records == []
