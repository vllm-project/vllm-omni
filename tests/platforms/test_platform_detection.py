# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import builtins
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

import vllm_omni.platforms as platforms

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _install_fake_torch_and_torch_npu(monkeypatch, *, preinstalled_npu: bool, torch_npu_loader):
    fake_torch = ModuleType("torch")
    if preinstalled_npu:
        fake_torch.npu = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch_npu":
            return torch_npu_loader(fake_torch)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return fake_torch


def test_failing_oot_plugin_is_logged_and_falls_back(monkeypatch, caplog):
    plugin_name = "raising_oot_plugin"

    def raising_plugin():
        raise RuntimeError("injected platform detection failure")

    monkeypatch.setattr(platforms, "builtin_omni_platform_plugins", {})
    monkeypatch.setattr(
        platforms,
        "load_omni_plugins_by_group",
        lambda _group: {plugin_name: raising_plugin},
    )

    target_logger = logging.getLogger(platforms.__name__)
    target_logger.addHandler(caplog.handler)
    previous_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    try:
        platform_cls = platforms.resolve_current_omni_platform_cls_qualname()
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(previous_level)

    assert platform_cls == "vllm_omni.platforms.interface.UnspecifiedOmniPlatform"
    assert plugin_name in caplog.text
    assert "RuntimeError: injected platform detection failure" in caplog.text


def test_npu_plugin_imports_torch_npu_when_torch_has_no_npu_attr(monkeypatch):
    imports = []

    def load_torch_npu(fake_torch):
        imports.append("torch_npu")
        fake_torch.npu = SimpleNamespace(is_available=lambda: True)
        mod = ModuleType("torch_npu")
        sys.modules["torch_npu"] = mod
        return mod

    fake_torch = _install_fake_torch_and_torch_npu(
        monkeypatch,
        preinstalled_npu=False,
        torch_npu_loader=load_torch_npu,
    )
    assert not hasattr(fake_torch, "npu")

    monkeypatch.setattr(
        platforms,
        "builtin_omni_platform_plugins",
        {"npu": platforms.npu_omni_platform_plugin},
    )
    monkeypatch.setattr(platforms, "load_omni_plugins_by_group", lambda _group: {})

    assert platforms.npu_omni_platform_plugin() == "vllm_omni.platforms.npu.platform.NPUOmniPlatform"
    assert imports == ["torch_npu"]
    assert platforms.resolve_current_omni_platform_cls_qualname() == "vllm_omni.platforms.npu.platform.NPUOmniPlatform"
    assert imports == ["torch_npu"]


def test_npu_plugin_returns_none_when_torch_npu_is_missing(monkeypatch):
    def missing_torch_npu(_fake_torch):
        raise ImportError("No module named 'torch_npu'")

    _install_fake_torch_and_torch_npu(
        monkeypatch,
        preinstalled_npu=False,
        torch_npu_loader=missing_torch_npu,
    )

    assert platforms.npu_omni_platform_plugin() is None


def test_npu_plugin_returns_none_when_torch_npu_raises_runtime_error(monkeypatch):
    def broken_torch_npu(_fake_torch):
        raise RuntimeError("broken CANN")

    _install_fake_torch_and_torch_npu(
        monkeypatch,
        preinstalled_npu=False,
        torch_npu_loader=broken_torch_npu,
    )

    assert platforms.npu_omni_platform_plugin() is None


def test_npu_plugin_uses_existing_torch_npu_without_reimporting(monkeypatch):
    def unexpected_import(_fake_torch):
        raise AssertionError("torch_npu should not be imported when torch.npu is already present")

    _install_fake_torch_and_torch_npu(
        monkeypatch,
        preinstalled_npu=True,
        torch_npu_loader=unexpected_import,
    )

    assert platforms.npu_omni_platform_plugin() == "vllm_omni.platforms.npu.platform.NPUOmniPlatform"


def test_npu_plugin_returns_none_when_npu_not_available(monkeypatch):
    def unavailable_torch_npu(fake_torch):
        fake_torch.npu = SimpleNamespace(is_available=lambda: False)
        return ModuleType("torch_npu")

    _install_fake_torch_and_torch_npu(
        monkeypatch,
        preinstalled_npu=False,
        torch_npu_loader=unavailable_torch_npu,
    )

    assert platforms.npu_omni_platform_plugin() is None
