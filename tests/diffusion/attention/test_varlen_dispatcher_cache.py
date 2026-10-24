# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import builtins
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.attention.backends.utils import fa

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def clear_dispatcher_cache():
    fa._vllm_varlen_dispatcher.cache_clear()
    yield
    fa._vllm_varlen_dispatcher.cache_clear()


def test_unavailable_dispatcher_is_imported_only_once(monkeypatch):
    original_import = builtins.__import__
    attempts = []

    def unavailable(name, *args, **kwargs):
        if name == "vllm.vllm_flash_attn":
            attempts.append(name)
            raise ImportError("CUDA extensions unavailable")
        return original_import(name, *args, **kwargs)

    resolver = Mock(side_effect=AssertionError("Non-vLLM dispatch must not resolve a version"))
    monkeypatch.setattr(builtins, "__import__", unavailable)
    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", resolver)
    for func in (Mock(), Mock(), Mock()):
        assert fa.vllm_varlen_version_kwargs(func) == {}
    assert len(attempts) == 1
    resolver.assert_not_called()


def test_available_dispatcher_uses_identity_and_retains_version_selection(monkeypatch):
    def dispatcher():
        pass

    def other():
        pass

    # Matching names must not make a compiled variant or wrapper the dispatcher.
    other.__name__ = dispatcher.__name__
    other.__module__ = dispatcher.__module__
    module = ModuleType("vllm.vllm_flash_attn")
    setattr(module, "flash_attn_varlen_func", dispatcher)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    resolver = Mock(return_value=3)
    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", resolver)

    assert fa.vllm_varlen_version_kwargs(dispatcher) == {"fa_version": 3}
    assert fa.vllm_varlen_version_kwargs(other) == {}
    # Availability is fixed until the cache is explicitly cleared.
    monkeypatch.setitem(sys.modules, module.__name__, None)
    assert fa.vllm_varlen_version_kwargs(dispatcher) == {"fa_version": 3}
    assert fa._vllm_varlen_dispatcher.cache_info().misses == 1
    assert resolver.call_count == 2


def test_missing_dispatcher_can_be_resolved_after_cache_clear(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", None)
    assert fa._vllm_varlen_dispatcher() is None
    module = ModuleType("vllm.vllm_flash_attn")
    dispatcher = Mock()
    setattr(module, "flash_attn_varlen_func", dispatcher)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert fa._vllm_varlen_dispatcher() is None
    fa._vllm_varlen_dispatcher.cache_clear()
    assert fa._vllm_varlen_dispatcher() is dispatcher


def test_none_skips_dispatcher_lookup_and_version_resolution(monkeypatch):
    resolver = Mock(side_effect=AssertionError("None must not resolve a version"))
    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", resolver)
    assert fa.vllm_varlen_version_kwargs(None) == {}
    assert fa._vllm_varlen_dispatcher.cache_info().misses == 0
    resolver.assert_not_called()
