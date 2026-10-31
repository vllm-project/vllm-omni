# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import builtins
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl, _Qwen21DecodeVarlen
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


def test_prepared_masked_forward_preserves_dispatcher_identity_and_cache(monkeypatch):
    """Exercise CUDA-path Python dispatch on CPU with fake kernel providers."""
    calls = []

    def dispatcher(q, k, v, **kwargs):
        calls.append(("vllm", q, k, v, kwargs))
        return q + 1

    def other_provider(q, k, v, **kwargs):
        calls.append(("other", q, k, v, kwargs))
        return q + 1

    other_provider.__name__ = dispatcher.__name__
    other_provider.__module__ = dispatcher.__module__
    module = ModuleType("vllm.vllm_flash_attn")
    setattr(module, "flash_attn_varlen_func", dispatcher)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    resolver = Mock(return_value=3)
    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", resolver)
    monkeypatch.setattr(fa, "_unpad_input", Mock(side_effect=AssertionError("Prepared metadata must be reused")))

    query = torch.arange(16, dtype=torch.float32).reshape(2, 2, 1, 4)
    key = torch.arange(32, dtype=torch.float32).reshape(2, 4, 1, 4)
    value = key + 100
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    prepared = _Qwen21DecodeVarlen(
        indices_k=torch.tensor([0, 1, 2, 4, 5]),
        cu_seqlens_k=torch.tensor([0, 3, 5], dtype=torch.int32),
        max_seqlen_k=3,
        cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
        max_seqlen_q=2,
    )
    metadata = AttentionMetadata(
        attn_mask=mask,
        extra={"attention_mask_mode": "padding", "qwen21_decode_varlen": prepared},
    )
    impl = FlashAttentionImpl(num_heads=1, head_size=4, softmax_scale=0.5)

    for provider in (dispatcher, other_provider):
        monkeypatch.setattr(fa, "flash_attn_varlen_func", provider)
        for _ in range(2):
            torch.testing.assert_close(impl.forward_cuda(query, key, value, metadata), query + 1)

    assert len(calls) == 4
    for name, q, k, v, kwargs in calls:
        torch.testing.assert_close(q, query.flatten(0, 1))
        torch.testing.assert_close(k, key.flatten(0, 1)[prepared.indices_k])
        torch.testing.assert_close(v, value.flatten(0, 1)[prepared.indices_k])
        assert kwargs["cu_seqlens_q"] is prepared.cu_seqlens_q
        assert kwargs["cu_seqlens_k"] is prepared.cu_seqlens_k
        assert kwargs["max_seqlen_q"] == 2
        assert kwargs["max_seqlen_k"] == 3
        assert kwargs["causal"] is False
        assert kwargs["softmax_scale"] == 0.5
        if name == "vllm":
            assert kwargs["fa_version"] == 3
        else:
            assert "fa_version" not in kwargs
    assert resolver.call_count == 2
    assert all(not call.args and not call.kwargs for call in resolver.call_args_list)
    assert fa._vllm_varlen_dispatcher.cache_info().misses == 1


def test_dispatcher_runtime_import_failure_propagates_and_is_retried(monkeypatch):
    original_import = builtins.__import__
    attempts = []

    def broken_import(name, *args, **kwargs):
        if name == "vllm.vllm_flash_attn":
            attempts.append(name)
            raise RuntimeError("Broken extension initialization")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Broken extension initialization"):
            fa.vllm_varlen_version_kwargs(Mock())
    assert len(attempts) == 2
    assert fa._vllm_varlen_dispatcher.cache_info().misses == 2
    assert fa._vllm_varlen_dispatcher.cache_info().currsize == 0
