# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.forward_context import set_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_attention_uses_compile_boundary_for_hsdp(monkeypatch):
    attention = object.__new__(Attention)
    calls = []

    def _boundary(query, key, value, attn_metadata=None):
        calls.append("boundary")
        return query

    def _impl(query, key, value, attn_metadata=None):
        calls.append("impl")
        return query

    attention._forward_hsdp_compile_boundary = _boundary
    attention._forward_impl = _impl
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    config = SimpleNamespace(parallel_config=SimpleNamespace(use_hsdp=True))
    query = torch.empty(1)
    with set_forward_context(omni_diffusion_config=config):
        assert Attention.forward(attention, query, query, query) is query

    assert calls == ["boundary"]


def test_attention_keeps_compiled_impl_without_hsdp(monkeypatch):
    attention = object.__new__(Attention)
    calls = []

    def _boundary(query, key, value, attn_metadata=None):
        calls.append("boundary")
        return query

    def _impl(query, key, value, attn_metadata=None):
        calls.append("impl")
        return query

    attention._forward_hsdp_compile_boundary = _boundary
    attention._forward_impl = _impl
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    config = SimpleNamespace(parallel_config=SimpleNamespace(use_hsdp=False))
    query = torch.empty(1)
    with set_forward_context(omni_diffusion_config=config):
        assert Attention.forward(attention, query, query, query) is query

    assert calls == ["impl"]


def test_attention_keeps_compiled_impl_without_diffusion_config(monkeypatch):
    attention = object.__new__(Attention)
    calls = []

    def _boundary(query, key, value, attn_metadata=None):
        calls.append("boundary")
        return query

    def _impl(query, key, value, attn_metadata=None):
        calls.append("impl")
        return query

    attention._forward_hsdp_compile_boundary = _boundary
    attention._forward_impl = _impl
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    query = torch.empty(1)
    with set_forward_context(vllm_config=None):
        assert Attention.forward(attention, query, query, query) is query

    assert calls == ["impl"]
