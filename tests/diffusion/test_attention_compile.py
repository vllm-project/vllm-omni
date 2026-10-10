# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_attention_uses_compile_boundary_for_hsdp(monkeypatch):
    attention = object.__new__(Attention)
    attention._hsdp_compile_boundary_enabled = False
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
    attention._hsdp_compile_boundary_enabled = False
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


@pytest.mark.parametrize("initialized_with_hsdp", [False, True])
def test_attention_compile_boundary_without_diffusion_config(monkeypatch, initialized_with_hsdp):
    attention = object.__new__(Attention)
    attention._hsdp_compile_boundary_enabled = initialized_with_hsdp
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

    assert calls == (["boundary"] if initialized_with_hsdp else ["impl"])


@pytest.mark.parametrize(
    ("configured", "initialized_with_hsdp", "use_hsdp", "with_context", "compiling", "expected"),
    [
        (True, False, False, True, True, ["schedule"]),
        (True, False, True, True, True, ["schedule"]),
        (True, False, False, False, True, ["schedule"]),
        (True, False, False, True, False, ["impl"]),
    ],
    ids=[
        "scheduled",
        "scheduled-with-hsdp",
        "scheduled-without-forward-context",
        "scheduled-eager",
    ],
)
def test_attention_uses_schedule_boundary_only_for_scheduled_layers(
    monkeypatch, configured, initialized_with_hsdp, use_hsdp, with_context, compiling, expected
):
    # While compiling, a layer built with a startup schedule calls
    # _forward_schedule_compile_boundary (a torch.compiler.disable method); the schedule flag set at
    # construction alone decides this, also when HSDP is enabled at construction or in the forward
    # context. Unscheduled layers call the HSDP boundary or the compiled impl.
    attention = object.__new__(Attention)
    attention._hsdp_compile_boundary_enabled = initialized_with_hsdp
    attention._schedule_configured = configured
    calls: list[str] = []

    def _recorder(name):
        def _call(query, key, value, attn_metadata=None):
            calls.append(name)
            return query

        return _call

    attention._forward_schedule_compile_boundary = _recorder("schedule")
    attention._forward_hsdp_compile_boundary = _recorder("hsdp")
    attention._forward_impl = _recorder("impl")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)

    config = OmniDiffusionConfig(parallel_config=DiffusionParallelConfig(use_hsdp=use_hsdp, hsdp_shard_size=1))
    query = torch.empty(1)
    with set_forward_context(omni_diffusion_config=config) if with_context else nullcontext():
        assert Attention.forward(attention, query, query, query) is query

    assert calls == expected
