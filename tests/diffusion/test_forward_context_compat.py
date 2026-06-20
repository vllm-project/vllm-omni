# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.forward_context import set_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_set_forward_context_allows_missing_ir_priority_and_vllm_ir(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    @contextmanager
    def _set_current_vllm_config(cfg):
        calls.append(("set_current_vllm_config", cfg))
        yield
        calls.append(("set_current_vllm_config_exit", cfg))

    @contextmanager
    def _enable_torch_wrap(flag):
        calls.append(("enable_torch_wrap", flag))
        yield
        calls.append(("enable_torch_wrap_exit", flag))

    monkeypatch.setattr("vllm.config.vllm.set_current_vllm_config", _set_current_vllm_config)
    monkeypatch.setattr("vllm_omni.diffusion.forward_context.enable_torch_wrap", _enable_torch_wrap)

    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(ir_enable_torch_wrap=True),
    )

    with set_forward_context(vllm_config=vllm_config):
        pass

    assert calls == [
        ("set_current_vllm_config", vllm_config),
        ("enable_torch_wrap", True),
        ("enable_torch_wrap_exit", True),
        ("set_current_vllm_config_exit", vllm_config),
    ]
