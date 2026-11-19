# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import vllm.envs as vllm_envs

from vllm_omni.diffusion.forward_context import vllm_batch_invariant_scope

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_batch_invariant_scope_restores_lazy_env(monkeypatch):
    monkeypatch.delitem(vars(vllm_envs), "VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    with vllm_batch_invariant_scope(True):
        assert vllm_envs.VLLM_BATCH_INVARIANT
        with vllm_batch_invariant_scope(True):
            assert vllm_envs.VLLM_BATCH_INVARIANT
        assert vllm_envs.VLLM_BATCH_INVARIANT
    assert "VLLM_BATCH_INVARIANT" not in vars(vllm_envs)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert vllm_envs.VLLM_BATCH_INVARIANT


def test_batch_invariant_scope_restores_explicit_override_on_error(monkeypatch):
    monkeypatch.setitem(vars(vllm_envs), "VLLM_BATCH_INVARIANT", False)
    with pytest.raises(RuntimeError), vllm_batch_invariant_scope(True):
        assert vllm_envs.VLLM_BATCH_INVARIANT
        raise RuntimeError("failed forward")
    assert vars(vllm_envs)["VLLM_BATCH_INVARIANT"] is False
