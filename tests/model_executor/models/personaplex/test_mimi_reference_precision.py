# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Test-only codec reference precision lifecycle; no weights or GPU allocation."""

import pytest
import torch

from tests.model_executor.models.personaplex.test_mimi_prefix_flush import _mimi_reference_precision

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("initial_tf32", [False, True])
def test_reference_precision_restores_after_success(monkeypatch, initial_tf32):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", initial_tf32)
    other_settings = {
        name: getattr(torch.backends.cudnn, name)
        for name in ("enabled", "benchmark", "benchmark_limit", "deterministic")
    }
    with _mimi_reference_precision("cuda"):
        assert torch.backends.cudnn.allow_tf32 is False
        assert {name: getattr(torch.backends.cudnn, name) for name in other_settings} == other_settings
    assert torch.backends.cudnn.allow_tf32 is initial_tf32
    assert {name: getattr(torch.backends.cudnn, name) for name in other_settings} == other_settings


@pytest.mark.parametrize("initial_tf32", [False, True])
def test_reference_precision_restores_after_exception(monkeypatch, initial_tf32):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", initial_tf32)
    with pytest.raises(RuntimeError, match="codec setup failed"):
        with _mimi_reference_precision("cuda"):
            assert torch.backends.cudnn.allow_tf32 is False
            raise RuntimeError("codec setup failed")
    assert torch.backends.cudnn.allow_tf32 is initial_tf32


@pytest.mark.parametrize("initial_tf32", [False, True])
def test_reference_precision_nested_exit_preserves_outer_scope(monkeypatch, initial_tf32):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", initial_tf32)
    with _mimi_reference_precision("cuda"):
        with _mimi_reference_precision("cuda"):
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is initial_tf32


@pytest.mark.parametrize("initial_tf32", [False, True])
def test_cpu_reference_does_not_change_cudnn_precision(monkeypatch, initial_tf32):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", initial_tf32)
    with _mimi_reference_precision("cpu"):
        assert torch.backends.cudnn.allow_tf32 is initial_tf32
    assert torch.backends.cudnn.allow_tf32 is initial_tf32
