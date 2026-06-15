# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for _ensure_tensor_values boundary enforcement."""

import pytest
import torch

from vllm_omni.worker.gpu_ar_model_runner import _ensure_tensor_values

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_tensors_pass_through():
    """Tensor values pass through unchanged."""
    t = torch.randn(10)
    result = _ensure_tensor_values({"audio": t})
    assert torch.equal(result["audio"], t)


def test_scalar_int_wrapped():
    """Integer scalars are wrapped in torch.tensor."""
    result = _ensure_tensor_values({"sr": 24000})
    assert isinstance(result["sr"], torch.Tensor)
    assert result["sr"].item() == 24000


def test_scalar_float_wrapped():
    """Float scalars are wrapped in torch.tensor."""
    result = _ensure_tensor_values({"gain": 0.5})
    assert isinstance(result["gain"], torch.Tensor)
    assert result["gain"].item() == pytest.approx(0.5)


def test_list_converted():
    """Lists of numbers are converted to tensors."""
    result = _ensure_tensor_values({"ids": [1, 2, 3]})
    assert isinstance(result["ids"], torch.Tensor)
    assert list(result["ids"]) == [1, 2, 3]


def test_non_tensorizable_dropped():
    """Values that cannot be tensorized are dropped with warning."""
    result = _ensure_tensor_values({"ok": torch.randn(5), "bad": {"nested": "dict"}})
    assert "ok" in result
    assert "bad" not in result


def test_mixed_payload():
    """Mixed tensor + scalar + list payload is properly sanitized."""
    payload = {
        "audio": torch.randn(100),
        "sr": 24000,
        "codes": [1, 2, 3],
        "hidden": torch.randn(10, 64),
    }
    result = _ensure_tensor_values(payload)
    assert len(result) == 4
    for v in result.values():
        assert isinstance(v, torch.Tensor)
