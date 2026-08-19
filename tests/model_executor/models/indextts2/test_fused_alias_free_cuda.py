# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.common.alias_free_activation import AliasFreeActivation1d
from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta
from vllm_omni.model_executor.models.indextts2.s2mel.modules.alias_free_cuda.activation import (
    OfficialFusedAliasFreeActivation1d,
)

pytestmark = pytest.mark.core_model


def test_official_fused_alias_free_falls_back_on_cpu():
    torch.manual_seed(37)
    activation = SnakeBeta(3, alpha_logscale=True)
    eager = AliasFreeActivation1d(copy.deepcopy(activation)).eval()
    fused = OfficialFusedAliasFreeActivation1d(copy.deepcopy(activation)).eval()
    hidden = torch.randn(1, 3, 257)

    with torch.inference_mode():
        expected = eager(hidden)
        actual = fused(hidden)

    torch.testing.assert_close(actual, expected)
    assert fused.fused_activation_active is False


@pytest.mark.parametrize(
    ("error", "fatal"),
    [
        (RuntimeError("unsupported fused kernel input"), False),
        (torch.cuda.OutOfMemoryError("CUDA out of memory"), True),
        (RuntimeError("CUDA error: illegal memory access"), True),
    ],
)
def test_official_fused_alias_free_handles_extension_failure(monkeypatch, error, fatal):
    extension = object()
    warning_once = Mock()
    monkeypatch.setattr(OfficialFusedAliasFreeActivation1d, "_extension", extension)
    monkeypatch.setattr(OfficialFusedAliasFreeActivation1d, "_extension_unavailable", False)
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.indextts2.s2mel.modules.alias_free_cuda.activation.logger.warning_once",
        warning_once,
    )

    if fatal:
        with pytest.raises(type(error)) as exc_info:
            OfficialFusedAliasFreeActivation1d._handle_extension_failure(error)
        assert exc_info.value is error
        warning_once.assert_not_called()
        assert OfficialFusedAliasFreeActivation1d._extension is extension
        assert OfficialFusedAliasFreeActivation1d._extension_unavailable is False
    else:
        OfficialFusedAliasFreeActivation1d._handle_extension_failure(error)
        warning_once.assert_called_once_with(
            "Official BigVGAN fused alias-free activation failed (%s); disabling the extension",
            str(error),
        )
        assert OfficialFusedAliasFreeActivation1d._extension is None
        assert OfficialFusedAliasFreeActivation1d._extension_unavailable is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
def test_official_fused_alias_free_oom_propagates_from_cuda_forward(monkeypatch):
    device = torch.device("cuda")
    fused = OfficialFusedAliasFreeActivation1d(SnakeBeta(3, alpha_logscale=True)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    hidden = torch.randn(1, 3, 257, device=device, dtype=torch.bfloat16)
    error = torch.cuda.OutOfMemoryError("CUDA out of memory")
    extension = Mock()
    extension.forward.side_effect = error
    monkeypatch.setattr(OfficialFusedAliasFreeActivation1d, "_extension", extension)
    monkeypatch.setattr(OfficialFusedAliasFreeActivation1d, "_extension_unavailable", False)

    with pytest.raises(type(error)) as exc_info, torch.inference_mode():
        fused(hidden)

    assert exc_info.value is error
    assert fused.fused_activation_active is False
    assert OfficialFusedAliasFreeActivation1d._extension is extension
    assert OfficialFusedAliasFreeActivation1d._extension_unavailable is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
def test_official_fused_alias_free_extension_matches_eager():
    torch.manual_seed(131)
    device = torch.device("cuda")
    activation = SnakeBeta(3, alpha_logscale=True)
    eager = AliasFreeActivation1d(copy.deepcopy(activation)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    fused = OfficialFusedAliasFreeActivation1d(copy.deepcopy(activation)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    hidden = torch.randn(1, 3, 4103, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = eager(hidden)
        actual = fused(hidden)

    error = (actual.float() - expected.float()).abs()
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    )
    assert fused.fused_activation_loaded is True
    assert fused.fused_activation_active is True
    assert error.max().item() < 0.08
    assert error.mean().item() < 0.01
    assert relative_l2.item() < 0.03
    assert cosine.item() > 0.999
