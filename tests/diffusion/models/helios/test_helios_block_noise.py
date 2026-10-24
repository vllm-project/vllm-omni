# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.helios.pipeline_helios import _block_noise_cholesky_factor
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("is_npu", "expected_factor_device", "expected_factor_dtype"),
    [
        (True, "cpu", torch.float64),
        (False, "meta", torch.float32),
    ],
)
def test_block_noise_cholesky_uses_cpu_only_on_npu(
    monkeypatch: pytest.MonkeyPatch,
    is_npu: bool,
    expected_factor_device: str,
    expected_factor_dtype: torch.dtype,
) -> None:
    seen_device: torch.device | None = None
    seen_dtype: torch.dtype | None = None

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: is_npu)

    def fake_cholesky(cov: torch.Tensor) -> torch.Tensor:
        nonlocal seen_device, seen_dtype
        seen_device = cov.device
        seen_dtype = cov.dtype
        assert cov.shape == (4, 4)
        return cov

    monkeypatch.setattr(torch.linalg, "cholesky", fake_cholesky)

    factor = _block_noise_cholesky_factor(4, 1 / 3, torch.device("meta"))

    assert seen_device is not None
    assert seen_device.type == expected_factor_device
    assert seen_dtype == expected_factor_dtype
    assert factor.device.type == "meta"
    assert factor.dtype == torch.float32


def test_npu_fallback_factors_the_singular_plus_ridge_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    factor = _block_noise_cholesky_factor(4, 1 / 3, torch.device("cpu"))

    assert factor.dtype == torch.float32
    assert torch.isfinite(factor).all()
    torch.testing.assert_close(factor[-1, -1], torch.tensor(2e-4), rtol=1e-4, atol=0)
