# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ROCm diffusion workers disable MIOpen's nondeterministic NHWC GTC conv solver."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GTC_NHWC_FWD = "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC"


@pytest.mark.parametrize(("user_value", "expected"), [(None, "0"), ("1", "1")], ids=["default", "user-override"])
def test_rocm_diffusion_worker_disables_gtc_nhwc_solver(monkeypatch, user_value, expected):
    # ROCm import reads the architecture; this policy test needs no device.
    with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(gcnArchName="gfx942")):
        from vllm_omni.platforms.rocm.platform import RocmOmniPlatform

    # Record the variable so monkeypatch also undoes the worker's setdefault.
    monkeypatch.setenv(GTC_NHWC_FWD, user_value or "")
    if user_value is None:
        monkeypatch.delenv(GTC_NHWC_FWD)

    RocmOmniPlatform.init_diffusion_worker_vllm_config(vllm_config=None)

    assert os.environ[GTC_NHWC_FWD] == expected
