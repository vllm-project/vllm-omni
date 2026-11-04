# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for Ascend MXFP8 scale_alg selection."""

from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize("value,expected", [(None, 0), ("0", 0), ("1", 1)])
def test_mxfp8_scale_alg(monkeypatch, value, expected):
    from vllm_omni.quantization.mxfp8_config import _mx_scale_alg

    monkeypatch.delenv("VLLM_OMNI_MXFP8_SCALE_ALG", raising=False)
    if value is not None:
        monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_ALG", value)
    assert _mx_scale_alg() == expected


def test_mxfp8_activation_passes_scale_alg(monkeypatch, mocker: MockerFixture):
    import sys

    from vllm_omni.quantization.mxfp8_config import NPUMxfp8LinearMethod

    monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_ALG", "1")
    quantize = mocker.Mock(return_value=("quantized", "scale"))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_dynamic_mx_quant=quantize, float8_e4m3fn="fp8"))
    method = NPUMxfp8LinearMethod(mocker.Mock())
    x = torch.ones(2, 64, dtype=torch.bfloat16)
    assert method._quantize_activation(x) == ("quantized", "scale")
    quantize.assert_called_once_with(x, dst_type="fp8", scale_alg=1)
