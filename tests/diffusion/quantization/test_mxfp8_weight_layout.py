# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Loading must give CANN canonical scale strides even for a single group."""

import sys

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.quantization.mxfp8_config import (
    DiffusionMXFP8Config,
    NPUMxfp8LinearMethod,
    NPUMxfp8OnlineLinearMethod,
)

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("k", [32, 64, 96, 128])
def test_loaded_scale_has_canonical_strides(mocker: MockerFixture, offline: bool, k: int):
    n = 33
    groups = (k + 31) // 32
    raw = torch.arange(n * groups, dtype=torch.uint8).reshape(n, groups)
    padded = torch.nn.functional.pad(raw, (0, groups % 2))
    quantize = mocker.Mock(return_value=(torch.zeros(n, k, dtype=torch.bfloat16), padded))
    # Stub only the hardware cast/quantizer; execute the actual loader layout.
    mocker.patch.dict(
        sys.modules,
        {"torch_npu": mocker.Mock(float8_e4m3fn=torch.bfloat16, npu_dynamic_mx_quant=quantize)},
    )
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(torch.zeros(n, k, dtype=torch.bfloat16)))
    if offline:
        layer.register_parameter("weight_scale", torch.nn.Parameter(raw, requires_grad=False))
    config = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=offline)
    method = NPUMxfp8LinearMethod(config) if offline else NPUMxfp8OnlineLinearMethod(config)

    method.process_weights_after_loading(layer)

    assert layer.weight_scale.shape == ((groups + 1) // 2, n, 2)
    assert layer.weight_scale.stride() == (n * 2, 2, 1)
    torch.testing.assert_close(layer.weight_scale.transpose(0, 1).reshape(n, -1), padded, rtol=0, atol=0)
    assert quantize.call_count == (0 if offline else 1)
