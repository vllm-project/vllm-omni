# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import runpy
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3DiTBlock
from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu.models import minimax_h3_modulation as modulation

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _block():
    block = MiniMaxH3DiTBlock.__new__(MiniMaxH3DiTBlock)
    nn.Module.__init__(block)
    return block


@pytest.mark.cpu
@pytest.mark.parametrize("sdk_available", [True, False])
def test_platform_binds_only_loaded_h3_blocks(monkeypatch, mocker, sdk_available):
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", SimpleNamespace(NPUPlatform=object))
    platform = runpy.run_module("vllm_omni.platforms.npu.platform")["NPUOmniPlatform"]
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda: "Ascend950PR"), raising=False)
    sdk = (
        SimpleNamespace(rms_norm_indexed_scale_shift=mocker.Mock(), indexed_gate=mocker.Mock())
        if sdk_available
        else None
    )
    monkeypatch.setitem(sys.modules, "mindiesd.layers.index_modulation", sdk)
    block = _block()

    platform.apply_diffusion_model_patches(nn.Sequential(block), SimpleNamespace(model_class_name="MiniMaxH3Pipeline"))

    assert (block._indexed_gate is not modulation.indexed_gate) == sdk_available
    assert (block._rms_norm_indexed_scale_shift is not modulation.rms_norm_indexed_scale_shift) == sdk_available
    assert MiniMaxH3DiTBlock._indexed_gate is modulation.indexed_gate
    assert MiniMaxH3DiTBlock._rms_norm_indexed_scale_shift is modulation.rms_norm_indexed_scale_shift


@pytest.mark.cpu
@pytest.mark.parametrize("width", [8192, 8193])
def test_rms_sdk_width_limit(mocker, width):
    # Isolate the SDK width boundary from the hardware limits of the fallback kernel.
    mocker.patch.object(modulation, "_supports_indexed_inputs", return_value=True)
    sdk_rms = mocker.Mock()
    omni_rms = mocker.patch.object(modulation, "rms_norm_indexed_scale_shift")
    x = torch.empty(1, width, dtype=torch.bfloat16)
    args = (x, x[0], x, x, torch.zeros(1, dtype=torch.long), 1e-6)

    result = modulation._rms_norm_indexed_scale_shift(sdk_rms, *args)

    selected, unused = (sdk_rms, omni_rms) if width == 8192 else (omni_rms, sdk_rms)
    selected.assert_called_once_with(*args)
    unused.assert_not_called()
    assert result is selected.return_value


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    ("dtype", "eps", "col_stride", "expected_sdk_calls"),
    [
        pytest.param(torch.bfloat16, 1e-6, 1, (1, 1), id="h3-bf16"),
        pytest.param(torch.float16, 1e-6, 1, (0, 0), id="fp16"),
        pytest.param(torch.bfloat16, 0.0, 1, (0, 1), id="zero-eps"),
        pytest.param(torch.bfloat16, 1e-50, 1, (0, 1), id="eps-rounds-to-zero"),
        pytest.param(torch.bfloat16, 1e-6, 2, (0, 0), id="strided-columns"),
    ],
)
def test_modulation_dispatch_and_outputs(mocker, dtype, eps, col_stride, expected_sdk_calls):
    sdk = pytest.importorskip("mindiesd.layers.index_modulation")
    sdk_rms = mocker.spy(sdk, "rms_norm_indexed_scale_shift")
    sdk_gate = mocker.spy(sdk, "indexed_gate")
    block = _block()
    modulation.apply_minimax_h3_modulation(block)

    width = 5376
    device = current_omni_platform.get_torch_device()
    generator = torch.Generator().manual_seed(1234)
    x, other, gate, shift, scale = [
        torch.randn(rows, width * col_stride, generator=generator).to(dtype) for rows in (5, 5, 3, 3, 3)
    ]
    weight = torch.randn(width, generator=generator).to(dtype)
    indices = torch.tensor([2, 0, 2, 1, 0])
    npu_x, npu_other, npu_gate, npu_shift, npu_scale = (
        t.to(device)[:, ::col_stride] for t in (x, other, gate, shift, scale)
    )
    npu_indices = indices.to(device)
    result_rms = block._rms_norm_indexed_scale_shift(npu_x, weight.to(device), npu_shift, npu_scale, npu_indices, eps)
    result_gate = block._indexed_gate(npu_x, npu_gate, npu_other, npu_indices)
    assert (sdk_rms.call_count, sdk_gate.call_count) == expected_sdk_calls

    x, other, gate, shift, scale = (t[:, ::col_stride].float() for t in (x, other, gate, shift, scale))
    norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()
    torch.testing.assert_close(result_rms.cpu(), (norm * (1 + scale[indices]) + shift[indices]).to(dtype))
    torch.testing.assert_close(result_gate.cpu(), (x + gate[indices] * other).to(dtype))
