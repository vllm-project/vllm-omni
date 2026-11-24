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
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu.models import minimax_h3_modulation as modulation

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _pipeline():
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    block = MiniMaxH3DiTBlock.__new__(MiniMaxH3DiTBlock)
    nn.Module.__init__(block)
    pipeline.transformer = nn.Sequential(block)
    return pipeline


@pytest.mark.cpu
@pytest.mark.parametrize("sdk_available", [True, False])
def test_platform_binds_only_loaded_h3_blocks(monkeypatch, sdk_available):
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", SimpleNamespace(NPUPlatform=object))
    platform = runpy.run_module("vllm_omni.platforms.npu.platform")["NPUOmniPlatform"]
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda: "Ascend950PR"), raising=False)
    sdk = SimpleNamespace(rms_norm_indexed_scale_shift=object(), indexed_gate=object()) if sdk_available else None
    monkeypatch.setitem(sys.modules, "mindiesd", sdk)
    monkeypatch.setitem(sys.modules, "mindiesd.layers.index_modulation", sdk)
    pipeline = _pipeline()

    platform.apply_diffusion_model_patches(pipeline, None)

    block = pipeline.transformer[0]
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
    ("width", "dtype", "eps", "col_stride", "expect_rms", "expect_gate"),
    [
        pytest.param(5376, torch.bfloat16, 1e-6, 1, True, True, id="h3-bf16"),
        pytest.param(5376, torch.float16, 1e-6, 1, False, False, id="fp16"),
        pytest.param(128, torch.bfloat16, 1e-50, 1, False, True, id="eps-rounds-to-zero"),
        pytest.param(128, torch.bfloat16, 1e-6, 2, False, False, id="strided-columns"),
    ],
)
def test_modulation_dispatch_and_outputs(mocker, width, dtype, eps, col_stride, expect_rms, expect_gate):
    sdk = pytest.importorskip("mindiesd")
    sdk_module = pytest.importorskip("mindiesd.layers.index_modulation")
    sdk_rms = mocker.spy(sdk, "rms_norm_indexed_scale_shift")
    sdk_gate = mocker.spy(sdk_module, "indexed_gate")
    omni_rms = mocker.spy(modulation, "rms_norm_indexed_scale_shift")
    omni_gate = mocker.spy(modulation, "indexed_gate")
    pipeline = _pipeline()
    modulation.apply_minimax_h3_modulation(pipeline)
    block = pipeline.transformer[0]

    device = current_omni_platform.get_torch_device()
    generator = torch.Generator().manual_seed(1234)
    x, other, gate, shift, scale = [
        torch.randn(rows, width * col_stride, generator=generator).to(dtype) for rows in (5, 5, 3, 3, 3)
    ]
    weight = torch.randn(width, generator=generator).to(dtype)
    indices = torch.tensor([2, 0, 2, 1, 0])
    result_rms = block._rms_norm_indexed_scale_shift(
        x.to(device)[:, ::col_stride],
        weight.to(device),
        shift.to(device)[:, ::col_stride],
        scale.to(device)[:, ::col_stride],
        indices.to(device),
        eps,
    )
    result_gate = block._indexed_gate(
        x.to(device)[:, ::col_stride],
        gate.to(device)[:, ::col_stride],
        other.to(device)[:, ::col_stride],
        indices.to(device),
    )
    assert (sdk_rms.call_count, omni_rms.call_count) == (int(expect_rms), int(not expect_rms))
    assert (sdk_gate.call_count, omni_gate.call_count) == (int(expect_gate), int(not expect_gate))

    x, other, gate, shift, scale = (t[:, ::col_stride].float() for t in (x, other, gate, shift, scale))
    norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()
    torch.testing.assert_close(result_rms.cpu(), (norm * (1 + scale[indices]) + shift[indices]).to(dtype))
    torch.testing.assert_close(result_gate.cpu(), (x + gate[indices] * other).to(dtype))
