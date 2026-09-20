# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-device public FIR tests without model checkpoints."""

import pytest
import torch
from tests.model_executor.models.common.test_alias_free_activation import ENV, native_resample
from vllm_omni.model_executor.models.common import alias_free_activation as fir

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.npu,
    pytest.mark.skipif(not fir.current_omni_platform.is_npu(), reason="requires Ascend NPU"),
]


@pytest.fixture(autouse=True)
def full_precision_convolution(monkeypatch):
    pytest.importorskip("torch_npu")
    monkeypatch.setattr(torch.npu.conv, "allow_hf32", False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,channels,length", [(1, 32, 65), (2, 3, 6145), (1, 32, 12288)])
def test_public_npu_resampler(monkeypatch, dtype, batch, channels, length):
    monkeypatch.setenv(ENV, "6144")
    with torch.device("npu"):
        module = fir.UpSample1d().to(dtype=dtype)
    assert module._max_conv_length == 6144
    generator = torch.Generator(device="cpu").manual_seed(42)
    x_cpu = torch.randn(batch, channels, length, generator=generator, device="cpu").to(dtype)
    x = x_cpu.to("npu")
    expected = native_resample(module, x_cpu)
    native = fir.F.conv_transpose1d
    lengths = []

    def capture(segment, *args, **kwargs):
        lengths.append(segment.shape[-1])
        return native(segment, *args, **kwargs)

    monkeypatch.setattr(fir.F, "conv_transpose1d", capture)
    with torch.inference_mode():
        actual = module(x)
    assert actual.device.type == "npu" and actual.dtype == dtype
    assert max(lengths) <= 6144
    assert (len(lengths) > 1) == (length > 6144)
    assert torch.isfinite(actual).all()
    tolerance = {torch.float32: (1e-4, 1e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}
    rtol, atol = tolerance[dtype]
    torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
    monkeypatch.setenv(ENV, "0")
    assert module._max_conv_length == 6144


@pytest.mark.parametrize("allow_hf32", [False, True])
def test_npu_whole_vs_chunked(monkeypatch, allow_hf32):
    monkeypatch.setattr(torch.npu.conv, "allow_hf32", allow_hf32)
    monkeypatch.setenv(ENV, "32")
    with torch.device("npu"):
        module = fir.UpSample1d()
    x = torch.randn(2, 3, 129, device="cpu").to("npu")
    with torch.inference_mode():
        chunked = module(x)
        module._max_conv_length = 0
        whole = module(x)
    torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-6)
