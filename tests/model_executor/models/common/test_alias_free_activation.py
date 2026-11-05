# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU correctness and dispatch tests; no checkpoints or serving processes."""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.common import alias_free_activation as fir

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ENV = "VLLM_OMNI_NPU_ANTIALIAS_MAX_CONV_LENGTH"


@pytest.fixture(autouse=True)
def cpu_platform(monkeypatch):
    monkeypatch.setattr(fir.current_omni_platform, "is_npu", lambda: False)
    monkeypatch.setattr(fir.current_omni_platform, "is_xpu", lambda: False)
    monkeypatch.delenv(ENV, raising=False)
    with torch.device("cpu"):
        yield


def native_resample(module, x):
    """Independent whole-tensor reference, including the existing cast order."""
    dtype = x.dtype
    channels = x.shape[1]
    weight = module.filter.to(x.device).expand(channels, -1, -1)
    if isinstance(module, fir.UpSample1d):
        padded = F.pad(x.float(), (module.pad, module.pad), mode="replicate").to(weight.dtype)
        out = module.ratio * F.conv_transpose1d(padded, weight, stride=module.stride, groups=channels).to(dtype)
        return out[..., module.pad_left : -module.pad_right]
    padded = F.pad(x.float(), (module.pad_left, module.pad_right), mode="replicate").to(weight.dtype)
    return F.conv1d(padded, weight, stride=module.stride, groups=channels).to(dtype)


@pytest.mark.parametrize("kernel,stride", [(1, 1), (7, 1), (11, 2), (12, 2), (12, 3), (12, 4), (15, 5)])
@pytest.mark.parametrize("length", [31, 32, 33, 63, 64, 65, 127])
def test_chunked_matches_native(monkeypatch, kernel, stride, length):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2, 3, length * 2, generator=generator)[..., ::2]
    weight = torch.randn(3, 1, kernel, generator=generator)
    native = F.conv_transpose1d
    expected = native(x, weight, stride=stride, groups=3)
    lengths = []

    def capture(segment, *args, **kwargs):
        lengths.append(segment.shape[-1])
        return native(segment, *args, **kwargs)

    monkeypatch.setattr(F, "conv_transpose1d", capture)
    actual = fir._chunked_conv_transpose1d(x, weight, stride, 32)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert max(lengths) <= 32
    assert (len(lengths) > 1) == (length > 32)
    assert actual.dtype == x.dtype and actual.device == x.device
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("length", [6143, 6144, 6145, 8160, 8192, 12288, 12289])
def test_large_boundary_and_strict_cap(monkeypatch, length):
    x = torch.randn(1, 2, length, generator=torch.Generator().manual_seed(1))
    weight = torch.randn(2, 1, 12, generator=torch.Generator().manual_seed(2))
    native = F.conv_transpose1d
    lengths = []

    def capture(segment, *args, **kwargs):
        lengths.append(segment.shape[-1])
        return native(segment, *args, **kwargs)

    expected = native(x, weight, stride=2, groups=2)
    monkeypatch.setattr(F, "conv_transpose1d", capture)
    torch.testing.assert_close(fir._chunked_conv_transpose1d(x, weight, 2, 6144), expected, rtol=1e-5, atol=1e-5)
    assert max(lengths) <= 6144


@pytest.mark.parametrize("kind", ["zero", "constant", "impulse"])
def test_boundary_signals(kind):
    x = torch.zeros(1, 2, 105)
    if kind == "constant":
        x.fill_(1)
    elif kind == "impulse":
        x[..., [0, 25, 26, 27, 51, 52, 53, 104]] = 1
    weight = fir.kaiser_sinc_filter1d(0.25, 0.3, 12).expand(2, -1, -1)
    torch.testing.assert_close(
        fir._chunked_conv_transpose1d(x, weight, 2, 32), F.conv_transpose1d(x, weight, stride=2, groups=2)
    )


@pytest.mark.parametrize("cap,length", [(0, 129), (32, 16), (32, 32)])
def test_native_fast_path_is_one_call(monkeypatch, cap, length):
    x = torch.ones(1, 2, length)
    weight = torch.ones(2, 1, 12)
    calls = []
    sentinel = torch.empty(0)

    def capture(segment, *args, **kwargs):
        calls.append(segment)
        return sentinel

    monkeypatch.setattr(F, "conv_transpose1d", capture)
    assert fir._chunked_conv_transpose1d(x, weight, 2, cap) is sentinel
    assert len(calls) == 1 and calls[0] is x


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ratio,kernel", [(2, 12), (3, 15)])
def test_public_npu_branch_on_cpu(monkeypatch, dtype, ratio, kernel):
    module = fir.UpSample1d(ratio, kernel)
    x = torch.randn(2, 3, 129, generator=torch.Generator().manual_seed(9)).to(dtype)
    expected = native_resample(module, x)
    monkeypatch.setattr(fir.current_omni_platform, "is_npu", lambda: True)
    module._max_conv_length = 32
    native = F.conv_transpose1d
    lengths = []

    def capture(segment, *args, **kwargs):
        lengths.append(segment.shape[-1])
        return native(segment, *args, **kwargs)

    monkeypatch.setattr(F, "conv_transpose1d", capture)
    actual = module(x)
    torch.testing.assert_close(actual, expected)
    assert len(lengths) > 1 and max(lengths) <= 32
    assert actual.dtype == dtype
    assert module.state_dict() == {}


@pytest.mark.parametrize("module_type", [fir.UpSample1d, fir.DownSample1d])
def test_non_npu_does_not_chunk(monkeypatch, module_type):
    monkeypatch.setenv(ENV, "32")
    module = module_type()
    assert getattr(module, "_max_conv_length", 0) == 0

    def unexpected(*args, **kwargs):
        pytest.fail("non-NPU path called a chunking helper")

    monkeypatch.setattr(fir, "_chunked_conv_transpose1d", unexpected)
    x = torch.randn(1, 2, 129)
    torch.testing.assert_close(module(x), native_resample(module, x))


def test_downsampling_is_unchanged(monkeypatch):
    module = fir.DownSample1d()
    monkeypatch.setattr(fir.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setenv(ENV, "32")
    x = torch.randn(1, 2, 129)
    expected = native_resample(module, x)
    native = F.conv1d
    lengths = []

    def capture(segment, *args, **kwargs):
        lengths.append(segment.shape[-1])
        return native(segment, *args, **kwargs)

    monkeypatch.setattr(F, "conv1d", capture)
    torch.testing.assert_close(module(x), expected)
    assert lengths == [129 + module.pad_left + module.pad_right]
    assert not hasattr(module, "_max_conv_length")


@pytest.mark.parametrize("value,expected", [(None, 0), ("0", 0), ("12", 12), ("6144", 6144)])
def test_npu_configuration(monkeypatch, value, expected):
    monkeypatch.setattr(fir.current_omni_platform, "is_npu", lambda: True)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert fir._npu_conv_max_length(12, 2) == expected


@pytest.mark.parametrize("value", ["-1", "1", "11", "invalid"])
def test_invalid_npu_configuration(monkeypatch, value):
    monkeypatch.setattr(fir.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError):
        fir._npu_conv_max_length(12, 2)


def test_unsupported_transpose_kernel_keeps_native(monkeypatch):
    monkeypatch.setattr(fir.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setenv(ENV, "32")
    assert fir._npu_conv_max_length(2, 3) == 0
    x, weight = torch.randn(1, 2, 65), torch.randn(2, 1, 2)
    torch.testing.assert_close(
        fir._chunked_conv_transpose1d(x, weight, 3, 32), F.conv_transpose1d(x, weight, stride=3, groups=2)
    )


def test_cap_smaller_than_kernel_rejected():
    with pytest.raises(ValueError, match="kernel"):
        fir._chunked_conv_transpose1d(torch.ones(1, 2, 65), torch.ones(2, 1, 12), 2, 11)


def test_alias_free_activation_state_dict_roundtrip(monkeypatch):
    module = fir.AliasFreeActivation1d(torch.nn.PReLU(3))
    restored = fir.AliasFreeActivation1d(torch.nn.PReLU(3))
    restored.load_state_dict(module.state_dict(), strict=True)
    assert set(module.state_dict()) == {"act.weight"}
    monkeypatch.setattr(fir.current_omni_platform, "is_npu", lambda: True)
    restored.upsample._max_conv_length = 32
    x = torch.randn(2, 3, 129)
    torch.testing.assert_close(restored(x), module(x), rtol=1e-5, atol=1e-6)
