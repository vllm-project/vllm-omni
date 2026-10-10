# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_omni.model_executor.models.common.ming.audio_dsp import ISTFT

pytestmark = [pytest.mark.core_model, pytest.mark.tts]
cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or not HAS_TRITON,
    reason="NVIDIA CUDA and Triton required",
)


def _assert_outputs_close(actual, expected):
    for value, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert value is None
        else:
            assert value.dtype == reference.dtype
            torch.testing.assert_close(value, reference, atol=5e-7, rtol=1e-5)


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("hop", [4, 320, 882])
@pytest.mark.parametrize("window_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch,frames", [(1, 2), (1, 25), (3, 7), (2, 100)])
@pytest.mark.parametrize("mode", ["offline", "first", "middle", "last", "first_last"])
@torch.inference_mode()
def test_fused_istft_matches_native(hop, window_dtype, batch, frames, mode, monkeypatch):
    torch.manual_seed(0)
    module = ISTFT(4 * hop, hop, 4 * hop).to(device="cuda", dtype=window_dtype)
    # Slice a larger spectrum so irfft must handle non-contiguous inputs.
    spec = torch.randn(batch, 2 * hop + 1, 2 * frames, device="cuda", dtype=torch.complex64)[..., ::2]
    kwargs = {"streaming": mode != "offline", "last_chunk": mode in ("last", "first_last")}
    if mode in ("middle", "last"):
        _, audio, window = module.forward_native(spec, streaming=True)
        kwargs.update(audio_buffer=audio, window_buffer=window)
        saved_audio, saved_window = audio.clone(), window.clone()
    expected = module.forward_native(spec, **kwargs)

    def no_fallback(*args, **kwargs):
        pytest.fail("supported input fell back to native ISTFT")

    monkeypatch.setattr(module, "forward_native", no_fallback)
    _assert_outputs_close(module(spec, **kwargs), expected)
    if mode in ("middle", "last"):
        torch.testing.assert_close(audio, saved_audio, rtol=0, atol=0)
        torch.testing.assert_close(window, saved_window, rtol=0, atol=0)


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("hop", [320, 882])
@pytest.mark.parametrize("window_dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fused_istft_streaming_state(hop, window_dtype):
    torch.manual_seed(1)
    module = ISTFT(4 * hop, hop, 4 * hop).to(device="cuda", dtype=window_dtype)
    spec = torch.randn(2, 2 * hop + 1, 40, device="cuda", dtype=torch.complex64)
    # Reusing the same module for a new request must not reuse the old tails.
    for _ in range(2):
        actual_state = reference_state = (None, None)
        actual_chunks, reference_chunks = [], []
        for start, end in [(0, 4), (4, 29), (29, 31), (31, 40)]:
            chunk = spec[..., start:end]
            actual = module(chunk, *actual_state, streaming=True, last_chunk=end == 40)
            expected = module.forward_native(chunk, *reference_state, streaming=True, last_chunk=end == 40)
            _assert_outputs_close(actual, expected)
            actual_state, reference_state = actual[1:], expected[1:]
            actual_chunks.append(actual[0])
            reference_chunks.append(expected[0])
        output = torch.cat(actual_chunks, dim=-1)
        torch.testing.assert_close(output, torch.cat(reference_chunks, dim=-1), atol=5e-7, rtol=1e-5)
        if window_dtype == torch.float32:
            # Low-precision window envelopes already round differently between
            # streaming and offline native execution; preserve each contract.
            torch.testing.assert_close(output, module.forward_native(spec)[0], atol=5e-7, rtol=1e-5)


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("invalid", [False, True])
@torch.inference_mode()
def test_fused_istft_silence_and_invalid_envelope(invalid):
    module = ISTFT(1280, 320, 1280).cuda()
    spec = torch.zeros(2, 641, 25, device="cuda", dtype=torch.complex64)
    if invalid:
        module.window.zero_()
        for forward in (module.forward_native, module.forward):
            with pytest.raises(RuntimeError, match="window envelope underflowed"):
                forward(spec)
    else:
        _assert_outputs_close(module(spec), module.forward_native(spec))
        assert torch.count_nonzero(module(spec)[0]) == 0


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=[pytest.mark.cuda, cuda])],
)
@pytest.mark.parametrize("padding", ["same", "center"])
def test_istft_autograd_fallback(padding, device, monkeypatch):
    module = ISTFT(32, 8, 32, padding=padding).to(device)
    spec = torch.randn(2, 17, 7, device=device, dtype=torch.complex64, requires_grad=True)
    native = module.forward_native
    calls = []

    def tracked(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(module, "forward_native", tracked)
    result = module(spec)
    assert calls == [True]
    reference = native(spec)
    if padding == "same":
        _assert_outputs_close(result, reference)
        result = result[0]
    else:
        torch.testing.assert_close(result, reference)
    result.square().sum().backward()
    assert spec.grad is not None and torch.isfinite(spec.grad).all()


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize(
    "n_fft,hop,frames,dtype", [(32, 8, 1, torch.complex64), (30, 10, 7, torch.complex64), (32, 8, 7, torch.complex128)]
)
@torch.inference_mode()
def test_istft_unsupported_cuda_falls_back(n_fft, hop, frames, dtype, monkeypatch):
    module = ISTFT(n_fft, hop, n_fft).cuda()
    spec = torch.randn(1, n_fft // 2 + 1, frames, device="cuda", dtype=dtype)
    native = module.forward_native
    calls = []

    def tracked(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(module, "forward_native", tracked)
    _assert_outputs_close(module(spec), native(spec))
    assert calls == [True]
