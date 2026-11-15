# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SHAPES = [(1024, 256, 1024), (400, 160, 400), (16, 4, 16), (1024, 256, 800)]


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _load_module():
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "a5_fft.py"
    spec = importlib.util.spec_from_file_location("vllm_omni_test_a5_fft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_npu(monkeypatch: pytest.MonkeyPatch, *, a5: bool) -> None:
    for name in ("vllm_omni", "vllm_omni.platforms"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    npu = types.ModuleType("vllm_omni.platforms.npu")
    npu.__path__ = []
    setattr(npu, "is_a5", lambda: a5)
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.npu", npu)


def _save_torch_ops():
    return (
        torch.stft,
        torch.istft,
        torch.hann_window,
        torch.functional.stft,
        torch.functional.istft,
        torch.Tensor.stft,
        torch.Tensor.istft,
    )


def _restore_torch_ops(saved) -> None:
    torch.stft, torch.istft, torch.hann_window, torch.functional.stft, torch.functional.istft = saved[:5]
    setattr(torch.Tensor, "stft", saved[5])
    setattr(torch.Tensor, "istft", saved[6])


@pytest.mark.parametrize("window_length", [0, 1, 2, 5, 16, 400, 1024])
@pytest.mark.parametrize("periodic", [True, False])
def test_hann_numpy_matches_torch(window_length: int, periodic: bool) -> None:
    module = _load_module()
    actual = module._hann_numpy(window_length, periodic)
    expected = torch.hann_window(window_length, periodic=periodic).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_hann_window_uses_numpy_for_non_cpu_device() -> None:
    module = _load_module()
    window = module.hann_window(16, device="meta")
    assert window.device.type == "meta"
    assert window.shape == (16,)
    assert window.dtype == torch.float32


def test_hann_window_accepts_pin_memory() -> None:
    module = _load_module()
    window = module.hann_window(4, pin_memory=False)
    assert window.shape == (4,)


@pytest.mark.parametrize("n_fft,hop_length,win_length", _SHAPES)
@pytest.mark.parametrize("center", [True, False])
def test_stft_fft_matches_torch(
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool,
) -> None:
    module = _load_module()
    torch.manual_seed(0)
    audio = torch.randn(2, n_fft * 3 + 100)
    window = torch.hann_window(win_length)
    expected = torch.stft(
        audio,
        n_fft,
        hop_length,
        win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    actual = module._stft_fft(
        audio,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        "reflect",
        False,
        None,
        True,
    )
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_stft_fft_matches_torch_normalized_real_and_full_spectrum() -> None:
    module = _load_module()
    torch.manual_seed(1)
    audio = torch.randn(1, 64)
    window = torch.hann_window(16)
    expected = torch.stft(
        audio,
        16,
        4,
        16,
        window=window,
        normalized=True,
        onesided=False,
        return_complex=False,
    )
    actual = module._stft_fft(
        audio,
        16,
        4,
        16,
        window,
        True,
        "reflect",
        True,
        False,
        False,
    )
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("center", [True, False])
def test_stft_fft_matches_torch_1d(center: bool) -> None:
    module = _load_module()
    torch.manual_seed(4)
    audio = torch.randn(1000)
    window = torch.hann_window(400)
    expected = torch.stft(
        audio,
        400,
        160,
        400,
        window=window,
        center=center,
        return_complex=True,
    )
    actual = module._stft_fft(audio, 400, 160, 400, window, center, "reflect", False, None, True)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_stft_fft_rejects_short_input() -> None:
    module = _load_module()
    window = torch.hann_window(16)
    with pytest.raises(module._FallbackError):
        module._stft_fft(torch.randn(8), 16, 4, 16, window, False, "reflect", False, None, True)


def test_stft_fft_falls_back_for_complex_window() -> None:
    module = _load_module()
    torch.manual_seed(8)
    audio = torch.randn(1, 64)
    window = torch.ones(16, dtype=torch.complex64) * 1j
    with pytest.raises(module._FallbackError):
        module._stft_fft(audio, 16, 4, 16, window, True, "reflect", False, False, True)

    expected = torch.stft(audio, 16, 4, 16, window=window, center=True, onesided=False, return_complex=True)
    actual = module._stft_on_cpu(audio, 16, 4, 16, window, True, "reflect", False, False, True)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_stft_fft_accepts_low_precision_input(dtype: torch.dtype) -> None:
    module = _load_module()
    torch.manual_seed(5)
    audio = torch.randn(2, 64, dtype=dtype)
    window = torch.hann_window(16, dtype=dtype)
    expected = module._stft_fft(
        audio.float(),
        16,
        4,
        16,
        window.float(),
        True,
        "reflect",
        False,
        None,
        True,
    )
    actual = module._stft_fft(audio, 16, 4, 16, window, True, "reflect", False, None, True)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("n_fft,hop_length,win_length", _SHAPES)
def test_istft_fft_matches_torch(
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> None:
    module = _load_module()
    torch.manual_seed(2)
    audio = torch.randn(2, n_fft * 3 + 100)
    window = torch.hann_window(win_length)
    spec = torch.stft(
        audio,
        n_fft,
        hop_length,
        win_length,
        window=window,
        return_complex=True,
    )
    expected = torch.istft(
        spec,
        n_fft,
        hop_length,
        win_length,
        window=window,
        length=audio.shape[-1],
    )
    actual = module._istft_fft(
        spec,
        n_fft,
        hop_length,
        win_length,
        window,
        True,
        False,
        None,
        audio.shape[-1],
        False,
    )
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_istft_fft_matches_torch_normalized_and_full_spectrum() -> None:
    module = _load_module()
    torch.manual_seed(3)
    audio = torch.randn(1, 64)
    window = torch.hann_window(16)
    spec = torch.stft(
        audio,
        16,
        4,
        16,
        window=window,
        normalized=True,
        onesided=False,
        return_complex=True,
    )
    expected = torch.istft(
        spec,
        16,
        4,
        16,
        window=window,
        normalized=True,
        onesided=False,
        length=50,
    )
    actual = module._istft_fft(spec, 16, 4, 16, window, True, True, False, 50, False)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_istft_fft_matches_torch_length_beyond_reconstruction() -> None:
    module = _load_module()
    torch.manual_seed(6)
    audio = torch.randn(1, 64)
    window = torch.hann_window(16)
    spec = torch.stft(audio, 16, 4, 16, window=window, return_complex=True)
    expected = torch.istft(spec, 16, 4, 16, window=window, length=100)
    actual = module._istft_fft(spec, 16, 4, 16, window, True, False, None, 100, False)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_istft_fft_full_spectrum_real_output_matches_torch() -> None:
    module = _load_module()
    spec = torch.zeros(1, 4, 1, dtype=torch.complex64)
    spec[0, 1, 0] = 1
    window = torch.ones(4)
    expected = torch.istft(spec, 4, 4, 4, window=window, center=False, onesided=False)
    actual = module._istft_fft(spec, 4, 4, 4, window, False, False, False, None, False)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_istft_fft_rejects_complex_output_with_full_spectrum() -> None:
    module = _load_module()
    spec = torch.zeros(1, 4, 1, dtype=torch.complex64)
    spec[0, 1, 0] = 1
    with pytest.raises(module._FallbackError):
        module._istft_fft(spec, 4, 4, 4, torch.ones(4), False, False, False, None, True)


def test_istft_envelope_cache_recomputes_after_window_mutation() -> None:
    module = _load_module()
    torch.manual_seed(7)
    audio = torch.randn(1, 64)
    window = torch.hann_window(16)
    spec = torch.stft(audio, 16, 4, 16, window=window, return_complex=True)
    expected = torch.istft(spec, 16, 4, 16, window=window)
    actual = module._istft_fft(spec, 16, 4, 16, window, True, False, None, None, False)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    window.mul_(0.5)
    expected = torch.istft(spec, 16, 4, 16, window=window)
    actual = module._istft_fft(spec, 16, 4, 16, window, True, False, None, None, False)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_istft_fft_handles_inference_window() -> None:
    module = _load_module()
    with torch.inference_mode():
        torch.manual_seed(9)
        audio = torch.randn(1, 64)
        window = torch.hann_window(16)
        spec = torch.stft(audio, 16, 4, 16, window=window, return_complex=True)
        expected = torch.istft(spec, 16, 4, 16, window=window)
        actual = module._istft_fft(spec, 16, 4, 16, window, True, False, None, None, False)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert not module._ENVELOPE_CACHE


def test_stft_falls_back_on_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_WARNED", set())

    def fail(*_args, **_kwargs):
        raise RuntimeError("unsupported op")

    monkeypatch.setattr(module, "_stft_fft", fail)
    monkeypatch.setattr(module, "_stft_on_cpu", lambda *_args, **_kwargs: torch.empty(0))
    with pytest.warns(RuntimeWarning, match="A5 FFT STFT failed"):
        result = module.stft(torch.empty(0, device="meta"), 16, 4, 16)
    assert result.numel() == 0


def test_istft_falls_back_on_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_WARNED", set())

    def fail(*_args, **_kwargs):
        raise RuntimeError("unsupported op")

    monkeypatch.setattr(module, "_istft_fft", fail)
    monkeypatch.setattr(module, "_istft_on_cpu", lambda *_args, **_kwargs: torch.empty(0))
    with pytest.warns(RuntimeWarning, match="A5 FFT ISTFT failed"):
        result = module.istft(torch.empty(0, device="meta"), 16, 4, 16)
    assert result.numel() == 0


def test_stft_propagates_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()

    def fail(*_args, **_kwargs):
        raise torch.OutOfMemoryError("oom")

    monkeypatch.setattr(module, "_stft_fft", fail)
    with pytest.raises(torch.OutOfMemoryError):
        module.stft(torch.empty(0, device="meta"), 16, 4, 16)


def test_fallback_counter_tracks_repeated_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_WARNED", set())
    monkeypatch.setattr(module, "_FALLBACK_COUNTS", {})
    with pytest.warns(RuntimeWarning, match="A5 FFT STFT failed"):
        module._warn_fallback("STFT", RuntimeError("first"))
        module._warn_fallback("STFT", RuntimeError("second"))
    assert module._FALLBACK_COUNTS == {"STFT": 2}
    assert module.fallback_counts() == {"STFT": 2}


def test_istft_propagates_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()

    def fail(*_args, **_kwargs):
        raise torch.OutOfMemoryError("oom")

    monkeypatch.setattr(module, "_istft_fft", fail)
    with pytest.raises(torch.OutOfMemoryError):
        module.istft(torch.empty(0, device="meta"), 16, 4, 16)


def test_install_patches_torch_ops() -> None:
    module = _load_module()
    saved = _save_torch_ops()
    try:
        module._install()
        assert torch.stft is module.stft
        assert torch.istft is module.istft
        assert torch.hann_window is module.hann_window
        assert torch.functional.stft is module.stft
        assert torch.functional.istft is module.istft
        assert torch.Tensor.stft is module.stft
        assert torch.Tensor.istft is module.istft
    finally:
        _restore_torch_ops(saved)


def test_apply_a5_fft_patches_skips_non_a5(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    _install_fake_npu(monkeypatch, a5=False)
    saved = _save_torch_ops()
    try:
        module.apply_a5_fft_patches()
        assert torch.stft is saved[0]
    finally:
        _restore_torch_ops(saved)
        module._PATCHED = False


def test_apply_a5_fft_patches_installs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    _install_fake_npu(monkeypatch, a5=True)
    saved = _save_torch_ops()
    try:
        module.apply_a5_fft_patches()
        assert torch.stft is module.stft
        module.apply_a5_fft_patches()
        assert torch.stft is module.stft
    finally:
        _restore_torch_ops(saved)
        module._PATCHED = False
