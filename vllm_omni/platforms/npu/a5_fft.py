# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FFT-based STFT/ISTFT replacements for Ascend A5."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

_TORCH_STFT = torch.stft
_TORCH_ISTFT = torch.istft
_TORCH_HANN_WINDOW = torch.hann_window
_PATCHED = False


class _FallbackError(Exception):
    pass


@lru_cache(maxsize=16)
def _hann_numpy(window_length: int, periodic: bool) -> np.ndarray:
    if window_length == 0:
        return np.empty(0, dtype=np.float64)
    if window_length == 1:
        return np.ones(1, dtype=np.float64)
    if periodic:
        return np.hanning(window_length + 1)[:-1]
    return np.hanning(window_length)


def hann_window(
    window_length: int,
    periodic: bool = True,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    target = torch.device(device) if device is not None else torch.get_default_device()
    if layout != torch.strided or target.type == "cpu":
        return _TORCH_HANN_WINDOW(
            window_length,
            periodic,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )
    data = _hann_numpy(int(window_length), bool(periodic))
    tensor = torch.from_numpy(data.copy()).to(
        dtype=dtype if dtype is not None else torch.get_default_dtype(),
        device=target,
    )
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def _stft_on_cpu(
    input: torch.Tensor,
    n_fft: int,
    hop_length: int | None,
    win_length: int | None,
    window: torch.Tensor | None,
    center: bool,
    pad_mode: str,
    normalized: bool,
    onesided: bool | None,
    return_complex: bool,
) -> torch.Tensor:
    device = input.device
    result = _TORCH_STFT(
        input.cpu(),
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.cpu() if window is not None else None,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
        return_complex=return_complex,
    )
    return result.to(device)


def _stft_fft(
    input: torch.Tensor,
    n_fft: int,
    hop_length: int | None,
    win_length: int | None,
    window: torch.Tensor | None,
    center: bool,
    pad_mode: str,
    normalized: bool,
    onesided: bool | None,
    return_complex: bool,
) -> torch.Tensor:
    if input.dim() not in (1, 2) or torch.is_complex(input) or n_fft <= 0:
        raise _FallbackError

    hop_length = n_fft // 4 if hop_length is None else int(hop_length)
    win_length = n_fft if win_length is None else int(win_length)
    if hop_length <= 0 or win_length <= 0 or win_length > n_fft:
        raise _FallbackError

    work_dtype = torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
    if window is None:
        window = torch.ones(win_length, device=input.device, dtype=work_dtype)
    elif window.ndim != 1 or window.numel() != win_length:
        raise _FallbackError
    else:
        window = window.to(device=input.device, dtype=work_dtype)

    if center:
        pad = n_fft // 2
        input = F.pad(input, (pad, pad), mode=pad_mode)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        window = F.pad(window, (left, n_fft - win_length - left))

    if onesided is None:
        onesided = True
    freq = n_fft // 2 + 1 if onesided else n_fft
    if input.shape[-1] < n_fft:
        complex_dtype = torch.complex64 if work_dtype == torch.float32 else torch.complex128
        shape = (*input.shape[:-1], freq, 0)
        return torch.empty(shape, dtype=complex_dtype if return_complex else work_dtype, device=input.device)

    frames = input.unfold(-1, n_fft, hop_length).to(work_dtype) * window
    spec = torch.fft.rfft(frames, dim=-1) if onesided else torch.fft.fft(frames, dim=-1)
    if normalized:
        spec = spec / (n_fft**0.5)
    spec = spec.transpose(-2, -1)
    if return_complex:
        return spec
    return torch.view_as_real(spec)


def stft(
    input: torch.Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: torch.Tensor | None = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool | None = None,
    return_complex: bool = False,
) -> torch.Tensor:
    if input.device.type == "cpu":
        return _stft_on_cpu(
            input,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
            normalized,
            onesided,
            return_complex,
        )
    try:
        return _stft_fft(
            input,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
            normalized,
            onesided,
            return_complex,
        )
    except _FallbackError:
        return _stft_on_cpu(
            input,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
            normalized,
            onesided,
            return_complex,
        )


def _istft_on_cpu(
    input: torch.Tensor,
    n_fft: int,
    hop_length: int | None,
    win_length: int | None,
    window: torch.Tensor | None,
    center: bool,
    normalized: bool,
    onesided: bool | None,
    length: int | None,
    return_complex: bool,
) -> torch.Tensor:
    device = input.device
    result = _TORCH_ISTFT(
        input.cpu(),
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.cpu() if window is not None else None,
        center=center,
        normalized=normalized,
        onesided=onesided,
        length=length,
        return_complex=return_complex,
    )
    return result.to(device)


def _istft_fft(
    input: torch.Tensor,
    n_fft: int,
    hop_length: int | None,
    win_length: int | None,
    window: torch.Tensor | None,
    center: bool,
    normalized: bool,
    onesided: bool | None,
    length: int | None,
    return_complex: bool,
) -> torch.Tensor:
    if input.dim() not in (2, 3) or return_complex or n_fft <= 0:
        raise _FallbackError

    hop_length = n_fft // 4 if hop_length is None else int(hop_length)
    win_length = n_fft if win_length is None else int(win_length)
    if hop_length <= 0 or win_length <= 0 or win_length > n_fft:
        raise _FallbackError

    if not torch.is_complex(input):
        if input.shape[-1] != 2 or input.dtype not in (torch.float32, torch.float64):
            raise _FallbackError
        input = torch.view_as_complex(input.contiguous())
    if input.shape[-2] == n_fft // 2 + 1:
        onesided = True
    elif input.shape[-2] == n_fft:
        onesided = False
    else:
        raise _FallbackError

    real_dtype = torch.float64 if input.dtype == torch.complex128 else torch.float32
    if window is None:
        window = torch.ones(win_length, device=input.device, dtype=real_dtype)
    elif window.ndim != 1 or window.numel() != win_length:
        raise _FallbackError
    else:
        window = window.to(device=input.device, dtype=real_dtype)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        window = F.pad(window, (left, n_fft - win_length - left))

    n_frames = input.shape[-1]
    if n_frames <= 0:
        raise _FallbackError
    expected_len = n_fft + hop_length * (n_frames - 1)
    envelope = torch.ops.aten.unfold_backward(
        window.square().expand(n_frames, n_fft),
        [expected_len],
        0,
        n_fft,
        hop_length,
    )

    if normalized:
        input = input * (n_fft**0.5)
    spectrum = input.transpose(-2, -1).contiguous()
    if onesided:
        frames = torch.fft.irfft(spectrum, n=n_fft, dim=-1)
    else:
        frames = torch.fft.ifft(spectrum, dim=-1).real
    frames = frames * window

    batch_shape = frames.shape[:-2]
    flat = frames.reshape(-1, n_frames, n_fft)
    signal = torch.ops.aten.unfold_backward(flat, [flat.shape[0], expected_len], 1, n_fft, hop_length)
    signal = signal.reshape(*batch_shape, expected_len)

    if length is None:
        target_len = expected_len - (n_fft if center else 0)
    else:
        target_len = int(length)
    if target_len < 0:
        raise _FallbackError
    offset = n_fft // 2 if center else 0
    available = min(target_len, expected_len - offset)
    signal = signal[..., offset : offset + available]
    envelope = envelope[offset : offset + available]
    if envelope.numel() == 0 or envelope.min() <= 0:
        raise _FallbackError
    signal = signal / envelope
    if target_len > available:
        signal = F.pad(signal, (0, target_len - available))
    return signal


def istft(
    input: torch.Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: torch.Tensor | None = None,
    center: bool = True,
    normalized: bool = False,
    onesided: bool | None = None,
    length: int | None = None,
    return_complex: bool = False,
) -> torch.Tensor:
    if input.device.type == "cpu":
        return _istft_on_cpu(
            input,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            normalized,
            onesided,
            length,
            return_complex,
        )
    try:
        return _istft_fft(
            input,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            normalized,
            onesided,
            length,
            return_complex,
        )
    except _FallbackError:
        return _istft_on_cpu(
            input,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            normalized,
            onesided,
            length,
            return_complex,
        )


def _install() -> None:
    setattr(torch, "stft", stft)
    setattr(torch, "istft", istft)
    setattr(torch, "hann_window", hann_window)
    setattr(torch.functional, "stft", stft)
    setattr(torch.functional, "istft", istft)


def apply_a5_fft_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from vllm_omni.platforms.npu import is_a5

    if not is_a5():
        return
    _install()
    _PATCHED = True
