# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FFT-based STFT/ISTFT replacements for Ascend A5.

References bound before platform initialization are not replaced.
"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_TORCH_STFT = torch.stft
_TORCH_ISTFT = torch.istft
_TORCH_HANN_WINDOW = torch.hann_window
_PATCHED = False
_ENVELOPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, int, bool]] = {}
_ENVELOPE_CACHE_MAX = 16
_WARNED: set[str] = set()
_FALLBACK_COUNTS: dict[str, int] = {}


class _FallbackError(Exception):
    pass


def _warn_fallback(op: str, error: Exception) -> None:
    count = _FALLBACK_COUNTS.get(op, 0) + 1
    _FALLBACK_COUNTS[op] = count
    logger.debug("A5 FFT %s fallback #%d: %s", op, count, error)
    if op in _WARNED:
        return
    _WARNED.add(op)
    warnings.warn(f"A5 FFT {op} failed ({error}); falling back to CPU", RuntimeWarning, stacklevel=3)


def fallback_counts() -> dict[str, int]:
    """Return A5 FFT CPU-fallback counts by operation."""
    return dict(_FALLBACK_COUNTS)


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
    pin_memory: bool = False,
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
            pin_memory=pin_memory,
        )
    data = _hann_numpy(int(window_length), bool(periodic))
    tensor = torch.from_numpy(data.copy()).to(dtype=dtype if dtype is not None else torch.get_default_dtype())
    if pin_memory:
        tensor = tensor.pin_memory()
    tensor = tensor.to(target)
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
    elif window.ndim != 1 or window.numel() != win_length or torch.is_complex(window):
        raise _FallbackError
    else:
        window = window.to(device=input.device, dtype=work_dtype)

    if center:
        pad = n_fft // 2
        if input.dim() == 1:
            input = F.pad(input.unsqueeze(0), (pad, pad), mode=pad_mode).squeeze(0)
        else:
            input = F.pad(input, (pad, pad), mode=pad_mode)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        window = F.pad(window, (left, n_fft - win_length - left))

    if input.shape[-1] < n_fft:
        raise _FallbackError
    if onesided is None:
        onesided = True

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
    except torch.OutOfMemoryError:
        raise
    except (RuntimeError, NotImplementedError) as error:
        _warn_fallback("STFT", error)
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


def _istft_envelope(
    window: torch.Tensor,
    n_frames: int,
    n_fft: int,
    hop_length: int,
    center: bool,
    target_len: int,
) -> tuple[torch.Tensor, int, bool]:
    expected_len = n_fft + hop_length * (n_frames - 1)
    offset = n_fft // 2 if center else 0
    available = min(target_len, expected_len - offset)
    cache_key = None
    if not window.is_inference():
        cache_key = (
            id(window),
            window._version,
            n_frames,
            n_fft,
            hop_length,
            center,
            target_len,
        )
        cached = _ENVELOPE_CACHE.get(cache_key)
        if cached is not None and cached[0] is window:
            return cached[1], cached[2], cached[3]
    envelope = torch.ops.aten.unfold_backward(
        window.square().expand(n_frames, n_fft),
        [expected_len],
        0,
        n_fft,
        hop_length,
    )
    envelope = envelope[offset : offset + available]
    valid = envelope.numel() > 0 and bool(envelope.abs().min() > 1e-11)
    if cache_key is not None:
        if len(_ENVELOPE_CACHE) >= _ENVELOPE_CACHE_MAX:
            _ENVELOPE_CACHE.pop(next(iter(_ENVELOPE_CACHE)))
        _ENVELOPE_CACHE[cache_key] = (window, envelope, available, valid)
    return envelope, available, valid


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
    elif window.ndim != 1 or window.numel() != win_length or torch.is_complex(window):
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
    if length is None:
        target_len = expected_len - (n_fft if center else 0)
    else:
        target_len = int(length)
    if target_len < 0:
        raise _FallbackError
    envelope, available, valid = _istft_envelope(window, n_frames, n_fft, hop_length, center, target_len)
    if not valid:
        raise _FallbackError

    if normalized:
        input = input * (n_fft**0.5)
    spectrum = input.transpose(-2, -1).contiguous()
    if not onesided:
        spectrum = spectrum[..., : n_fft // 2 + 1]
    frames = torch.fft.irfft(spectrum, n=n_fft, dim=-1)
    frames = frames * window

    batch_shape = frames.shape[:-2]
    flat = frames.reshape(-1, n_frames, n_fft)
    signal = torch.ops.aten.unfold_backward(flat, [flat.shape[0], expected_len], 1, n_fft, hop_length)
    signal = signal.reshape(*batch_shape, expected_len)
    offset = n_fft // 2 if center else 0
    signal = signal[..., offset : offset + available] / envelope
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
    except torch.OutOfMemoryError:
        raise
    except (RuntimeError, NotImplementedError) as error:
        _warn_fallback("ISTFT", error)
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
    setattr(torch.Tensor, "stft", stft)
    setattr(torch.Tensor, "istft", istft)


def apply_a5_fft_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from vllm_omni.platforms.npu import is_a5

    if not is_a5():
        return
    _install()
    _PATCHED = True
