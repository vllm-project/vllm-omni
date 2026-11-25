# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pure-PyTorch FFT/STFT re-implementation for Ascend 310P.

310P (Atlas 300I Duo) ships no FFT-family aclnn kernels at all: ``torch.fft.*``,
``torch.stft``, ``torch.istft`` and even ``torch.hann_window`` fail at runtime
(the official torch_npu support matrix lists only Atlas A2/A3 and Ascend 950DT
for ``torch.fft``). 310P therefore cannot use the A5 approach from #9869
(rebuild ``stft`` from ``rfft``) — there is no ``rfft`` to rebuild from. This
module rebuilds the FFT itself from operators a 70-operator on-device probe
confirmed working on 310P3: matmul, gather, unfold, roll, stack/cat,
view_as_real/complex and elementwise math.

Design constraints discovered on device (310P3, CANN 9.2, torch_npu
2.10.post7):

- fp32 ``matmul`` shows ~350x worse accumulation error than CPU (suspected
  low-precision internal accumulation; ``bmm``, blocked-K and fp16-split
  compensation do not recover it). A direct DFT-matrix multiply lands at
  rel ~2.5e-4 — not enough for audio front-ends. Arbitrary-N DFTs therefore
  go through **Bluestein's algorithm**, which reduces them to a convolution
  evaluated by radix-2 butterflies (no large-K reduction), landing at
  rel ~3e-7.
- Complex kernels ``abs``/``conj``/``where``/``clamp``/scalar-mul are
  unsupported (``DT_COMPLEX64`` not in the op support list) and complex
  tensors cannot be transferred NPU<->CPU at all (``ERR01007``). Complex
  values are only materialized at function boundaries; all intermediate
  state keeps real/imaginary parts as separate fp32 tensors.

All functions are torch-API-compatible drop-ins (batched, same layout and
default arguments as ``torch.stft``/``torch.istft``/``torch.fft.*``).

Standalone repo with tests and bilingual README:
https://github.com/green-hand-shandong/npu-fft-310p
"""

from __future__ import annotations

import math
from typing import Optional

import torch

__all__ = [
    "fft_pow2", "fft_dft", "fft", "ifft", "rfft", "irfft",
    "stft", "istft", "hann_window", "hamming_window", "kaiser_window",
]


# ---------------------------------------------------------------- butterfly
def _bit_reverse_permute(n: int, device) -> torch.Tensor:
    """bit-reversal permutation (Cooley-Tukey step 0). CPU-generated."""
    bits = n.bit_length() - 1
    idx = torch.arange(n)
    rev = torch.zeros_like(idx)
    for b in range(bits):
        rev = (rev << 1) | ((idx >> b) & 1)
    return rev.to(device)


def _twiddles_pow2(n: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """radix-2 twiddle factors per stage: (cos, sin), each (log2N, N/2)."""
    half = n // 2
    stages = n.bit_length() - 1
    cos_s = torch.empty(stages, half, dtype=dtype)
    sin_s = torch.empty(stages, half, dtype=dtype)
    for s in range(stages):
        m = 1 << (s + 1)
        k = torch.arange(half, dtype=torch.float64)
        k = (k % (m // 2)) * (2 * math.pi / m)
        cos_s[s] = torch.cos(k).to(dtype)
        sin_s[s] = torch.sin(k).to(dtype)
    return cos_s.to(device), sin_s.to(device)


def _fft_pow2_re_im(re: torch.Tensor, im: torch.Tensor, inverse: bool):
    """radix-2 iterative FFT on separate real/imag tensors (..., N), N=2^k."""
    n = re.shape[-1]
    assert n & (n - 1) == 0 and n >= 2, f"fft_pow2 requires N=2^k, got {n}"
    device, dtype = re.device, re.dtype

    perm = _bit_reverse_permute(n, device)
    re = re.gather(-1, perm.expand_as(re))
    im = im.gather(-1, perm.expand_as(im))

    sign = 1.0 if inverse else -1.0
    cos_s, sin_s = _twiddles_pow2(n, device, dtype)
    log2n = n.bit_length() - 1

    for s in range(log2n):
        m = 1 << (s + 1)
        mh = m // 2
        re = re.view(*re.shape[:-1], n // m, m)
        im = im.view(*im.shape[:-1], n // m, m)
        wr = cos_s[s][:mh]
        wi = sin_s[s][:mh] * sign
        xr, xi = re[..., :mh], im[..., :mh]
        yr, yi = re[..., mh:], im[..., mh:]
        tr = yr * wr - yi * wi
        ti = yr * wi + yi * wr
        re = torch.cat([xr + tr, xr - tr], dim=-1)
        im = torch.cat([xi + ti, xi - ti], dim=-1)
        re = re.view(*re.shape[:-2], n)
        im = im.view(*im.shape[:-2], n)
    return re, im


def fft_pow2(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """FFT for N=2^k. x: (..., N) real or last-dim real/imag pair -> complex64."""
    if x.is_complex():
        pair = torch.view_as_real(x)
        re, im = pair[..., 0], pair[..., 1]
    else:
        re, im = x, torch.zeros_like(x)
    n = re.shape[-1]
    re, im = _fft_pow2_re_im(re.contiguous(), im.contiguous(), inverse)
    if inverse:
        re = re / n
        im = im / n
    return torch.view_as_complex(torch.stack([re, im], dim=-1))


def fft_dft(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Arbitrary-N DFT via Bluestein's algorithm. x: (..., N) -> complex64.

    Reduces an N-point DFT to a linear convolution evaluated by radix-2
    butterflies, avoiding the large-K matmul reduction whose accumulation
    accuracy is degraded on 310P.

    Derivation: with nk = (k^2 + n^2 - (k-n)^2)/2,
        X[k] = w^{k^2/2} * conv(a, b)[k],  j = k-n in [-(N-1), N-1]
        a[n] = x[n] * e^{-i*pi*n^2/N},     b[j] = e^{+i*pi*j^2/N}
    (chirps generated in fp64 on CPU, stored fp32; kernel j is signed —
    taking j mod N here is incorrect).
    """
    if x.is_complex():
        pair = torch.view_as_real(x)
        re, im = pair[..., 0], pair[..., 1]
    else:
        re, im = x, torch.zeros_like(x)
    n = re.shape[-1]
    device, dtype = re.device, re.dtype
    sign = 1.0 if inverse else -1.0

    m = 1
    while m < 2 * n - 1:
        m *= 2
    k = torch.arange(n, dtype=torch.float64)
    w_half_re = torch.cos(sign * math.pi * k * k / n)
    w_half_im = torch.sin(sign * math.pi * k * k / n)
    j = torch.arange(-(n - 1), n, dtype=torch.float64)
    b_re = torch.cos(-sign * math.pi * j * j / n)
    b_im = torch.sin(-sign * math.pi * j * j / n)
    w_half_re = w_half_re.to(device).to(dtype)
    w_half_im = w_half_im.to(device).to(dtype)
    b_re = b_re.to(device).to(dtype)
    b_im = b_im.to(device).to(dtype)

    ar = re * w_half_re - im * w_half_im
    ai = re * w_half_im + im * w_half_re
    ar = torch.nn.functional.pad(ar, (0, m - n))
    ai = torch.nn.functional.pad(ai, (0, m - n))
    a_f = fft_pow2(torch.view_as_complex(torch.stack([ar, ai], dim=-1)), inverse=False)
    br = torch.nn.functional.pad(b_re, (0, m - (2 * n - 1)))
    bi = torch.nn.functional.pad(b_im, (0, m - (2 * n - 1)))
    b_f = fft_pow2(torch.view_as_complex(torch.stack([br, bi], dim=-1)), inverse=False)

    pa = torch.view_as_real(a_f)
    pb = torch.view_as_real(b_f)
    prod = torch.view_as_complex(torch.stack(
        [pa[..., 0] * pb[..., 0] - pa[..., 1] * pb[..., 1],
         pa[..., 0] * pb[..., 1] + pa[..., 1] * pb[..., 0]], dim=-1))
    conv = fft_pow2(prod, inverse=True)[..., n - 1: 2 * n - 1]

    pc = torch.view_as_real(conv)
    out_re = pc[..., 0] * w_half_re - pc[..., 1] * w_half_im
    out_im = pc[..., 0] * w_half_im + pc[..., 1] * w_half_re
    if inverse:
        out_re = out_re / n
        out_im = out_im / n
    return torch.view_as_complex(torch.stack([out_re, out_im], dim=-1))


# ---------------------------------------------------------------- top level
def fft(input: torch.Tensor, n: Optional[int] = None, dim: int = -1,
        norm: Optional[str] = None) -> torch.Tensor:
    """torch.fft.fft drop-in (last dim only; 'ortho' norm supported)."""
    assert dim == -1 or dim == input.ndim - 1, "only the last dim is supported"
    x = input
    if n is not None and n != x.shape[-1]:
        pad = n - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, max(pad, 0)))[:n if pad < 0 else None]
    N = x.shape[-1]
    impl = "pow2" if N & (N - 1) == 0 else "dft"
    out = fft_pow2(x, inverse=False) if impl == "pow2" else fft_dft(x, inverse=False)
    if norm == "ortho":
        out = out / math.sqrt(N)
    return out


def ifft(input: torch.Tensor, n: Optional[int] = None, dim: int = -1,
         norm: Optional[str] = None) -> torch.Tensor:
    """torch.fft.ifft drop-in (last dim only; 'ortho' norm supported)."""
    assert dim == -1 or dim == input.ndim - 1, "only the last dim is supported"
    x = input
    if n is not None and n != x.shape[-1]:
        pad = n - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, max(pad, 0)))[:n if pad < 0 else None]
    N = x.shape[-1]
    impl = "pow2" if N & (N - 1) == 0 else "dft"
    out = fft_pow2(x, inverse=True) if impl == "pow2" else fft_dft(x, inverse=True)
    if norm == "ortho":
        out = out * math.sqrt(N)
    return out


def rfft(input: torch.Tensor, n: Optional[int] = None, dim: int = -1,
         norm: Optional[str] = None) -> torch.Tensor:
    """torch.fft.rfft drop-in: real -> one-sided complex64 spectrum."""
    assert dim == -1 or dim == input.ndim - 1
    x = input
    if not x.is_complex():
        x = x.float()
    full = fft(x, n=n, dim=dim, norm=norm)
    return full[..., : full.shape[-1] // 2 + 1]


def irfft(input: torch.Tensor, n: Optional[int] = None, dim: int = -1,
          norm: Optional[str] = None) -> torch.Tensor:
    """torch.fft.irfft drop-in: one-sided spectrum -> real signal."""
    assert dim == -1 or dim == input.ndim - 1
    x = input
    if n is None:
        n = 2 * (x.shape[-1] - 1)
    # conjugate-symmetric reconstruction; 310P lacks complex conj, so the
    # mirrored half is assembled from real/imag parts directly
    if x.is_complex():
        pair = torch.view_as_real(x.resolve_conj())
        re, im = pair[..., 0], pair[..., 1]
        mid_re, mid_im = re[..., 1:-1], im[..., 1:-1]
        full = torch.view_as_complex(torch.stack(
            [torch.cat([re, mid_re.flip(-1)], dim=-1),
             torch.cat([im, (-mid_im).flip(-1)], dim=-1)], dim=-1))
    else:
        full = torch.cat([x, x[..., 1:-1].flip(-1).conj()], dim=-1)
    out = ifft(full, n=n, dim=dim, norm=norm)
    return out.real if torch.is_complex(out) else out


# ---------------------------------------------------------------- windows
def hann_window(window_length: int, periodic: bool = True, dtype=None,
                device=None, requires_grad: bool = False) -> torch.Tensor:
    """torch.hann_window drop-in: generated on CPU (310P cos_ kernel missing)."""
    w = torch.hann_window(window_length, periodic=periodic,
                          dtype=dtype or torch.float32, device="cpu")
    return w.to(device) if device is not None else w


def hamming_window(window_length: int, periodic: bool = True, dtype=None,
                   device=None, requires_grad: bool = False) -> torch.Tensor:
    w = torch.hamming_window(window_length, periodic=periodic,
                             dtype=dtype or torch.float32, device="cpu")
    return w.to(device) if device is not None else w


def kaiser_window(window_length: int, beta: float, periodic: bool = True,
                  dtype=None, device=None, requires_grad: bool = False) -> torch.Tensor:
    w = torch.kaiser_window(window_length, periodic=periodic, beta=beta,
                            dtype=dtype or torch.float32, device="cpu")
    return w.to(device) if device is not None else w


# ---------------------------------------------------------------- STFT/ISTFT
def stft(input: torch.Tensor, n_fft: int, hop_length: Optional[int] = None,
         win_length: Optional[int] = None, window: Optional[torch.Tensor] = None,
         center: bool = True, pad_mode: str = "reflect",
         normalized: bool = False, onesided: bool = True,
         return_complex: bool = True) -> torch.Tensor:
    """torch.stft drop-in (common signatures; onesided complex output)."""
    assert onesided and return_complex, "only onesided complex output is supported"
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    assert win_length == n_fft, "win_length must equal n_fft"
    if window is None:
        window = hann_window(n_fft, device=input.device)
    window = window.to(input.device, torch.float32)

    x = input
    if center:
        x = torch.nn.functional.pad(x, (n_fft // 2, n_fft // 2), mode=pad_mode)

    frames = x.unfold(-1, n_fft, hop_length) * window
    spec = fft_pow2(frames, inverse=False)
    if normalized:
        spec = spec / math.sqrt(n_fft)
    spec = spec.transpose(-1, -2)
    return spec[..., : n_fft // 2 + 1, :]


def istft(input: torch.Tensor, n_fft: int, hop_length: Optional[int] = None,
          win_length: Optional[int] = None, window: Optional[torch.Tensor] = None,
          center: bool = True, normalized: bool = False,
          length: Optional[int] = None, return_complex: bool = False) -> torch.Tensor:
    """torch.istft drop-in: one-sided spectrum -> waveform (overlap-add)."""
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = hann_window(n_fft, device=input.device)
    # NB: never cast the window to input.dtype — input is complex and a
    # complex window would poison the real-domain overlap-add below.
    window = window.to(input.device, torch.float32)

    spec = input
    if spec.is_complex():
        pair = torch.view_as_real(spec.resolve_conj())
        re, im = pair[..., 0], pair[..., 1]
        mid_re, mid_im = re[..., 1:-1, :], im[..., 1:-1, :]
        full = torch.view_as_complex(torch.stack(
            [torch.cat([re, mid_re.flip(-2)], dim=-2),
             torch.cat([im, (-mid_im).flip(-2)], dim=-2)], dim=-1))
    else:
        full = torch.cat([spec, spec[..., 1:-1, :].flip(-2).conj()], dim=-2)
    full = full.transpose(-1, -2)
    if normalized:
        fp = torch.view_as_real(full)
        full = torch.view_as_complex(fp * math.sqrt(n_fft))

    n = full.shape[-1]
    impl = "pow2" if n & (n - 1) == 0 else "dft"
    frames_c = fft_pow2(full, inverse=True) if impl == "pow2" else fft_dft(full, inverse=True)
    frames = frames_c.real * window

    T = frames.shape[-2]
    out_len = hop_length * (T - 1) + n_fft
    out = frames.new_zeros(*frames.shape[:-2], out_len)
    denom = frames.new_zeros(out_len)
    win_sq = (window * window).float()
    for t in range(T):
        out[..., t * hop_length: t * hop_length + n_fft] += frames[..., t, :]
        denom[..., t * hop_length: t * hop_length + n_fft] += win_sq
    denom = torch.where(denom > 1e-11, denom, torch.full_like(denom, 1e-11))
    out = out / denom

    if center:
        out = out[..., n_fft // 2: out.shape[-1] - n_fft // 2]
    if length is not None:
        out = out[..., :length]
    return out
