# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: N803
"""Bit-exact fused RMSNorm epilogue for the diffusers Wan VAE decoder.

``WanRMS_norm.forward`` evaluates, for fp16/bf16 activations::

    normalized = F.normalize(x.float(), dim=1).to(x.dtype)
    out = normalized * scale * gamma + 0.0

That is an fp32 copy of the activation, an fp32 reduction, an fp32 divide, a
narrowing cast and three low-precision elementwise kernels: six or seven full
passes over an activation that reaches 472 MB in the Cosmos3 720p decoder.

The fast path keeps the reduction in ATen (``torch.linalg.vector_norm`` with
``dtype=torch.float32`` reads the low-precision tensor once and runs the same
reduction kernel ``F.normalize(x.float())`` runs, so the fp32 denominators are
identical) and folds everything after it into one Triton kernel that reads
``x`` once and writes the result once. The kernel is bit-exact with ATen
rather than merely close:

* ``tl.math.div_rn`` is IEEE round-to-nearest fp32 division, the operation
  ATen's ``div`` performs in fp32 opmath. Triton's ``/`` lowers to an
  approximate reciprocal-multiply and is not usable here.
* Every value ATen materializes in ``x.dtype`` is rounded here at the same
  point with ``.to(dtype).to(tl.float32)``, including the final ``+ 0.0``
  (which turns ``-0.0`` into ``+0.0`` exactly like upstream's ``+ self.bias``).
* ``enable_fp_fusion=False`` stops LLVM from contracting a multiply and its
  neighbour into an FMA, which would skip one of those roundings.

The optional SiLU epilogue reproduces ATen's ``x / (1 + expf(-x))`` with the
libdevice ``expf`` and an IEEE divide. Whether that is bit-identical depends
on the toolkit, so :func:`silu_epilogue_is_exact` verifies it exhaustively
over all 65,536 bf16/fp16 values before callers enable it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# These kernels are HBM bound; a wide block gives every thread more independent
# loads to overlap. 2048/4 measured ~6% faster than 1024/4 on the decoder's
# real shapes and is bit-identical (the kernel is elementwise).
_BLOCK = 2048
_NUM_WARPS = 4
_MAX_GRID_Y = 65535

if HAS_TRITON:

    @triton.jit
    def _rms_norm_scale_kernel(
        X,
        DENOM,
        GAMMA,
        OUT,
        spatial,
        channels,
        scale,
        SILU: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``out = ((x / denom) * scale * gamma + 0.0)`` [then SiLU] with ATen's rounding points.

        ``X``/``OUT`` are dense ``(rows, spatial)`` with ``rows = batch * channels``;
        ``DENOM`` is fp32 ``(batch, spatial)``; ``GAMMA`` is ``(channels,)``.
        """
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < spatial

        channel = row % channels
        batch = row // channels
        base = row.to(tl.int64) * spatial + col
        denom_base = batch.to(tl.int64) * spatial + col

        dtype = X.dtype.element_ty
        x = tl.load(X + base, mask=mask, other=0).to(tl.float32)
        d = tl.load(DENOM + denom_base, mask=mask, other=1.0)
        g = tl.load(GAMMA + channel).to(tl.float32)

        v = tl.math.div_rn(x, d).to(dtype).to(tl.float32)
        v = (v * scale).to(dtype).to(tl.float32)
        v = (v * g).to(dtype).to(tl.float32)
        v = (v + 0.0).to(dtype)
        if SILU:
            v32 = v.to(tl.float32)
            v = tl.math.div_rn(v32, 1.0 + tldevice.exp(-v32)).to(dtype)
        tl.store(OUT + base, v, mask=mask)


def rms_norm_scale(
    x: torch.Tensor,
    denom: torch.Tensor,
    gamma: torch.Tensor,
    scale: float,
    *,
    silu: bool = False,
) -> torch.Tensor | None:
    """``(x / denom) * scale * gamma + 0.0`` (optionally followed by SiLU), or ``None`` to decline.

    ``x`` must be a contiguous ``(N, C, S)`` tensor, ``denom`` a contiguous fp32
    ``(N, 1, S)`` tensor and ``gamma`` a contiguous ``(C,)`` tensor of ``x``'s dtype.
    """
    if not HAS_TRITON or not x.is_cuda:
        return None
    if x.dim() != 3 or x.dtype not in _SUPPORTED_DTYPES:
        return None
    if denom.dtype is not torch.float32 or gamma.dtype is not x.dtype:
        return None
    # Raw pointers go to one kernel, which does no cross-device checking.
    if denom.device != x.device or gamma.device != x.device:
        return None
    if not (x.is_contiguous() and denom.is_contiguous() and gamma.is_contiguous()):
        return None
    batch, channels, spatial = x.shape
    if denom.shape != (batch, 1, spatial) or gamma.shape != (channels,):
        return None
    if spatial == 0 or batch * channels == 0:
        return None
    spatial_blocks = triton.cdiv(spatial, _BLOCK)
    if spatial_blocks > _MAX_GRID_Y:
        return None

    out = torch.empty_like(x)
    grid = (batch * channels, spatial_blocks)
    with torch.get_device_module().device(x.device):
        _rms_norm_scale_kernel[grid](
            x,
            denom,
            gamma,
            out,
            spatial,
            channels,
            float(scale),
            SILU=silu,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
            # FMA contraction would swallow one of the low-precision roundings.
            enable_fp_fusion=False,
        )
    return out


def _bitwise_equal_allow_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    int_dtype = torch.int16 if a.element_size() == 2 else torch.int32
    same_bits = a.view(int_dtype) == b.view(int_dtype)
    both_nan = torch.isnan(a) & torch.isnan(b)
    return bool((same_bits | both_nan).all().item())


_SILU_EXACT_CACHE: dict[tuple[str, torch.dtype], bool] = {}


def silu_epilogue_is_exact(device: torch.device, dtype: torch.dtype) -> bool:
    """Exhaustively verify the fused SiLU epilogue against ``F.silu`` for one dtype.

    bf16 and fp16 have 65,536 values each, so the check is complete and costs
    about a millisecond. The result is cached per device. fp32 cannot be
    checked exhaustively and always returns ``False``.
    """
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    key = (str(device), dtype)
    cached = _SILU_EXACT_CACHE.get(key)
    if cached is not None:
        return cached
    result = False
    if HAS_TRITON and device.type == "cuda":
        try:
            bits = torch.arange(0, 65536, dtype=torch.int32, device=device).to(torch.int16)
            x = bits.view(dtype).reshape(1, 1, -1)
            denom = torch.ones((1, 1, x.shape[-1]), dtype=torch.float32, device=device)
            gamma = torch.ones((1,), dtype=dtype, device=device)
            out = rms_norm_scale(x, denom, gamma, 1.0, silu=True)
            if out is not None:
                # The reference epilogue also ends with ``+ 0.0`` before SiLU.
                reference = F.silu(x + 0.0)
                result = _bitwise_equal_allow_nan(out, reference)
        except Exception:  # pragma: no cover - depends on the installed toolkit
            result = False
    _SILU_EXACT_CACHE[key] = result
    return result


__all__ = ["rms_norm_scale", "silu_epilogue_is_exact"]
