# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pack independent Audio VAE FIR channels into dense diagonal convolutions.

The MiniMax-H3 Audio VAE decoder runs a 12-tap FIR per channel through
``UpSample1d`` and ``LowPassFilter1d``. A depthwise convolution pays one
NC1HWC0 C0=16 tile per C1 item even though one channel uses a single input
channel and 12 taps, so most of the cube tile is idle. Packing ``w``
independent ``(batch, channel)`` FIRs into one dense convolution with
diagonal weights keeps every channel independent while filling the tile.

``install(model)`` registers the packed weights and wraps the checkpoint's
``UpSample1d`` / ``LowPassFilter1d`` modules. The upsampler uses a
polyphase decomposition because a dense ``conv_transpose1d`` regresses on
long sequences. The wrapper falls back to the original forward on CPU,
during training, or when the packed width is 1.
"""

import types
from collections.abc import Callable
from functools import partial

import torch
from torch.nn import functional as F

_DEFAULT_WIDTH = 16


def _pack_width(count: int) -> int:
    """Largest 16/8/4/2/1 divisor of ``count``."""
    for width in (16, 8, 4, 2, 1):
        if count % width == 0:
            return width
    return 1


def _diagonal_filter(filter_: torch.Tensor, width: int) -> torch.Tensor:
    """Build ``[width, width, taps]`` weights with the taps on the diagonal."""
    return torch.eye(width, device=filter_.device, dtype=filter_.dtype)[:, :, None] * filter_.reshape(1, 1, -1)


def _phase_filter(filter_: torch.Tensor, width: int) -> torch.Tensor:
    """Build ``[2*width, width, 6]`` even/odd polyphase weights."""
    taps = filter_.reshape(-1)
    if taps.numel() != 12:
        raise ValueError(f"expected 12 FIR taps, got {taps.numel()}")
    result = filter_.new_zeros((width * 2, width, 6))
    for channel in range(width):
        result[2 * channel, channel] = taps[0::2].flip(-1)
        result[2 * channel + 1, channel] = taps[1::2].flip(-1)
    return result


def _up_phase(module, x: torch.Tensor, width: int | None = None) -> torch.Tensor:
    """Apply the packed even and odd filters and restore time order."""
    batch, channels, _ = x.shape
    width = _pack_width(batch * channels) if width is None else width
    weight = module._h3_phase_fir[: width * 2, :width, :].contiguous()
    padded = F.pad(x, (module.pad, module.pad), mode="replicate")
    phases = F.conv1d(padded.reshape(-1, width, padded.shape[-1]), weight, padding=5)
    samples = phases.reshape(-1, width, 2, phases.shape[-1]).transpose(-1, -2)
    samples = samples.reshape(batch, channels, -1) * module.ratio
    return samples[..., module.pad_left : -module.pad_right]


def _low(module, x: torch.Tensor, width: int | None = None) -> torch.Tensor:
    """Apply the packed low-pass convolution with native padding and stride."""
    batch, channels, _ = x.shape
    width = _pack_width(batch * channels) if width is None else width
    weight = module._h3_dense_fir[:width, :width, :].contiguous()
    if module.padding:
        x = F.pad(x, (module.pad_left, module.pad_right), mode=module.padding_mode)
    samples = F.conv1d(x.reshape(-1, width, x.shape[-1]), weight, stride=module.stride)
    return samples.reshape(batch, channels, -1)


def _fir_forward(
    module: torch.nn.Module,
    x: torch.Tensor,
    *,
    packed_forward: Callable[[torch.nn.Module, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """Dispatch to packed FP32 inference, counting executions and NPU fallbacks separately."""
    if x.device.type != "npu":
        return module._h3_fir_original(x)
    if x.ndim == 3 and x.dtype == module.filter.dtype == torch.float32 and x.numel() and not module.training:
        width = _pack_width(x.shape[0] * x.shape[1])
        if width > 1:
            output = packed_forward(module, x, width)
            module._h3_fir_c16_calls = getattr(module, "_h3_fir_c16_calls", 0) + 1
            return output
    module._h3_fir_fallback_calls = getattr(module, "_h3_fir_fallback_calls", 0) + 1
    return module._h3_fir_original(x)


def install(model: torch.nn.Module) -> int:
    """Install the packed FIR path on ``model.decoder`` and return the count."""
    installed = 0
    for module in model.decoder.modules():
        name = type(module).__name__
        if name not in ("UpSample1d", "LowPassFilter1d") or hasattr(module, "_h3_dense_fir"):
            continue
        filter_ = getattr(module, "filter", None)
        if not isinstance(filter_, torch.Tensor) or filter_.shape != (1, 1, 12):
            continue
        if name == "UpSample1d" and (module.stride != 2 or module.ratio != 2 or module.pad_right <= 0):
            continue
        module.register_buffer("_h3_dense_fir", _diagonal_filter(filter_, _DEFAULT_WIDTH), persistent=False)
        if name == "UpSample1d":
            module.register_buffer("_h3_phase_fir", _phase_filter(filter_, _DEFAULT_WIDTH), persistent=False)
        module._h3_fir_original = module.forward
        packed_forward = _up_phase if name == "UpSample1d" else _low
        module.forward = types.MethodType(partial(_fir_forward, packed_forward=packed_forward), module)
        installed += 1
    return installed


__all__ = ["install"]
