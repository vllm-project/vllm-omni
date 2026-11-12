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

Environment switches:
- ``H3_VAE_OPERATOR_FIR_C16_INSTALL=1``: install at construction time.
- ``H3_VAE_OPERATOR_FIR_C16=1``: execute the packed path per request.
"""

import os
import types

import torch
from torch.nn import functional as F

_INSTALL_ENV = "H3_VAE_OPERATOR_FIR_C16_INSTALL"
_EXECUTE_ENV = "H3_VAE_OPERATOR_FIR_C16"
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
        # ``flip(-1)`` matches conv1d's cross-correlation convention.
        result[2 * channel, channel] = taps[0::2].flip(-1)
        result[2 * channel + 1, channel] = taps[1::2].flip(-1)
    return result


def _up_phase(module, x: torch.Tensor) -> torch.Tensor:
    batch, channels, _ = x.shape
    width = _pack_width(batch * channels)
    if width == 1:
        return module._h3_fir_original(x)
    if module.stride != 2 or module.filter.numel() != 12:
        return module._h3_fir_original(x)
    weight = module._h3_phase_fir[: width * 2, :width, :].contiguous()
    padded = F.pad(x, (module.pad, module.pad), mode="replicate")
    value = F.conv1d(padded.reshape(-1, width, padded.shape[-1]), weight, padding=5)
    value = value.reshape(-1, width, 2, value.shape[-1]).transpose(-1, -2).contiguous()
    value = value.reshape(batch, channels, -1) * module.ratio
    return value[..., module.pad_left : -module.pad_right]


def _low(module, x: torch.Tensor) -> torch.Tensor:
    batch, channels, _ = x.shape
    width = _pack_width(batch * channels)
    if width == 1:
        return module._h3_fir_original(x)
    weight = module._h3_dense_fir[:width, :width, :].contiguous()
    if module.padding:
        x = F.pad(x, (module.pad_left, module.pad_right), mode=module.padding_mode)
    value = F.conv1d(x.reshape(-1, width, x.shape[-1]), weight, stride=module.stride)
    return value.reshape(batch, channels, -1)


def install(model: torch.nn.Module) -> int:
    """Install the packed FIR path on ``model.decoder`` and return the count."""
    installed = 0
    for module in model.decoder.modules():
        name = type(module).__name__
        if name not in ("UpSample1d", "LowPassFilter1d") or hasattr(module, "_h3_dense_fir"):
            continue
        filter_ = getattr(module, "filter", None)
        if not isinstance(filter_, torch.Tensor) or filter_.numel() != 12:
            continue
        if name == "UpSample1d" and (module.stride != 2 or module.ratio != 2):
            continue
        module.register_buffer(
            "_h3_dense_fir",
            _diagonal_filter(module.filter, _DEFAULT_WIDTH),
            persistent=False,
        )
        if name == "UpSample1d":
            module.register_buffer(
                "_h3_phase_fir",
                _phase_filter(module.filter, _DEFAULT_WIDTH),
                persistent=False,
            )
        module._h3_fir_original = module.forward
        fn = _up_phase if name == "UpSample1d" else _low

        def forward(self, x, _fn=fn):
            if os.environ.get(_EXECUTE_ENV, "0") != "1" or x.device.type != "npu" or self.training:
                return self._h3_fir_original(x)
            self._h3_fir_c16_calls = getattr(self, "_h3_fir_c16_calls", 0) + 1
            return _fn(self, x)

        module.forward = types.MethodType(forward, module)
        installed += 1
    return installed


__all__ = ["install"]
