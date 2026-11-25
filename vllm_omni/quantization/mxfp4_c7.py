# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""C7 encoding for native E2M1/E8M0 kernels, without a floating GEMM fallback."""

import torch


def c7_elements_and_exponents(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve the training recipe's input-dtype arithmetic and K groups."""
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = torch.ceil(torch.log2(maximum.clamp(min=1.17e-38) * (1.0 / 7.0))).clamp(-127, 127)
    normalized = blocks * torch.exp2(-exponent)
    private = torch.floor(torch.log2(normalized.abs().clamp(min=1.17e-38))).clamp(min=0.0)
    scaled = normalized * torch.exp2(-private) * 2.0
    rounded = torch.sign(scaled) * torch.floor(scaled.abs() + 0.5)
    elements = (rounded * 0.5 * torch.exp2(private)).clamp(-6.0, 6.0)
    return elements.reshape(x.shape), exponent.squeeze(-1)


def pack_mxfp4_c7(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return low-nibble-first E2M1 bytes and biased E8M0 exponent bytes.

    The caller owns the NPU kernel's scale relayout. This encoder also runs on
    CPU so packed bytes can be checked independently of the device kernel.
    Native GEMM needs K divisible by 64, even though the C7 group size is 32.
    Inputs must be finite; NaN/Inf have no corresponding finite E2M1 encoding.
    """
    if x.ndim < 1 or x.shape[-1] == 0 or x.shape[-1] % 64:
        raise ValueError("native C7 requires a nonempty K divisible by 64, including TP shards")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("native C7 requires BF16 or FP32 input")
    width = x.shape[-1]
    flat = x.reshape(-1, width)
    packed = torch.empty((flat.shape[0], width // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((flat.shape[0], width // 32), dtype=torch.uint8, device=x.device)
    chunk_rows = 1024 if x.device.type == "npu" else 64
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        values = flat[start:stop]
        elements, exponent = c7_elements_and_exponents(values)
        # E2M1 magnitude codes 0..7 encode 0,.5,1,1.5,2,3,4,6.
        magnitude = elements.abs()
        codes = torch.zeros_like(elements, dtype=torch.uint8)
        for code, level in enumerate((0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), 1):
            codes = torch.where(magnitude >= level, code, codes)
        codes = codes | ((elements < 0).to(torch.uint8) << 3)
        packed[start:stop] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales[start:stop] = (exponent.to(torch.int16) + 127).to(torch.uint8)
    return packed.reshape(*x.shape[:-1], width // 2).view(torch.int8), scales.reshape(*x.shape[:-1], width // 32)
