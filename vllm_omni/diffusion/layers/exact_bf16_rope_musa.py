# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Interleaved RoPE Triton kernel for Qwen-Image-2.1 on MUSA.

It follows the CUDA kernel in ``exact_bf16_pointwise`` with two differences: the rotation rounds
every product, as MUSA's native real-arithmetic rotation does, and the batch is a grid axis with the
per-batch size passed at runtime, so a new sequence length reuses the compiled kernel and the index
math needs no runtime division. The kernel runs through an opaque custom op registered on MUSA only,
which keeps the launch, and its enable_fp_fusion=False, out of regionally compiled graphs.
"""

import torch
from torch.library import Library
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.platforms import current_omni_platform

# The op is registered on MUSA only; the helper below declines wherever it is not.
_HAS_MUSA_OPS = HAS_TRITON and current_omni_platform.is_musa()

if HAS_TRITON:

    @triton.jit
    def _rope_kernel(
        input_ptr,
        freq_ptr,
        output_ptr,
        pairs_per_batch,
        heads: tl.constexpr,
        x_stride0,
        x_stride1: tl.constexpr,
        x_stride2: tl.constexpr,
        freq_stride0: tl.constexpr,
        freq_stride1: tl.constexpr,
        block_size: tl.constexpr,
    ):
        # Grid: (blocks of one batch, batch).
        index = tl.program_id(0) * block_size + tl.arange(0, block_size)
        batch = tl.program_id(1)
        mask = index < pairs_per_batch
        pair = index % 64
        head = (index // 64) % heads
        token = index // (64 * heads)
        base = batch * x_stride0 + token * x_stride1 + head * x_stride2 + pair * 2
        real = tl.load(input_ptr + base, mask, 0).to(tl.float32)
        imag = tl.load(input_ptr + base + 1, mask, 0).to(tl.float32)
        cosine = tl.load(freq_ptr + token * freq_stride0 + pair * freq_stride1, mask, 0)
        sine = tl.load(freq_ptr + token * freq_stride0 + pair * freq_stride1 + 1, mask, 0)
        # The real-arithmetic rotation rounds every product first.
        rotated_real = real * cosine - imag * sine
        rotated_imag = imag * cosine + real * sine
        output = (batch * pairs_per_batch + index) * 2
        tl.store(output_ptr + output, rotated_real.to(tl.bfloat16), mask)
        tl.store(output_ptr + output + 1, rotated_imag.to(tl.bfloat16), mask)


def _interleaved_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    real_freqs = torch.view_as_real(freqs)
    _rope_kernel[(triton.cdiv(x[0].numel() // 2, 256), x.shape[0])](
        x,
        real_freqs,
        output,
        x[0].numel() // 2,
        x.shape[2],
        *x.stride()[:3],
        *real_freqs.stride()[:2],
        256,
        enable_fp_fusion=False,
    )
    return output


def _interleaved_rope_fake(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


if _HAS_MUSA_OPS and not hasattr(torch.ops.vllm_omni, "musa_interleaved_rope"):
    _ROPE_OP_LIB = Library("vllm_omni", "FRAGMENT")
    direct_register_custom_op(
        op_name="musa_interleaved_rope",
        op_func=_interleaved_rope,
        fake_impl=_interleaved_rope_fake,
        mutates_args=[],
        target_lib=_ROPE_OP_LIB,
    )


def musa_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor | None:
    """Interleaved RoPE on MUSA with the rounding of the real-arithmetic rotation, or ``None``.

    It also runs inside compiled regions, through the opaque op.
    """
    if (
        not _HAS_MUSA_OPS
        or x.device.type != "musa"
        or x.dtype != torch.bfloat16
        or x.ndim != 4
        or x.shape[-1] != 128
        or x.stride(-1) != 1
        or freqs.dtype != torch.complex64
        or freqs.device != x.device
        or freqs.shape != (x.shape[1], 64)
        or torch.is_grad_enabled()
        or x.numel() == 0
    ):
        return None
    return torch.ops.vllm_omni.musa_interleaved_rope(x, freqs)
