# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bit-exact RMSNorm fast path for SANA's low-precision affine contract.

Diffusers' SANA RMSNorm rounds the FP32-normalized activation to the
parameter dtype *before* multiplying by the learned weight.  Generic fused
RMSNorm kernels usually multiply the weight in FP32 and round only once, so
they cannot replace that expression without changing model output.

This CUDA fast path leaves aten's reduction and ``rsqrt`` untouched.  It
only fuses the two bandwidth-heavy pointwise regions around them:

* BF16-to-FP32 conversion plus squaring before ``aten::mean`` or ``aten::sum``;
* normalization, the explicit BF16 rounding point, and weight multiplication.

The mean variant serves SanaRMSNorm.  The sum/count variant serves the video
Q/K SanaDistributedRMSNorm at TP=1, retaining its FP32 sum and division rather
than substituting a mean reduction.  TP>1 stays on the model's original path.

The first eligible input signature is compared bit-for-bit with the eager
expression.  A mismatch or non-OOM launch failure permanently disables that
signature, while unsupported platforms, layouts, dtypes, small tensors,
autograd, and compiled regions keep the original PyTorch expression.
Out-of-memory errors propagate without changing the signature state.
"""

from __future__ import annotations

import torch
from torch.library import Library
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_BLOCK_SIZE = 1024
# The [2, 300, 2240] text norms are launch-bound and slightly faster in eager
# mode. Only video-token tensors meeting this threshold are eligible.
_MIN_ELEMENTS = 2_000_000

_Signature = tuple[torch.device, torch.dtype, int, int]
_VERIFIED_SIGNATURES: set[_Signature] = set()
_DISABLED_SIGNATURES: set[_Signature] = set()
# The mean and sum/count reductions have different numerical contracts and
# must never share first-sight verification or failure state.
_VERIFIED_SUM_SIGNATURES: set[_Signature] = set()
_DISABLED_SUM_SIGNATURES: set[_Signature] = set()


if HAS_TRITON:

    @triton.jit
    def _round_bf16_to_fp32(value):
        """RNE-round FP32 to BF16 while retaining an FP32 register."""
        return tl.inline_asm_elementwise(
            asm="{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }",
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn_f32(left, right):
        """Prevent contraction or reassociation across eager boundaries."""
        return tl.inline_asm_elementwise(
            asm="mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit(do_not_specialize=["numel"])
    def _square_bf16_to_fp32_kernel(
        output_ptr,
        input_ptr,
        numel,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + offsets, _mul_rn_f32(values, values), mask=mask)

    @triton.jit(do_not_specialize=["numel"])
    def _rms_norm_affine_tail_kernel(
        output_ptr,
        input_ptr,
        inverse_rms_ptr,
        weight_ptr,
        numel,
        hidden_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        columns = offsets % hidden_size
        rows = offsets // hidden_size

        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        inverse_rms = tl.load(inverse_rms_ptr + rows, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)

        # Match Diffusers exactly: the normalized value materializes as BF16
        # before the BF16 weight multiplication.  The final BF16 store provides
        # the second rounding boundary.
        normalized = _round_bf16_to_fp32(_mul_rn_f32(values, inverse_rms))
        weighted = _mul_rn_f32(normalized, weight)
        tl.store(output_ptr + offsets, weighted, mask=mask)


def _eager_sana_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        hidden_states = hidden_states.to(weight.dtype)
    return hidden_states * weight


def _eager_sana_rms_norm_sum(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Match SanaDistributedRMSNorm at TP=1, including x.float() for the
    # normalization multiply and a separate FP32 division after aten::sum.
    x_float = hidden_states.float()
    sum_sq = x_float.pow(2).sum(dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(sum_sq / hidden_states.shape[-1] + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        normalized = normalized.to(weight.dtype)
    return normalized * weight


def _kernel_inputs_supported(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and hidden_states.is_cuda
        and hidden_states.dtype is torch.bfloat16
        and hidden_states.ndim >= 2
        and hidden_states.numel() > 0
        and hidden_states.shape[-1] > 0
        and hidden_states.is_contiguous()
        and weight.is_cuda
        and weight.device == hidden_states.device
        and weight.dtype == hidden_states.dtype
        and weight.ndim == 1
        and weight.shape[0] == hidden_states.shape[-1]
        and weight.is_contiguous()
    )


def _can_use_exact_sana_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    if not _kernel_inputs_supported(hidden_states, weight):
        return False
    return hidden_states.numel() >= _MIN_ELEMENTS and not torch.is_grad_enabled()


def _square_bf16_to_fp32(hidden_states: torch.Tensor) -> torch.Tensor:
    output = torch.empty(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
    if hidden_states.numel() == 0:
        return output
    _square_bf16_to_fp32_kernel[(triton.cdiv(hidden_states.numel(), _BLOCK_SIZE),)](
        output,
        hidden_states,
        hidden_states.numel(),
        block_size=_BLOCK_SIZE,
        num_warps=4,
    )
    return output


def _rms_norm_affine_tail(
    hidden_states: torch.Tensor,
    inverse_rms: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(hidden_states)
    if hidden_states.numel() == 0:
        return output
    _rms_norm_affine_tail_kernel[(triton.cdiv(hidden_states.numel(), _BLOCK_SIZE),)](
        output,
        hidden_states,
        inverse_rms,
        weight,
        hidden_states.numel(),
        hidden_size=hidden_states.shape[-1],
        block_size=_BLOCK_SIZE,
        num_warps=4,
    )
    return output


def _launch_exact_sana_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    squares = _square_bf16_to_fp32(hidden_states)
    variance = squares.mean(-1, keepdim=True)
    del squares
    inverse_rms = torch.rsqrt(variance + eps)
    return _rms_norm_affine_tail(hidden_states, inverse_rms, weight)


def _exact_sana_rms_norm_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not _kernel_inputs_supported(hidden_states, weight):
        return _eager_sana_rms_norm(hidden_states, weight, eps)
    return _launch_exact_sana_rms_norm(hidden_states, weight, eps)


def _launch_exact_sana_rms_norm_sum(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    squares = _square_bf16_to_fp32(hidden_states)
    sum_sq = squares.sum(dim=-1, keepdim=True)
    del squares
    inverse_rms = torch.rsqrt(sum_sq / hidden_states.shape[-1] + eps)
    return _rms_norm_affine_tail(hidden_states, inverse_rms, weight)


def _exact_sana_rms_norm_sum_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not _kernel_inputs_supported(hidden_states, weight):
        return _eager_sana_rms_norm_sum(hidden_states, weight, eps)
    return _launch_exact_sana_rms_norm_sum(hidden_states, weight, eps)


def _exact_sana_rms_norm_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del weight, eps
    return torch.empty_like(hidden_states)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "exact_sana_rms_norm"):
    direct_register_custom_op(
        op_name="exact_sana_rms_norm",
        op_func=_exact_sana_rms_norm_impl,
        fake_impl=_exact_sana_rms_norm_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )

if not hasattr(torch.ops.vllm_omni, "exact_sana_rms_norm_sum"):
    direct_register_custom_op(
        op_name="exact_sana_rms_norm_sum",
        op_func=_exact_sana_rms_norm_sum_impl,
        fake_impl=_exact_sana_rms_norm_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def _signature(hidden_states: torch.Tensor) -> _Signature:
    hidden_size = hidden_states.shape[-1]
    rows = hidden_states.numel() // hidden_size
    return hidden_states.device, hidden_states.dtype, rows, hidden_size


def _storage_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare BF16 payloads, including signed zero and NaN bit patterns."""
    return torch.equal(left.view(torch.int16), right.view(torch.int16))


def _verified_sana_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    sum_reduction: bool = False,
) -> torch.Tensor:
    eager = _eager_sana_rms_norm_sum if sum_reduction else _eager_sana_rms_norm
    # Preserve the model's pre-existing full-compile behavior.  The custom-op
    # fast path is intentionally an eager/CUDA-graph optimization; regional
    # compilation of other SANA submodules does not enter this branch.
    if torch.compiler.is_compiling():
        return eager(hidden_states, weight, eps)

    if not _can_use_exact_sana_rms_norm(hidden_states, weight):
        return eager(hidden_states, weight, eps)

    verified = _VERIFIED_SUM_SIGNATURES if sum_reduction else _VERIFIED_SIGNATURES
    disabled = _DISABLED_SUM_SIGNATURES if sum_reduction else _DISABLED_SIGNATURES
    reduction = "sum/count" if sum_reduction else "mean"
    signature = _signature(hidden_states)
    if signature in disabled:
        return eager(hidden_states, weight, eps)

    if signature not in verified:
        if torch.cuda.is_current_stream_capturing():
            # Synchronizing a first-sight bit comparison is illegal during
            # graph capture.  A prior eager warmup can verify the signature.
            return eager(hidden_states, weight, eps)

    try:
        if sum_reduction:
            output = torch.ops.vllm_omni.exact_sana_rms_norm_sum(hidden_states, weight, eps)
        else:
            output = torch.ops.vllm_omni.exact_sana_rms_norm(hidden_states, weight, eps)
    except torch.OutOfMemoryError:
        # Temporary memory pressure does not invalidate this kernel signature.
        raise
    except Exception as error:
        disabled.add(signature)
        logger.warning_once(
            "SANA exact RMSNorm (%s) failed for %s; disabling this signature: %s",
            reduction,
            signature,
            error,
        )
        return eager(hidden_states, weight, eps)

    if signature not in verified:
        reference = eager(hidden_states, weight, eps)
        if not _storage_equal(output, reference):
            disabled.add(signature)
            logger.warning_once(
                "SANA exact RMSNorm (%s) did not match eager output for %s; disabling this signature",
                reduction,
                signature,
            )
            return reference
        verified.add(signature)

    return output


def exact_sana_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply SANA's mean-based no-bias affine RMSNorm with verified fusion."""
    return _verified_sana_rms_norm(hidden_states, weight, eps)


def exact_sana_rms_norm_sum(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply SanaDistributedRMSNorm's sum/count contract at TP=1 only.

    The caller must guard TP=1. This helper deliberately has no collective;
    TP>1 must retain the original global sum and global element count.
    """
    return _verified_sana_rms_norm(hidden_states, weight, eps, sum_reduction=True)


__all__ = ["exact_sana_rms_norm", "exact_sana_rms_norm_sum"]
