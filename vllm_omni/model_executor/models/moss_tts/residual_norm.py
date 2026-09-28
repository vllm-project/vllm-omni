# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""One-pass MOSS residual RMSNorm, FP32 accumulation and BF16 outputs.

Matches the existing Inductor arithmetic: retain the FP32 residual sum for
normalization, multiply weight in FP32, round only the stored BF16 tensors.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice


@triton.jit
def _norm(
    x_ptr,
    r_ptr,
    w_ptr,
    o_ptr,
    ro_ptr,
    xs: tl.constexpr,
    rs: tl.constexpr,
    n: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, block)
    x = tl.load(x_ptr + row * xs + d, d < n, 0).to(tl.float32)
    r = tl.load(r_ptr + row * rs + d, d < n, 0).to(tl.float32)
    z = x + r
    variance = tl.sum(z * z, 0) / n
    inv = libdevice.rsqrt(variance + eps)
    w = tl.load(w_ptr + d, d < n, 0).to(tl.float32)
    out = (z * inv) * w
    tl.store(o_ptr + row * n + d, out, d < n)
    tl.store(ro_ptr + row * n + d, z, d < n)


@torch.library.custom_op("vllm_omni::moss_persistent_residual_norm", mutates_args=())
def norm(x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    res = torch.empty_like(out)
    _norm[(x.shape[0],)](
        x,
        r,
        w,
        out,
        res,
        x.stride(0),
        r.stride(0),
        x.shape[-1],
        eps,
        triton.next_power_of_2(x.shape[-1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, res


@norm.register_fake
def _(x, r, w, eps=1e-6):
    return torch.empty(x.shape, device=x.device, dtype=x.dtype), torch.empty(x.shape, device=x.device, dtype=x.dtype)


def _forward(module, x, residual=None):
    if residual is not None and x.is_cuda and x.dtype == torch.bfloat16:
        return norm(x, residual, module.weight.data, module.variance_epsilon)
    return module.forward_native(x, residual)


def install(model):
    import types

    from vllm.model_executor.layers.layernorm import RMSNorm

    count = 0
    for module in model.modules():
        if isinstance(module, RMSNorm) and module.hidden_size == 2560:
            assert module.variance_size_override is None and module.has_weight
            module.forward = types.MethodType(_forward, module)
            count += 1
    return count
