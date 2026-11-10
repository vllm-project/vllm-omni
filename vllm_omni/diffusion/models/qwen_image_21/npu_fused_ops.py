# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from functools import cache

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


@cache
def _available(device_index: int | None) -> bool:
    if not HAS_TRITON or "ascend" not in getattr(triton.backends, "backends", {}):
        return False
    return torch.npu.get_device_name(device_index) == "Ascend910B1"


def can_fuse_swiglu(hidden_states: torch.Tensor) -> bool:
    return (
        hidden_states.device.type == "npu"
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 3
        and hidden_states.shape[0] == 1
        and hidden_states.shape[1] >= 4096
        and hidden_states.shape[2] == 4096
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and _available(hidden_states.device.index)
    )


@triton.jit
def _swiglu_kernel(gate, up, output, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    g = tl.load(gate + offsets).to(tl.float32)
    u = tl.load(up + offsets).to(tl.float32)
    # Preserve the rounding at the original SiLU output before multiplying.
    activated = (g / (1 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, activated * u)


def swiglu_inplace(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Consume a fresh gate projection; leave the independent up projection intact."""
    import torch_npu

    if (
        gate.numel() == 0
        or gate.ndim == 0
        or gate.device.type != "npu"
        or gate.shape != up.shape
        or gate.dtype != torch.bfloat16
        or up.dtype != torch.bfloat16
        or gate.device != up.device
        or not gate.is_contiguous()
        or not up.is_contiguous()
        or gate.shape[-1] != 12288
        or torch_npu.get_npu_format(gate) != 2
        or torch_npu.get_npu_format(up) != 2
    ):
        return torch.nn.functional.silu(gate, inplace=True).mul_(up)
    # Width 12288 guarantees full 4096-element tiles. Omitting a length
    # argument lets all eligible sequence lengths share one compiled kernel.
    _swiglu_kernel[(gate.numel() // 4096,)](gate, up, gate, 4096)
    return gate
