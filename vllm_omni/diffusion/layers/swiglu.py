# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SwiGLU layer with platform-specific fused implementations."""

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.custom_op import CustomOp


class SwiGLU(CustomOp):
    """SwiGLU over a packed ``[gate, up]`` activation tensor.

    The input's final dimension is split evenly into gate and up projections.
    Ascend uses ``torch_npu.npu_swiglu`` to fuse SiLU and the elementwise
    multiplication. Other platforms retain the native reference expression.
    """

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        return torch_npu.npu_swiglu(x, dim=-1)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_musa(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
