# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib.util import find_spec

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.diffusion.layers.sglangnorm import _layer_norm_fwd_1pass_kernel

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None


class LayerNorm(nn.LayerNorm, CustomOp):
    """
    LayerNorm implementation that inherits from both ``nn.LayerNorm`` and ``CustomOp``.
    NPU:
        Uses ``mindiesd.fast_layernorm(self, x)`` when MindIE-SD is installed.
    CUDA:
        Uses a single-pass Triton kernel (adapted from SGLang) that avoids the
        FP32 cast round-trip and operates natively in bf16/fp16.
        Falls back to ``forward_native`` for CPU tensors or when
        ``elementwise_affine=False``.
    HIP / XPU / native:
        Falls back to FP32 nn.LayerNorm implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__(normalized_shape=dim, eps=eps, elementwise_affine=elementwise_affine)
        # CustomOp.__init__ cannot be called here because it would re-run
        # nn.Module initialization and clear LayerNorm parameters.
        self._forward_method = CustomOp.dispatch_forward(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_method(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or self.weight is None:
            return self.forward_native(x)

        weight = self.weight
        bias = self.bias
        has_bias = bias is not None
        eps = self.eps
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])  # (M, N)
        M, N = x_2d.shape

        y = torch.empty_like(x_2d)
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        BLOCK_N = triton.next_power_of_2(N)
        assert BLOCK_N <= 131072, f"hidden_dim {N} too large for single-block kernel"

        bias_ptr = bias if has_bias else weight  # harmless dummy when HAS_BIAS=False
        grid = (M,)
        _layer_norm_fwd_1pass_kernel[grid](
            # data pointers
            x_2d,
            y,
            weight,
            bias_ptr,
            # unused optional pointers — pass x_2d as a harmless dummy
            x_2d,
            x_2d,
            weight,
            bias_ptr,
            y,
            x_2d,
            x_2d,
            x_2d,
            x_2d,
            x_2d,
            mean,
            rstd,
            # strides
            x_2d.stride(0),
            y.stride(0),
            x_2d.stride(0),
            x_2d.stride(0),
            x_2d.stride(0),
            y.stride(0),
            # scalars
            M,
            N,
            eps,
            0.0,  # dropout_p
            False,  # zero_centered_weight
            # constexprs — only the ones we actually need are True
            IS_RMS_NORM=False,
            BLOCK_N=BLOCK_N,
            HAS_RESIDUAL=False,
            STORE_RESIDUAL_OUT=False,
            HAS_WEIGHT=True,
            HAS_BIAS=has_bias,
            HAS_DROPOUT=False,
            STORE_DROPOUT_MASK=False,
            HAS_ROWSCALE=False,
            HAS_X1=False,
            HAS_W1=False,
            HAS_B1=False,
        )
        return y.reshape(orig_shape)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_MINDIESD:
            try:
                from mindiesd import fast_layernorm

                return fast_layernorm(self, x)
            except ImportError as e:
                logger.warning_once(
                    "mindiesd.fast_layernorm import failed, falling back to FP32 layer_norm: %s",
                    e,
                )

        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class RMSNorm(CustomOp):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        output = torch_npu.npu_rms_norm(x, gamma=self.weight, epsilon=self.variance_epsilon)[0]

        return output

    def forward_xpu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight.to(torch.float32) * out
        return out.to(input_dtype)


class RMSNormVAE(CustomOp):
    """Root Mean Square Layer Normalization for Channel-First or Last"""

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None
        self.epsilon = epsilon

        self.gamma_rmsnorm = None

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        if self.gamma_rmsnorm is None:
            self.gamma_rmsnorm = self.gamma.reshape(-1)

        if self.channel_first:
            x = x.transpose(1, -1)
            out = torch_npu.npu_rms_norm(x, self.gamma_rmsnorm, epsilon=self.epsilon)[0].transpose(1, -1)
        else:
            out = torch_npu.npu_rms_norm(x, self.gamma_rmsnorm, epsilon=self.epsilon)[0]

        if self.bias is not None:
            out = out + self.bias
        return out

    def forward_xpu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = (
            F.normalize(
                x,
                dim=(1 if self.channel_first else -1),
                eps=self.epsilon,
            )
            * self.scale
            * self.gamma
        )
        if self.bias is not None:
            out = out + self.bias
        return out
