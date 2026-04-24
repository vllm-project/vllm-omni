from importlib.util import find_spec

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None


class LayerNorm(nn.LayerNorm, CustomOp):
    """
    LayerNorm implementation that inherits from both ``nn.LayerNorm`` and ``CustomOp``.
    NPU:
        Uses ``mindiesd.fast_layernorm(self, x)`` when MindIE-SD is installed.
    CUDA / HIP / XPU / native:
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
        return self.forward_native(x)

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
