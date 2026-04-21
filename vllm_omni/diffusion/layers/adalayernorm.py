from importlib.util import find_spec
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.diffusion.layers.norm import LayerNorm

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None


class AdaLayerNorm(CustomOp):
    """
    AdaLayerNorm:
        out = layernorm(x) * (1 + scale) + shift
    """

    def __init__(self, hidden_size: int, elementwise_affine: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.hidden_size = hidden_size
        self.layernorm = LayerNorm(self.hidden_size, elementwise_affine=self.elementwise_affine, eps=self.eps)

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_hip(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_npu(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        if _HAS_MINDIESD:
            try:
                from mindiesd import layernorm_scale_shift

                output = layernorm_scale_shift(self.layernorm, x, scale, shift, fused=True)

                return output
            except ImportError as e:
                logger.warning_once(f"mindiesd import failed, falling back to torch_npu: {e}")

        import torch_npu

        output = (
            torch_npu.npu_layer_norm_eval(x, normalized_shape=[self.hidden_size], eps=self.eps) * (1 + scale) + shift
        )

        return output

    def forward_xpu(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.layernorm(x) * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.emb = None
        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            embedding_dim,
            6 * embedding_dim,
            bias=bias,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        if isinstance(emb, tuple):
            emb = emb[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            embedding_dim,
            3 * embedding_dim,
            bias=bias,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        if isinstance(emb, tuple):
            emb = emb[0]
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            conditioning_embedding_dim,
            embedding_dim * 2,
            bias=bias,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        if isinstance(emb, tuple):
            emb = emb[0]
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
