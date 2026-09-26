import os
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
_TRITON_ADALN_MAX_BLOCK_SIZE = 8192

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _adalayernorm_scale_shift_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        scale_mode: tl.constexpr,
        rows_per_batch: tl.constexpr,
        scale_stride_batch: tl.constexpr,
        shift_stride_batch: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_size)
        mask = cols < n_cols

        x_offsets = row * n_cols + cols
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        x_masked = tl.where(mask, x, 0.0)
        mean = tl.sum(x_masked, axis=0) / n_cols
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / n_cols
        normed = centered * tl.rsqrt(var + eps)

        if scale_mode == 0:
            scale_offsets = row * n_cols + cols
            shift_offsets = row * n_cols + cols
        elif scale_mode == 1:
            batch = row // rows_per_batch
            scale_offsets = batch * scale_stride_batch + cols
            shift_offsets = batch * shift_stride_batch + cols
        else:
            scale_offsets = cols
            shift_offsets = cols

        scale = tl.load(scale_ptr + scale_offsets, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + shift_offsets, mask=mask, other=0.0).to(tl.float32)
        out = normed * (1.0 + scale) + shift
        tl.store(out_ptr + x_offsets, out, mask=mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


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
        if os.environ.get("VLLM_OMNI_ENABLE_TRITON_ADALN", "1") != "1":
            return self.forward_native(x, scale, shift)

        triton_args = self._get_triton_args(x, scale, shift)
        if triton_args is None:
            return self.forward_native(x, scale, shift)

        scale_mode, rows_per_batch, scale_stride_batch, shift_stride_batch, block_size, num_warps = triton_args
        out = torch.empty_like(x)
        rows = x.numel() // self.hidden_size
        _adalayernorm_scale_shift_kernel[(rows,)](
            x,
            scale,
            shift,
            out,
            self.hidden_size,
            self.eps,
            scale_mode,
            rows_per_batch,
            scale_stride_batch,
            shift_stride_batch,
            block_size,
            num_warps=num_warps,
        )
        return out

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

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.layernorm(x) * (1 + scale) + shift

    def _get_triton_args(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> tuple[int, int, int, int, int, int] | None:
        if (
            not _HAS_TRITON
            or self.elementwise_affine
            or x.ndim < 2
            or x.shape[-1] != self.hidden_size
            or not x.is_cuda
            or not scale.is_cuda
            or not shift.is_cuda
            or x.device != scale.device
            or x.device != shift.device
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or scale.dtype != x.dtype
            or shift.dtype != x.dtype
            or not x.is_contiguous()
            or scale.shape[-1] != self.hidden_size
            or shift.shape[-1] != self.hidden_size
            or scale.stride(-1) != 1
            or shift.stride(-1) != 1
        ):
            return None

        rows = x.numel() // self.hidden_size
        block_size = _next_power_of_2(self.hidden_size)
        if block_size > _TRITON_ADALN_MAX_BLOCK_SIZE:
            return None

        if scale.shape == x.shape and shift.shape == x.shape and scale.is_contiguous() and shift.is_contiguous():
            scale_mode = 0
            rows_per_batch = 1
            scale_stride_batch = self.hidden_size
            shift_stride_batch = self.hidden_size
        elif (
            scale.shape[0] == x.shape[0]
            and shift.shape[0] == x.shape[0]
            and scale.numel() == x.shape[0] * self.hidden_size
            and shift.numel() == x.shape[0] * self.hidden_size
            and rows % x.shape[0] == 0
        ):
            scale_mode = 1
            rows_per_batch = rows // x.shape[0]
            scale_stride_batch = scale.stride(0)
            shift_stride_batch = shift.stride(0)
        elif scale.numel() == self.hidden_size and shift.numel() == self.hidden_size:
            scale_mode = 2
            rows_per_batch = 1
            scale_stride_batch = 0
            shift_stride_batch = 0
        else:
            return None

        num_warps = 8 if block_size >= 2048 else 4
        return scale_mode, rows_per_batch, scale_stride_batch, shift_stride_batch, block_size, num_warps


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
