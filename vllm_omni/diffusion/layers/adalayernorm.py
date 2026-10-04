# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Callable
from importlib.util import find_spec
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.triton_utils import HAS_TRITON as _HAS_TRITON
from vllm.triton_utils import tl, triton

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.diffusion.layers.norm import LayerNorm

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None

# Fast-path bound: one Triton program covers the whole normalized dimension, so
# BLOCK_C = next_power_of_2(C) grows with the hidden size. The diffusion models
# consuming AdaLayerNorm use hidden sizes 2240 (Sana-WM), 3072/3584 (Wan2.2,
# Qwen-Image) and 5120 (Wan2.2 A14B), i.e. BLOCK_C up to 8192, which compiles
# and runs efficiently on H200 (measured; C=5120 -> 30-111 us depending on L).
# Beyond that, masked-lane waste and register pressure grow without a measured
# benefit, so the fast path falls back to forward_native.
_MAX_BLOCK_C = 8192

_ADALN_CONFIGS: dict = {}
_ADALN_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# Runtime configs whose Triton compile/launch failed synchronously; they fall
# back to forward_native and are not retried (mirrors the LTX2
# residual-AdaLN failed-key cache in residual_adaln.py).
_FAILED_ADALN_KEYS: set = set()
_adaln_fused_forward: Callable[["AdaLayerNorm", torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor | None] | None


if _HAS_TRITON:

    @triton.jit
    def _adaln_scale_shift_layernorm_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        weight_ptr,
        bias_ptr,
        eps,
        channels,
        seq_len,
        scale_row_stride,
        shift_row_stride,
        has_weight: tl.constexpr,
        has_bias: tl.constexpr,
        is_half: tl.constexpr,
        per_sample_scale: tl.constexpr,
        per_sample_shift: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """One program per LayerNorm row: out = ln(x) * (1 + scale) + shift.

        Rows enumerate contiguous (B, L, C) or (B, F, S, C) input.
        sample_idx = row // seq_len selects a sample, or a flattened
        batch/frame group with seq_len = S. Per-group modulation is
        (B, 1, C) or (B, F, 1, C); explicit row strides support projection
        chunk views. Shared modulation addresses one channel vector.
        (B, C) at B > 1 is a native-path broadcast error and never reaches
        this kernel.

        Mean/variance use a shift-invariant two-pass: the row is centered on
        its first element (x0) before the fp32 tree sum, so for inputs with a
        large constant offset and small variance the summation and the
        x - mean cancellation stay accurate where a plain fp32 sum of
        large-offset values loses the low-order bits (the reason
        fused_adaptive_group_norm_silu uses Welford/Chan). Reductions
        accumulate in fp32 (matches the golden path, which computes
        F.layer_norm on x.float()). For half-precision outputs the golden
        chain rounds after every torch op (LN result, 1+scale, product, sum),
        so the kernel replicates exactly those roundings - keeping the output
        bit-faithful to the frozen semantics. fp32 outputs have no
        intermediate rounding in the golden path and use a plain fp32 chain.
        block_c covers the whole normalized dim (masked); it is never
        autotuned. Non-contiguous last-dim inputs never reach this kernel
        (routed to forward_native upstream).
        """
        row = tl.program_id(0).to(tl.int64)
        sample_idx = row // seq_len
        cols = tl.arange(0, block_c)
        mask = cols < channels
        x = tl.load(x_ptr + row * channels + cols, mask=mask, other=0.0).to(tl.float32)
        # Center on the first element before summing: for x ~ offset + noise
        # this keeps every value entering the tree sum at noise magnitude.
        x0 = tl.load(x_ptr + row * channels).to(tl.float32)
        xs = tl.where(mask, x - x0, 0.0)
        mean_xs = tl.sum(xs, axis=0) / channels
        # xm = xs - mean_xs is exactly x - (x0 + mean_xs): the row mean enters
        # only through this centering, and the shift keeps it accurate.
        xm = tl.where(mask, xs - mean_xs, 0.0)
        var = tl.sum(xm * xm, axis=0) / channels
        xn = xm * tl.rsqrt(var + eps)
        if has_weight:
            xn = xn * tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if has_bias:
            xn = xn + tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if per_sample_scale:
            s = tl.load(scale_ptr + sample_idx * scale_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            s = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if per_sample_shift:
            sh = tl.load(shift_ptr + sample_idx * shift_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            sh = tl.load(shift_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out_dtype = out_ptr.dtype.element_ty
        if is_half:
            # Golden bf16/fp16 chain: round after each op, exactly as the
            # sequence of torch elementwise kernels does (fp32 internal math).
            ln16 = xn.to(out_dtype)
            t1 = (1.0 + s).to(out_dtype)
            t2 = (ln16.to(tl.float32) * t1.to(tl.float32)).to(out_dtype)
            y = (t2.to(tl.float32) + sh).to(out_dtype)
        else:
            y = xn * (1.0 + s) + sh
        # out 与输入形状相同，按连续行写入。
        tl.store(out_ptr + row * channels + cols, y.to(out_dtype), mask=mask)

    def _adaln_modulation_mode(t: torch.Tensor, x: torch.Tensor):
        """Classify scale/shift against 3D/4D x for the fused kernel.

        "shared"     every leading dim is 1: (C,), (1, C), (1, 1, C)
        "per_sample" (B, 1, C) or (B, F, 1, C): one row per sample/frame
        None         anything else - native torch broadcasting on the
                     forward_native path handles it (or raises, for the
                     frozen B > 1 (B, C) RuntimeError)
        """
        shape = t.shape
        if t.ndim < 1 or t.ndim > x.ndim or shape[-1] != x.shape[-1]:
            return None
        if all(d == 1 for d in shape[:-1]):
            return "shared"
        if x.ndim == 3 and t.ndim == 3 and shape == (x.shape[0], 1, x.shape[-1]):
            return "per_sample"
        if x.ndim == 4 and t.ndim == 4 and shape == (x.shape[0], x.shape[1], 1, x.shape[-1]):
            # batch/frame 必须能用单个 row stride 线性寻址；保留 chunk view。
            if shape[0] > 1 and shape[1] > 1 and t.stride(0) != shape[1] * t.stride(1):
                return None
            return "per_sample"
        return None

    def _adaln_param_matches(x: torch.Tensor, p: torch.Tensor) -> bool:
        return (
            p.dtype is x.dtype
            and p.is_cuda
            and p.device == x.device
            and p.ndim == 1
            and p.shape[0] == x.shape[-1]
            and p.is_contiguous()
        )

    def _adaln_inputs_supported(module: "AdaLayerNorm", x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        """Strict fast-path eligibility (mirrors the LTX2 residual-AdaLN
        guard pattern): every tensor the kernel dereferences must be a dense
        CUDA tensor on x's device with a supported layout and dtype; anything
        else falls back to forward_native, which reproduces torch semantics
        exactly - including the frozen B > 1 (B, C) RuntimeError. Raw pointers
        go to one kernel, which does no cross-device checking. Runs native
        under torch.compile so regional compilation is not disrupted."""
        if torch.compiler.is_compiling():
            return None
        if (
            not _HAS_TRITON
            or x.device.type != "cuda"
            or x.ndim not in (3, 4)
            or x.numel() == 0
            or not x.is_contiguous()
        ):
            return None
        if x.dtype not in _ADALN_DTYPES:
            return None
        if scale.dtype is not x.dtype or shift.dtype is not x.dtype:
            return None
        if not (scale.is_cuda and shift.is_cuda):
            return None
        if scale.device != x.device or shift.device != x.device:
            return None
        mode_s = _adaln_modulation_mode(scale, x)
        mode_h = _adaln_modulation_mode(shift, x)
        if mode_s is None or mode_h is None:
            return None
        # The kernel addresses the modulation rows densely per channel and
        # strides per sample: the last dim must be unit-stride; other dims
        # are handled by the explicit row stride (chunk views supported).
        # 0-dim (scalar) modulation is classified None above and falls back.
        if scale.stride(-1) != 1 or shift.stride(-1) != 1:
            return None
        if x.data_ptr() % 16 or scale.data_ptr() % 16 or shift.data_ptr() % 16:
            return None
        channels = x.shape[-1]
        block_c = triton.next_power_of_2(channels)
        if block_c > _MAX_BLOCK_C:
            return None
        ln = module.layernorm
        for p in (ln.weight, ln.bias):
            if p is not None and not _adaln_param_matches(x, p):
                return None
        return channels, block_c, mode_s == "per_sample", mode_h == "per_sample"

    def _adaln_fused_forward_impl(module: "AdaLayerNorm", x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        """Fused fast path. Returns the output tensor, or None when the inputs
        are unsupported or the optimized path failed synchronously (the caller
        then uses forward_native). The runtime fallback is best-effort: it
        catches synchronous Triton compile/launch errors, not asynchronous
        CUDA execution faults."""
        supported = _adaln_inputs_supported(module, x, scale, shift)
        if supported is None:
            return None
        channels, block_c, per_sample_scale, per_sample_shift = supported
        seq_len = x.shape[-2]
        rows = x.numel() // channels
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ln = module.layernorm
        weight = ln.weight
        bias = ln.bias
        has_weight = weight is not None
        has_bias = bias is not None
        is_half = x.dtype is not torch.float32
        cfg = _ADALN_CONFIGS.get(block_c)
        if cfg is None:
            num_warps = 4 if block_c <= 1024 else (8 if block_c <= 4096 else 16)
            cfg = (block_c, num_warps)
            _ADALN_CONFIGS[block_c] = cfg
        # F > 1 时按 frame 跨行；F = 1 时按 batch 跨行。
        scale_row_stride = scale.stride(1) if scale.ndim == 4 and scale.shape[1] > 1 else scale.stride(0)
        shift_row_stride = shift.stride(1) if shift.ndim == 4 and shift.shape[1] > 1 else shift.stride(0)
        # Dummy pointer args for disabled branches: never dereferenced because
        # the loads are constexpr-pruned when has_weight/has_bias is False.
        args = (
            x,
            scale,
            shift,
            out,
            weight if has_weight else scale,
            bias if has_bias else scale,
            module.eps,
            channels,
            seq_len,
            scale_row_stride,
            shift_row_stride,
            has_weight,
            has_bias,
            is_half,
            per_sample_scale,
            per_sample_shift,
            block_c,
        )
        # The key carries every field that selects the compiled binary
        # (dtype covers is_half; per-sample flags and affine presence select
        # constexpr variants; eps is a runtime scalar). A failure in one
        # variant must not disable the others.
        runtime_key = (
            x.device.index,
            channels,
            x.dtype,
            has_weight,
            has_bias,
            per_sample_scale,
            per_sample_shift,
        )
        if runtime_key in _FAILED_ADALN_KEYS:
            return None
        try:
            _adaln_scale_shift_layernorm_kernel[(rows,)](*args, num_warps=cfg[1])
        except Exception as exc:  # noqa: BLE001 - fail closed after optimized-path failure
            _FAILED_ADALN_KEYS.add(runtime_key)
            logger.warning(
                "Disabling the AdaLayerNorm fused fast path on %s after failure: %s",
                x.device,
                exc,
            )
            return None
        return out

    _adaln_fused_forward = _adaln_fused_forward_impl


else:
    # No Triton: the fused fast path is unavailable; forward_cuda falls back
    # to forward_native below.
    _adaln_fused_forward = None


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
        if _adaln_fused_forward is None:
            return self.forward_native(x, scale, shift)
        out = _adaln_fused_forward(self, x, scale, shift)
        if out is not None:
            return out
        return self.forward_native(x, scale, shift)

    def forward_hip(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_musa(
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
