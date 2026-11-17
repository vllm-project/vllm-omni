# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in signed Hadamard64 for MiniMax-H3-Image's online FP8 MLPs.

Matching right-side rotations preserve ``X @ W.T`` before quantization. The
fc2 weights rotate in FP32 once while loading; activation rotation and token
quantization share one Triton kernel. The native row-wise FP8 GEMM is retained.
This changes FP8 rounding, and does not establish BF16-equivalent image quality.
"""

from __future__ import annotations

import hashlib
from functools import cache
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.triton_utils import tl, triton

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8

    from vllm_omni.diffusion.data import OmniDiffusionConfig

_FC2_SHAPE = (5376, 14336)
_MLP_NAMES = frozenset(f"blocks.{index}.mlp.fc{side}" for index in range(50) for side in (1, 2))
_SEED = 171117


def _signs(name: str, width: int = 14336) -> torch.Tensor:
    domain = f"h3-fc2-signed-hadamard64-v1:{_SEED}:{name}:{width}".encode()
    values = hashlib.shake_256(domain).digest(width)
    return torch.tensor([1.0 if value & 1 else -1.0 for value in values], dtype=torch.float32)


def _rotate_rows(value: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """FP32 weight transform; never round a rotated weight through BF16."""
    if value.ndim != 2 or value.shape[1] % 64 or signs.shape != (value.shape[1],):
        raise ValueError("Hadamard64 requires matrices with widths divisible by 64 and one sign per column")
    result = value.float() * signs.to(device=value.device, dtype=torch.float32)
    shape = result.shape
    for stride in (1, 2, 4, 8, 16, 32):
        grouped = result.reshape(shape[0], -1, 2, stride)
        a, b = grouped[:, :, 0, :], grouped[:, :, 1, :]
        result = torch.stack((a + b, a - b), dim=2).reshape(shape)
    return result * 0.125


@triton.jit
def _fwht_stage(values, n: tl.constexpr, stride: tl.constexpr):
    grouped = tl.reshape(values, (n // (2 * stride), 2, stride))
    grouped = tl.trans(grouped, 0, 2, 1)
    a, b = tl.split(grouped)
    grouped = tl.trans(tl.join(a + b, a - b), 0, 2, 1)
    return tl.reshape(grouped, (n,))


@triton.jit
def _hadamard64_quant_kernel(x_ptr, signs_ptr, q_ptr, scales_ptr, k: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    valid = columns < k
    values = tl.load(x_ptr + row * k + columns, mask=valid, other=0).to(tl.float32)
    signs = tl.load(signs_ptr + columns, mask=valid, other=0)
    values = values * signs
    # Every butterfly span divides 64, so padding never mixes into real groups.
    values = _fwht_stage(values, block, 1)
    values = _fwht_stage(values, block, 2)
    values = _fwht_stage(values, block, 4)
    values = _fwht_stage(values, block, 8)
    values = _fwht_stage(values, block, 16)
    values = _fwht_stage(values, block, 32)
    values = values * 0.125
    maximum = tl.max(tl.where(valid, tl.abs(values), 0), axis=0)
    # Preserve the tested nonzero-row reciprocal/rounding recipe. Zero rows use
    # scale one and encode exactly zero instead of forming 0 * reciprocal(0).
    scale = tl.where(maximum > 0, maximum * 0.0022321429569274187, 1.0)
    inverse = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1;", constraints="=v,v", args=[scale], dtype=tl.float32, is_pure=True, pack=1
    )
    scaled = values * inverse
    clamped = tl.inline_asm_elementwise(
        "v_med3_f32 $0, $1, $2, $3;",
        constraints="=v,v,v,v",
        args=[scaled, tl.full((block,), -448.0, tl.float32), tl.full((block,), 448.0, tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    encoded = clamped.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(q_ptr + row * k + columns, encoded, mask=valid)
    tl.store(scales_ptr + row, scale)


@cache
def _validate_device(device: torch.device) -> None:
    if not current_omni_platform.is_rocm() or device.type != "cuda":
        raise ValueError("MiniMax-H3-Image Hadamard64 FP8 requires ROCm gfx950")
    capability = current_omni_platform.get_device_capability(device.index or 0)
    if capability is None or capability.to_int() != 95:
        raise ValueError("MiniMax-H3-Image Hadamard64 FP8 is validated only on gfx950")
    if current_omni_platform.fp8_dtype() != torch.float8_e4m3fn:
        raise ValueError("MiniMax-H3-Image Hadamard64 requires OCP E4M3 FP8")


def hadamard64_quant(x: torch.Tensor, signs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused signed rotation and whole-token dynamic FP8 quantization."""
    _validate_device(x.device)
    if x.ndim != 2 or x.shape[1] != _FC2_SHAPE[1] or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Hadamard64 requires contiguous BF16 activations with shape [M, 14336]")
    if signs.dtype != torch.float32 or signs.shape != (x.shape[1],) or signs.device != x.device:
        raise ValueError("Hadamard64 signs must be FP32[14336] on the activation device")
    if not signs.is_contiguous():
        raise ValueError("Hadamard64 signs must be contiguous")
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    if x.shape[0]:
        _hadamard64_quant_kernel[(x.shape[0],)](
            x, signs, quantized, scales, _FC2_SHAPE[1], 16384, num_warps=8, enable_fp_fusion=False
        )
    return quantized, scales


class Hadamard64QuantFP8:
    """Replace the activation quantizer while retaining the native FP8 GEMM."""

    def __init__(self, original: QuantFP8, signs: torch.Tensor) -> None:
        if original.static or not original.group_shape.is_per_token() or original.is_group_quant:
            raise ValueError("Hadamard64 requires dynamic per-token FP8 quantization")
        if not original.use_aiter:
            raise ValueError("Hadamard64 requires the ROCm AITER per-token quantization path")
        self.signs = signs

    def __call__(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scale is not None or scale_ub is not None or use_triton:
            raise ValueError("Hadamard64 supports only dynamic unbounded per-token FP8 quantization")
        return hadamard64_quant(x, self.signs)


def _validate_config(od_config: OmniDiffusionConfig, device: torch.device) -> None:
    from vllm import envs

    from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload
    from vllm_omni.quantization import resolve_component_quant_config

    _validate_device(device)
    parallel = od_config.parallel_config
    if (
        any(
            int(getattr(parallel, name, 1)) != 1
            for name in (
                "tensor_parallel_size",
                "sequence_parallel_size",
                "pipeline_parallel_size",
                "cfg_parallel_size",
            )
        )
        or parallel.use_hsdp
    ):
        raise ValueError("MiniMax-H3-Image Hadamard64 FP8 requires TP1/SP1 without HSDP")
    if resolve_offload(od_config).strategy is not OffloadStrategy.NONE or od_config.enable_sleep_mode:
        raise ValueError("MiniMax-H3-Image Hadamard64 FP8 requires resident weights without offload or sleep")
    if od_config.lora_path or od_config.lora_application != "runtime" or od_config.weight_transfer_config is not None:
        raise ValueError("MiniMax-H3-Image Hadamard64 FP8 does not support LoRA or runtime weight updates")
    if envs.VLLM_BATCH_INVARIANT:
        raise ValueError("MiniMax-H3-Image Hadamard64 FP8 does not support the batch-invariant BF16 fallback")
    for component in ("text_encoder", "vae"):
        if resolve_component_quant_config(od_config.quantization_config, component) is not None:
            raise ValueError(f"Hadamard64 MLP-only FP8 requires quantization disabled for {component}")


def _install_linear(name: str, module: nn.Module) -> None:
    """Install before loading the BF16 checkpoint, preserving native setup."""
    method = module.quant_method
    original = method.process_weights_after_loading
    signs = _signs(name)
    prepared = False

    def process(layer: nn.Module) -> None:
        from vllm.model_executor.layers.quantization.online.fp8 import _fp8_channel_scale, _fp8_quant_per_channel
        from vllm.model_executor.utils import replace_parameter

        nonlocal prepared
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            original(layer)
            return
        if prepared or layer is not module or int(layer.tp_size) != 1:
            raise ValueError("Hadamard64 supports a fresh TP1 BF16 weight load only")
        if getattr(method, "requantization_source", None) is not None:
            raise ValueError("Hadamard64 cannot load pre-quantized weights")
        source = layer.weight.detach()
        if source.dtype != torch.bfloat16 or tuple(source.shape) != _FC2_SHAPE or source.device.type == "meta":
            raise ValueError("Hadamard64 requires original complete BF16 fc2 weights [5376, 14336]")
        qweight = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty((source.shape[0], 1), device=source.device, dtype=torch.float32)
        local_signs = signs.to(source.device)
        # Bound temporary FP32 memory; no rotated BF16 intermediate is created.
        for begin in range(0, source.shape[0], 128):
            end = min(source.shape[0], begin + 128)
            rotated = _rotate_rows(source[begin:end], local_signs)
            scales = _fp8_channel_scale(rotated.abs().amax(-1, keepdim=True))
            qweight[begin:end] = _fp8_quant_per_channel(rotated, scales)
            weight_scale[begin:end] = scales
        if not bool(torch.isfinite(qweight.float()).all() and torch.isfinite(weight_scale).all()):
            raise ValueError("Nonfinite Hadamard64 FP8 weights or scales")
        # Native conversion configures the GEMM and lifecycle; replace only its
        # primary weight/scale and activation quantizer after that setup.
        original(layer)
        replace_parameter(layer, "weight", qweight.t())
        replace_parameter(layer, "weight_scale", weight_scale)
        kernel = layer.quant_method.fp8_linear
        kernel.quant_fp8 = Hadamard64QuantFP8(kernel.quant_fp8, local_signs)
        prepared = True

    method.process_weights_after_loading = process


def install_fc2_hadamard_fp8(dit: nn.Module, *, od_config: OmniDiffusionConfig, device: torch.device) -> None:
    """Transform the explicit 100-MLP online PTPC recipe; never select FP8."""
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import RowWiseTorchFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod

    _validate_config(od_config, device)
    if getattr(dit, "_minimax_h3_mlp_fp8_hadamard", False):
        raise ValueError("MiniMax-H3-Image Hadamard64 is already installed")
    modules = dict(dit.named_modules())
    for name in _MLP_NAMES:
        module = modules.get(name)
        method = getattr(module, "quant_method", None)
        if (
            type(method) is not Fp8PtpcOnlineLinearMethod
            or type(method.fp8_linear) is not RowWiseTorchFP8ScaledMMLinearKernel
        ):
            raise ValueError("Hadamard64 requires explicit online fp8_per_channel for all 100 MLP linears")
        if getattr(module, "_already_called_process_weights_after_loading", False):
            raise ValueError("Install Hadamard64 before loading the BF16 checkpoint")
    if len({id(modules[name].quant_method.fp8_linear) for name in _MLP_NAMES}) != len(_MLP_NAMES):
        raise ValueError("Hadamard64 requires independent per-layer FP8 kernels")
    for name, module in modules.items():
        if isinstance(module, LinearBase) and name not in _MLP_NAMES:
            if not isinstance(module.quant_method, UnquantizedLinearMethod):
                raise ValueError(f"Hadamard64 MLP-only FP8 requires an unquantized transformer layer: {name}")
    for index in range(50):
        name = f"blocks.{index}.mlp.fc2"
        _install_linear(name, modules[name])
    dit._minimax_h3_mlp_fp8_hadamard = True
