# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend native MX execution for ModelSlim H3 component deliveries.

Delivery dual-scale FP4 uses groups of 32, unlike the runtime's dual-level
kernel (512). The reader applies both scales before this adapter converts
weights once to native single-scale MX. No floating GEMM fallback is used.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import replace_parameter

from vllm_omni.quantization.int8_config import LazyWeightMixin

if TYPE_CHECKING:
    from .offline_quantization import OfflineComponentCheckpoint

logger = init_logger(__name__)
_MX4 = "W4A4_MXFP4_DUALSCALE"
_MX8 = "W8A8_MXFP8"
_OUTPUT_CHUNK = 32768
_EXECUTED_BITS: set[int] = set()


def _target_bits(scheme: str) -> int | None:
    if scheme in {_MX4, "W4A4_DYNAMIC"}:
        return 4
    return 8 if scheme in {_MX8, "W8A8_DYNAMIC"} else None


@torch.no_grad()
def quantize_mx_weight(weight: torch.Tensor, bits: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a reconstructed matrix once; return plain transportable bytes."""
    import torch_npu

    if device.type != "npu" or bits not in (4, 8):
        raise ValueError("H3 native MX requires an NPU and bits=4 or bits=8")
    from .offline_quantization import require_native_family

    require_native_family("mx", device)
    if weight.ndim != 2 or min(weight.shape) <= 0 or not torch.isfinite(weight).all():
        raise ValueError("native MX weight must be a finite matrix")
    padded_k = (weight.shape[1] + 63) // 64 * 64
    dtype = torch_npu.float4_e2m1fn_x2 if bits == 4 else torch_npu.float8_e4m3fn
    weights, scales = [], []
    for chunk in weight.split(_OUTPUT_CHUNK, dim=0):
        chunk = chunk.to(device=device, dtype=torch.bfloat16)
        chunk = nn.functional.pad(chunk, (0, padded_k - weight.shape[1]))
        q, scale = torch_npu.npu_dynamic_mx_quant(
            chunk, dst_type=dtype, axis=-1, block_size=32, round_mode="rint", scale_alg=0
        )
        weights.append(q.view(torch.uint8))
        scales.append(scale.reshape(chunk.shape[0], -1, 2))
    return torch.cat(weights).contiguous(), torch.cat(scales).contiguous()


def native_mx_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bits: int, bias: torch.Tensor | None
) -> torch.Tensor:
    """Dynamic MX activations and native low-bit GEMM, including wide H3 AdaLN."""
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "native MX activations must be FP16, BF16, or FP32; packed weight dtype is not an activation dtype"
        )

    import torch_npu

    if x.device.type != "npu" or weight.device != x.device or scale.device != x.device:
        raise ValueError("native MX inputs, weight, and scale must reside on the same NPU")
    from .offline_quantization import require_native_family

    require_native_family("mx", x.device)
    if bits not in (4, 8) or x.ndim < 1 or weight.ndim != 2 or weight.dtype != torch.uint8:
        raise ValueError("native MX requires bits=4/8, byte matrix storage, and a floating activation")
    if scale.dtype != torch.uint8 or scale.shape != (
        weight.shape[0],
        (weight.shape[1] * (2 if bits == 4 else 1)) // 64,
        2,
    ):
        raise ValueError("native MX scale layout must match weight output rows and K groups")
    if bias is not None and (
        bias.shape != (weight.shape[0],) or not bias.is_floating_point() or bias.device != x.device
    ):
        raise ValueError("native MX bias must match floating output channels on the same NPU")
    shape, output_dtype = x.shape, x.dtype
    padded_k = weight.shape[1] * (2 if bits == 4 else 1)
    if shape[-1] > padded_k:
        raise ValueError("native MX input exceeds the prepared weight width")
    x = x.reshape(-1, shape[-1]).to(torch.bfloat16)
    x = nn.functional.pad(x, (0, padded_k - shape[-1]))
    mx_dtype = torch_npu.float4_e2m1fn_x2 if bits == 4 else torch_npu.float8_e4m3fn
    xq, xs = torch_npu.npu_dynamic_mx_quant(
        x, dst_type=mx_dtype, axis=-1, block_size=32, round_mode="rint", scale_alg=0
    )
    outputs = []
    for offset in range(0, weight.shape[0], _OUTPUT_CHUNK):
        w = weight[offset : offset + _OUTPUT_CHUNK]
        if bits == 4 and w.storage_offset() != 0:
            # The FP4 override reinterprets uint8 storage offsets in nibble
            # units. Rebase DLO views and later AdaLN chunks before dispatch;
            # contiguous() can return the same view with its nonzero offset.
            # Copy at most one output chunk, keeping native low-bit compute.
            w = w.clone()
        elif bits == 8:
            w = w.view(torch.float8_e4m3fn)
        # Keep BF16 output for the documented native MX kernel contract. Audio
        # VAE's public FP32 boundary is restored after the low-bit multiply.
        y = torch_npu.npu_quant_matmul(
            xq,
            w.transpose(0, 1),
            scale[offset : offset + _OUTPUT_CHUNK].transpose(0, 1),
            x1_dtype=mx_dtype,
            x2_dtype=mx_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=xs,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            output_dtype=torch.bfloat16,
            group_sizes=[1, 1, 32],
        )
        outputs.append(y)
    output = torch.cat(outputs, dim=-1).to(output_dtype)
    if bits not in _EXECUTED_BITS:
        logger.info(
            "H3 native MX executed W%dA%d: npu_dynamic_mx_quant + npu_quant_matmul; "
            "activation=%s weight_storage=%s output=%s group=32",
            bits,
            bits,
            xq.dtype,
            weight.dtype,
            output.dtype,
        )
        _EXECUTED_BITS.add(bits)
    if bias is not None:
        output = output + bias.to(output_dtype)
    return output.reshape(*shape[:-1], weight.shape[0])


class NativeMXLinear(nn.Module):
    """VAE Linear with byte storage and a floating input/output contract.

    ``weight.dtype`` describes packed storage, never the activation dtype.
    Checkpoint callers that inspect it must use the adapter installed below.
    """

    def __init__(self, linear: nn.Linear, bits: int, device: torch.device) -> None:
        super().__init__()
        self.in_features, self.out_features, self.bits = linear.in_features, linear.out_features, bits
        weight, scale = quantize_mx_weight(linear.weight, bits, device)
        storage_device = linear.weight.device
        self.weight = nn.Parameter(weight.to(storage_device), requires_grad=False)
        self.weight_scale = nn.Parameter(scale.to(storage_device), requires_grad=False)
        self.register_parameter("bias", linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("native MX input does not match logical Linear width")
        return native_mx_linear(x, self.weight, self.weight_scale, self.bits, self.bias)


def _with_native_mx_activation_dtype(original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    @wraps(original)
    def linear_with_module_dtype(
        linear: nn.Module, tensor: torch.Tensor, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        if not isinstance(linear, NativeMXLinear):
            return original(linear, tensor, out_dtype)
        # Do not infer activation precision from uint8 MX weight storage. The
        # cast must be avoided here: converting back inside forward is too late.
        output = linear(tensor)
        return output if out_dtype is None else output.to(out_dtype)

    setattr(linear_with_module_dtype, "_h3_native_mx_activation_aware", True)
    return linear_with_module_dtype


def _native_mx_causal_attention(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Checkpoint audio attention, routing QKV through the native module.

    The checkpoint uses F.linear(qkv.weight, external_bias), bypassing forward.
    Its QKV has no internal bias; apply its separate Q/K/V biases exactly once.
    Attention itself and the checkpoint's head reduction remain floating point.
    """
    batch, sequence, _ = x.shape
    qkv = self.qkv(x)
    bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
    qkv = qkv + bias.to(dtype=qkv.dtype)
    q, k, v = qkv.reshape(batch, sequence, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
    if self.in_dim > self.out_dim:
        x = torch.mean(x, dim=1)
        if self.in_dim // self.num_heads != self.out_dim:
            x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
    else:
        x = x.transpose(1, 2).reshape(batch, sequence, -1)
    return self.proj(x)


def _install_native_mx_vae_callers(model: nn.Module) -> None:
    for module in model.modules():
        # Remote video VAE functions share their checkpoint module namespace.
        # Dispatch only NativeMXLinear differently; floating models using that
        # same namespace still delegate to the original helper unchanged.
        namespace = getattr(type(module).forward, "__globals__", {})
        helper = namespace.get("_linear_with_module_dtype")
        if callable(helper) and not getattr(helper, "_h3_native_mx_activation_aware", False):
            namespace["_linear_with_module_dtype"] = _with_native_mx_activation_dtype(helper)

        qkv = getattr(module, "qkv", None)
        if not isinstance(qkv, NativeMXLinear) or not hasattr(module, "q_bias"):
            continue
        required = ("zero_k_bias", "v_bias", "in_dim", "out_dim", "num_heads", "head_dim", "proj")
        if not all(hasattr(module, attr) for attr in required) or qkv.bias is not None:
            raise ValueError(
                "unsupported native MX audio attention contract: expected bias-free QKV and external biases"
            )
        if qkv.in_features != module.in_dim or qkv.out_features != 3 * module.num_heads * module.head_dim:
            raise ValueError("unsupported native MX audio attention QKV shape")
        module.forward = MethodType(_native_mx_causal_attention, module)


def install_native_mx_vae(model: nn.Module, checkpoint: OfflineComponentCheckpoint, device: torch.device) -> int:
    """Replace only delivery-quantized Linear modules before staging is built."""
    count = 0
    for name in checkpoint.tensors:
        if not name.endswith(".weight"):
            continue
        bits = _target_bits(checkpoint.description.get(name, "FLOAT"))
        if bits is None:
            continue
        path = name.removesuffix(".weight")
        linear = model.get_submodule(path)
        if not isinstance(linear, nn.Linear):
            raise ValueError(f"native MX VAE weight does not target Linear: {path}")
        parent_path, _, child = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child, NativeMXLinear(linear, bits, device))
        count += 1
    if count:
        _install_native_mx_vae_callers(model)
    logger.info("Installed H3 native MX VAE mode=%s matrices=%d", checkpoint.execution_mode, count)
    return count


class NativeMXLinearMethod(LazyWeightMixin, LinearMethodBase):
    """TP-aware lazy loading followed by native MX conversion once per layer."""

    def __init__(self, bits: int) -> None:
        self.bits = bits

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        layer._native_mx_storage_device = layer._load_device

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        from vllm_omni.diffusion.distributed.utils import get_local_device

        storage_device = layer._native_mx_storage_device
        weight, scale = quantize_mx_weight(layer.weight, self.bits, get_local_device())
        replace_parameter(layer, "weight", weight.to(storage_device))
        layer.register_parameter("weight_scale", nn.Parameter(scale.to(storage_device), requires_grad=False))
        layer._already_called_process_weights_after_loading = True

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != layer.input_size_per_partition:
            raise ValueError("native MX input does not match logical TP width")
        return native_mx_linear(x, layer.weight, layer.weight_scale, self.bits, bias)


class H3NativeMXConfig(QuantizationConfig):
    """Delivery-driven model-local MX config; no global quantization override."""

    delivery_schemes = {_MX4, _MX8}

    def __init__(self, targets: dict[str, int]) -> None:
        super().__init__()
        self.targets = targets

    def get_name(self) -> str:
        return "h3_native_mx"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> H3NativeMXConfig:
        raise ValueError("H3 native MX is selected through offline_quantization.execution_mode")

    def get_quant_method(self, layer: nn.Module, prefix: str) -> LinearMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        bits = self.targets.get(prefix.removeprefix("transformer."))
        return NativeMXLinearMethod(bits) if bits else UnquantizedLinearMethod()

    def validate_model_targets(self, model: nn.Module) -> None:
        installed = {
            name: module.quant_method.bits
            for name, module in model.named_modules()
            if isinstance(getattr(module, "quant_method", None), NativeMXLinearMethod)
        }
        if installed != self.targets:
            missing = sorted(self.targets.keys() - installed.keys())
            raise ValueError(f"native MX targets must all use native methods; missing={missing}")

    @classmethod
    def from_checkpoint(cls, checkpoint: OfflineComponentCheckpoint) -> H3NativeMXConfig:
        from .fasth3 import _resolve_native_target

        schemes: dict[str, set[str]] = {}
        for name in checkpoint.tensors:
            if not name.endswith(".weight"):
                continue
            module = name.removesuffix(".weight")
            if checkpoint.diffusers_weights:
                target = _resolve_native_target(module)
                if target is None:
                    raise ValueError(f"unsupported native MX H3 projection: {module}")
                module = target[0]
            scheme = checkpoint.description.get(name, "FLOAT")
            allowed = cls.delivery_schemes
            if scheme != "FLOAT" and scheme not in allowed:
                raise ValueError(f"{cls.__name__} incompatible delivery format: {scheme}")
            schemes.setdefault(module, set()).add(scheme)
        targets = {}
        for name, values in schemes.items():
            if len(values) != 1:
                raise ValueError(f"native MX fused projections must share one delivery scheme: {name}")
            if (bits := _target_bits(next(iter(values)))) is not None:
                targets[name] = bits
        logger.info("H3 native MX mode=%s target_modules=%d", checkpoint.execution_mode, len(targets))
        return cls(targets)
