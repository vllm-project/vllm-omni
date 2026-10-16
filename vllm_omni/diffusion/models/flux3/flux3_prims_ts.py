# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer PrimsTS FP8 and NVFP4 implementation for Flux3."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, TypeAlias
from weakref import ref

import torch
from torch import Tensor
from torch.library import Library
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion.layers.prims_ts_fp8 import (
    prepare_prims_ts_fp8_weights,
    prims_ts_fp8_linear,
    prims_ts_fp8_qkv_qknorm_rope,
    prims_ts_fp8_quantize,
    prims_ts_fp8_swiglu,
    register_prims_ts_fp8_config_provider,
)
from vllm_omni.diffusion.layers.prims_ts_nvfp4 import (
    _CACHE_ID_ATTR as _NVFP4_CACHE_ID_ATTR,
)
from vllm_omni.diffusion.layers.prims_ts_nvfp4 import (
    _nvfp4_linear_fake,
    _nvfp4_linear_impl,
    _nvfp4_qkv_qknorm_rope_fake,
    _nvfp4_qkv_qknorm_rope_impl,
    _nvfp4_quantize_fake,
    _nvfp4_swiglu_fake,
    _nvfp4_swiglu_impl,
    get_prims_ts_nvfp4_cache_id,
    nvfp4_quantize_128x4,
    prepare_prims_ts_nvfp4_weights,
    prims_ts_nvfp4_linear,
    prims_ts_nvfp4_qkv_qknorm_rope,
    register_prims_ts_nvfp4_config_provider,
    register_prims_ts_nvfp4_layer,
)
from vllm_omni.diffusion.models.flux3.flux3_kernel_adapter import (
    Flux3BlockKernelPlan,
    Flux3KernelAdapter,
)
from vllm_omni.diffusion.models.flux3.flux3_kernel_adapter_factory import (
    Flux3PrimsTSConfig,
    Flux3PrimsTSMode,
    Flux3PrimsTSSupport,
)
from vllm_omni.diffusion.models.flux3.flux3_prims_ts_config import (
    flux3_fp8_gemm_config,
    flux3_nvfp4_gemm_config,
    prims_ts_autotune_enabled,
)

if TYPE_CHECKING:
    from torch import nn

Fp8Activation: TypeAlias = tuple[torch.Tensor, torch.Tensor]
NvFp4Activation: TypeAlias = tuple[torch.Tensor, torch.Tensor]
QuantizedActivation: TypeAlias = Fp8Activation | NvFp4Activation
PreparedInput: TypeAlias = tuple[torch.Tensor, QuantizedActivation | None]

_FP8_WEIGHT_BUFFER = "_flux3_prims_ts_fp8_weight"
_FP8_SCALE_BUFFER = "_flux3_prims_ts_fp8_scale"
_NVFP4_WEIGHT_BUFFER = "_flux3_prims_ts_nvfp4_weight"
_NVFP4_SCALE_BUFFER = "_flux3_prims_ts_nvfp4_scale"
_NVFP4_QKV_SCALE_BUFFER = "_flux3_prims_ts_nvfp4_qkv_scale"
_NVFP4_QKV_RESCALE_BUFFER = "_flux3_prims_ts_nvfp4_qkv_rescale"
_NVFP4_INPUT_SCALE_INV_BUFFER = "_flux3_prims_ts_nvfp4_input_scale_inv"

# Per-layer NVFP4 scalars (cache id, input and weight global scales, host-float input
# quantization scale), looked up by the prepared weight's current address. The
# flux3_nvfp4_* ops below take only tensors and resolve these inside the op, so
# torch.compile sees no per-layer Python constants and one compiled graph serves every
# block (passing them as arguments guards each block on its own values and exhausts the
# recompile limit). Layerwise offload swaps weight storage in place, so every hit is
# checked against the layer's current weight and the index is rebuilt from the
# registered layers when addresses change.
_NVFP4_LAYERS: list[ref] = []
_NVFP4_BY_ADDRESS: dict[int, ref] = {}


def _register_nvfp4_operands(layer: nn.Module) -> None:
    if getattr(layer, _NVFP4_CACHE_ID_ATTR, None) is None:
        return  # not configured as a PrimsTS block projection
    if not any(r() is layer for r in _NVFP4_LAYERS):
        _NVFP4_LAYERS.append(ref(layer))
    _NVFP4_BY_ADDRESS[layer._buffers[_NVFP4_WEIGHT_BUFFER].data_ptr()] = ref(layer)


def _nvfp4_layer_for(weight: Tensor) -> nn.Module | None:
    address = weight.data_ptr()
    owner_ref = _NVFP4_BY_ADDRESS.get(address)
    owner = owner_ref() if owner_ref is not None else None
    if owner is not None and owner._buffers[_NVFP4_WEIGHT_BUFFER].data_ptr() == address:
        return owner
    return None


def _nvfp4_owner(weight: Tensor) -> nn.Module:
    owner = _nvfp4_layer_for(weight)
    if owner is None:
        _NVFP4_LAYERS[:] = [r for r in _NVFP4_LAYERS if r() is not None]
        _NVFP4_BY_ADDRESS.clear()
        for r in _NVFP4_LAYERS:
            layer = r()
            _NVFP4_BY_ADDRESS[layer._buffers[_NVFP4_WEIGHT_BUFFER].data_ptr()] = r
        owner = _nvfp4_layer_for(weight)
        if owner is None:
            raise RuntimeError("Flux3 PrimsTS NVFP4 weight was not registered after loading")
    return owner


def _nvfp4_operands(weight: Tensor) -> tuple[int, float, float]:
    owner = _nvfp4_owner(weight)
    return (
        get_prims_ts_nvfp4_cache_id(owner),
        owner._flux3_prims_ts_nvfp4_input_global_scale,
        owner._flux3_prims_ts_nvfp4_weight_global_scale,
    )


def _flux3_nvfp4_quantize_impl(activation: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    # Quantize with the scale of the layer that owns ``weight``, passed as a host float.
    return nvfp4_quantize_128x4(activation, _nvfp4_owner(weight)._flux3_prims_ts_nvfp4_host_input_scale_inv)


def _flux3_nvfp4_quantize_fake(activation: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    return _nvfp4_quantize_fake(activation, weight)


def _flux3_nvfp4_linear_impl(
    activation: Tensor, activation_scale: Tensor, weight: Tensor, weight_scale: Tensor, packed_input: bool
) -> Tensor:
    cache_id, input_global_scale, weight_global_scale = _nvfp4_operands(weight)
    # A packed SwiGLU output carries unit global scale.
    input_global_scale = 1.0 if packed_input else input_global_scale
    return _nvfp4_linear_impl(
        activation, activation_scale, weight, weight_scale, input_global_scale, weight_global_scale, cache_id
    )


def _flux3_nvfp4_linear_fake(
    activation: Tensor, activation_scale: Tensor, weight: Tensor, weight_scale: Tensor, packed_input: bool
) -> Tensor:
    return _nvfp4_linear_fake(activation, activation_scale, weight, weight_scale, 1.0, 1.0, 0)


def _flux3_nvfp4_swiglu_impl(
    activation: Tensor,
    activation_scale: Tensor,
    hidden_states: Tensor,
    quantized_by: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    cache_id, input_global_scale, weight_global_scale = _nvfp4_operands(weight)
    if _nvfp4_operands(quantized_by)[1] != input_global_scale:
        # The shared activation was quantized with the QKV checkpoint scale.
        activation, activation_scale = _flux3_nvfp4_quantize_impl(hidden_states, weight)
    return _nvfp4_swiglu_impl(
        activation, activation_scale, weight, weight_scale, input_global_scale, weight_global_scale, cache_id
    )


def _flux3_nvfp4_swiglu_fake(
    activation: Tensor,
    activation_scale: Tensor,
    hidden_states: Tensor,
    quantized_by: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    return _nvfp4_swiglu_fake(activation, activation_scale, weight, weight_scale, 1.0, 1.0, 0)


def _flux3_nvfp4_qkv_qknorm_rope_impl(
    activation: Tensor,
    activation_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    qkv_scale: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    cos_sin: Tensor,
) -> Tensor:
    cache_id, input_global_scale, weight_global_scale = _nvfp4_operands(weight)
    return _nvfp4_qkv_qknorm_rope_impl(
        activation,
        activation_scale,
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        qkv_scale,
        q_norm,
        k_norm,
        cos_sin,
        None,
        cache_id,
    )


def _flux3_nvfp4_qkv_qknorm_rope_fake(
    activation: Tensor,
    activation_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    qkv_scale: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    cos_sin: Tensor,
) -> Tensor:
    return _nvfp4_qkv_qknorm_rope_fake(
        activation, activation_scale, weight, weight_scale, 1.0, 1.0, qkv_scale, q_norm, k_norm, cos_sin, None, 0
    )


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
for _name, _impl, _fake in (
    ("flux3_nvfp4_quantize", _flux3_nvfp4_quantize_impl, _flux3_nvfp4_quantize_fake),
    ("flux3_nvfp4_linear", _flux3_nvfp4_linear_impl, _flux3_nvfp4_linear_fake),
    ("flux3_nvfp4_swiglu", _flux3_nvfp4_swiglu_impl, _flux3_nvfp4_swiglu_fake),
    ("flux3_nvfp4_qkv_qknorm_rope", _flux3_nvfp4_qkv_qknorm_rope_impl, _flux3_nvfp4_qkv_qknorm_rope_fake),
):
    if not hasattr(torch.ops.vllm_omni, _name):
        direct_register_custom_op(
            op_name=_name, op_func=_impl, fake_impl=_fake, mutates_args=[], target_lib=_OMNI_OP_LIB
        )
_FLUX3_NVFP4_QUANTIZE_OP = torch.ops.vllm_omni.flux3_nvfp4_quantize
_FLUX3_NVFP4_LINEAR_OP = torch.ops.vllm_omni.flux3_nvfp4_linear
_FLUX3_NVFP4_SWIGLU_OP = torch.ops.vllm_omni.flux3_nvfp4_swiglu
_FLUX3_NVFP4_QKV_QKNORM_ROPE_OP = torch.ops.vllm_omni.flux3_nvfp4_qkv_qknorm_rope


def _fp8_config(x: Tensor, weight: Tensor, epilogue: str, *, head_dim: int | None = None):
    return flux3_fp8_gemm_config(x, weight, epilogue, head_dim=head_dim)


def _fp8_config_provider(x: Tensor, weight: Tensor, epilogue: str, head_dim: int | None):
    return _fp8_config(x, weight, epilogue, head_dim=head_dim)


register_prims_ts_fp8_config_provider("flux3", _fp8_config_provider)


def _nvfp4_config_provider(
    device: torch.device,
    epilogue: str,
    k: int,
    n: int,
    m: int,
    head_dim: int | None,
    qkv_scale: bool,
):
    return flux3_nvfp4_gemm_config(
        device,
        epilogue=epilogue,
        k=k,
        n=n,
        m=m,
        head_dim=head_dim,
        qkv_scale=qkv_scale,
    )


register_prims_ts_nvfp4_config_provider("flux3", _nvfp4_config_provider, cache_key_provider=prims_ts_autotune_enabled)


def _register_prepared_buffer(layer: nn.Module, name: str, value: Tensor) -> None:
    if name in layer._buffers:
        layer._buffers[name] = value
        layer._non_persistent_buffers_set.add(name)
    else:
        layer.register_buffer(name, value, persistent=False)


def prepare_fp8_layer(layer: nn.Module, layout: Literal["linear", "swiglu"]) -> None:
    """Attach prepared PrimsTS FP8 operands to the layer after loading."""
    packed_weight, packed_scale = prepare_prims_ts_fp8_weights(
        layer.weight,
        layer.weight_scale,
        layout=layout,
    )
    _register_prepared_buffer(layer, _FP8_WEIGHT_BUFFER, packed_weight)
    _register_prepared_buffer(layer, _FP8_SCALE_BUFFER, packed_scale)


class _PrimsTSFp8LinearMethod(LinearMethodBase):
    """Add preparation and local GEMM dispatch to the checkpoint method."""

    def __init__(self, base_method: LinearMethodBase, layout: Literal["linear", "swiglu"]) -> None:
        self.base_method = base_method
        self.layout = layout

    def create_weights(self, *args, **kwargs) -> None:
        self.base_method.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self.base_method.process_weights_after_loading(layer)
        prepare_fp8_layer(layer, self.layout)

    def apply(self, layer: nn.Module, x: Tensor, bias: Tensor | None = None) -> Tensor:
        if self.layout == "linear" and bias is None:
            return _fp8_linear(layer, x)
        return self.base_method.apply(layer, x, bias)


def _nvfp4_global_scale(layer: nn.Module) -> float:
    scale = getattr(layer, "weight_global_scale", None)
    if scale is None or scale.numel() != 1:
        raise ValueError("PrimsTS NVFP4 requires one weight global scale per projection")
    return float(scale.detach().cpu().item())


def _register_nvfp4_input_scale(layer: nn.Module) -> None:
    # Reuse the scalar operands prepared by ModelOpt after loading.
    input_scale = float(layer.input_global_scale.detach().cpu().item())
    if not math.isfinite(input_scale) or input_scale <= 0:
        raise ValueError(f"PrimsTS NVFP4 requires a positive finite input global scale, got {input_scale}")
    _register_prepared_buffer(layer, _NVFP4_INPUT_SCALE_INV_BUFFER, layer.input_global_scale_inv.detach().reshape(1))
    layer._flux3_prims_ts_nvfp4_input_global_scale = input_scale
    layer._flux3_prims_ts_nvfp4_host_input_scale_inv = float(layer.input_global_scale_inv.detach().cpu().item())


def prepare_nvfp4_layer(layer: nn.Module, layout: Literal["linear", "swiglu"]) -> None:
    """Prepare a ModelOpt NVFP4 projection without changing its base weights."""
    weight = getattr(layer, "weight")
    packed_weight, packed_scales = prepare_prims_ts_nvfp4_weights(
        weight,
        layer.weight_scale,
        layout=layout,
    )

    # These are inference-only representations consumed by a DLPack-backed
    # kernel; do not retain an autograd graph from the checkpoint parameters.
    packed_weight = packed_weight.detach()
    packed_scales = packed_scales.detach()
    _register_prepared_buffer(layer, _NVFP4_WEIGHT_BUFFER, packed_weight)
    _register_prepared_buffer(layer, _NVFP4_SCALE_BUFFER, packed_scales)
    _register_nvfp4_input_scale(layer)
    layer._flux3_prims_ts_nvfp4_weight_global_scale = _nvfp4_global_scale(layer)
    _register_nvfp4_operands(layer)


class _PrimsTSNvFp4LinearMethod(LinearMethodBase):
    """Retain native checkpoint processing and derive PrimsTS NVFP4 buffers."""

    def __init__(
        self,
        base_method: LinearMethodBase,
        layout: Literal["linear", "swiglu", "qkv"],
    ) -> None:
        self.base_method = base_method
        self.layout = layout

    def create_weights(self, *args, **kwargs) -> None:
        self.base_method.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        qkv_scales = None
        if self.layout == "qkv":
            qkv_scales = layer.weight_scale_2.data.to(torch.float32).detach().clone()
            common_scale = qkv_scales.max()
            if qkv_scales.numel() != 3 or not torch.isfinite(common_scale) or common_scale <= 0:
                raise ValueError("PrimsTS fused NVFP4 QKV requires three positive finite global scales")
            # The native method stores one shared multiplier. The PrimsTS QKV
            # op receives the original per-projection values separately.
            layer.weight_scale_2.data = common_scale.expand_as(qkv_scales).clone()
            # Per-output-channel correction for that shared multiplier, used only when
            # the checkpoint method runs this layer (LoRA-wrapped blocks).
            rescale = (qkv_scales / common_scale).repeat_interleave(layer.output_size_per_partition // 3)
        self.base_method.process_weights_after_loading(layer)
        if self.layout == "qkv":
            packed_weight, packed_scale = prepare_prims_ts_nvfp4_weights(
                layer.weight,
                layer.weight_scale,
                layout="linear",
            )
            _register_prepared_buffer(layer, _NVFP4_WEIGHT_BUFFER, packed_weight)
            _register_prepared_buffer(layer, _NVFP4_SCALE_BUFFER, packed_scale)
            _register_prepared_buffer(layer, _NVFP4_QKV_SCALE_BUFFER, qkv_scales)
            _register_prepared_buffer(layer, _NVFP4_QKV_RESCALE_BUFFER, rescale)
            _register_nvfp4_input_scale(layer)
            # qkv_scale replaces the GEMM multiplier, so include the activation scale.
            qkv_scales.mul_(layer._flux3_prims_ts_nvfp4_input_global_scale)
            layer._flux3_prims_ts_nvfp4_weight_global_scale = _nvfp4_global_scale(layer)
            _register_nvfp4_operands(layer)
        else:
            prepare_nvfp4_layer(layer, self.layout)

    def apply(self, layer: nn.Module, x: Tensor | NvFp4Activation, bias: Tensor | None = None) -> Tensor:
        if self.layout == "linear":
            if isinstance(x, tuple) and bias is not None:
                raise ValueError("Packed NVFP4 activations do not support a separate bias")
            if bias is None:
                return _nvfp4_linear(layer, x)
        out = self.base_method.apply(layer, x, bias)
        if self.layout == "qkv":
            # The checkpoint method dequantizes Q, K and V with their largest multiplier.
            out = (out.float() * layer._buffers[_NVFP4_QKV_RESCALE_BUFFER]).to(out.dtype)
        return out


def _prepared_fp8_weight(layer: nn.Module) -> tuple[Tensor, Tensor]:
    try:
        return layer._buffers[_FP8_WEIGHT_BUFFER], layer._buffers[_FP8_SCALE_BUFFER]
    except KeyError as exc:
        weight = getattr(layer, "weight", None)
        quant_method = getattr(layer, "quant_method", None)
        raise RuntimeError(
            "Flux3 PrimsTS FP8 weights were not prepared after loading. "
            f"weight_dtype={getattr(weight, 'dtype', None)}, "
            f"weight_device={getattr(weight, 'device', None)}, "
            f"quant_method={type(quant_method).__name__}, "
            f"available_buffers={sorted(getattr(layer, '_buffers', {}))}."
        ) from exc


def _fp8_linear(layer: nn.Module, x: Tensor | Fp8Activation) -> Tensor:
    weight, scale = _prepared_fp8_weight(layer)
    output_shape = x.shape if isinstance(x, Tensor) else None
    out = prims_ts_fp8_linear(x, weight, scale, config_provider="flux3")
    return out if output_shape is None else out.view(*output_shape[:-1], out.shape[-1])


def _fp8_swiglu(layer: nn.Module, x: Tensor | Fp8Activation) -> Tensor:
    weight, scale = _prepared_fp8_weight(layer)
    output_shape = x.shape if isinstance(x, Tensor) else None
    out = prims_ts_fp8_swiglu(x, weight, scale, config_provider="flux3")
    return out if output_shape is None else out.view(*output_shape[:-1], out.shape[-1])


def _rope_cache(rope: Tensor) -> Tensor:
    # One table row per token; the fused ops then use identity positions, which they
    # build once per row count instead of an arange per block.
    if rope.ndim != 3:
        raise ValueError(f"Flux3 PrimsTS expects packed [B, T, 2D] RoPE, got shape {tuple(rope.shape)}")
    return rope.reshape(-1, rope.shape[-1]).float().contiguous()


def _norm_weight(norm: nn.Module, name: str) -> Tensor:
    module = getattr(norm, name)
    weight = getattr(module, "weight", None)
    return weight if weight is not None else getattr(module, "scale")


def _fp8_qkv_qknorm_rope(
    layer: nn.Module,
    norm: nn.Module,
    x: Tensor | Fp8Activation,
    rope: Tensor,
) -> Tensor:
    weight, scale = _prepared_fp8_weight(layer)
    q_norm = _norm_weight(norm, "query_norm").detach()
    k_norm = _norm_weight(norm, "key_norm").detach()
    cache = _rope_cache(rope)
    head_dim = q_norm.numel()
    heads = weight.shape[0] // (3 * head_dim)
    out = prims_ts_fp8_qkv_qknorm_rope(
        x,
        weight,
        scale,
        q_norm,
        k_norm,
        cache,
        None,
        num_q_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        config_provider="flux3",
    )
    return out if isinstance(x, tuple) else out.view(*x.shape[:-1], out.shape[-1])


def _prepared_nvfp4_operands(layer: nn.Module) -> tuple[Tensor, Tensor, float, float]:
    try:
        weight = layer._buffers[_NVFP4_WEIGHT_BUFFER]
        scale = layer._buffers[_NVFP4_SCALE_BUFFER]
        input_global_scale = layer._flux3_prims_ts_nvfp4_input_global_scale
        weight_global_scale = layer._flux3_prims_ts_nvfp4_weight_global_scale
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(
            "Flux3 PrimsTS NVFP4 weights were not prepared after loading. "
            f"quant_method={type(getattr(layer, 'quant_method', None)).__name__}, "
            f"available_buffers={sorted(getattr(layer, '_buffers', {}))}."
        ) from exc
    return weight, scale, input_global_scale, weight_global_scale


def _flux3_nvfp4_quantize(x: Tensor, layer: nn.Module) -> NvFp4Activation:
    """Quantize ``x`` with ``layer``'s checkpoint input scale."""
    x_2d = x.reshape(-1, x.shape[-1]).contiguous().detach()
    return _FLUX3_NVFP4_QUANTIZE_OP(x_2d, layer._buffers[_NVFP4_WEIGHT_BUFFER])


def _nvfp4_linear(
    layer: nn.Module,
    x: Tensor | NvFp4Activation,
    *,
    packed_input_global_scale: float = 1.0,
) -> Tensor:
    if packed_input_global_scale == 1.0:
        weight, weight_scale = layer._buffers[_NVFP4_WEIGHT_BUFFER], layer._buffers[_NVFP4_SCALE_BUFFER]
        if isinstance(x, Tensor):
            packed, scale = _flux3_nvfp4_quantize(x, layer)
        else:
            packed, scale = x
        out = _FLUX3_NVFP4_LINEAR_OP(packed, scale, weight, weight_scale, not isinstance(x, Tensor))
        return out if not isinstance(x, Tensor) else out.view(*x.shape[:-1], out.shape[-1])
    weight, weight_scale, input_global_scale, weight_global_scale = _prepared_nvfp4_operands(layer)
    cache_id = get_prims_ts_nvfp4_cache_id(layer)
    output_shape = x.shape if isinstance(x, Tensor) else None
    if isinstance(x, Tensor):
        effective_input_global_scale = input_global_scale
        input_scale_inv = layer._buffers[_NVFP4_INPUT_SCALE_INV_BUFFER]
    else:
        effective_input_global_scale = packed_input_global_scale
        input_scale_inv = None
    out = prims_ts_nvfp4_linear(
        x,
        weight,
        weight_scale,
        effective_input_global_scale,
        weight_global_scale,
        cache_id,
        input_scale_inv=input_scale_inv,
    )
    return out if output_shape is None else out.view(*output_shape[:-1], out.shape[-1])


def _nvfp4_swiglu_shared(
    layer: nn.Module,
    activation: NvFp4Activation,
    hidden_states: Tensor,
    quantized_by: nn.Module,
) -> NvFp4Activation:
    packed, scale = activation
    return _FLUX3_NVFP4_SWIGLU_OP(
        packed,
        scale,
        hidden_states.reshape(-1, hidden_states.shape[-1]),
        quantized_by._buffers[_NVFP4_WEIGHT_BUFFER],
        layer._buffers[_NVFP4_WEIGHT_BUFFER],
        layer._buffers[_NVFP4_SCALE_BUFFER],
    )


def _nvfp4_qkv_qknorm_rope(
    qkv: nn.Module,
    norm: nn.Module,
    x: Tensor | NvFp4Activation,
    rope: Tensor,
) -> Tensor:
    if isinstance(x, tuple):
        packed, scale = x
        return _FLUX3_NVFP4_QKV_QKNORM_ROPE_OP(
            packed,
            scale,
            qkv._buffers[_NVFP4_WEIGHT_BUFFER],
            qkv._buffers[_NVFP4_SCALE_BUFFER],
            qkv._buffers[_NVFP4_QKV_SCALE_BUFFER],
            _norm_weight(norm, "query_norm").detach(),
            _norm_weight(norm, "key_norm").detach(),
            _rope_cache(rope),
        )
    weight, weight_scale, input_global_scale, weight_global_scale = _prepared_nvfp4_operands(qkv)
    qkv_scale = qkv._buffers[_NVFP4_QKV_SCALE_BUFFER]
    q_norm = _norm_weight(norm, "query_norm").detach()
    k_norm = _norm_weight(norm, "key_norm").detach()
    input_scale_inv = qkv._buffers[_NVFP4_INPUT_SCALE_INV_BUFFER] if isinstance(x, Tensor) else None
    cache = _rope_cache(rope)
    out = prims_ts_nvfp4_qkv_qknorm_rope(
        x,
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        qkv_scale,
        q_norm,
        k_norm,
        cache,
        None,
        get_prims_ts_nvfp4_cache_id(qkv),
        input_scale_inv=input_scale_inv,
    )
    return out


def _is_target(prefix: str) -> bool:
    return prefix.startswith("content_mode_blocks.image.") or prefix.startswith("single_blocks.")


_PROJECTIONS = ("qkv", "attn_out", "mlp_in", "mlp_out")


def _has_lora(block: nn.Module) -> bool:
    # LoRA, at startup or when an adapter is first loaded, wraps projections and keeps
    # the prepared layer under ``base_layer``. The fused paths call PrimsTS on the
    # prepared buffers directly and would skip the adapter, so such a block calls its
    # projection modules instead: the base layers still run their PrimsTS (or
    # checkpoint) GEMM and the wrappers add the adapter.
    return any(isinstance(getattr(block, name, None), BaseLayerWithLoRA) for name in _PROJECTIONS)


class Flux3PrimsTSKernelAdapter(Flux3KernelAdapter):
    """PrimsTS projections for supported Flux3 image and joint blocks."""

    backend = "prims_ts"

    def uses_fused_qkv(self, prefix: str, head_dim: int) -> bool:
        return _is_target(prefix) and head_dim == 128

    def __init__(
        self,
        qk_norm_rope,
        *,
        prims_ts_config: Flux3PrimsTSConfig,
        prims_ts_support: Flux3PrimsTSSupport,
        quant_config: object,
        quant_format: Literal["fp8", "nvfp4"],
    ) -> None:
        super().__init__(qk_norm_rope)
        self.prims_ts_config = prims_ts_config
        self.prims_ts_support = prims_ts_support
        self.quant_config = quant_config
        self.quant_format = quant_format
        self._configured_blocks = 0

    def _linear_supported(self, layer: nn.Module) -> bool:
        method = getattr(layer, "quant_method", None)
        if self.quant_format == "fp8":
            expected_method = getattr(self.quant_config, "LinearMethodCls", None)
            method_supported = isinstance(expected_method, type) and isinstance(method, expected_method)
            alignment = 128
        else:
            method_supported = getattr(method, "quant_config", None) is self.quant_config
            alignment = 256
        return bool(
            method_supported
            and (self.quant_format == "nvfp4" or getattr(method, "out_dtype", None) is torch.bfloat16)
            and getattr(layer, "bias", None) is None
            and getattr(layer, "input_size_per_partition", 0) > 0
            and getattr(layer, "output_size_per_partition", 0) > 0
            and layer.input_size_per_partition % alignment == 0
            and (self.quant_format != "nvfp4" or layer.output_size_per_partition % 512 == 0)
        )

    def _wrap(
        self,
        layer: nn.Module,
        layout: Literal["linear", "swiglu", "qkv"],
    ) -> None:
        if self.quant_format == "nvfp4":
            register_prims_ts_nvfp4_layer(layer, config_provider="flux3")
        method = getattr(layer, "quant_method")
        wrapper_type = _PrimsTSFp8LinearMethod if self.quant_format == "fp8" else _PrimsTSNvFp4LinearMethod
        if isinstance(method, wrapper_type):
            if method.layout != layout:
                raise RuntimeError(f"Conflicting PrimsTS layouts for one layer: {method.layout} and {layout}")
            return
        if self.quant_format == "fp8":
            layer.quant_method = _PrimsTSFp8LinearMethod(method, layout)
        else:
            layer.quant_method = _PrimsTSNvFp4LinearMethod(method, layout)

    def configure_block(self, block: nn.Module, prefix: str) -> Flux3BlockKernelPlan:
        if not _is_target(prefix):
            return Flux3BlockKernelPlan()

        if self.quant_format == "nvfp4" and not self.uses_fused_qkv(prefix, block.head_dim):
            return Flux3BlockKernelPlan()

        if not hasattr(block.qkv, "weight"):
            raise RuntimeError("PrimsTS requires fused QKV projections")
        qkv_linear = self._linear_supported(block.qkv)
        attention_output = self._linear_supported(block.attn_out)
        mlp_swiglu = self._linear_supported(block.mlp_in)
        mlp_output = self._linear_supported(block.mlp_out)
        if self.quant_format == "nvfp4" and mlp_swiglu and not mlp_output:
            # NVFP4 SwiGLU returns a packed activation that only the PrimsTS
            # NVFP4 down-projection can consume. Keep this pair on the default
            # path when the down-projection is excluded or unsupported.
            mlp_swiglu = False
        num_heads = getattr(block, "num_heads_local", 0)
        num_kv_heads = getattr(block, "num_kv_heads_local", 0)
        qkv_qknorm_rope = bool(
            qkv_linear
            and hasattr(block.qkv, "weight")
            and getattr(block, "head_dim", 0) == 128
            and num_heads > 0
            and num_heads == num_kv_heads
            and getattr(getattr(block, "norm", None), "query_norm", None) is not None
        )
        plan = Flux3BlockKernelPlan(
            qkv_linear=qkv_linear,
            qkv_qknorm_rope=qkv_qknorm_rope,
            attention_output=attention_output,
            mlp_swiglu=mlp_swiglu,
            mlp_output=mlp_output,
        )
        if not plan.enabled:
            return plan

        if qkv_linear:
            layout = "qkv" if self.quant_format == "nvfp4" and qkv_qknorm_rope else "linear"
            self._wrap(block.qkv, layout)
        if attention_output:
            self._wrap(block.attn_out, "linear")
        if mlp_swiglu:
            self._wrap(block.mlp_in, "swiglu")
        if mlp_output:
            self._wrap(block.mlp_out, "linear")
        self._configured_blocks += 1
        return plan

    def prepare_input(self, block: nn.Module, hidden_states: Tensor) -> PreparedInput:
        plan = block.kernel_plan
        if _has_lora(block) or not (plan.qkv_linear or plan.mlp_swiglu):
            activation = None
        elif self.quant_format == "fp8":
            activation = prims_ts_fp8_quantize(hidden_states)
        else:
            # Quantized once with the first consumer's checkpoint scale; the SwiGLU op
            # requantizes inside when its own scale differs.
            owner = block.qkv if plan.qkv_linear else block.mlp_in
            activation = _flux3_nvfp4_quantize(hidden_states, owner)
        return hidden_states, activation

    def qkv_qknorm_rope(
        self,
        block: nn.Module,
        prepared_input: PreparedInput,
        rope: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden_states, activation = prepared_input
        if _has_lora(block):
            return super().qkv_qknorm_rope(block, hidden_states, rope)
        plan = block.kernel_plan
        if plan.qkv_qknorm_rope:
            if self.quant_format == "fp8":
                if activation is None:
                    raise RuntimeError("PrimsTS fused QKV requires a prepared quantized activation")
                qkv = _fp8_qkv_qknorm_rope(block.qkv, block.norm, activation, rope)
                qkv = qkv.view(*hidden_states.shape[:-1], qkv.shape[-1])
            else:
                qkv = _nvfp4_qkv_qknorm_rope(
                    block.qkv, block.norm, activation if activation is not None else hidden_states, rope
                )
                qkv = qkv.view(*hidden_states.shape[:-1], qkv.shape[-1])
            q, k, v = qkv.chunk(3, dim=-1)
            return (
                q.unflatten(-1, (block.num_heads_local, block.head_dim)),
                k.unflatten(-1, (block.num_kv_heads_local, block.head_dim)),
                v.unflatten(-1, (block.num_kv_heads_local, block.head_dim)),
            )
        if plan.qkv_linear:
            if self.quant_format == "fp8":
                if activation is None:
                    raise RuntimeError("PrimsTS QKV requires a prepared quantized activation")
                qkv = _fp8_linear(block.qkv, activation)
            else:
                qkv = _nvfp4_linear(
                    block.qkv,
                    activation if activation is not None else hidden_states,
                    packed_input_global_scale=block.qkv._flux3_prims_ts_nvfp4_input_global_scale,
                )
            qkv = qkv.view(*hidden_states.shape[:-1], qkv.shape[-1])
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.unflatten(-1, (block.num_heads_local, block.head_dim))
            k = k.unflatten(-1, (block.num_kv_heads_local, block.head_dim))
            v = v.unflatten(-1, (block.num_kv_heads_local, block.head_dim))
            q, k = self._qk_norm_rope(q, k, v, block.norm, rope)
            return q, k, v
        return super().qkv_qknorm_rope(block, hidden_states, rope)

    def attention_output(self, block: nn.Module, hidden_states: Tensor) -> Tensor:
        # Enter through RowParallelLinear so its tensor-parallel reduction is
        # preserved. The wrapped quant method replaces only the local GEMM.
        return block.attn_out(hidden_states)

    def mlp(self, block: nn.Module, prepared_input: PreparedInput) -> Tensor:
        hidden_states, activation = prepared_input
        if _has_lora(block):
            return super().mlp(block, hidden_states)
        plan = block.kernel_plan
        if plan.mlp_swiglu:
            if self.quant_format == "fp8":
                if activation is None:
                    raise RuntimeError("PrimsTS fused SwiGLU requires a prepared quantized activation")
                mlp_activation = _fp8_swiglu(block.mlp_in, activation)
                mlp_activation = mlp_activation.view(*hidden_states.shape[:-1], mlp_activation.shape[-1])
            else:
                quantized_by = block.qkv if plan.qkv_linear else block.mlp_in
                mlp_activation = _nvfp4_swiglu_shared(block.mlp_in, activation, hidden_states, quantized_by)
        else:
            mlp_activation = block.mlp_act(block.mlp_in(hidden_states))
        # As above, RowParallelLinear remains responsible for TP reduction.
        mlp_out = block.mlp_out(mlp_activation)
        if self.quant_format == "nvfp4" and plan.mlp_swiglu:
            # The packed input is flattened to [rows, features] for the GEMM.
            # Restore the original token dimensions after RowParallelLinear has
            # performed its usual tensor-parallel reduction.
            return mlp_out.view(*hidden_states.shape[:-1], mlp_out.shape[-1])
        return mlp_out

    def finalize(self) -> None:
        if self.prims_ts_config.mode is Flux3PrimsTSMode.REQUIRED and self._configured_blocks == 0:
            raise RuntimeError(
                "Flux3 PrimsTS was required but no compatible "
                f"{self.quant_format.upper()} transformer blocks were found"
            )


__all__ = ["Flux3PrimsTSKernelAdapter", "prepare_fp8_layer", "prepare_nvfp4_layer"]
