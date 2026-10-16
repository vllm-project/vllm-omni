# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable NVFP4 custom operations backed by FlashInfer PrimsTS GEMMs."""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias
from weakref import ref

import torch
from torch import Tensor
from torch.library import Library
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion.layers.prims_ts_fp8 import token_positions

if TYPE_CHECKING:
    from flashinfer.prims_ts.gemm import PrimsTsGemmConfig
    from torch import nn

ConfigProvider = Callable[[torch.device, str, int, int, int, int | None, bool], "PrimsTsGemmConfig | None"]
CacheKeyProvider = Callable[[], Hashable]
NvFp4Activation: TypeAlias = tuple[Tensor, Tensor]

_CONFIG_PROVIDERS: dict[str, tuple[ConfigProvider, CacheKeyProvider | None]] = {}
_LAYER_REFS: dict[int, ref] = {}
_CACHE_IDS = itertools.count(1)
_CACHE_LOCK = threading.RLock()
_CACHE_ID_ATTR = "_prims_ts_nvfp4_cache_id"
_CONFIG_PROVIDER_ATTR = "_prims_ts_nvfp4_config_provider"
_PREPARED_ATTR = "_prims_ts_nvfp4_prepared"
_PREPARED_CACHE_SIZE = 8


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def nvfp4_scales_to_linear(scales: Tensor) -> Tensor:
    """Unpack logical NVFP4 scales from the serialized 128x4 byte layout."""
    original_ndim = scales.ndim
    if original_ndim == 2:
        scales = scales.unsqueeze(0)
    if scales.ndim != 3:
        raise ValueError(f"NVFP4 scales must be 2D or 3D, got {tuple(scales.shape)}")
    batches, rows, cols = scales.shape
    padded_rows, padded_cols = _round_up(rows, 128), _round_up(cols, 4)
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((batches, padded_rows, padded_cols), dtype=scales.dtype, device=scales.device)
        padded[:, :rows, :cols] = scales
        scales = padded
    linear = scales.reshape(batches, padded_rows // 128, padded_cols // 4, 32, 4, 4)
    linear = linear.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(batches, padded_rows, padded_cols)
    linear = linear[:, :rows, :cols]
    return linear.squeeze(0) if original_ndim == 2 else linear


def nvfp4_scales_to_128x4(scales: Tensor) -> Tensor:
    """Pack logical block scales into PrimsTS' contiguous 128x4 byte layout."""
    original_ndim = scales.ndim
    if original_ndim == 2:
        scales = scales.unsqueeze(0)
    if scales.ndim != 3:
        raise ValueError(f"NVFP4 scales must be 2D or 3D, got {tuple(scales.shape)}")
    batches, rows, cols = scales.shape
    padded_rows, padded_cols = _round_up(rows, 128), _round_up(cols, 4)
    padded = torch.zeros((batches, padded_rows, padded_cols), dtype=scales.dtype, device=scales.device)
    padded[:, :rows, :cols] = scales
    packed = padded.reshape(batches, padded_rows // 128, 4, 32, padded_cols // 4, 4)
    packed = packed.permute(0, 1, 4, 3, 2, 5).contiguous()
    return packed.view(torch.uint8).reshape(-1).contiguous()


def prepare_prims_ts_nvfp4_weights(
    weight: Tensor,
    weight_scale: Tensor,
    *,
    layout: Literal["linear", "swiglu"],
) -> tuple[Tensor, Tensor]:
    """Prepare packed NVFP4 weights and scales for a PrimsTS projection.

    For SwiGLU, source rows must be ``[gate | up]``; they are reordered to
    adjacent ``[up, gate]`` pairs expected by the fused PrimsTS epilogue.
    """
    if weight.dtype is not torch.uint8:
        raise TypeError(f"PrimsTS NVFP4 preparation expected packed uint8 weights, got {weight.dtype}")
    if weight_scale.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"PrimsTS NVFP4 preparation expected float8_e4m3fn scales, got {weight_scale.dtype}")

    if layout == "swiglu":
        rows = weight.shape[0]
        if rows % 2:
            raise ValueError(f"PrimsTS NVFP4 SwiGLU requires an even output width, got {rows}")
        half = rows // 2
        packed_weight = torch.stack((weight[half:], weight[:half]), dim=1).reshape_as(weight).contiguous()
        logical_scales = nvfp4_scales_to_linear(weight_scale)
        logical_scales = torch.stack((logical_scales[half:], logical_scales[:half]), dim=1).reshape_as(logical_scales)
        packed_scale = nvfp4_scales_to_128x4(logical_scales)
    else:
        packed_weight = weight.detach()
        packed_scale = weight_scale.detach().view(torch.uint8).reshape(-1).contiguous()
    return packed_weight.detach(), packed_scale.detach()


def prepare_prims_ts_nvfp4_qkv_weights(
    weights: Sequence[Tensor],
    weight_scales: Sequence[Tensor],
    weight_global_scales: Sequence[float],
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Assemble three NVFP4 projections and normalize their global scales.

    Returns concatenated packed weights, packed block scales, complete
    per-projection global-scale factors, and the shared maximum global scale.
    FlashInfer's ``qkv_scale`` replaces the GEMM's shared global-scale
    multiplier, so it must contain each projection's full scale, not just its
    ratio to the shared maximum.
    """
    if len(weights) != 3 or len(weight_scales) != 3 or len(weight_global_scales) != 3:
        raise ValueError("Fused NVFP4 QKV preparation requires exactly three projections")
    packed_weight = torch.cat([weight.detach() for weight in weights], dim=0).contiguous()
    logical_scales = torch.cat([nvfp4_scales_to_linear(scale.detach()) for scale in weight_scales], dim=0)
    global_scales = torch.tensor(weight_global_scales, device=packed_weight.device, dtype=torch.float32)
    common_scale = float(global_scales.max().detach().cpu().item())
    if common_scale <= 0:
        raise ValueError(f"PrimsTS NVFP4 QKV requires positive weight global scales, got {global_scales}")
    return (
        packed_weight,
        nvfp4_scales_to_128x4(logical_scales),
        global_scales,
        common_scale,
    )


def register_prims_ts_nvfp4_config_provider(
    name: str,
    provider: ConfigProvider,
    *,
    cache_key_provider: CacheKeyProvider | None = None,
) -> None:
    """Register optional model-specific tactic policy for NVFP4 GEMMs.

    The cache-key callback should return values that invalidate prepared
    callables when the provider's tactic policy changes (for example, whether
    autotuning is enabled).
    """
    if not name:
        raise ValueError("Config provider name must not be empty")
    policy = (provider, cache_key_provider)
    existing = _CONFIG_PROVIDERS.setdefault(name, policy)
    if existing != policy:
        raise ValueError(f"PrimsTS NVFP4 config provider {name!r} is already registered")


def register_prims_ts_nvfp4_layer(layer: nn.Module, *, config_provider: str | None = None) -> int:
    """Register a layer that owns prepared NVFP4 GEMM callables.

    Call this while building/configuring a model, before its first forward.
    Each model can register its own tactic provider or omit one to use the
    FlashInfer defaults.
    """
    provider_name = config_provider or ""
    if provider_name and provider_name not in _CONFIG_PROVIDERS:
        raise ValueError(f"PrimsTS NVFP4 config provider {provider_name!r} is not registered")

    cache_id = getattr(layer, _CACHE_ID_ATTR, None)
    existing_provider = getattr(layer, _CONFIG_PROVIDER_ATTR, provider_name)
    if cache_id is not None:
        if existing_provider != provider_name:
            raise ValueError("NVFP4 cache owner was registered with a different config provider")
        return cache_id

    with _CACHE_LOCK:
        cache_id = getattr(layer, _CACHE_ID_ATTR, None)
        if cache_id is None:
            cache_id = next(_CACHE_IDS)

            def remove_owner(_owner_ref: ref, *, key: int = cache_id) -> None:
                with _CACHE_LOCK:
                    _LAYER_REFS.pop(key, None)

            _LAYER_REFS[cache_id] = ref(layer, remove_owner)
            setattr(layer, _CACHE_ID_ATTR, cache_id)
            setattr(layer, _CONFIG_PROVIDER_ATTR, provider_name)
        elif getattr(layer, _CONFIG_PROVIDER_ATTR, "") != provider_name:
            raise ValueError("NVFP4 cache owner was registered with a different config provider")
    return cache_id


def get_prims_ts_nvfp4_cache_id(layer: nn.Module) -> int:
    """Return a layer's ID for passing to a PrimsTS NVFP4 custom op."""
    cache_id = getattr(layer, _CACHE_ID_ATTR, None)
    if cache_id is None:
        raise RuntimeError("PrimsTS NVFP4 layer was not registered during model configuration")
    return cache_id


def _fp4_scale_numel(rows: int, k: int) -> int:
    return ((rows + 127) // 128) * 128 * ((k // 16 + 3) // 4) * 4


def _gemm():
    from flashinfer.prims_ts import gemm

    return gemm


def _owner_and_policy(cache_id: int):
    owner_ref = _LAYER_REFS.get(cache_id)
    owner = owner_ref() if owner_ref is not None else None
    if owner is None:
        raise RuntimeError("PrimsTS NVFP4 prepared operator has no live owning layer")
    provider_name = getattr(owner, _CONFIG_PROVIDER_ATTR, "")
    policy = _CONFIG_PROVIDERS.get(provider_name) if provider_name else None
    return owner, policy


def _cache_policy_key(policy: tuple[ConfigProvider, CacheKeyProvider | None] | None) -> Hashable | None:
    return policy[1]() if policy is not None and policy[1] is not None else None


def _make_config(
    policy: tuple[ConfigProvider, CacheKeyProvider | None] | None,
    device: torch.device,
    epilogue: str,
    k: int,
    n: int,
    m: int,
    head_dim: int | None = None,
    qkv_scale: bool = False,
) -> PrimsTsGemmConfig | None:
    return policy[0](device, epilogue, k, n, m, head_dim, qkv_scale) if policy is not None else None


def _prepare_nvfp4_linear(
    cache_id: int,
    kind: Literal["linear", "swiglu"],
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    rows: int,
):
    owner, policy = _owner_and_policy(cache_id)
    key = (
        rows,
        weight.data_ptr(),
        weight._version,
        weight_scale.data_ptr(),
        weight_scale._version,
        input_global_scale,
        weight_global_scale,
        _cache_policy_key(policy),
    )
    prepared_cache = getattr(owner, _PREPARED_ATTR, None)
    kind_cache = prepared_cache.get(kind) if prepared_cache is not None else None
    cached = kind_cache.get(key) if kind_cache is not None else None
    if cached is not None:
        return cached

    # Only serialize cache misses; steady-state execution is a lock-free lookup.
    with _CACHE_LOCK:
        prepared_cache = getattr(owner, _PREPARED_ATTR, None)
        if prepared_cache is None:
            prepared_cache = {}
            setattr(owner, _PREPARED_ATTR, prepared_cache)
        kind_cache = prepared_cache.setdefault(kind, OrderedDict())
        cached = kind_cache.get(key)
        if cached is not None:
            kind_cache.move_to_end(key)
            return cached
        config = _make_config(
            policy,
            weight.device,
            kind,
            k=weight.shape[1] * 2,
            n=weight.shape[0],
            m=rows,
        )
        if kind == "linear":
            prepared = _gemm().prepare_fp4_linear(
                weight.detach(), weight_scale.detach(), input_global_scale, weight_global_scale, config=config
            )
        else:
            # Preserve packed output for the following NVFP4 down projection.
            output_quant_scale = torch.ones(1, device=weight.device, dtype=torch.float32)
            prepared = _gemm().prepare_fp4_linear_swiglu(
                weight.detach(),
                weight_scale.detach(),
                input_global_scale,
                weight_global_scale,
                out_dtype=torch.uint8,
                output_quant_scale=output_quant_scale,
                config=config,
            )
        kind_cache[key] = prepared
        if len(kind_cache) > _PREPARED_CACHE_SIZE:
            kind_cache.popitem(last=False)
        return prepared


def _prepare_nvfp4_qkv(
    cache_id: int,
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    qkv_scale: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    rows: int,
):
    owner, policy = _owner_and_policy(cache_id)
    key = (
        rows,
        weight.data_ptr(),
        weight._version,
        weight_scale.data_ptr(),
        weight_scale._version,
        input_global_scale,
        weight_global_scale,
        qkv_scale.data_ptr(),
        qkv_scale._version,
        q_norm.data_ptr(),
        q_norm._version,
        k_norm.data_ptr(),
        k_norm._version,
        _cache_policy_key(policy),
    )
    prepared_cache = getattr(owner, _PREPARED_ATTR, None)
    kind_cache = prepared_cache.get("qkv_qknorm_rope") if prepared_cache is not None else None
    cached = kind_cache.get(key) if kind_cache is not None else None
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        prepared_cache = getattr(owner, _PREPARED_ATTR, None)
        if prepared_cache is None:
            prepared_cache = {}
            setattr(owner, _PREPARED_ATTR, prepared_cache)
        kind_cache = prepared_cache.setdefault("qkv_qknorm_rope", OrderedDict())
        cached = kind_cache.get(key)
        if cached is not None:
            kind_cache.move_to_end(key)
            return cached
        head_dim = q_norm.numel()
        heads = weight.shape[0] // (3 * head_dim)
        config = _make_config(
            policy,
            weight.device,
            "qkv_qknorm_rope",
            k=weight.shape[1] * 2,
            n=weight.shape[0],
            m=rows,
            head_dim=head_dim,
            qkv_scale=True,
        )
        prepared = _gemm().prepare_fp4_qkv_qknorm_rope(
            weight.detach(),
            weight_scale.detach(),
            input_global_scale,
            weight_global_scale,
            q_norm.detach(),
            k_norm.detach(),
            num_q_heads=heads,
            num_kv_heads=heads,
            head_dim=head_dim,
            is_neox=False,
            qkv_scale=qkv_scale.detach(),
            config=config,
        )
        kind_cache[key] = prepared
        if len(kind_cache) > _PREPARED_CACHE_SIZE:
            kind_cache.popitem(last=False)
        return prepared


def _nvfp4_quantize_impl(activation: Tensor, input_scale_inv: Tensor) -> tuple[Tensor, Tensor]:
    return nvfp4_quantize_128x4(activation, input_scale_inv)


def nvfp4_quantize_128x4(activation: Tensor, input_scale_inv: Tensor | float) -> tuple[Tensor, Tensor]:
    """Eager NVFP4 quantization to the 128x4 scale layout.

    A host-float ``input_scale_inv`` saves FlashInfer's per-call tensor handling
    (about 6 us of host time per call); both forms give identical output.
    """
    from flashinfer import SfLayout, nvfp4_quantize

    packed, scales = nvfp4_quantize(
        activation.detach(),
        input_scale_inv.detach() if isinstance(input_scale_inv, Tensor) else input_scale_inv,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    return packed.view(torch.uint8).contiguous(), scales.view(torch.uint8).reshape(-1).contiguous()


def _nvfp4_quantize_fake(activation: Tensor, input_scale_inv: Tensor) -> tuple[Tensor, Tensor]:
    del input_scale_inv
    rows, k = activation.shape
    packed = torch.empty((rows, k // 2), device=activation.device, dtype=torch.uint8)
    scales = torch.empty((_fp4_scale_numel(rows, k),), device=activation.device, dtype=torch.uint8)
    return packed, scales


def _nvfp4_linear_impl(
    activation: Tensor,
    activation_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    cache_id: int,
) -> Tensor:
    prepared = _prepare_nvfp4_linear(
        cache_id, "linear", weight, weight_scale, input_global_scale, weight_global_scale, activation.shape[0]
    )
    return prepared(activation.detach(), activation_scale.detach())


def _nvfp4_linear_fake(
    activation: Tensor,
    activation_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    cache_id: int,
) -> Tensor:
    del activation_scale, weight_scale, input_global_scale, weight_global_scale, cache_id
    return torch.empty((activation.shape[0], weight.shape[0]), device=activation.device, dtype=torch.bfloat16)


def _nvfp4_swiglu_impl(
    activation: Tensor,
    activation_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    cache_id: int,
) -> tuple[Tensor, Tensor]:
    prepared = _prepare_nvfp4_linear(
        cache_id, "swiglu", weight, weight_scale, input_global_scale, weight_global_scale, activation.shape[0]
    )
    return prepared(activation.detach(), activation_scale.detach())


def _nvfp4_swiglu_fake(
    activation: Tensor,
    activation_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    cache_id: int,
) -> tuple[Tensor, Tensor]:
    del activation_scale, input_global_scale, weight_global_scale, cache_id
    rows = activation.shape[0]
    logical_width = weight.shape[0] // 2
    packed = torch.empty((rows, logical_width // 2), device=activation.device, dtype=torch.uint8)
    scales = torch.empty((_fp4_scale_numel(rows, logical_width),), device=activation.device, dtype=torch.uint8)
    del weight_scale
    return packed, scales


def _nvfp4_qkv_qknorm_rope_impl(
    activation: Tensor,
    activation_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    qkv_scale: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    cos_sin: Tensor,
    positions: Tensor | None,
    cache_id: int,
) -> Tensor:
    if positions is None:
        positions = token_positions(activation.shape[0], activation.device)
    prepared = _prepare_nvfp4_qkv(
        cache_id,
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        qkv_scale,
        q_norm,
        k_norm,
        activation.shape[0],
    )
    return prepared(
        activation.detach(),
        activation_scale.detach(),
        cos_sin=cos_sin.detach(),
        positions=positions.detach(),
    )


def _nvfp4_qkv_qknorm_rope_fake(
    activation: Tensor,
    activation_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    qkv_scale: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    cos_sin: Tensor,
    positions: Tensor | None,
    cache_id: int,
) -> Tensor:
    del activation_scale, weight_scale, input_global_scale, weight_global_scale, qkv_scale, q_norm, k_norm, cos_sin
    del positions, cache_id
    return torch.empty((activation.shape[0], weight.shape[0]), device=activation.device, dtype=torch.bfloat16)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "prims_ts_nvfp4_quantize"):
    direct_register_custom_op(
        op_name="prims_ts_nvfp4_quantize",
        op_func=_nvfp4_quantize_impl,
        fake_impl=_nvfp4_quantize_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )
if not hasattr(torch.ops.vllm_omni, "prims_ts_nvfp4_linear"):
    direct_register_custom_op(
        op_name="prims_ts_nvfp4_linear",
        op_func=_nvfp4_linear_impl,
        fake_impl=_nvfp4_linear_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )
if not hasattr(torch.ops.vllm_omni, "prims_ts_nvfp4_swiglu"):
    direct_register_custom_op(
        op_name="prims_ts_nvfp4_swiglu",
        op_func=_nvfp4_swiglu_impl,
        fake_impl=_nvfp4_swiglu_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )
if not hasattr(torch.ops.vllm_omni, "prims_ts_nvfp4_qkv_qknorm_rope"):
    direct_register_custom_op(
        op_name="prims_ts_nvfp4_qkv_qknorm_rope",
        op_func=_nvfp4_qkv_qknorm_rope_impl,
        fake_impl=_nvfp4_qkv_qknorm_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


_QUANTIZE_OP = torch.ops.vllm_omni.prims_ts_nvfp4_quantize
_LINEAR_OP = torch.ops.vllm_omni.prims_ts_nvfp4_linear
_SWIGLU_OP = torch.ops.vllm_omni.prims_ts_nvfp4_swiglu
_QKV_QKNORM_ROPE_OP = torch.ops.vllm_omni.prims_ts_nvfp4_qkv_qknorm_rope


def prims_ts_nvfp4_quantize(activation: Tensor, input_scale_inv: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize activations to packed NVFP4 using the 128x4 layout."""
    activation_2d = activation.reshape(-1, activation.shape[-1]).contiguous().detach()
    return _QUANTIZE_OP(activation_2d, input_scale_inv)


def _prepare_nvfp4_activation(
    activation: Tensor | NvFp4Activation,
    input_scale_inv: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if isinstance(activation, tuple):
        if len(activation) != 2:
            raise ValueError("prequantized NVFP4 activation must be a (packed_activation, scale) pair")
        if input_scale_inv is not None:
            raise ValueError("input_scale_inv must be omitted for a prequantized activation pair")
        packed, scale = activation
        if packed.dtype is not torch.uint8:
            raise TypeError("prequantized NVFP4 activation must use packed uint8 data")
        if packed.ndim != 2:
            raise ValueError("prequantized activation must have shape [M, K]")
        return packed, scale
    if activation.dtype is torch.uint8:
        raise TypeError("pass packed NVFP4 input as a (packed_activation, scale) pair")
    if input_scale_inv is None:
        raise ValueError("input_scale_inv is required when passing floating-point activations")
    return prims_ts_nvfp4_quantize(activation, input_scale_inv)


def prims_ts_nvfp4_linear(
    activation: Tensor | NvFp4Activation,
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    cache_id: int,
    *,
    input_scale_inv: Tensor | None = None,
) -> Tensor:
    """Run NVFP4 linear, quantizing tensor input or reusing a packed pair.

    Pass a floating-point tensor with ``input_scale_inv`` to quantize on demand,
    or a ``(packed_activation, scale)`` pair to reuse a previous quantization.
    """
    activation, activation_scale = _prepare_nvfp4_activation(activation, input_scale_inv)
    return _LINEAR_OP(
        activation,
        activation_scale,
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        cache_id,
    )


def prims_ts_nvfp4_swiglu(
    activation: Tensor | NvFp4Activation,
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    cache_id: int,
    *,
    input_scale_inv: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run NVFP4 linear+SwiGLU with tensor or prequantized-pair input."""
    activation, activation_scale = _prepare_nvfp4_activation(activation, input_scale_inv)
    return _SWIGLU_OP(
        activation,
        activation_scale,
        weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        cache_id,
    )


def prims_ts_nvfp4_qkv_qknorm_rope(
    activation: Tensor | NvFp4Activation,
    weight: Tensor,
    weight_scale: Tensor,
    input_global_scale: float,
    weight_global_scale: float,
    qkv_scale: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    cos_sin: Tensor,
    positions: Tensor | None,
    cache_id: int,
    *,
    input_scale_inv: Tensor | None = None,
) -> Tensor:
    """Run fused NVFP4 QKV+QK norm+RoPE with tensor or pair input.

    ``positions=None`` means token i uses ``cos_sin`` row i.
    """
    activation, activation_scale = _prepare_nvfp4_activation(activation, input_scale_inv)
    if cos_sin.ndim != 2 or cos_sin.shape[0] != activation.shape[0]:
        raise ValueError("cos_sin must have one row per activation token")
    if positions is not None and positions.shape != (activation.shape[0],):
        raise ValueError("positions must have one index per activation token")
    return _QKV_QKNORM_ROPE_OP(
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
        positions,
        cache_id,
    )


__all__ = [
    "NvFp4Activation",
    "get_prims_ts_nvfp4_cache_id",
    "nvfp4_scales_to_128x4",
    "nvfp4_scales_to_linear",
    "prepare_prims_ts_nvfp4_qkv_weights",
    "prepare_prims_ts_nvfp4_weights",
    "prims_ts_nvfp4_linear",
    "prims_ts_nvfp4_qkv_qknorm_rope",
    "prims_ts_nvfp4_quantize",
    "prims_ts_nvfp4_swiglu",
    "register_prims_ts_nvfp4_config_provider",
    "register_prims_ts_nvfp4_layer",
]
