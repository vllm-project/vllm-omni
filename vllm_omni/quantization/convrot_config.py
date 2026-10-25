# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ConvRot online quantization for diffusion transformers.

ConvRot (arXiv:2512.03673) splits the input dimension of each linear into
groups of ``rotation_group_size`` channels and rotates every group of both the
activation and the weight by the same normalized regular Hadamard matrix ``H``
(Kronecker powers of a regular 4x4 Hadamard). ``H`` is orthogonal, so
``X W^T = sum_i (X_i H)(W_i H)^T`` is unchanged, while row- and column-wise
outliers are spread across their group before per-token/per-channel
quantization. The group-wise rotation is a reshape plus one small matmul, so it
costs linear rather than quadratic time in the hidden size.

Weights are rotated in FP32 once at load time; activations are rotated on every
forward. ``convrot_int8`` quantizes both to INT8 (dynamic per-token
activations, per-output-channel weights) through vLLM's INT8 kernels.
``convrot_int4`` quantizes both to INT4 with the same granularity, keeps the
weight packed two values per byte, and runs the product on INT8 tensor cores,
which compute the INT4 values exactly on GPUs without INT4 MMA (e.g. Hopper).
Its ``int8_layers`` keep sensitive layers on rotated INT8, the paper's mixed
W4A4 + INT8 setting.

``convrot_mxfp4`` keeps the same rotation and stores both operands in the
block-32 MXFP4 format: E2M1 values with one E8M0 scale per 32 input channels.
``convrot_mxfp8`` uses the same block layout with E4M3 values. vLLM selects the
MXFP linear kernel for the current device and shape. ``convrot_mxfp4`` can
additionally route sensitive layers to MXFP8 through ``mxfp8_layers``.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear import (
    init_mxfp4_linear_kernel,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization._copy_missing_attrs import copy_missing_attrs as _copy_missing_attrs
from vllm_omni.quantization.int8_config import (
    BaseInt8LinearMethod,
    DiffusionInt8Config,
    Int8OnlineLinearMethod,
    LazyWeightMixin,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

DEFAULT_ROTATION_GROUP_SIZE = 256
INT4_MAX = 7

# Eq. (9) of the paper: every row and column sums to 2 = sqrt(4).
_REGULAR_HADAMARD_4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))


@functools.cache
def regular_hadamard(size: int) -> torch.Tensor:
    """Normalized regular Hadamard matrix of a power-of-four ``size`` (FP32, CPU).

    Its entries are ``+-2^-k``, so they are exact in BF16 and FP16 as well.
    """
    if size < 4 or 4 ** round(math.log(size, 4)) != size:
        raise ValueError(f"ConvRot rotation group size must be a power of four >= 4, got {size}")
    base = torch.tensor(_REGULAR_HADAMARD_4, dtype=torch.float32)
    matrix = base
    while matrix.shape[0] < size:
        matrix = torch.kron(matrix, base)
    return matrix / math.sqrt(size)


def rotation_group_size(in_features: int, requested: int) -> int | None:
    """The largest power-of-four group <= ``requested`` that tiles ``in_features``, if any."""
    size = requested
    while size >= 4:
        if in_features % size == 0:
            return size
        size //= 4
    return None


def rotate(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """Rotate each contiguous group of the last dimension by ``hadamard``."""
    size = hadamard.shape[0]
    return torch.matmul(x.reshape(*x.shape[:-1], x.shape[-1] // size, size), hadamard).reshape(x.shape)


def quantize_int4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row INT4 values (held in INT8) and FP32 scales of shape ``[rows, 1]``."""
    x = x.float()
    scale = (x.abs().amax(dim=-1, keepdim=True) / INT4_MAX).clamp_(min=torch.finfo(torch.float32).tiny)
    return torch.round(x / scale).clamp_(-INT4_MAX - 1, INT4_MAX).to(torch.int8), scale


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack INT4 values in INT8 storage two per byte, low nibble first, along the last dimension."""
    if values.shape[-1] % 2:
        raise ValueError("INT4 packing needs an even last dimension")
    nibbles = values.to(torch.uint8) & 0xF
    return nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4`, sign-extending each nibble to INT8."""
    low = (packed & 0xF).to(torch.int8)
    high = (packed >> 4).to(torch.int8)
    values = torch.stack(((low ^ 8) - 8, (high ^ 8) - 8), dim=-1)
    return values.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _materialized_weight(layer: Module) -> torch.Tensor:
    """The loaded BF16/FP16 weight; a never-loaded meta weight gets dummy values (dummy load format)."""
    if layer.weight.device == torch.device("meta"):
        weight = ModelWeightParameter(
            data=torch.empty_like(layer.weight, device=layer._load_device),
            input_dim=1,
            output_dim=0,
            weight_loader=layer.weight.weight_loader,
        )
        _copy_missing_attrs(layer.weight, weight)
        layer.register_parameter("weight", weight)
        initialize_single_dummy_weight(layer.weight)
    return layer.weight


def _rotated_weight(layer: Module, requested_group: int) -> torch.Tensor:
    """The FP32 weight rotated in place of its input groups; registers the activation rotation.

    A layer whose input no power-of-four group tiles keeps no rotation.
    """
    weight = _materialized_weight(layer).data.float()
    group = rotation_group_size(layer.input_size_per_partition, requested_group)
    hadamard = None if group is None else regular_hadamard(group).to(device=weight.device)
    layer.register_buffer(
        "convrot_hadamard", None if hadamard is None else hadamard.to(layer.orig_dtype), persistent=False
    )
    return weight if hadamard is None else rotate(weight, hadamard)


class DiffusionConvRotConfig(QuantizationConfig):
    """Online ConvRot quantization of every non-ignored diffusion linear layer."""

    weight_bits: int = 8

    def __init__(
        self,
        rotation_group_size: int = DEFAULT_ROTATION_GROUP_SIZE,
        ignored_layers: list[str] | None = None,
        int8_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        regular_hadamard(rotation_group_size)  # validate eagerly
        if int8_layers and self.weight_bits != 4:
            raise ValueError("int8_layers selects rotated INT8 layers inside convrot_int4 only")
        self.rotation_group_size = rotation_group_size
        self.ignored_layers = ignored_layers or []
        self.int8_layers = int8_layers or []
        # The INT8 layers reuse vLLM's dynamic per-token INT8 kernel setup.
        self._int8 = DiffusionInt8Config(ignored_layers=self.ignored_layers)

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)
        self.int8_layers = hf_to_vllm_mapper.apply_list(self.int8_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionConvRotConfig:
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None) or cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        kwargs: dict[str, Any] = {
            "rotation_group_size": cls.get_from_keys_or(config, ["rotation_group_size"], DEFAULT_ROTATION_GROUP_SIZE),
            "ignored_layers": ignored_layers,
        }
        if cls.weight_bits == 4:
            kwargs["int8_layers"] = cls.get_from_keys_or(config, ["int8_layers"], None)
        return cls(**kwargs)

    def _matches(self, prefix: str, layers: list[str]) -> bool:
        """Whether ``prefix`` contains one of ``layers`` as whole path components.

        ``attn.out_proj`` selects that module in every block and ``token_refiner``
        every layer below it; ``proj`` does not select ``out_proj``. A fused
        layer matches through its own name or any shard name.
        """
        if not layers:
            return False
        name = prefix.rsplit(".", 1)[-1]
        base = prefix[: len(prefix) - len(name)]
        candidates = [prefix, *(base + shard for shard in self.packed_modules_mapping.get(name, ()))]
        return any(f".{layer.strip('.')}." in f".{candidate}." for candidate in candidates for layer in layers)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if self._matches(prefix, self.ignored_layers):
            return UnquantizedLinearMethod()
        if not current_omni_platform.is_cuda():
            raise NotImplementedError("ConvRot quantization currently runs on CUDA only.")
        if self.weight_bits == 8 or self._matches(prefix, self.int8_layers):
            return ConvRotInt8OnlineLinearMethod(self)
        return ConvRotInt4OnlineLinearMethod(self)


class DiffusionConvRotInt8Config(DiffusionConvRotConfig):
    weight_bits = 8

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "convrot_int8"  # type: ignore[return-value]


class DiffusionConvRotInt4Config(DiffusionConvRotConfig):
    weight_bits = 4

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "convrot_int4"  # type: ignore[return-value]


class DiffusionConvRotMxfp4Config(DiffusionConvRotConfig):
    """ConvRot rotated W4A4 MXFP4, optionally mixed with W8A8 MXFP8 layers."""

    weight_bits = 4

    def __init__(
        self,
        rotation_group_size: int = DEFAULT_ROTATION_GROUP_SIZE,
        ignored_layers: list[str] | None = None,
        int8_layers: list[str] | None = None,
        mxfp8_layers: list[str] | None = None,
    ) -> None:
        if int8_layers:
            raise ValueError(
                "convrot_mxfp4 does not support int8_layers; use mxfp8_layers for MXFP8 "
                "fallback or ignored_layers for BF16"
            )
        super().__init__(
            rotation_group_size=rotation_group_size,
            ignored_layers=ignored_layers,
            int8_layers=None,
        )
        self.mxfp8_layers = mxfp8_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "convrot_mxfp4"  # type: ignore[return-value]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        self.mxfp8_layers = hf_to_vllm_mapper.apply_list(self.mxfp8_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionConvRotMxfp4Config:
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None) or cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(
            rotation_group_size=cls.get_from_keys_or(
                config, ["rotation_group_size"], DEFAULT_ROTATION_GROUP_SIZE
            ),
            ignored_layers=ignored_layers,
            mxfp8_layers=cls.get_from_keys_or(config, ["mxfp8_layers"], None),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if self._matches(prefix, self.ignored_layers):
            return UnquantizedLinearMethod()
        if self._matches(prefix, self.mxfp8_layers):
            return ConvRotMxfp8OnlineLinearMethod(self)
        return ConvRotMxfp4OnlineLinearMethod(self)


class DiffusionConvRotMxfp8Config(DiffusionConvRotConfig):
    """ConvRot rotated W8A8 with native block-32 MXFP8 tensor-core GEMM."""

    weight_bits = 8

    def __init__(
        self,
        rotation_group_size: int = DEFAULT_ROTATION_GROUP_SIZE,
        ignored_layers: list[str] | None = None,
        int8_layers: list[str] | None = None,
    ) -> None:
        if int8_layers:
            raise ValueError("convrot_mxfp8 does not support int8_layers; use ignored_layers for BF16 fallback")
        super().__init__(
            rotation_group_size=rotation_group_size,
            ignored_layers=ignored_layers,
            int8_layers=None,
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "convrot_mxfp8"  # type: ignore[return-value]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionConvRotMxfp8Config:
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None) or cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(
            rotation_group_size=cls.get_from_keys_or(
                config, ["rotation_group_size"], DEFAULT_ROTATION_GROUP_SIZE
            ),
            ignored_layers=ignored_layers,
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if self._matches(prefix, self.ignored_layers):
            return UnquantizedLinearMethod()
        return ConvRotMxfp8OnlineLinearMethod(self)


class ConvRotInt8OnlineLinearMethod(Int8OnlineLinearMethod):
    """Rotated W8A8: per-output-channel INT8 weights and dynamic per-token INT8 activations."""

    def __init__(self, quant_config: DiffusionConvRotConfig):
        super().__init__(quant_config._int8)
        self.rotation_group_size = quant_config.rotation_group_size

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        layer.weight.data = _rotated_weight(layer, self.rotation_group_size).to(layer.orig_dtype)
        super().process_weights_after_loading(layer)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.convrot_hadamard is not None:
            x = rotate(x, layer.convrot_hadamard)
        return super().apply(layer, x, bias)


class ConvRotInt4OnlineLinearMethod(LazyWeightMixin, BaseInt8LinearMethod):
    """Rotated W4A4: packed per-output-channel INT4 weights and dynamic per-token INT4 activations."""

    def __init__(self, quant_config: DiffusionConvRotConfig):
        super().__init__(quant_config._int8)
        self.rotation_group_size = quant_config.rotation_group_size

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        qweight, scale = quantize_int4(_rotated_weight(layer, self.rotation_group_size))
        replace_parameter(layer, "weight", torch.nn.Parameter(pack_int4(qweight), requires_grad=False))
        replace_parameter(layer, "weight_scale", torch.nn.Parameter(scale.t().contiguous(), requires_grad=False))
        layer._already_called_process_weights_after_loading = True

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.convrot_hadamard is not None:
            x = rotate(x, layer.convrot_hadamard)
        rows = x.reshape(-1, x.shape[-1])
        qx, x_scale = quantize_int4(rows)
        weight = unpack_int4(layer.weight)
        # INT4 values are exact in INT8, so INT8 tensor cores compute the W4A4 product.
        out = ops.cutlass_scaled_mm(qx, weight.t(), x_scale, layer.weight_scale, x.dtype, bias)
        return out.reshape(*x.shape[:-1], -1)


class ConvRotMxfp4OnlineLinearMethod(LazyWeightMixin, BaseInt8LinearMethod):
    """Rotated block-32 MXFP4 using vLLM's device-selected linear kernel.

    ConvRot rotates the weights and activations. Weight conversion and the
    dtype/layout required by the activation are delegated to vLLM's MXFP4
    kernel factory and the selected kernel.
    """

    def __init__(self, quant_config: DiffusionConvRotMxfp4Config):
        super().__init__(quant_config._int8)
        self.rotation_group_size = quant_config.rotation_group_size
        self.kernel = None

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if layer.orig_dtype != torch.bfloat16:
            raise ValueError("convrot_mxfp4 requires bfloat16 activations")

        weight = _rotated_weight(layer, self.rotation_group_size).to(torch.bfloat16).contiguous()
        if weight.shape[-1] % 32:
            raise ValueError(f"convrot_mxfp4 needs an input size divisible by 32, got {weight.shape[-1]}")

        self.kernel = init_mxfp4_linear_kernel(activation_quant_key=kMxfp4Dynamic)
        qweight, weight_scale = mxfp4_quantize(weight)
        layer.params_dtype = layer.orig_dtype
        replace_parameter(layer, "weight", torch.nn.Parameter(qweight, requires_grad=False))
        replace_parameter(layer, "weight_scale", torch.nn.Parameter(weight_scale, requires_grad=False))
        self.kernel.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.convrot_hadamard is not None:
            x = rotate(x, layer.convrot_hadamard)
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        out = self.kernel.apply_weights(layer, rows, bias)
        return out.reshape(*x.shape[:-1], out.shape[-1])


class ConvRotMxfp8OnlineLinearMethod(LazyWeightMixin, BaseInt8LinearMethod):
    """Rotated block-32 MXFP8 using vLLM's device-selected linear kernel."""

    def __init__(self, quant_config: DiffusionConvRotConfig):
        super().__init__(quant_config._int8)
        self.rotation_group_size = quant_config.rotation_group_size
        self.kernel = None

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if layer.orig_dtype != torch.bfloat16:
            raise ValueError("convrot_mxfp8 requires bfloat16 activations")

        weight = _rotated_weight(layer, self.rotation_group_size).to(torch.bfloat16).contiguous()
        if weight.shape[-1] % 32:
            raise ValueError(f"convrot_mxfp8 needs an input size divisible by 32, got {weight.shape[-1]}")

        self.kernel = init_mxfp8_linear_kernel(weight_shape=tuple(weight.shape))
        qweight, weight_scale = mxfp8_e4m3_quantize(weight)
        layer.params_dtype = layer.orig_dtype
        replace_parameter(layer, "weight", torch.nn.Parameter(qweight, requires_grad=False))
        replace_parameter(layer, "weight_scale", torch.nn.Parameter(weight_scale, requires_grad=False))
        self.kernel.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.convrot_hadamard is not None:
            x = rotate(x, layer.convrot_hadamard)
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        out = self.kernel.apply_weights(layer, rows, bias)
        return out.reshape(*x.shape[:-1], out.shape[-1])
