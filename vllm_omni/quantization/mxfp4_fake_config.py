# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit MXFP4 QDQ recipes with floating storage and GEMM."""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


_MXFP4_BLOCK_SIZE = 32
_E8M0_MIN_EXPONENT = -127
_E8M0_MAX_EXPONENT = 127
_MXFP4_E2M1_MAX = 6.0
_MXFP4_TORCH_CHUNK_ROWS = 64
_MXFP4_C7_SCALE_MAX = 7.0
_MXFP4_C7_MIN_PRIVATE_EXPONENT = 0.0
_MXFP4_C7_SCALE_FACTOR = 2.0
_MXFP4_C7_INV_SCALE_FACTOR = 0.5
_MXFP4_C7_EPSILON = 1.17e-38


def _mxfp4_c7_qdq_chunk(chunk: torch.Tensor) -> torch.Tensor:
    """Delivered C7 arithmetic, including input-dtype intermediate rounding."""
    shape = tuple(chunk.shape)
    blocks = chunk.reshape(*shape[:-1], shape[-1] // _MXFP4_BLOCK_SIZE, _MXFP4_BLOCK_SIZE)
    max_val = blocks.abs().amax(dim=-1, keepdim=True)
    shared_exp = torch.ceil(torch.log2(max_val.clamp(min=_MXFP4_C7_EPSILON) * (1.0 / _MXFP4_C7_SCALE_MAX))).clamp(
        _E8M0_MIN_EXPONENT, _E8M0_MAX_EXPONENT
    )
    scale = torch.exp2(shared_exp)
    normalized_values = blocks * torch.exp2(-shared_exp)
    private_exp = torch.floor(torch.log2(normalized_values.abs().clamp(min=_MXFP4_C7_EPSILON))).clamp(
        min=_MXFP4_C7_MIN_PRIVATE_EXPONENT
    )
    scaled = normalized_values * torch.exp2(-private_exp) * _MXFP4_C7_SCALE_FACTOR
    rounded = torch.sign(scaled) * torch.floor(scaled.abs() + 0.5)
    elements = (rounded * _MXFP4_C7_INV_SCALE_FACTOR * torch.exp2(private_exp)).clamp(-_MXFP4_E2M1_MAX, _MXFP4_E2M1_MAX)
    restored = elements * scale
    return restored.reshape(shape)


def _mxfp4_c7_qdq(x: torch.Tensor) -> torch.Tensor:
    """Per-32-block C7 QDQ without silently promoting BF16 to FP32.

    FP16 cannot represent the delivered epsilon or the E8M0 scale range.
    Keep its input unsupported rather than changing the supplied arithmetic.
    The row chunk size is an execution choice; groups never cross K rows.
    """
    if x.ndim == 0 or x.shape[-1] == 0 or x.shape[-1] % _MXFP4_BLOCK_SIZE:
        raise ValueError("mxfp4_fake requires a nonempty K dimension divisible by 32, including TP shards")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("mxfp4_c7 requires BF16 or FP32 input; the delivered epsilon is not representable in FP16")
    if x.numel() == 0:
        return torch.empty_like(x)
    orig_dtype = x.dtype
    flat = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(flat)
    for start in range(0, flat.shape[0], _MXFP4_TORCH_CHUNK_ROWS):
        stop = min(start + _MXFP4_TORCH_CHUNK_ROWS, flat.shape[0])
        out[start:stop].copy_(_mxfp4_c7_qdq_chunk(flat[start:stop]))
    return out.reshape(x.shape).to(orig_dtype)


def fake_quantize_mxfp4(x: torch.Tensor, recipe: str = "e2m1_rne") -> torch.Tensor:
    """Group 32 on K, power-of-two scales, E2M1 nearest-even, then dequantize.

    The e2m1_rne reference uses exponent=clamp(floor(log2(amax))-2, -127, 127).
    C7 is an explicit alternate recipe. Nonfinite inputs follow each recipe.
    The E2M1 reference bounds temporary FP32 memory by chunking.
    No scale tensors are serialized. This is inference-only, without an STE.
    """
    if recipe == "mxfp4_c7":
        return _mxfp4_c7_qdq(x)
    if recipe != "e2m1_rne":
        raise ValueError("mxfp4_fake recipe must be e2m1_rne or mxfp4_c7")
    if x.ndim == 0 or x.shape[-1] == 0 or x.shape[-1] % 32:
        raise ValueError("mxfp4_fake requires a nonempty K dimension divisible by 32, including TP shards")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("mxfp4_fake requires floating-point input")
    groups = x.reshape(-1, 32)
    result = torch.empty_like(groups)
    for start in range(0, groups.shape[0], 65536):
        values = groups[start : start + 65536].float()
        magnitude = values.abs()
        maximum = magnitude.amax(dim=-1, keepdim=True)
        exponent = (maximum.clamp_min(torch.finfo(torch.float32).tiny).log2().floor() - 2).clamp(-127, 127)
        scale = torch.exp2(exponent)
        scaled = magnitude / scale
        rounded = torch.zeros_like(scaled)
        # E2M1 positive values: 0, .5, 1, 1.5, 2, 3, 4, 6.
        for index, (midpoint, upper) in enumerate(
            ((0.25, 0.5), (0.75, 1.0), (1.25, 1.5), (1.75, 2.0), (2.5, 3.0), (3.5, 4.0), (5.0, 6.0))
        ):
            select = scaled > midpoint
            if index % 2:
                select = select | (scaled == midpoint)
            rounded = torch.where(select, upper, rounded)
        restored = rounded * scale * values.sign()
        restored = torch.where(values.isfinite(), restored, values)
        result[start : start + 65536] = restored.to(x.dtype)
    return result.reshape(x.shape)


class DiffusionMXFP4FakeConfig(QuantizationConfig):
    """Opt-in floating QDQ; a checkpoint name alone never selects a recipe."""

    def __init__(self, recipe: str = "e2m1_rne", ignored_layers: list[str] | None = None) -> None:
        super().__init__()
        if recipe not in {"e2m1_rne", "mxfp4_c7"}:
            raise ValueError("mxfp4_fake recipe must be e2m1_rne or mxfp4_c7")
        if ignored_layers is not None and (
            not isinstance(ignored_layers, list)
            or any(not isinstance(name, str) or not name for name in ignored_layers)
        ):
            raise ValueError("ignored_layers must be a list of nonempty layer names")
        self.recipe = recipe
        self.ignored_layers = list(ignored_layers or [])
        logger.warning_once(
            "mxfp4_fake uses floating storage/GEMM with recipe=%s; this does not provide FP4 acceleration",
            recipe,
        )

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4_fake"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper") -> None:
        self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiffusionMXFP4FakeConfig":
        return cls(recipe=config.get("recipe", "e2m1_rne"), ignored_layers=config.get("ignored_layers"))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix=prefix, ignored_layers=self.ignored_layers, fused_mapping=self.packed_modules_mapping
        ):
            return UnquantizedLinearMethod()
        return MXFP4FakeLinearMethod(self.recipe)


class MXFP4FakeLinearMethod(UnquantizedLinearMethod):
    """Reuse the ordinary vLLM TP loaders; quantize weights after loading."""

    supports_pre_processed_weights = False

    def __init__(self, recipe: str = "e2m1_rne") -> None:
        self.recipe = recipe

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        with torch.no_grad():
            layer.weight.copy_(fake_quantize_mxfp4(layer.weight, self.recipe))

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        logger.info_once("Executed MXFP4 fake quantization: recipe=%s; floating GEMM", self.recipe)
        return F.linear(fake_quantize_mxfp4(x, self.recipe), layer.weight, bias)
