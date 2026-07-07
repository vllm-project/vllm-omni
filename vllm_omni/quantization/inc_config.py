# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended INC/AutoRound config for multi-stage omni models."""

from __future__ import annotations

from os.path import commonprefix
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

_REGEX_SPECIAL_CHARS = frozenset(r"*+?^$()[]{}|\\")


def _stage_prefix(prefix_map: dict[str, str | None]) -> str:
    """Derive the container/stage prefix from mapper source keys."""
    cp = commonprefix(list(prefix_map.keys()))
    dot = cp.rfind(".")
    return cp[: dot + 1] if dot >= 0 else ""


def _map_with_stage_prefix(
    items: list[str],
    prefix_map: dict[str, str | None],
    stage: str,
) -> list[str]:
    """Apply *prefix_map* to each item and prepend *stage* to mapped items."""
    sorted_keys = sorted(prefix_map, key=len, reverse=True)
    result: list[str] = []
    for item in items:
        new_item = item
        for orig in sorted_keys:
            if item.startswith(orig):
                new_val = prefix_map[orig] or ""
                new_item = stage + new_val + item[len(orig) :]
                break
        result.append(new_item)
    return result


class OmniINCConfig(INCConfig):
    """INCConfig extended with multi-stage prefix remapping and MXFP8 support.

    AutoRound MXFP8 checkpoints declare quant_method="auto-round" with data_type="mx_fp".
    This config detects that case and dispatches to IncMxfp8OfflineLinearMethod instead
    of the standard INT quantization path.

    Architecture:
      - AutoRound INT4/INT8 → standard INCConfig path (xpu_w4a16_quant_layer, etc.)
      - AutoRound MXFP8 (data_type="mx_fp") → IncMxfp8OfflineLinearMethod
      - Native MXFP8 (quant_method="mxfp8") → DiffusionMXFP8Config → NPUMxfp8LinearMethod (NPU only)
    """

    # Extend supported data types and formats to include MXFP8
    SUPPORTED_DTYPES = {"int", "mx_fp"}
    SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq", "auto_round:llm_compressor"}

    # ------------------------------------------------------------------
    # Core integration: called by vLLM's configure_quant_config()
    # ------------------------------------------------------------------

    def get_quant_method(self, layer, prefix: str):
        """Get quantization method, handling AutoRound MXFP8 as a special case."""
        # Check if this is an AutoRound MXFP8 checkpoint (data_type="mx_fp")
        if hasattr(self, "data_type") and self.data_type == "mx_fp" and self.weight_bits == 8:
            from vllm.model_executor.layers.linear import LinearBase

            if isinstance(layer, LinearBase):
                return IncMxfp8OfflineLinearMethod()
            return None

        # Otherwise, use parent INCConfig logic
        return super().get_quant_method(layer, prefix)

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        """Remap HF checkpoint names to vLLM runtime prefixes."""
        prefix_map = getattr(hf_to_vllm_mapper, "orig_to_new_prefix", None) or {}
        stage = _stage_prefix(prefix_map) if prefix_map else ""

        # -- Normalize CSV string -----------------------------------------
        if isinstance(self.block_name_to_quantize, str):
            self.block_name_to_quantize = [b.strip() for b in self.block_name_to_quantize.split(",") if b.strip()]

        # -- block_name_to_quantize ----------------------------------------
        if self.block_name_to_quantize is not None:
            if prefix_map and stage:
                self.block_name_to_quantize = _map_with_stage_prefix(
                    self.block_name_to_quantize,
                    prefix_map,
                    stage,
                )
            else:
                self.block_name_to_quantize = hf_to_vllm_mapper.apply_list(self.block_name_to_quantize)

        # -- extra_config --------------------------------------------------
        if self.extra_config is not None and prefix_map:
            new_extra: dict[str, Any] = {}
            sorted_keys = sorted(prefix_map, key=len, reverse=True)

            # Build escaped-dot map for regex pattern keys
            escaped_map: dict[str, str] = {}
            for orig, new in prefix_map.items():
                escaped_map[orig.replace(".", r"\.")] = (new or "").replace(".", r"\.")
            escaped_sorted = sorted(escaped_map, key=len, reverse=True)

            for key, val in self.extra_config.items():
                is_regex = any(c in _REGEX_SPECIAL_CHARS for c in key)
                if is_regex:
                    # Regex keys: escaped-dot substring replacement.
                    # re.search matches anywhere so no stage prefix needed.
                    new_key = key
                    for esc_orig in escaped_sorted:
                        if esc_orig in new_key:
                            new_key = new_key.replace(
                                esc_orig,
                                escaped_map[esc_orig],
                                1,
                            )
                            break
                else:
                    # Plain keys: prefix replacement + stage prefix.
                    new_key = key
                    for orig in sorted_keys:
                        if key.startswith(orig):
                            new_val = prefix_map[orig] or ""
                            new_key = stage + new_val + key[len(orig) :]
                            break
                new_extra[new_key] = val
            self.extra_config = new_extra
        elif self.extra_config is not None:
            self.extra_config = hf_to_vllm_mapper.apply_dict(self.extra_config)

    # ------------------------------------------------------------------
    # Upgrading a vanilla INCConfig created by vLLM
    # ------------------------------------------------------------------

    @classmethod
    def from_inc_config(cls, inc: INCConfig) -> OmniINCConfig:
        """Promote a vanilla :class:`INCConfig` to :class:`OmniINCConfig`.

        Copies all attributes so that the new instance is a drop-in
        replacement.
        """
        omni = object.__new__(cls)
        omni.__dict__.update(inc.__dict__)
        return omni

    @classmethod
    def maybe_upgrade(cls, quant_config: QuantizationConfig | None) -> QuantizationConfig | None:
        """Upgrade *quant_config* to :class:`OmniINCConfig` if applicable.

        Returns the original config unchanged when it is not an INC
        config or is already an :class:`OmniINCConfig`.
        """
        if quant_config is None:
            return None
        if isinstance(quant_config, cls):
            return quant_config
        if isinstance(quant_config, INCConfig):
            return cls.from_inc_config(quant_config)
        return quant_config


# ---------------------------------------------------------------------------
# IncMxfp8OfflineLinearMethod - used by AutoRound MXFP8 checkpoints
# ---------------------------------------------------------------------------


class IncMxfp8OfflineLinearMethod(LinearMethodBase):
    """Offline MXFP8 linear method for AutoRound MXFP8 checkpoints.

    For pre-quantized AutoRound MXFP8 checkpoints (data_type="mx_fp") where weights
    are stored as BF16 (losslessly representing FP8 E4M3 values) with uint8 MX block scales.
    Delegates to the platform-appropriate kernel (XPUMxFp8LinearKernel, etc.).

    Used by OmniINCConfig when data_type="mx_fp".
    """

    def __init__(self) -> None:
        from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            MXFP8_BLOCK_SIZE,
            MXFP8_SCALE_DTYPE,
        )

        self.kernel = init_mxfp8_linear_kernel()
        self.block_size = MXFP8_BLOCK_SIZE
        self.scale_dtype = MXFP8_SCALE_DTYPE

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if input_size_per_partition % self.block_size != 0:
            raise ValueError(
                f"MXFP8 requires input_size_per_partition "
                f"({input_size_per_partition}) to be divisible by "
                f"{self.block_size}."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Allocate as params_dtype (BF16) to match checkpoint dtype.
        # Will be cast to FP8 in process_weights_after_loading.
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

        num_groups = input_size_per_partition // self.block_size
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, num_groups, dtype=self.scale_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Cast BF16 weight to FP8.
        # Checkpoint stores weights as BF16 (losslessly representing FP8 E4M3 values).
        w = layer.weight.data
        if w.dtype != torch.float8_e4m3fn:
            # Cast to FP8 E4M3. AutoRound stores FP8 values as BF16, so this is lossless.
            w = w.to(torch.float8_e4m3fn).contiguous()
            replace_parameter(layer, "weight", w)

        # Delegate layout transforms (transpose, scale reinterpret) to the platform kernel.
        self.kernel.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ori_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, ori_shape[-1])
        output = self.kernel.apply_weights(layer, x, bias)
        if len(ori_shape) > 2:
            output = output.reshape(*ori_shape[:-1], -1)
        return output
