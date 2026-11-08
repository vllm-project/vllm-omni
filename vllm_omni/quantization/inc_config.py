# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Extended INC/AutoRound config for multi-stage omni models."""

from __future__ import annotations

import logging
from os.path import commonprefix
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.models.utils import WeightsMapper

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

_REGEX_SPECIAL_CHARS = frozenset(r"*+?^$()[]{}|\\")
logger = logging.getLogger(__name__)


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


def _normalize_block_names(value: str | list[str] | None) -> list[str] | None:
    if isinstance(value, str):
        return [name.strip() for name in value.split(",") if name.strip()]
    return value


def _remap_name(name: str, prefix_map: dict[str, str]) -> str:
    """Replace the first matching checkpoint prefix in *name*."""
    for source in sorted(prefix_map, key=len, reverse=True):
        if name.startswith(source):
            return prefix_map[source] + name[len(source) :]
    return name


def _is_regex_key(key: str) -> bool:
    return any(char in _REGEX_SPECIAL_CHARS for char in key)


def _remap_regex_key(key: str, prefix_map: dict[str, str]) -> str:
    """Remap a checkpoint prefix in a regex key in either dot style."""
    for source in sorted(prefix_map, key=len, reverse=True):
        target = prefix_map[source] or ""
        for source_form, target_form in (
            (source, target),
            (re.escape(source), re.escape(target)),
        ):
            if source_form != target_form and source_form in key:
                return key.replace(source_form, target_form, 1)
    return key


def _remap_extra_config_key(key: str, prefix_map: dict[str, str]) -> str:
    """Remap an ``extra_config`` key, preserving regex patterns."""
    if _is_regex_key(key):
        new_key = _remap_regex_key(key, prefix_map)
        if new_key == key:
            logger.debug("extra_config regex key did not match any prefix: %s", key)
        return new_key
    return _remap_name(key, prefix_map)


class OmniINCConfig(INCConfig):
    """INC config with stage-prefix mapping and AutoRound MXFP8/NVFP4 dispatch."""

    block_name_to_quantize: list[str] | None
    extra_config: dict[str, Any] | None

    SUPPORTED_DTYPES = {"int", "mx_fp", "nv_fp"}
    SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq", "auto_round:llm_compressor"}

    # ------------------------------------------------------------------
    # Quantization method selection and checkpoint name mapping
    # ------------------------------------------------------------------

    def get_quant_method(self, layer, prefix: str):
        """Get quantization methods for AutoRound checkpoints."""
        if self.data_type == "nv_fp":
            from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
                CompressedTensorsLinearMethod,
            )
            from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
                CompressedTensorsW4A4Fp4,
            )

            layer_config = self.config_parser.resolve(layer, prefix)
            if isinstance(layer, LinearBase):
                if not layer_config.quantized:
                    return UnquantizedLinearMethod()
                scheme = CompressedTensorsW4A4Fp4()
                layer.scheme = scheme
                return CompressedTensorsLinearMethod(self)
            return None

        # AutoRound MXFP8/MXFP4: vLLM's method, wrapped for diffusion's 3-D activations.
        method = super().get_quant_method(layer, prefix)
        if self.is_mxfp and isinstance(method, INCLinearMethod):
            return _MXFlattenedLinearMethod(method.scheme)
        return method

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        """Remap HF checkpoint names to vLLM runtime prefixes."""
        prefix_map = getattr(hf_to_vllm_mapper, "orig_to_new_prefix", None) or {}
        stage = _stage_prefix(prefix_map) if prefix_map else ""

        # -- block_name_to_quantize ----------------------------------------
        block_names = _normalize_block_names(self.block_name_to_quantize)
        if block_names is not None:
            if prefix_map and stage:
                block_names = _map_with_stage_prefix(block_names, prefix_map, stage)
            else:
                block_names = hf_to_vllm_mapper.apply_list(block_names)
        self.block_name_to_quantize = block_names

        # -- extra_config --------------------------------------------------
        extra_config = self.extra_config
        if extra_config is not None and prefix_map:
            new_extra: dict[str, Any] = {}
            sorted_keys = sorted(prefix_map, key=len, reverse=True)

            for key, val in extra_config.items():
                is_regex = _is_regex_key(key)
                if is_regex:
                    # re.search matches anywhere, so no stage prefix is needed.
                    new_key = _remap_extra_config_key(key, prefix_map)
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
        elif extra_config is not None:
            self.extra_config = hf_to_vllm_mapper.apply_dict(extra_config)

    def apply_prefix_map(self, prefix_map: dict[str, str]) -> None:
        """Replace checkpoint prefixes with runtime prefixes in-place.

        ``block_name_to_quantize`` and ``extra_config`` both carry checkpoint
        module names, so models that rename submodules between checkpoint and
        runtime must remap both. Otherwise a per-layer ``extra_config``
        exception (for example a BF16 ``down_proj``) stops matching at
        runtime and the layer is built quantized even though its checkpoint
        stores an unquantized weight.

        Unlike :meth:`apply_vllm_mapper`, keys in *prefix_map* are replaced
        verbatim; no container/stage prefix is inferred. Entries without a
        matching prefix are left unchanged.
        """
        if not prefix_map:
            return
        block_names = _normalize_block_names(self.block_name_to_quantize)
        if block_names is not None:
            block_names = [_remap_name(name, prefix_map) for name in block_names]
        self.block_name_to_quantize = block_names

        extra_config = self.extra_config
        if extra_config is not None:
            self.extra_config = {_remap_extra_config_key(key, prefix_map): value for key, value in extra_config.items()}

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
        # The copied parser still references the original INCConfig. Rebuild it
        # so layer resolution reads the upgraded config after prefix remapping.
        omni.config_parser = type(inc.config_parser)(omni)
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
# AutoRound MXFP8/MXFP4 checkpoints - vLLM's INC MX schemes, 3-D activations
# ---------------------------------------------------------------------------


class _MXFlattenedLinearMethod(INCLinearMethod):
    """vLLM's INC MX linear method with diffusion activations flattened to 2-D.

    The MX GEMMs behind the INC MXFP4/MXFP8 schemes (``fp4_gemm`` / ``fp8_gemm``
    on XPU) only accept 2-D activations, while diffusion transformers run
    (B, S, K) hidden states. Weight creation, layout and the GEMM stay vLLM's;
    only the reshape is added.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if not hasattr(layer, "weight_packed") and hasattr(layer, "weight"):
            from vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_linear import (
                INCMxfp4LinearMethod,
            )

            if isinstance(self.scheme, INCMxfp4LinearMethod):
                # INCMxfp4LinearMethod destructively replaces weight_packed with
                # weight. A second finalize (generic post-load plus an explicit
                # repack) would otherwise raise AttributeError.
                layer._already_called_process_weights_after_loading = True
                return
        super().process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() <= 2:
            return super().apply(layer, x, bias)
        ori_shape = x.shape
        output = super().apply(layer, x.reshape(-1, ori_shape[-1]), bias)
        return output.reshape(*ori_shape[:-1], -1)
