# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization helpers for the Diffusers backend."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from diffusers import PipelineQuantizationConfig, TorchAoConfig

logger = logging.getLogger(__name__)

_DIFFUSERS_TRANSFORMER_COMPONENTS = ("transformer", "transformer_2")
_QuantizationConfigSpec = tuple[Callable[[Any], None], str]


def ensure_supported_diffusers_quantization(quant_config: Any) -> None:
    """Validate that a vLLM-Omni quantization config has a Diffusers mapping."""

    method = _get_quant_method_name(quant_config)
    validator, _ = _get_quantization_config_spec(method)
    validator(quant_config)


def convert_diffusers_quantization_config(load_kwargs: dict[str, Any]) -> None:
    """Build Diffusers-native quantization config objects from dict kwargs."""

    quant_config = load_kwargs.get("quantization_config")
    if isinstance(quant_config, dict):
        load_kwargs["quantization_config"] = _build_diffusers_native_quantization_config(quant_config)


def apply_diffusers_quantization_config(
    od_config: Any,
    load_kwargs: dict[str, Any],
    component_names: dict[str, Any],
) -> bool:
    """Inject a courtesy-converted quantization_config into load kwargs.

    ``diffusers_load_kwargs`` is the canonical Diffusers backend configuration
    path, so an explicit ``quantization_config`` already present in ``load_kwargs``
    is never replaced. Returns whether a config was injected by this helper.
    """

    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is None:
        return False

    if "quantization_config" in load_kwargs:
        logger.warning(
            "Both vLLM-Omni quantization_config and diffusers_load_kwargs.quantization_config "
            "were provided for the diffusers backend. Using the Diffusers-native "
            "quantization_config from diffusers_load_kwargs."
        )
        return False

    transformer_components = _get_diffusers_transformer_components(component_names)
    load_kwargs["quantization_config"] = _build_diffusers_quantization_config(
        quant_config,
        transformer_components,
    )
    return True


def _normalize_method_name(method: Any) -> str:
    method = getattr(method, "value", method)
    return str(method).lower().replace("-", "_")


def _get_quant_method_name(quant_config: Any) -> str:
    get_name = getattr(quant_config, "get_name", None)
    if get_name is not None:
        return _normalize_method_name(get_name() if callable(get_name) else get_name)

    method = getattr(quant_config, "quant_method", None)
    if method is None:
        method = getattr(quant_config, "method", None)
    if method is None:
        raise NotImplementedError(
            "Diffusers backend quantization conversion requires a quantization "
            "config with get_name(), quant_method, or method."
        )
    return _normalize_method_name(method)


def _ensure_no_ignored_layers(quant_config: Any) -> None:
    ignored_layers = getattr(quant_config, "ignored_layers", None)
    if not ignored_layers:
        ignored_layers = getattr(quant_config, "modules_to_not_convert", None)
    if ignored_layers:
        raise NotImplementedError(
            "Diffusers backend quantization conversion does not map vLLM "
            "ignored_layers/modules_to_not_convert names to Diffusers module "
            "names. Use diffusers_load_kwargs for a native Diffusers config."
        )


def _get_torchao_quant_type_cls(class_name: str) -> type[Any]:
    try:
        import torchao.quantization as torchao_quantization
    except ImportError as exc:
        raise ImportError(
            "Diffusers backend quantization conversion for fp8/int8 requires "
            "torchao. Install torchao or pass a Diffusers-native "
            "quantization_config through diffusers_load_kwargs."
        ) from exc

    try:
        return getattr(torchao_quantization, class_name)
    except AttributeError as exc:
        raise ImportError(f"torchao.quantization.{class_name} is required for this quantization mapping.") from exc


def _build_diffusers_native_quantization_config(config: dict[str, Any]) -> Any:
    config = config.copy()
    quant_mapping = config.get("quant_mapping")
    if isinstance(quant_mapping, dict):
        config["quant_mapping"] = {
            component_name: _build_diffusers_component_quant_config(component_config)
            for component_name, component_config in quant_mapping.items()
        }
    return PipelineQuantizationConfig(**config)


def _build_diffusers_component_quant_config(config: Any) -> Any:
    if not isinstance(config, dict):
        return config

    quant_method = config.get("quant_method")
    if quant_method is None:
        raise ValueError("Diffusers quant_mapping entries must provide quant_method.")

    quant_method = _normalize_method_name(quant_method)
    config_cls = _get_diffusers_quantization_config_cls(quant_method)
    from_dict = getattr(config_cls, "from_dict", None)
    if from_dict is not None:
        return from_dict(config)

    init_kwargs = config.copy()
    init_kwargs.pop("quant_method", None)
    return config_cls(**init_kwargs)


def _get_diffusers_quantization_config_cls(quant_method: str) -> type[Any]:
    from diffusers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING as DIFFUSERS_QUANT_CONFIG_MAPPING

    config_cls = DIFFUSERS_QUANT_CONFIG_MAPPING.get(quant_method)
    if config_cls is not None:
        return config_cls

    try:
        from transformers.quantizers.auto import (
            AUTO_QUANTIZATION_CONFIG_MAPPING as TRANSFORMERS_QUANT_CONFIG_MAPPING,
        )
    except ImportError:
        TRANSFORMERS_QUANT_CONFIG_MAPPING = {}

    config_cls = TRANSFORMERS_QUANT_CONFIG_MAPPING.get(quant_method)
    if config_cls is None:
        raise ValueError(f"Diffusers quant_mapping entry uses unknown quant_method={quant_method!r}.")
    return config_cls


def _build_torchao_pipeline_quant_config(
    torchao_quant_type_name: str,
    transformer_components: list[str],
) -> Any:
    """Build a Diffusers config after the caller has validated the vLLM config."""

    quant_type_cls = _get_torchao_quant_type_cls(torchao_quant_type_name)
    return PipelineQuantizationConfig(
        quant_mapping={
            component_name: TorchAoConfig(
                quant_type=quant_type_cls(),
            )
            for component_name in transformer_components
        }
    )


def _get_diffusers_transformer_components(component_names: dict[str, Any]) -> list[str]:
    transformer_components = [
        component_name for component_name in component_names if component_name in _DIFFUSERS_TRANSFORMER_COMPONENTS
    ]
    if transformer_components:
        return transformer_components

    available_components = ", ".join(sorted(component_names.keys())) or "<none>"
    raise NotImplementedError(
        "Diffusers backend quantization conversion currently only supports "
        "pipelines with transformer or transformer_2 components. "
        f"Found pipeline components: {available_components}. Use "
        "diffusers_load_kwargs for a native Diffusers quantization config."
    )


def _validate_fp8_quant_config(quant_config: Any) -> None:
    if getattr(quant_config, "is_checkpoint_fp8_serialized", False):
        raise NotImplementedError(
            "Diffusers backend fp8 conversion only supports online/dynamic "
            "TorchAO quantization; serialized vLLM fp8 checkpoints are not mapped."
        )

    activation_scheme = getattr(quant_config, "activation_scheme", "dynamic")
    if activation_scheme != "dynamic":
        raise NotImplementedError(
            f"Diffusers backend fp8 conversion only supports activation_scheme='dynamic'. Got {activation_scheme!r}."
        )

    weight_block_size = getattr(quant_config, "weight_block_size", None)
    if weight_block_size is not None:
        raise NotImplementedError(
            "Diffusers backend fp8 conversion does not map vLLM weight_block_size "
            "to TorchAO. Use diffusers_load_kwargs for a native Diffusers config."
        )

    _ensure_no_ignored_layers(quant_config)


def _validate_int8_quant_config(quant_config: Any) -> None:
    if getattr(quant_config, "is_checkpoint_int8_serialized", False):
        raise NotImplementedError(
            "Diffusers backend int8 conversion only supports online/dynamic "
            "TorchAO quantization; serialized vLLM int8 checkpoints are not mapped."
        )

    activation_scheme = getattr(quant_config, "activation_scheme", "dynamic")
    if activation_scheme != "dynamic":
        raise NotImplementedError(
            f"Diffusers backend int8 conversion only supports activation_scheme='dynamic'. Got {activation_scheme!r}."
        )

    _ensure_no_ignored_layers(quant_config)


_QUANTIZATION_CONFIG_SPECS: dict[str, _QuantizationConfigSpec] = {
    "fp8": (_validate_fp8_quant_config, "Float8DynamicActivationFloat8WeightConfig"),
    "int8": (_validate_int8_quant_config, "Int8DynamicActivationInt8WeightConfig"),
}


def _get_quantization_config_spec(method: str) -> _QuantizationConfigSpec:
    spec = _QUANTIZATION_CONFIG_SPECS.get(method)
    if spec is None:
        raise NotImplementedError(
            f"Diffusers backend quantization conversion does not support {method!r}. "
            "Use diffusers_load_kwargs for a native Diffusers quantization config, "
            "or use a native vLLM-Omni pipeline for this quantization method."
        )
    return spec


def _build_diffusers_quantization_config(
    quant_config: Any,
    transformer_components: list[str],
) -> Any:
    """Build a Diffusers PipelineQuantizationConfig from a supported config."""

    method = _get_quant_method_name(quant_config)
    _, torchao_quant_type_name = _get_quantization_config_spec(method)
    return _build_torchao_pipeline_quant_config(
        torchao_quant_type_name,
        transformer_components,
    )
