# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for building quantization configs.

build_quant_config() delegates to vLLM's quantization registry.
The only extension point is _OVERRIDES for methods that need a
different QuantizationConfig subclass in the OMNI context (e.g. GGUF).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    get_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

from .component_config import ComponentQuantizationConfig

logger = init_logger(__name__)


def _build_gguf(**kw: Any) -> QuantizationConfig:
    """Lazy import to avoid pulling in CUDA/pynvml at module load time."""
    from .gguf_config import DiffusionGGUFConfig

    return DiffusionGGUFConfig(**kw)


def _build_int8(**kw: Any) -> QuantizationConfig:
    """Lazy import for Int8 diffusion config (supports CUDA + NPU)."""
    from .int8_config import DiffusionInt8Config

    return DiffusionInt8Config(**kw)


def _build_inc(**kw: Any) -> QuantizationConfig:
    """Lazy import for INC/AutoRound config with checkpoint kwarg normalization."""
    from .inc_config import OmniINCConfig

    # Map checkpoint key 'bits' to INCConfig's 'weight_bits'
    if "bits" in kw and "weight_bits" not in kw:
        kw["weight_bits"] = kw.pop("bits")

    # Filter to only valid INCConfig params
    valid = set(inspect.signature(OmniINCConfig.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in kw.items() if k in valid}
    return OmniINCConfig(**filtered)


_OVERRIDES: dict[str, Callable[..., QuantizationConfig]] = {
    "gguf": _build_gguf,
    "int8": _build_int8,
    "inc": _build_inc,
    "auto-round": _build_inc,
    "auto_round": _build_inc,
}

SUPPORTED_QUANTIZATION_METHODS: list[str] = list(dict.fromkeys(QUANTIZATION_METHODS + list(_OVERRIDES.keys())))


_MODEL_OPT_METHODS = {
    "modelopt",
}
_MODEL_OPT_FP8_ALGOS = {
    "FP8",
    "FP8_PER_CHANNEL_PER_TOKEN",
}


def _normalize_method_name(method: Any) -> str:
    return str(method).lower().replace("-", "_")


def _detect_modelopt_method(config: Mapping[str, Any]) -> str | None:
    quantization = config.get("quantization")
    if isinstance(quantization, Mapping):
        quant_algo = str(quantization.get("quant_algo", "")).upper()
    else:
        quant_algo = str(config.get("quant_algo", "")).upper()

    method = config.get("method", config.get("quant_method"))
    normalized_method = _normalize_method_name(method) if method is not None else None

    producer = config.get("producer")
    is_modelopt_config = normalized_method in _MODEL_OPT_METHODS or (
        isinstance(producer, Mapping) and str(producer.get("name", "")).lower() == "modelopt"
    )

    if not is_modelopt_config:
        return None

    if quant_algo:
        if quant_algo in _MODEL_OPT_FP8_ALGOS:
            return "modelopt"
        return None

    if method is not None:
        if normalized_method in _MODEL_OPT_METHODS:
            return normalized_method

    return None


def _build_modelopt_from_config(method: str, config: Mapping[str, Any]) -> QuantizationConfig:
    config_cls = get_quantization_config(method)
    normalized_config = dict(config)
    normalized_config.setdefault("quant_method", method)
    return config_cls.from_config(normalized_config)


def _pop_method_name(spec: dict[str, Any]) -> str | None:
    method = spec.pop("method", None)
    if method is None:
        method = spec.pop("quant_method", None)
    return method


def _build_from_method_and_config(method: str, config: Mapping[str, Any]) -> QuantizationConfig:
    normalized_config = {"quant_method": method, **config}
    modelopt_method = _detect_modelopt_method(normalized_config)
    if modelopt_method is not None:
        return _build_modelopt_from_config(modelopt_method, normalized_config)
    return _build_single(method, **config)


def _build_single(method: str, **kwargs: Any) -> QuantizationConfig:
    """Build a single QuantizationConfig by method name.

    Resolution: _OVERRIDES first, then vLLM registry via from_config().
    """
    method = _normalize_method_name(method)

    if method in _OVERRIDES:
        return _OVERRIDES[method](**kwargs)

    if method not in QUANTIZATION_METHODS:
        raise ValueError(f"Unknown quantization method: {method!r}. Supported: {SUPPORTED_QUANTIZATION_METHODS}")

    config_cls = get_quantization_config(method)

    try:
        return config_cls(**kwargs)
    except TypeError:
        sig = inspect.signature(config_cls.__init__)
        raise TypeError(
            f"Cannot instantiate {config_cls.__name__} with kwargs {kwargs}. Expected signature: {sig}"
        ) from None


def _is_per_component_dict(spec: dict[str, Any]) -> bool:
    """Check if a dict describes per-component quantization.

    A per-component dict has no "method" / "quant_method" key and all values are
    str, dict, or None. To avoid misdetecting a flat config with
    all-string values (e.g. {"activation_scheme": "static"}), we
    require at least one value to be None or a dict with "method" /
    "quant_method".
    """
    if "method" in spec or "quant_method" in spec:
        return False
    if not all(isinstance(v, (dict, str, type(None))) for v in spec.values()):
        return False
    return any(v is None or (isinstance(v, dict) and ("method" in v or "quant_method" in v)) for v in spec.values())


def _build_component_config(spec: dict[str, Any]) -> ComponentQuantizationConfig:
    """Build ComponentQuantizationConfig from a per-component dict."""
    component_configs: dict[str, QuantizationConfig | None] = {}
    default_config: QuantizationConfig | None = None

    for prefix, value in spec.items():
        if value is None:
            config = None
        elif isinstance(value, str):
            config = _build_single(value)
        elif isinstance(value, dict):
            value = dict(value)  # avoid mutating caller's dict
            method = _pop_method_name(value)
            if method is None:
                raise ValueError(f"Component '{prefix}' config dict must have a 'method' or 'quant_method' key")
            config = _build_from_method_and_config(method, value)
        else:
            raise TypeError(f"Component '{prefix}' config must be str, dict, or None, got {type(value).__name__}")

        if prefix == "default":
            default_config = config
        else:
            component_configs[prefix] = config

    logger.info(
        "Per-component quantization: %s",
        {k: (v.get_name() if v else None) for k, v in component_configs.items()},
    )
    return ComponentQuantizationConfig(component_configs, default_config)


def build_quant_config(
    spec: str | dict[str, Any] | QuantizationConfig | None,
    **kwargs: Any,
) -> QuantizationConfig | None:
    """Build a quantization config from a flexible specification.

    Args:
        spec: None/"none", method name str, dict with "method" key,
              per-component dict, or QuantizationConfig passthrough.
        **kwargs: Extra params merged with dict spec.
    """
    if spec is None:
        return None

    if isinstance(spec, QuantizationConfig):
        return spec

    if isinstance(spec, str):
        if spec.lower() == "none":
            return None
        logger.info("Building quantization config: %s", spec)
        return _build_single(spec, **kwargs)

    if isinstance(spec, Mapping):
        spec = dict(spec)

        if _is_per_component_dict(spec):
            return _build_component_config(spec)

        modelopt_method = _detect_modelopt_method(spec)
        if modelopt_method is not None:
            logger.info("Building quantization config: %s", modelopt_method)
            return _build_modelopt_from_config(modelopt_method, spec)

        method = _pop_method_name(spec)
        if method is None:
            raise ValueError(
                "Dict quantization config must have a 'method' or 'quant_method' key or "
                "be a per-component config with component prefixes as keys."
            )
        merged = {**spec, **kwargs}
        logger.info("Building quantization config: %s", method)
        return _build_from_method_and_config(method, merged)

    raise TypeError(f"quantization config must be str, dict, QuantizationConfig, or None, got {type(spec).__name__}")
