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
    from vllm.model_executor.layers.quantization.inc import INCConfig

    # Map checkpoint key 'bits' to INCConfig's 'weight_bits'
    if "bits" in kw and "weight_bits" not in kw:
        kw["weight_bits"] = kw.pop("bits")

    # Filter to only valid INCConfig params
    valid = set(inspect.signature(INCConfig.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in kw.items() if k in valid}
    return INCConfig(**filtered)


_OVERRIDES: dict[str, Callable[..., QuantizationConfig]] = {
    "gguf": _build_gguf,
    "int8": _build_int8,
    "inc": _build_inc,
    "auto-round": _build_inc,
}

SUPPORTED_QUANTIZATION_METHODS: list[str] = list(dict.fromkeys(QUANTIZATION_METHODS + list(_OVERRIDES.keys())))


def _build_single(method: str, **kwargs: Any) -> QuantizationConfig:
    """Build a single QuantizationConfig by method name.

    Resolution: _OVERRIDES first, then vLLM registry via from_config().
    """
    method = method.lower()

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

    A per-component dict has no "method" key and all values are
    str, dict, or None. To avoid misdetecting a flat config with
    all-string values (e.g. {"activation_scheme": "static"}), we
    require at least one value to be None or a dict with "method".
    """
    if "method" in spec:
        return False
    if not all(isinstance(v, (dict, str, type(None))) for v in spec.values()):
        return False
    return any(v is None or (isinstance(v, dict) and "method" in v) for v in spec.values())


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
            method = value.pop("method", None)
            if method is None:
                raise ValueError(f"Component '{prefix}' config dict must have a 'method' key")
            config = _build_single(method, **value)
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

        method = spec.pop("method", None)
        if method is None:
            raise ValueError(
                "Dict quantization config must have a 'method' key or "
                "be a per-component config with component prefixes as keys."
            )
        merged = {**spec, **kwargs}
        logger.info("Building quantization config: %s", method)
        return _build_single(method, **merged)

    raise TypeError(f"quantization config must be str, dict, QuantizationConfig, or None, got {type(spec).__name__}")
