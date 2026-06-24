# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for building quantization configs.

build_quant_config() delegates to vLLM's quantization registry.
The only extension point is _OVERRIDES for methods that need a
different QuantizationConfig subclass in the OMNI context (e.g. GGUF).
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from vllm.logger import init_logger


# ---------------------------------------------------------------------------
# Stub the ``humming`` package so that vLLM's lazy import inside
# ``get_quantization_config()`` (which unconditionally does
# ``from .humming import HummingConfig``) does not crash when the real
# ``humming`` wheel is not installed.  Only populate the bare-minimum
# names that ``humming.py`` accesses at module level.
# ---------------------------------------------------------------------------
def _register_humming_stubs() -> None:
    """Register stub ``humming`` sub-modules so that the optional
    humming quantization backend can be imported without the real wheel."""
    if "humming" in sys.modules:
        return  # already present (real or stub)

    # --- sub-modules ---
    submodules: dict[str, tuple[str, ...]] = {
        "humming": (),
        "humming.config": ("GemmType",),
        "humming.dtypes": ("DataType",),
        "humming.layer": ("HummingLayerMeta", "HummingMethod"),
        "humming.schema": (
            "BaseInputSchema",
            "BaseWeightSchema",
            "HummingInputSchema",
            "HummingWeightSchema",
        ),
        "humming.utils": (),
        "humming.utils.weight": ("quantize_weight",),
    }

    registry: dict[str, ModuleType] = {}
    for name, attrs in submodules.items():
        mod = ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, type(attr, (), {}))
        registry[name] = mod

    # wire parent references
    registry["humming"].config = registry["humming.config"]
    registry["humming"].dtypes = registry["humming.dtypes"]
    registry["humming"].layer = registry["humming.layer"]
    registry["humming"].schema = registry["humming.schema"]
    registry["humming"].utils = registry["humming.utils"]
    registry["humming.utils"].weight = registry["humming.utils.weight"]

    for name, mod in registry.items():
        sys.modules[name] = mod


_register_humming_stubs()

from vllm.model_executor.layers.quantization import (  # noqa: E402
    QUANTIZATION_METHODS,
    get_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E402
    QuantizationConfig,
)

from .component_config import ComponentQuantizationConfig  # noqa: E402

logger = init_logger(__name__)


def _build_gguf(**kw: Any) -> QuantizationConfig:
    """Lazy import to avoid pulling in CUDA/pynvml at module load time."""
    from .gguf_config import DiffusionGGUFConfig

    return DiffusionGGUFConfig(**kw)


def _build_int8(**kw: Any) -> QuantizationConfig:
    """Lazy import for Int8 diffusion config (supports CUDA + NPU)."""
    from .int8_config import DiffusionInt8Config

    return DiffusionInt8Config(**kw)


def _build_mxfp8(**kw: Any) -> QuantizationConfig:
    """Lazy import for W8A8 MXFP8 diffusion config (NPU only)."""
    from .mxfp8_config import DiffusionMXFP8Config

    return DiffusionMXFP8Config(**kw)


def _build_mxfp4(**kw: Any) -> QuantizationConfig:
    """Lazy import for W4A4 MXFP4 diffusion config (NPU only)."""
    from .mxfp4_config import DiffusionMXFP4Config

    return DiffusionMXFP4Config(**kw)


def _build_mxfp4_dualscale(**kw: Any) -> QuantizationConfig:
    """Lazy import for MXFP4 DualScale + BF16 mixed diffusion config (NPU only).

    Offline mode (is_checkpoint_serialized=True):
        ignored_layers from config.json marks interleaved BF16 fallback layers.
        All other linear layers use W4A4 MXFP4 DualScale.

    Online mode (is_checkpoint_serialized=False):
        num_bf16_fallback_layers leading transformer blocks use BF16 original weights
        (default 5 when not specified). Remaining blocks use W4A4 MXFP4 DualScale online.
    """
    from .mxfp4_config import DiffusionMXFP4DualScaleMixedConfig

    return DiffusionMXFP4DualScaleMixedConfig(**kw)


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
    "mxfp8": _build_mxfp8,
    "mxfp4": _build_mxfp4,
    "mxfp4_dualscale": _build_mxfp4_dualscale,
    "inc": _build_inc,
    "auto-round": _build_inc,
    "auto_round": _build_inc,
}

SUPPORTED_QUANTIZATION_METHODS: list[str] = list(dict.fromkeys(QUANTIZATION_METHODS + list(_OVERRIDES.keys())))


def _build_reverse_alias_map() -> dict[str, str]:
    """Build a mapping from normalized method aliases to canonical names.

    All keys in _OVERRIDES that share the same builder function are considered
    aliases of each other. The canonical name is the first key (in definition
    order) that maps to a given builder — i.e. the one returned by
    builder().get_name().
    """
    builder_to_first_key: dict[Callable[..., QuantizationConfig], str] = {}
    for key in _OVERRIDES:
        builder = _OVERRIDES[key]
        if builder not in builder_to_first_key:
            builder_to_first_key[builder] = key

    result: dict[str, str] = {}
    for key, builder in _OVERRIDES.items():
        canonical = builder_to_first_key[builder]
        result[key.lower().replace("-", "_")] = canonical
    return result


_CACHED_ALIAS_MAP: dict[str, str] | None = None


def _normalize_quant_method_alias(method: str | None) -> str | None:
    """Map a method name (or any of its aliases) to its canonical internal name.
    Returns the input unchanged if it is not a known alias.
    """
    if method is None:
        return None
    global _CACHED_ALIAS_MAP
    if _CACHED_ALIAS_MAP is None:
        _CACHED_ALIAS_MAP = _build_reverse_alias_map()
    normalized = method.lower().replace("-", "_")
    return _CACHED_ALIAS_MAP.get(normalized, normalized)


_MODEL_OPT_METHODS = {
    "modelopt",
    "modelopt_fp4",
    "modelopt_mixed",
}
_MODEL_OPT_FP8_ALGOS = {
    "FP8",
    "FP8_PER_CHANNEL_PER_TOKEN",
}
_MODEL_OPT_NVFP4_ALGOS = {
    "NVFP4",
    "W4A16_NVFP4",
}
_MODEL_OPT_MIXED_ALGOS = {
    "MIXED_PRECISION",
}


def _normalize_method_name(method: Any) -> str:
    return str(method).lower().replace("-", "_")


def _get_modelopt_quant_algo(config: Mapping[str, Any]) -> str:
    quantization = config.get("quantization")
    if isinstance(quantization, Mapping):
        quant_algo = quantization.get("quant_algo")
        if quant_algo:
            return str(quant_algo).upper()
    return str(config.get("quant_algo", "")).upper()


def _detect_modelopt_method(config: Mapping[str, Any]) -> str | None:
    quant_algo = _get_modelopt_quant_algo(config)

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
        if quant_algo in _MODEL_OPT_NVFP4_ALGOS:
            return "modelopt_fp4"
        if quant_algo in _MODEL_OPT_MIXED_ALGOS:
            return "modelopt_mixed"
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


def _disk_marks_serialized(qc_kwargs: dict[str, Any], quant_config: object) -> bool:
    """Return True when config.json says serialized but the active quant_config does not.

    Matches any flag following the is_checkpoint_*_serialized naming convention,
    so new quant methods don't require updating an explicit allowlist.
    """
    for key, val in qc_kwargs.items():
        if key.startswith("is_checkpoint_") and key.endswith("_serialized"):
            if val and hasattr(quant_config, key) and not getattr(quant_config, key):
                return True
    return False


def resolve_quant_config_from_disk(
    quant_config: QuantizationConfig | None,
    disk_qc: dict[str, Any] | str | None,
) -> QuantizationConfig | None:
    """Reconcile an active quant_config against quantization_config from a transformer's config.json.

    Used when loading individual transformer blocks that each have their own config.json
    (e.g. cascade models with separate transformer and transformer_2 directories).

    Rules:
      - disk_qc is None: return quant_config unchanged.
      - quant_config is None: auto-detect from disk_qc (full build).
      - Methods mismatch: raise ValueError — prevents silent weight corruption.
      - Disk marks serialized but quant_config is online: rebuild from disk.
      - ignored_layers differ: rebuild from disk (per-transformer BF16 routing).
    """
    if disk_qc is None:
        return quant_config

    if isinstance(disk_qc, str):
        if quant_config is None:
            logger.info("Auto-detected quantization from config.json: method=%s", disk_qc)
            return build_quant_config(disk_qc)
        return quant_config

    if not isinstance(disk_qc, Mapping):
        return quant_config

    disk_spec = dict(disk_qc)
    raw_method = disk_spec.get("quant_method", disk_spec.get("method"))
    detected_modelopt_method = _detect_modelopt_method(disk_spec)
    if detected_modelopt_method is None and raw_method is None:
        return quant_config

    disk_method = detected_modelopt_method or str(raw_method)
    qc_kwargs: dict[str, Any] = {k: v for k, v in disk_spec.items() if k not in {"method", "quant_method"}}

    if quant_config is None:
        logger.info(
            "Auto-detected quantization from config.json: method=%s kwargs=%s",
            disk_method,
            qc_kwargs,
        )
        return build_quant_config(disk_spec)

    active_method = _normalize_quant_method_alias(quant_config.get_name())
    normalized_disk_method = _normalize_quant_method_alias(disk_method)
    if active_method != normalized_disk_method:
        raise ValueError(
            f"Checkpoint config.json declares quant_method={raw_method!r} "
            f"(resolved as {disk_method!r}) but the "
            f"active quantization config is {quant_config.get_name()!r}. "
            "Pass a matching --quantization flag or omit it for auto-detection."
        )

    if detected_modelopt_method is not None:
        disk_quant_algo = _get_modelopt_quant_algo(disk_spec)
        if detected_modelopt_method == "modelopt_mixed":
            logger.info("Using per-transformer checkpoint ModelOpt mixed policy.")
            return build_quant_config(disk_spec)

        active_quant_algo = str(getattr(quant_config, "quant_method", "")).upper()
        serialized_flag = {
            "modelopt": "is_checkpoint_fp8_serialized",
            "modelopt_fp4": "is_checkpoint_nvfp4_serialized",
        }.get(detected_modelopt_method)
        active_is_serialized = serialized_flag is None or bool(getattr(quant_config, serialized_flag, False))
        if active_quant_algo != disk_quant_algo or not active_is_serialized:
            logger.info("Using checkpoint ModelOpt config for quant_algo=%s.", disk_quant_algo)
            return build_quant_config(disk_spec)

    if _disk_marks_serialized(qc_kwargs, quant_config):
        logger.info(
            "config.json marks checkpoint as serialized; switching to offline %s mode.",
            disk_method,
        )
        return build_quant_config(disk_spec)

    # AutoRound MXFP8 checkpoints use data_type="mx_fp" instead of
    # is_checkpoint_*_serialized; rebuild so the offline path is selected.
    if qc_kwargs.get("data_type") == "mx_fp":
        logger.info("config.json declares data_type='mx_fp'; rebuilding as offline AutoRound MXFP8.")
        return build_quant_config(disk_spec)

    if (
        "ignored_layers" in qc_kwargs
        and hasattr(quant_config, "ignored_layers")
        and set(qc_kwargs.get("ignored_layers") or []) != set(quant_config.ignored_layers or [])
    ):
        logger.info("config.json ignored_layers differs from active config; rebuilding quant_config.")
        return build_quant_config(disk_spec)

    return quant_config
