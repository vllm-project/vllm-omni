# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Factory for building quantization configs.

build_quantization_config() delegates to vLLM's quantization registry. Omni
configs are registered into that registry by register_omni_quantization_configs().
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.repo_utils import file_or_path_exists, get_hf_file_to_dict

from vllm_omni.utils.model_source import materialize_object_storage_configs


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
    setattr(registry["humming"], "config", registry["humming.config"])
    setattr(registry["humming"], "dtypes", registry["humming.dtypes"])
    setattr(registry["humming"], "layer", registry["humming.layer"])
    setattr(registry["humming"], "schema", registry["humming.schema"])
    setattr(registry["humming"], "utils", registry["humming.utils"])
    setattr(registry["humming.utils"], "weight", registry["humming.utils.weight"])

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
from vllm.model_executor.layers.quantization.fp8 import Fp8Config  # noqa: E402
from vllm.model_executor.layers.quantization.modelopt import (  # noqa: E402
    ModelOptFp8Config,
    ModelOptNvFp4Config,
)

from .component_config import ComponentQuantizationConfig  # noqa: E402

logger = init_logger(__name__)

# Aliased spec keys for the quant method HF. In the future, we should consider
# deprecating one of these; for now we use get/set utils below to that if we do,
# it is easy to rip the deprecated one out.
METHOD_KEY = "method"
QUANT_METHOD_KEY = "quant_method"


def get_quantization_method(spec: Mapping[str, Any]) -> str | None:
    """Read the method from a quant spec, accepting either key spelling.

    Raises when both keys are present but disagree: they are aliases, so a spec
    declaring method="int8" and quant_method="fp8" is contradictory input.
    """
    method = spec.get(METHOD_KEY)
    quant_method = spec.get(QUANT_METHOD_KEY)
    if method is not None and quant_method is not None and method != quant_method:
        raise ValueError(
            f"Conflicting quantization method keys: {METHOD_KEY}={method!r} vs "
            f"{QUANT_METHOD_KEY}={quant_method!r}. They are aliases and must agree."
        )
    return method if method is not None else quant_method


def set_quantization_method(spec: dict[str, Any], method: str) -> None:
    """Record the method under the checkpoint key, without clobbering an existing one."""
    spec.setdefault(QUANT_METHOD_KEY, method)


def register_omni_quantization_configs() -> None:
    """Import omni quant config modules so their @register_quantization_config
    decorators fire. This ensures that Omni's quantization definitions are registered
    over vLLM's quantization definitions, which ensures that the same quantization
    definitions are used in vLLM's ModelConfig.get_quantization_config() path (i.e.,
    for AR) as in the diffusion factory lookup.
    """
    from . import (  # noqa: F401  (import side-effect = decorator registration)
        bitsandbytes_config,
        inc_config,
        int8_config,
        mxfp4_config,
        mxfp8_config,
        torchao_config,
    )


# Omni configs registered into vLLM's registry. Static so membership/count is
# independent of when registration fires; auto-round spellings alias to inc.
_OMNI_QUANT_METHODS = (
    "int8",
    "bitsandbytes",
    "mxfp8",
    "mxfp4",
    "mxfp4_dualscale",
    "svdquant",
    "inc",
    "auto-round",
    "auto_round",
    "torchao",
    "torchao_float8_weight_only",
)
SUPPORTED_QUANTIZATION_METHODS: list[str] = list(dict.fromkeys([*QUANTIZATION_METHODS, *_OMNI_QUANT_METHODS]))

_QUANT_METHOD_ALIASES = {"auto-round": "inc", "auto_round": "inc"}


_GENERIC_FP8_NAMES = frozenset({"fp8"})
_GENERIC_NVFP4_NAMES = frozenset({"fp4", "nvfp4", "modelopt_fp4"})


def should_adopt_checkpoint_quant_config(active: QuantizationConfig | None, checkpoint: QuantizationConfig) -> bool:
    """Decide whether a checkpoint's own quant config should replace the active one.

    Adopt when nothing is set yet, or when the active config is a generic online
    request (bare "fp8"/fp4) but the checkpoint carries serialized weights of the
    same family — those must load with is_checkpoint_*_serialized=True.

    vLLM records serialized-checkpoint provenance only on the concrete fp8/nvfp4
    config classes (never on the QuantizationConfig base), so we isinstance-narrow
    to the carriers and read the flag as a typed property.
    """
    if active is None:
        return True
    name = active.get_name()
    if (
        isinstance(checkpoint, (Fp8Config, ModelOptFp8Config))
        and checkpoint.is_checkpoint_fp8_serialized
        and name in _GENERIC_FP8_NAMES
    ):
        return True
    if (
        isinstance(checkpoint, ModelOptNvFp4Config)
        and checkpoint.is_checkpoint_nvfp4_serialized
        and name in _GENERIC_NVFP4_NAMES
    ):
        return True
    return False


def _normalize_quant_method_alias(method: str | None) -> str | None:
    """Fold known aliases (auto-round/auto_round to inc); pass everything else through."""
    if method is None:
        return None
    return _QUANT_METHOD_ALIASES.get(method, method)


_MODEL_OPT_METHODS = {"modelopt", "modelopt_fp4", "modelopt_mixed"}
_MODEL_OPT_ALGO_TO_METHOD = {
    "FP8": "modelopt",
    "FP8_PER_CHANNEL_PER_TOKEN": "modelopt",
    "NVFP4": "modelopt_fp4",
    "MIXED_PRECISION": "modelopt_mixed",
}


# TODO(Alex): vLLM's ModelOpt configs probably already detect this with
# override_quantization_method, so we may be able to leverage the upstream code for this.
def _detect_modelopt_method(config: Mapping[str, Any]) -> str | None:
    quantization = config.get("quantization")
    if isinstance(quantization, Mapping):
        quant_algo = str(quantization.get("quant_algo", "")).upper()
    else:
        quant_algo = str(config.get("quant_algo", "")).upper()

    method = get_quantization_method(config)
    # NOTE: We normalize by replacing hyphens with underscores inline here
    # instead of doing it generally, since in other cases this may create
    # an invalid method.
    normalized_method = str(method).lower().replace("-", "_") if method is not None else None

    producer = config.get("producer")
    is_modelopt_config = normalized_method in _MODEL_OPT_METHODS or (
        isinstance(producer, Mapping) and str(producer.get("name", "")).lower() == "modelopt"
    )

    if not is_modelopt_config:
        return None

    if quant_algo:
        return _MODEL_OPT_ALGO_TO_METHOD.get(quant_algo)

    if normalized_method in _MODEL_OPT_METHODS:
        return normalized_method

    return None


def maybe_build_modelopt_from_config(config: Mapping[str, Any]) -> QuantizationConfig | None:
    """Build a ModelOpt config when config is a ModelOpt checkpoint, else None."""
    method = _detect_modelopt_method(config)
    if method is None:
        return None
    config_cls = get_quantization_config(method)
    normalized_config = dict(config)
    set_quantization_method(normalized_config, method)
    return config_cls.from_config(normalized_config)


def _pop_method_name(spec: dict[str, Any]) -> str | None:
    """Pops the method key (including for aliases) from the quantization config dict."""
    method = get_quantization_method(spec)  # validates the aliases agree before we drop them
    spec.pop(METHOD_KEY, None)
    spec.pop(QUANT_METHOD_KEY, None)
    if method is not None and not isinstance(method, str):
        raise TypeError(f"{METHOD_KEY!r}/{QUANT_METHOD_KEY!r} must be a string, got {type(method).__name__}")
    return method


def _is_per_component_dict(spec: dict[str, Any]) -> bool:
    """Check if a dict describes per-component quantization.

    A per-component dict has no "method" / "quant_method" key and all values are
    str, dict, or None. To avoid misdetecting a flat config with
    all-string values (e.g. {"activation_scheme": "static"}), we
    require at least one value to be None or a dict with "method" /
    "quant_method".
    """
    if METHOD_KEY in spec or QUANT_METHOD_KEY in spec:
        return False
    if not all(isinstance(v, (dict, str, type(None))) for v in spec.values()):
        return False
    return any(v is None or (isinstance(v, dict) and (METHOD_KEY in v or QUANT_METHOD_KEY in v)) for v in spec.values())


def _maybe_build_component_quant_config(
    spec: dict[str, Any],
    quant_config: dict[str, Any] | None,
) -> ComponentQuantizationConfig | None:
    if not _is_per_component_dict(spec):
        return None
    component_configs: dict[str, QuantizationConfig | None] = {}
    default_config: QuantizationConfig | None = None
    for prefix, value in spec.items():
        if not isinstance(value, (str, dict, QuantizationConfig, type(None))):
            raise TypeError(
                f"Per-component value for {prefix!r} must be str, dict, "
                f"QuantizationConfig, or None, got {type(value).__name__}"
            )
        resolved = build_quantization_config(value, quant_config)
        if prefix == "default":
            default_config = resolved
        else:
            component_configs[prefix] = resolved
    return ComponentQuantizationConfig(component_configs, default_config)


def build_quantization_config(
    quantization: str | dict[str, Any] | QuantizationConfig | None,
    quant_config: dict[str, Any] | None = None,
) -> QuantizationConfig | None:
    """Build a resolved QuantizationConfig.

    Examples::

        build_quantization_config("fp8")
        build_quantization_config("fp8", {"quant_method": "fp8", "is_checkpoint_fp8_serialized": True})
        build_quantization_config({"method": "fp8", "activation_scheme": "static"})
        build_quantization_config({"transformer": "fp8", "vae": None}) # component config

    Args:
        quantization: Method string, dict spec, QuantizationConfig passthrough, or None.
        quant_config: Checkpoint quantization metadata dict (e.g. from a model's
            config.json ``quantization_config`` field). Passed to ``from_config()``
            for checkpoint-quantized models. Omit for online quantization.
    """
    if isinstance(quantization, QuantizationConfig):
        return quantization

    # If we don't pass quantization, we can still grab it from the checkpoint's config
    if quantization is None:
        if isinstance(quant_config, Mapping):
            quantization = get_quantization_method(quant_config)
        if quantization is None:
            return None

    # Since we need to build a quant config, ensure Omni quant defs are registered
    register_omni_quantization_configs()

    if isinstance(quantization, Mapping):
        spec = dict(quantization)
        component_cfg = _maybe_build_component_quant_config(spec, quant_config)
        if component_cfg is not None:
            return component_cfg

        # NOTE: This is explicitly using 'quant_method' instead of the getters and
        # setters with the alias because int & fp8, the latter of which lives in vLLM,
        # derive is_checkpoint_*_serialized based on the method name.
        from_checkpoint = QUANT_METHOD_KEY in spec
        quantization = _pop_method_name(spec)
        if quantization is None:
            raise ValueError(
                f"Dict quantization config must have a {METHOD_KEY!r} or {QUANT_METHOD_KEY!r} key "
                "or be a per-component config with component prefixes as keys."
            )
        # A dict spec may inline the modelopt algo (even under the "method" key),
        # so always check it; a bare method string can't, hence the else gate.
        detect_modelopt = True
    else:
        spec = dict(quant_config) if isinstance(quant_config, dict) else {}
        from_checkpoint = QUANT_METHOD_KEY in spec
        # Only a checkpoint carries an algo to disambiguate; a bare method string
        # (no quant_config) is a user request and constructs directly below.
        detect_modelopt = from_checkpoint

    # ModelOpt records its algo (FP8/NVFP4/mixed) separately from
    # quant_method="modelopt"; disambiguate on the effective checkpoint dict so
    # the right class is picked no matter which arg carried the checkpoint.
    if detect_modelopt:
        modelopt = maybe_build_modelopt_from_config({QUANT_METHOD_KEY: quantization, **spec})
        if modelopt is not None:
            return modelopt

    method = _normalize_quant_method_alias(quantization)
    if method == "none":
        return None

    if method not in QUANTIZATION_METHODS:
        raise ValueError(f"Unknown quantization method: {method!r}. Supported: {SUPPORTED_QUANTIZATION_METHODS}")

    # Checkpoint dicts go through from_config (plucks only wanted keys); inline
    # specs construct directly. Restore quant_method popped above, since some
    # from_config impls read it (e.g. int8 derives is_checkpoint_*_serialized).
    quant_cls = get_quantization_config(method)
    if from_checkpoint:
        set_quantization_method(spec, quantization)
        return quant_cls.from_config(spec)
    return quant_cls(**spec)


@functools.cache
def read_checkpoint_quantization_config(model: str) -> dict[str, Any] | None:
    """Read a checkpoint's serialized quantization_config from config.json, or the
    hf_quant_config.json sidecar (ModelOpt<=0.29)."""
    source = materialize_object_storage_configs(model)
    quant = None
    if file_or_path_exists(source, "config.json", None):
        quant = get_hf_file_to_dict("config.json", source, revision=None).get("quantization_config")
    # See: https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/transformers_utils/config.py#L765
    if quant is None and file_or_path_exists(source, "hf_quant_config.json", None):
        quant = get_hf_file_to_dict("hf_quant_config.json", source, revision=None)

    if quant is not None and not isinstance(quant, dict):
        raise TypeError(f"quantization_config for {model!r} must be a dict or None, got {type(quant).__name__}")
    return quant


def _disk_marks_serialized(qc_kwargs: dict[str, Any], quant_config: QuantizationConfig) -> bool:
    """Return True when config.json says serialized but the active quant_config does not.

    Matches any flag following the is_checkpoint_*_serialized naming convention,
    so new quant methods don't require updating an explicit allowlist.
    """
    for key, val in qc_kwargs.items():
        if key.startswith("is_checkpoint_") and key.endswith("_serialized"):
            if val and hasattr(quant_config, key) and not getattr(quant_config, key):
                return True
    return False


def maybe_rebuild_quantization_config(quant_config: QuantizationConfig, disk_qc: dict[str, Any]) -> QuantizationConfig:
    """Produce the final quantization config, which will either be a newly built config ,
    or a handle to the original if we can reuse it. Currently this is only applicable for
    models that need to consider that case where we may have multiple quantization configs,
    E.g., wan2_2.
    """
    qc_method = get_quantization_method(disk_qc)
    qc_kwargs = {k: v for k, v in disk_qc.items() if k not in (METHOD_KEY, QUANT_METHOD_KEY)}
    if _disk_marks_serialized(qc_kwargs, quant_config):
        logger.info(
            "config.json marks checkpoint as serialized; switching to offline %s mode.",
            qc_method,
        )
        return build_quantization_config(qc_method, disk_qc)

    # AutoRound MXFP8 checkpoints use data_type="mx_fp" instead of
    # is_checkpoint_*_serialized; rebuild so the offline path is selected.
    if qc_kwargs.get("data_type") == "mx_fp":
        logger.info("config.json declares data_type='mx_fp'; rebuilding as offline AutoRound MXFP8.")
        return build_quantization_config(qc_method, disk_qc)

    if (
        "ignored_layers" in qc_kwargs
        and hasattr(quant_config, "ignored_layers")
        and set(qc_kwargs.get("ignored_layers") or []) != set(quant_config.ignored_layers or [])
    ):
        logger.info("config.json ignored_layers differs from active config; rebuilding quant_config.")
        return build_quantization_config(qc_method, disk_qc)
    return quant_config


def resolve_quantization_config_from_disk(
    quant_config: QuantizationConfig | None,
    disk_qc: dict[str, Any] | str | None,
) -> QuantizationConfig | None:
    """Reconcile an active quant_config against a transformer's config.json.

    Used for cascade models where individual transformer blocks have their
    own config.json (e.g. separate transformer and transformer_2 directories).
    Returns the disk config when it carries more specific info than the active one.
    """
    if disk_qc is None:
        return quant_config

    if quant_config is None:
        return build_quantization_config(disk_qc, disk_qc if isinstance(disk_qc, dict) else None)

    if isinstance(disk_qc, str):
        return quant_config

    disk_method = get_quantization_method(disk_qc)

    if not disk_method:
        return quant_config

    disk_method = _normalize_quant_method_alias(disk_method)
    active_method = _normalize_quant_method_alias(quant_config.get_name())
    if active_method != disk_method:
        raise ValueError(
            f"Checkpoint config.json declares quant_method={disk_method!r} "
            f"but the active quantization config is {quant_config.get_name()!r}."
        )

    return maybe_rebuild_quantization_config(quant_config, disk_qc)
