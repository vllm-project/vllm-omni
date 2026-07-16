from __future__ import annotations

import json
import os
import types
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config, get_hf_file_to_dict
from vllm.transformers_utils.repo_utils import file_or_path_exists

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.endpoint_policy import EndpointRestriction
from vllm_omni.config.yaml_util import create_config, load_yaml_config, merge_configs
from vllm_omni.diffusion.utils.hf_utils import _looks_like_dreamzero
from vllm_omni.platforms import current_omni_platform

PROJECT_ROOT = Path(__file__).parent.parent.parent

logger = init_logger(__name__)


_DIFFUSERS_CLASS_TO_CONFIG: dict[str, str] = {
    "GlmImagePipeline": "glm_image",
}


@dataclass(frozen=True)
class OmniConfigResolveRequest:
    """All source-selection inputs for one Omni configuration resolution."""

    model: str
    legacy_stage_configs_path: str | None = None
    trust_remote_code: bool = False
    deploy_config_path: str | None = None
    cli_overrides: Mapping[str, Any] = field(default_factory=dict)
    stage_overrides: Mapping[str, Mapping[str, Any]] | str | None = None
    strategy_config_path: str | None = None


@dataclass(frozen=True)
class OmniConfigResolution:
    """Resolved runtime input while stage consumers still use OmegaConf."""

    config_path: str | None
    stage_configs: tuple[Any, ...]
    omni_lb_policy: str | None = None
    endpoint_restrictions: tuple[EndpointRestriction, ...] = ()

    def stage_by_id(self, stage_id: int) -> Any:
        for stage in self.stage_configs:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"no stage {stage_id}")


def _try_get_class_name_from_diffusers_config(model: str) -> str | None:
    """Try to get class name from diffusers model configuration files.

    Args:
        model: Model name or path

    Returns:
        Model type string if found, None otherwise
    """
    model_index = get_hf_file_to_dict("model_index.json", model, revision=None)
    if model_index and isinstance(model_index, dict) and "_class_name" in model_index:
        logger.debug(f"Found model_type '{model_index['_class_name']}' in model_index.json")
        return model_index["_class_name"]

    return None


def _filter_dict_like_object(obj: dict | Any) -> dict:
    """Filter dict-like object by removing callables and recursively converting values.

    Converts dict-like objects to regular dicts while filtering out callable values
    that are incompatible with OmegaConf. Recursively processes values through
    _convert_dataclasses_to_dict for nested object conversion.

    Args:
        obj: Dict or dict-like object to filter

    Returns:
        Regular dict with callables filtered out and values recursively converted

    Raises:
        TypeError: If obj doesn't support .items() method
        ValueError: If dict conversion fails unexpectedly
    """

    def _is_callable_value(value: Any) -> bool:
        if callable(value):
            return True
        return isinstance(
            value,
            (
                types.FunctionType,
                types.MethodType,
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
            ),
        )

    result = {}
    filtered_keys = []
    for k, v in obj.items():
        # Preserve class objects by converting to a fully qualified name string
        # so callers that resolve via import path (e.g. custom_pipeline_args.pipeline_class)
        # still work after OmegaConf round-trip.
        if isinstance(v, type):
            module = getattr(v, "__module__", None)
            qualname = getattr(v, "__qualname__", getattr(v, "__name__", None))
            if module and qualname and module != "builtins":
                result[k] = f"{module}.{qualname}"
            else:
                result[k] = qualname
        elif _is_callable_value(v):
            filtered_keys.append(str(k))
        else:
            result[k] = _convert_dataclasses_to_dict(v)
    if filtered_keys:
        logger.warning(
            f"Filtered out {len(filtered_keys)} callable object(s) from base_engine_args "
            f"that are not compatible with OmegaConf: {filtered_keys}. "
        )
    return result


def _convert_dataclasses_to_dict(obj: Any) -> Any:
    """Recursively convert non-serializable objects to OmegaConf-compatible types.

    This is needed because OmegaConf cannot handle:
    - Dataclass objects with Literal type annotations (e.g., StructuredOutputsConfig)
    - Counter objects (from collections or vllm.utils)
    - Set objects
    - Callable objects (functions, methods, etc.)
    - Other non-primitive types
    """
    # IMPORTANT: Check Counter BEFORE dict, since Counter is a subclass of dict
    # Handle Counter objects (convert to dict)
    # Check by class name first to catch both collections.Counter and vllm.utils.Counter
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "Counter":
        try:
            return dict(obj)
        except (TypeError, ValueError):
            # If Counter can't be converted to dict, return empty dict
            return {}
    # Also check isinstance for collections.Counter (must be before dict check)
    if isinstance(obj, Counter):
        return dict(obj)
    # Handle set objects (convert to list)
    if isinstance(obj, set):
        return list(obj)
    # Handle dataclass objects
    # Use field iteration instead of asdict() to:
    # 1. Only include init fields (non-init fields cause "unexpected kwarg" errors)
    # 2. Skip None values matching field defaults (avoids Pydantic validation
    #    failures when None is explicitly passed for non-Optional typed fields,
    #    e.g. CompilationConfig.cudagraph_capture_sizes: list[int] = None)
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in fields(obj):
            if not f.init:
                continue
            value = getattr(obj, f.name)
            if value is None and f.default is None:
                continue
            result[f.name] = _convert_dataclasses_to_dict(value)
        return result
    # Handle dictionaries (recurse into values) and filter out callables(cause error in OmegaConf.create)
    # Note: This must come AFTER Counter check since Counter is a dict subclass
    if isinstance(obj, dict):
        return _filter_dict_like_object(obj)
    # Preserve class objects by converting to a fully qualified name string.
    if isinstance(obj, type):
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", None))
        if module and qualname and module != "builtins":
            return f"{module}.{qualname}"
        return qualname
    # Handle callable objects (functions, methods, etc.) - skip them
    # Note: This comes after dict/list checks to avoid misclassifying dict-like objects
    if callable(obj):
        return None
    # Handle lists and tuples (recurse into items)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_convert_dataclasses_to_dict(item) for item in obj if not callable(item))
    # Try to convert any dict-like object (has keys/values methods) to dict
    if hasattr(obj, "keys") and hasattr(obj, "values") and not isinstance(obj, (str, bytes)):
        try:
            return _filter_dict_like_object(obj)
        except (TypeError, ValueError, AttributeError):
            # If conversion fails, return as-is
            return obj
    # Primitive types and other objects that OmegaConf can handle
    return obj


def _try_resolve_omni_model_type(model: str) -> str | None:
    """Try to resolve model_type for omni models with empty config.json.

    Searches both the legacy ``stage_configs/*.yaml`` directory and the
    migrated ``deploy/*.yaml`` directory for a stem that substring-matches
    the model path (e.g. ``cosyvoice3`` in
    ``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``). The longest match wins so
    ``cosyvoice3`` beats ``cosyvoice`` and ``bagel_single_stage`` beats
    ``bagel``.
    """
    model_lower = model.lower().replace("-", "").replace("_", "")
    best_match: str | None = None
    best_len = 0
    for subdir in ("model_executor/stage_configs", "deploy"):
        config_dir = PROJECT_ROOT / "vllm_omni" / subdir
        if not config_dir.exists():
            continue
        for config_file in sorted(config_dir.glob("*.yaml")):
            candidate = config_file.stem.replace("-", "").replace("_", "")
            if candidate and candidate in model_lower and len(candidate) > best_len:
                best_match = config_file.stem
                best_len = len(candidate)
    return best_match


def resolve_model_config_path(model: str) -> str:
    """Resolve the stage config file path from the model name.

    Resolves stage configuration path based on the model type and device type.
    First tries to find a device-specific YAML file from stage_configs/{device_type}/
    directory. If not found, falls back to the default config file.

    Args:
        model: Model name or path (used to determine model_type)

    Returns:
        String path to the stage configuration file

    Raises:
        ValueError: If model_type cannot be determined
        FileNotFoundError: If no stage config file exists for the model type
    """
    # Try to get config from standard transformers format first
    try:
        hf_config = get_config(model, trust_remote_code=True)
        model_type = hf_config.model_type
    except (ValueError, Exception):
        # If standard transformers format fails, try diffusers format
        if file_or_path_exists(model, "model_index.json", revision=None):
            model_type = _try_get_class_name_from_diffusers_config(model)
            if model_type is None:
                raise ValueError(
                    f"Could not determine model_type for diffusers model: {model}. "
                    f"Please ensure the model has 'model_type' in transformer/config.json or model_index.json"
                )
        elif file_or_path_exists(model, "config.json", revision=None):
            # Try to read config.json manually for custom models like Bagel that fail get_config
            # but have a valid config.json with model_type
            try:
                config_dict = get_hf_file_to_dict("config.json", model, revision=None)
                if config_dict and "model_type" in config_dict:
                    model_type = config_dict["model_type"]
                else:
                    # For models with empty config.json (e.g. CosyVoice3),
                    # try matching against registered omni stage configs.
                    model_type = _try_resolve_omni_model_type(model)
                    if model_type is None:
                        raise ValueError(f"config.json found but missing 'model_type' for model: {model}")
            except Exception as e:
                raise ValueError(f"Failed to read config.json for model: {model}. Error: {e}") from e
        else:
            # No config.json at repo root (e.g. GLM-TTS stores configs in
            # subdirectories only).  Try matching against registered deploy
            # YAML filenames before giving up.
            model_type = _try_resolve_omni_model_type(model)
            if model_type is None:
                raise ValueError(
                    f"Could not determine model_type for model: {model}. "
                    f"Model is not in standard transformers format and does not have model_index.json. "
                    f"Please ensure the model has proper configuration files with 'model_type' field"
                )

    default_config_path = current_omni_platform.get_default_stage_config_path()
    if model_type == "vla" and _looks_like_dreamzero(model):
        model_type = "dreamzero"

    if model_type in _DIFFUSERS_CLASS_TO_CONFIG:
        normalized_model_type = _DIFFUSERS_CLASS_TO_CONFIG[model_type]
    else:
        normalized_model_type = model_type.replace("-", "_")
    model_type_str = f"{normalized_model_type}.yaml"
    complete_config_path = PROJECT_ROOT / default_config_path / model_type_str
    if os.path.exists(complete_config_path):
        return str(complete_config_path)

    deploy_config_path = PROJECT_ROOT / "vllm_omni" / "deploy" / model_type_str
    if os.path.exists(deploy_config_path):
        return str(deploy_config_path)

    stage_config_file = f"vllm_omni/model_executor/stage_configs/{normalized_model_type}.yaml"
    stage_config_path = PROJECT_ROOT / stage_config_file
    if not os.path.exists(stage_config_path):
        return None
    return str(stage_config_path)


def _load_stage_configs_from_model(
    model: str,
    *,
    trust_remote_code: bool,
    base_engine_args: dict | None = None,
    deploy_config_path: str | None = None,
    stage_overrides: dict[str, dict[str, Any]] | None = None,
    strategy_config_path: str | None = None,
) -> tuple[list, str | None]:
    """Load stage configurations from model's default config file.

    For models registered in the pipeline registry (new path), uses
    ``StageConfigFactory.create_legacy_stage_configs_from_model()`` which merges
    PipelineConfig + DeployConfig + CLI overrides.

    For other models (legacy path), loads stage configs from YAML.

    Args:
        model: Model name or path (used to determine model_type)
        trust_remote_code: Whether to trust remote model configuration code.
        base_engine_args: Base engine args to merge as CLI overrides.
        deploy_config_path: Optional explicit deploy config path.
        stage_overrides: Per-stage overrides from --stage-overrides.
        strategy_config_path: Optional path to a composable-parallel
            ``strategy.yaml`` whose derived sizing is overlaid onto the
            registry-merged stages (opt-in; ignored on the legacy YAML path).

    Returns:
        ``(stage_configs, omni_lb_policy)``: the list of stage configuration
        dictionaries plus the strategy-derived pipeline-wide ``omni_lb_policy``
        (``None`` when no strategy set one). The policy is returned rather than
        written into a caller-provided mutable dict.
    """
    if base_engine_args is None:
        base_engine_args = {}

    cli_overrides = _convert_dataclasses_to_dict(dict(base_engine_args))
    if stage_overrides:
        for stage_id_str, overrides in stage_overrides.items():
            for key, val in overrides.items():
                cli_overrides[f"stage_{stage_id_str}_{key}"] = val

    strategy_specs = None
    if strategy_config_path is not None:
        from vllm_omni.config.composable_parallel.strategy_loader import load_strategy_specs

        strategy_specs = load_strategy_specs(strategy_config_path)

    stages, omni_lb_policy = StageConfigFactory.create_legacy_stage_configs_from_model(
        model,
        trust_remote_code=trust_remote_code,
        cli_overrides=cli_overrides,
        deploy_config_path=deploy_config_path,
        strategy_specs=strategy_specs,
    )
    if stages is not None:
        # Convert StageConfig objects to OmegaConf for backward compat
        return [stage.to_omegaconf() for stage in stages], omni_lb_policy

    # Legacy fallback: load from YAML. A composable-parallel strategy cannot be
    # applied here (it overlays onto registry-merged stages), so warn rather than
    # silently dropping the operator's --strategy-config.
    if strategy_config_path is not None:
        logger.warning(
            "--strategy-config (%s) was provided but model %r resolves via the "
            "legacy stage_configs YAML path, which does not support "
            "composable-parallel strategies; the strategy is ignored. Use a "
            "registry-based model to apply it.",
            strategy_config_path,
            model,
        )
    stage_config_path = resolve_model_config_path(model)
    if stage_config_path is None:
        return [], None
    stage_configs = _load_stage_configs_from_yaml(
        config_path=stage_config_path,
        base_engine_args=base_engine_args,
        prefer_stage_engine_args=True,
    )
    return stage_configs, None


def _load_stage_configs_from_yaml(
    config_path: str,
    base_engine_args: dict | None = None,
    prefer_stage_engine_args: bool = True,
) -> list:
    """Load stage configurations from a YAML file (legacy OmegaConf path).

    TODO(@lishunyang12): remove once all models use PipelineConfig + DeployConfig.

    Args:
        config_path: Path to the YAML configuration file
        base_engine_args: Engine args supplied by the caller.
        prefer_stage_engine_args: When True, YAML stage args override caller
            engine args. When False, caller engine args override YAML defaults.

    Returns:
        List of stage configuration dictionaries from the file's stage_args
    """
    if base_engine_args is None:
        base_engine_args = {}
    config_data = load_yaml_config(config_path)
    stage_args = config_data.stage_args
    global_async_chunk = config_data.get("async_chunk", False)
    # Convert any nested dataclass objects to dicts before creating DictConfig
    base_engine_args = _convert_dataclasses_to_dict(base_engine_args)
    base_engine_args = create_config(base_engine_args)
    for stage_arg in stage_args:
        base_engine_args_tmp = base_engine_args.copy()
        # Update base_engine_args with stage-specific engine_args if they exist
        if hasattr(stage_arg, "engine_args") and stage_arg.engine_args is not None:
            if prefer_stage_engine_args:
                merged_engine_args = merge_configs(base_engine_args_tmp, stage_arg.engine_args)
            else:
                merged_engine_args = merge_configs(stage_arg.engine_args, base_engine_args_tmp)
            base_engine_args_tmp = create_config(merged_engine_args)
        stage_type = getattr(stage_arg, "stage_type", "llm")
        if hasattr(stage_arg, "runtime") and stage_arg.runtime is not None and stage_type != "diffusion":
            base_engine_args_tmp.async_chunk = global_async_chunk
        stage_arg.engine_args = base_engine_args_tmp
    return stage_args


def _filter_stages(
    config_path: str | None,
    stage_configs: list,
    kwargs: dict | None,
) -> list:
    """Filter stage configs by mode when YAML defines a `modes` section.

    The YAML can define, e.g.:

        modes:
          - mode: text-to-image
            stages: [1]
          - mode: image-to-text
            stages: [0]

    When users pass `mode="image-to-text"` into Omni(**kwargs), only the stages
    listed for that mode are returned. If no mode is provided, defaults to
    "text-to-image". If no modes are defined or filtering fails, returns the
    original stage_configs unchanged.

    Args:
        config_path: Path to the YAML config (used to read `modes`).
        stage_configs: Loaded list of stage configs.
        kwargs: Engine/caller kwargs; may contain "mode".

    Returns:
        Filtered list of stage configs (or original list if filtering not applied).
    """
    if not stage_configs or config_path is None:
        return stage_configs

    try:
        cfg = load_yaml_config(config_path)
        yaml_modes = getattr(cfg, "modes", None)
        if yaml_modes is None:
            return stage_configs

        mode_to_stage_ids: dict[str, list[int]] = {}
        if yaml_modes is not None:
            for entry in yaml_modes:
                mode_name = None
                stages = None
                if hasattr(entry, "mode") or hasattr(entry, "stages"):
                    mode_name = getattr(entry, "mode", None)
                    stages = getattr(entry, "stages", None)
                elif isinstance(entry, dict):
                    mode_name = entry.get("mode")
                    stages = entry.get("stages")

                if mode_name is None or stages is None:
                    continue

                if isinstance(stages, int):
                    stage_list = [stages]
                else:
                    stage_list = list(stages)

                mode_to_stage_ids[str(mode_name)] = [int(sid) for sid in stage_list]

        # No modes section or empty mapping: use all stages and return early.
        active_mode: str | None = None
        if isinstance(kwargs, dict):
            active_mode = kwargs.get("mode")

        if active_mode is None:
            active_mode = "text-to-image"

        if active_mode not in mode_to_stage_ids:
            logger.warning(
                "Requested mode '%s' not found in config '%s'; available modes: %s. Using all stages.",
                active_mode,
                config_path,
                sorted(mode_to_stage_ids.keys()),
            )
            return stage_configs

        allowed_ids = set(mode_to_stage_ids[active_mode])
        filtered_stage_configs = [sc for sc in stage_configs if getattr(sc, "stage_id", None) in allowed_ids]
        if not filtered_stage_configs:
            logger.warning(
                "Mode '%s' in config '%s' resolved to stage ids %s, but none matched loaded stage_args. "
                "Falling back to all stages.",
                active_mode,
                config_path,
                sorted(allowed_ids),
            )
            return stage_configs

        return filtered_stage_configs
    except Exception as e:
        logger.warning("Failed to apply mode-based stage filtering: %s", e)
        return stage_configs


def _parse_stage_overrides(value: Any) -> dict[str, dict[str, Any]] | None:
    """Parse the ``--stage-overrides`` value into a per-stage override dict.

    ``value`` may be a raw JSON string (as supplied on the CLI) or an
    already-parsed mapping. Returns ``None`` when no overrides are given.

    Raises:
        ValueError: when ``value`` is a string that is not valid JSON.
    """
    if not value:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--stage-overrides is not valid JSON: {exc}. Got: {value!r}") from exc
    return value


def _resolve_stage_configs(
    model: str,
    stage_configs_path: str | None,
    kwargs: dict | None,
    deploy_config_path: str | None = None,
    stage_overrides: dict[str, dict[str, Any]] | None = None,
    strategy_config_path: str | None = None,
) -> tuple[str | None, list, str | None]:
    """Private source selector used only by :func:`resolve_omni_config`.

    Args:
        model: Model name or path
        stage_configs_path: Optional path to legacy YAML (stage_args format)
        kwargs: Engine arguments to merge with stage configs
        deploy_config_path: Optional path to deploy YAML (new format).
            Mutually exclusive with ``stage_configs_path``.
        stage_overrides: Per-stage overrides from ``--stage-overrides`` JSON.
            Keys are stage_id strings, values are dicts of overrides.
        strategy_config_path: Optional path to a composable-parallel
            ``strategy.yaml`` overlaid onto the registry-merged stages.

    Returns:
        Tuple of ``(config_path, stage_configs, omni_lb_policy)`` — the last is
        the strategy-derived pipeline-wide load-balancer policy (``None`` when no
        strategy set one), returned for the engine to apply.
    """
    if stage_configs_path is not None and deploy_config_path is not None:
        raise ValueError(
            "--stage-configs-path and --deploy-config are mutually exclusive: "
            "they use different path resolution rules and loading paths. "
            "Use --deploy-config for new-format YAMLs (preferred); "
            "--stage-configs-path is kept only for the legacy `stage_args` format "
            "and will be removed in a future release."
        )
    if stage_configs_path is not None and deploy_config_path is None:
        if not os.path.exists(stage_configs_path):
            raise FileNotFoundError(
                f"--stage-configs-path {stage_configs_path!r} does not exist. "
                "Legacy `stage_configs/` yamls were replaced by `vllm_omni/deploy/<model>.yaml`; "
                "use --deploy-config. See docs/configuration/stage_configs.md."
            )
        with open(stage_configs_path, encoding="utf-8") as f:
            _peek = yaml.safe_load(f) or {}
        if "stages" in _peek and "stage_args" not in _peek:
            deploy_config_path = stage_configs_path
            stage_configs_path = None
        else:
            logger.warning(
                "--stage-configs-path is deprecated; migrate %r and use --deploy-config.",
                stage_configs_path,
            )

    omni_lb_policy: str | None = None
    if deploy_config_path is not None:
        config_path = deploy_config_path
        stage_configs, omni_lb_policy = _load_stage_configs_from_model(
            model,
            trust_remote_code=bool((kwargs or {}).get("trust_remote_code", False)),
            base_engine_args=kwargs,
            deploy_config_path=deploy_config_path,
            stage_overrides=stage_overrides,
            strategy_config_path=strategy_config_path,
        )
    elif stage_configs_path is None:
        config_path = resolve_model_config_path(model)
        stage_configs, omni_lb_policy = _load_stage_configs_from_model(
            model,
            trust_remote_code=bool((kwargs or {}).get("trust_remote_code", False)),
            base_engine_args=kwargs,
            stage_overrides=stage_overrides,
            strategy_config_path=strategy_config_path,
        )
    else:
        config_path = stage_configs_path
        stage_configs = _load_stage_configs_from_yaml(stage_configs_path, base_engine_args=kwargs)

    stage_configs = _filter_stages(config_path, stage_configs, kwargs)
    logger.debug(f"stage_configs: {stage_configs}")

    return config_path, stage_configs, omni_lb_policy


def _resolve_generic_diffusion_model_class(
    model: str,
    cli_overrides: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Detect generic diffusion support and its serving pipeline class."""
    from vllm_omni.diffusion.data import resolve_model_class_name
    from vllm_omni.diffusion.registry import DiffusionModelRegistry
    from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model

    model_class_name = cli_overrides.get("model_class_name") or resolve_model_class_name(
        model,
        str(cli_overrides.get("diffusion_load_format") or "default"),
    )
    supported = bool(model_class_name and DiffusionModelRegistry._try_load_model_cls(str(model_class_name)) is not None)
    if not supported:
        supported = is_diffusion_model(model)
    return supported, str(model_class_name) if model_class_name else None


def resolve_omni_config(request: OmniConfigResolveRequest) -> OmniConfigResolution:
    """Resolve all supported authoring sources through one public entrypoint."""
    cli_overrides = dict(request.cli_overrides)
    cli_overrides["trust_remote_code"] = request.trust_remote_code
    stage_overrides = _parse_stage_overrides(request.stage_overrides)
    config_path, stage_configs, omni_lb_policy = _resolve_stage_configs(
        request.model,
        request.legacy_stage_configs_path,
        cli_overrides,
        deploy_config_path=request.deploy_config_path,
        stage_overrides=stage_overrides,
        strategy_config_path=request.strategy_config_path,
    )
    if not stage_configs:
        supported, model_class_name = _resolve_generic_diffusion_model_class(request.model, cli_overrides)
        if not supported:
            raise ValueError(
                f"Model {request.model!r} did not resolve to a registered Omni pipeline, "
                "a legacy stage config, or a supported diffusion model."
            )
        if model_class_name is not None:
            cli_overrides.setdefault("model_class_name", model_class_name)
        default_stage = StageConfigFactory.create_default_diffusion(cli_overrides)
        stage_configs = create_config(_convert_dataclasses_to_dict(default_stage))

    return OmniConfigResolution(
        config_path=config_path,
        stage_configs=tuple(stage_configs),
        omni_lb_policy=omni_lb_policy,
        endpoint_restrictions=StageConfigFactory.get_pipeline_endpoint_restrictions(
            model=request.model,
            trust_remote_code=request.trust_remote_code,
            deploy_config_path=request.deploy_config_path,
        ),
    )


__all__ = [
    "OmniConfigResolution",
    "OmniConfigResolveRequest",
    "resolve_model_config_path",
    "resolve_omni_config",
]
