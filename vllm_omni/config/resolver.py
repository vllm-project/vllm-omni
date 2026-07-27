from __future__ import annotations

import types
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.config.composable_parallel.strategy_loader import load_strategy_specs
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.endpoint_policy import EndpointRestriction
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.yaml_util import create_config
from vllm_omni.diffusion.data import resolve_model_class_name
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model

logger = init_logger(__name__)


@dataclass(frozen=True)
class OmniConfigResolution:
    """Migration envelope returned by the production config resolver.

    ``stage_configs`` intentionally carries the current OmegaConf-compatible
    runtime ABI only until stage startup consumes ``VllmOmniConfig`` directly.
    It is not a stable authoring or extension API; new production callers
    should resolve through :func:`resolve_omni_config` and must not construct or
    merge this compatibility shape themselves.
    """

    config_path: str | None
    stage_configs: tuple[Any, ...]  # Temporary StageConfig/OmegaConf bridge.
    omni_lb_policy: str | None = None
    endpoint_restrictions: tuple[EndpointRestriction, ...] = ()

    def stage_by_id(self, stage_id: int) -> Any:
        for stage in self.stage_configs:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"no stage {stage_id}")


def _filter_dict_like_object(obj: dict | Any) -> dict:
    """Convert a dict-like object while dropping OmegaConf-incompatible callables."""

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
    for key, value in obj.items():
        # Preserve class objects as import paths for consumers such as
        # custom_pipeline_args.pipeline_class.
        if isinstance(value, type):
            module = getattr(value, "__module__", None)
            qualname = getattr(value, "__qualname__", getattr(value, "__name__", None))
            result[key] = f"{module}.{qualname}" if module and qualname and module != "builtins" else qualname
        elif _is_callable_value(value):
            filtered_keys.append(str(key))
        else:
            result[key] = _convert_dataclasses_to_dict(value)
    if filtered_keys:
        logger.warning(
            "Filtered out %d callable object(s) from base_engine_args that are not compatible with OmegaConf: %s.",
            len(filtered_keys),
            filtered_keys,
        )
    return result


def _convert_dataclasses_to_dict(obj: Any) -> Any:
    """Recursively convert caller values to OmegaConf-compatible types."""
    # Counter is a dict subclass, so it must be handled first. The class-name
    # check also covers vllm.utils.Counter.
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "Counter":
        try:
            return dict(obj)
        except (TypeError, ValueError):
            return {}
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, set):
        return list(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for config_field in fields(obj):
            if not config_field.init:
                continue
            value = getattr(obj, config_field.name)
            if value is None and config_field.default is None:
                continue
            result[config_field.name] = _convert_dataclasses_to_dict(value)
        return result
    if isinstance(obj, dict):
        return _filter_dict_like_object(obj)
    if isinstance(obj, type):
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", None))
        return f"{module}.{qualname}" if module and qualname and module != "builtins" else qualname
    if callable(obj):
        return None
    if isinstance(obj, (list, tuple)):
        return type(obj)(_convert_dataclasses_to_dict(item) for item in obj if not callable(item))
    if hasattr(obj, "keys") and hasattr(obj, "values") and not isinstance(obj, (str, bytes)):
        try:
            return _filter_dict_like_object(obj)
        except (TypeError, ValueError, AttributeError):
            return obj
    return obj


def _flatten_stage_overrides(
    cli_overrides: dict[str, Any],
    stage_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Translate per-stage mappings to the factory's flat CLI override ABI."""
    if not stage_overrides:
        return
    for stage_id, overrides in stage_overrides.items():
        for key, value in overrides.items():
            cli_overrides[f"stage_{stage_id}_{key}"] = value


def _load_strategy_specs(strategy_config_path: str | None) -> Mapping[Any, Any] | None:
    if strategy_config_path is None:
        return None

    return load_strategy_specs(strategy_config_path)


def _build_registered_resolution(
    structured_config: VllmOmniConfig,
    *,
    model: str,
    cli_overrides: dict[str, Any],
    trust_remote_code: bool,
    deploy_config_path: str | None,
    strategy_config_path: str | None,
) -> OmniConfigResolution:
    """Build the temporary runtime view for an already-resolved pipeline."""
    # Runtime consumers have not yet moved to typed per-stage configs. Use the
    # factory-owned compatibility bridge instead of reimplementing legacy YAML
    # discovery and merging in this resolver.
    legacy_stages, omni_lb_policy = StageConfigFactory.create_legacy_stage_configs_from_model(
        model,
        trust_remote_code=trust_remote_code,
        cli_overrides=cli_overrides,
        deploy_config_path=deploy_config_path,
        strategy_specs=_load_strategy_specs(strategy_config_path),
    )
    if legacy_stages is None:
        raise RuntimeError(
            f"Model {model!r} resolved to a structured Omni pipeline, "
            "but the runtime compatibility stage configs could not be built."
        )

    return OmniConfigResolution(
        config_path=structured_config.orchestrator_config.deploy_config_path,
        stage_configs=tuple(stage.to_omegaconf() for stage in legacy_stages),
        omni_lb_policy=omni_lb_policy,
        endpoint_restrictions=tuple(structured_config.pipeline_config.endpoint_restrictions),
    )


def _resolve_generic_diffusion_model_class(
    model: str,
    cli_overrides: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Detect generic diffusion support and its serving pipeline class."""
    model_class_name = cli_overrides.get("model_class_name") or resolve_model_class_name(
        model,
        str(cli_overrides.get("diffusion_load_format") or "default"),
    )
    supported = bool(model_class_name and DiffusionModelRegistry._try_load_model_cls(str(model_class_name)) is not None)
    if not supported:
        supported = is_diffusion_model(model)
    return supported, str(model_class_name) if model_class_name else None


def resolve_omni_config(
    model: str,
    *,
    trust_remote_code: bool,
    deploy_config_path: str | None,
    cli_overrides: Mapping[str, Any] | None,
    stage_overrides: Mapping[str, Mapping[str, Any]] | None,
    strategy_config_path: str | None,
) -> OmniConfigResolution:
    """Resolve registry/deploy inputs through the single public entrypoint."""
    normalized_overrides = _convert_dataclasses_to_dict(dict(cli_overrides or {}))
    normalized_overrides["trust_remote_code"] = trust_remote_code
    _flatten_stage_overrides(normalized_overrides, stage_overrides)

    structured_config = StageConfigFactory.create_from_model(
        model,
        trust_remote_code=trust_remote_code,
        cli_overrides=normalized_overrides,
        deploy_config_path=deploy_config_path,
    )
    if structured_config is not None:
        return _build_registered_resolution(
            structured_config,
            model=model,
            cli_overrides=normalized_overrides,
            trust_remote_code=trust_remote_code,
            deploy_config_path=deploy_config_path,
            strategy_config_path=strategy_config_path,
        )

    supported, model_class_name = _resolve_generic_diffusion_model_class(model, normalized_overrides)
    if not supported:
        raise ValueError(
            f"Model {model!r} did not resolve to a registered Omni pipeline or a supported diffusion model."
        )
    if model_class_name is not None:
        normalized_overrides.setdefault("model_class_name", model_class_name)
    default_stages = StageConfigFactory.create_default_diffusion(normalized_overrides)

    return OmniConfigResolution(
        config_path=deploy_config_path,
        stage_configs=tuple(create_config(_convert_dataclasses_to_dict(default_stages))),
    )


__all__ = [
    "OmniConfigResolution",
    "resolve_omni_config",
]
