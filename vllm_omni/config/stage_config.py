# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage configuration system for vLLM-Omni."""

from __future__ import annotations

import dataclasses
import re
import warnings
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from vllm_omni.config.yaml_util import create_config, load_yaml_config, to_dict
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

_MODELS_DIR = Path(__file__).resolve().parent.parent / "model_executor" / "models"


def get_pipeline_path(model_dir: str, filename: str) -> Path:
    return _MODELS_DIR / model_dir / filename


logger = init_logger(__name__)


def _warn_deprecated_kwargs(kwargs: dict[str, Any]) -> None:
    if "cli_explicit_keys" in kwargs:
        warnings.warn(
            "cli_explicit_keys= is deprecated and ignored. Remove the kwarg.",
            DeprecationWarning,
            stacklevel=3,
        )


_STAGE_OVERRIDE_PATTERN = re.compile(r"^stage_(\d+)_(.+)$")


def build_stage_runtime_overrides(
    stage_id: int,
    cli_overrides: dict[str, Any],
    *,
    internal_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Build per-stage runtime overrides from global and ``stage_<id>_*`` kwargs.

    ``internal_keys`` defaults to the union of
    ``arg_utils.internal_blacklist_keys()`` and ``arg_utils.SHARED_FIELDS``
    so that neither orchestrator-only fields nor shared-pipeline fields
    (``model`` / ``stage_configs_path`` / ``log_stats`` / ``stage_id``) leak
    into a stage's per-stage runtime overrides — the orchestrator sets those
    uniformly for every stage, they are not per-stage knobs. Callers can
    pass an explicit set for tests or specialized flows.
    """
    if internal_keys is None:
        from vllm_omni.engine.arg_utils import SHARED_FIELDS, internal_blacklist_keys

        internal_keys = internal_blacklist_keys() | SHARED_FIELDS

    result: dict[str, Any] = {}

    for key, value in cli_overrides.items():
        if value is None or key in internal_keys:
            continue

        match = _STAGE_OVERRIDE_PATTERN.match(key)
        if match is not None:
            override_stage_id = int(match.group(1))
            param_name = match.group(2)
            if override_stage_id == stage_id and param_name not in internal_keys:
                result[param_name] = value
            continue

        result[key] = value

    return result


def strip_parent_engine_args(
    kwargs: dict[str, Any],
    *,
    parent_fields: dict[str, dataclasses.Field],
    keep_keys: set[str] | frozenset[str] = frozenset(),
    strip_keys: set[str] | frozenset[str] = frozenset(),
    no_warn_keys: set[str] | frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], list[str]]:
    """Strip parent ``EngineArgs`` fields before merging into stage YAML."""
    overridden: list[str] = []
    result: dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in strip_keys:
            continue

        if key not in parent_fields or key in keep_keys:
            result[key] = value
            continue

        field_def = parent_fields[key]
        if field_def.default is not dataclasses.MISSING:
            default = field_def.default
        elif field_def.default_factory is not dataclasses.MISSING:
            default = field_def.default_factory()
        else:
            default = dataclasses.MISSING

        if default is dataclasses.MISSING or value is None:
            continue

        if dataclasses.is_dataclass(default) and not isinstance(default, type):
            default = asdict(default)

        if value != default and key not in no_warn_keys:
            overridden.append(key)

    return result, sorted(overridden)


class StageType(str, Enum):
    """Type of processing stage in the Omni pipeline."""

    # TODO(@lishunyang12): remove once all models migrate to StageExecutionType
    LLM = "llm"
    DIFFUSION = "diffusion"


class StageExecutionType(str, Enum):
    """Merged StageType + WorkerType — 3 combinations today."""

    LLM_AR = "llm_ar"
    LLM_GENERATION = "llm_generation"
    DIFFUSION = "diffusion"


# Mapping class refs (not dotted-path strings) so module/class renames fail
# at import time instead of lazily at scheduler resolution. YAML overrides
# and downstream serialization still use the dotted-path string form; the
# conversion happens at the map lookup site via _scheduler_path().
_EXECUTION_TYPE_TO_SCHEDULER: dict[StageExecutionType, type | None] = {
    StageExecutionType.LLM_AR: OmniARScheduler,
    StageExecutionType.LLM_GENERATION: OmniGenerationScheduler,
    StageExecutionType.DIFFUSION: None,
}


def _scheduler_path(cls: type | None) -> str | None:
    """Return the dotted import path for a scheduler class (``None`` passes through)."""
    if cls is None:
        return None
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class StagePipelineConfig:
    """Fixed topology for one stage (frozen, not user-configurable)."""

    stage_id: int
    model_stage: str
    execution_type: StageExecutionType = StageExecutionType.LLM_AR
    input_sources: tuple[int, ...] = ()
    final_output: bool = False
    final_output_type: str | None = None
    owns_tokenizer: bool = False
    requires_multimodal_data: bool = False
    hf_config_name: str | None = None
    engine_output_type: str | None = None
    model_arch: str | None = None
    sampling_constraints: dict[str, Any] = field(default_factory=dict)
    custom_process_input_func: str | None = None
    custom_process_next_stage_input_func: str | None = None
    # Alternates picked by ``merge_pipeline_deploy`` based on ``deploy.async_chunk``.
    async_chunk_process_next_stage_input_func: str | None = None
    sync_process_input_func: str | None = None
    prompt_expand_func: str | None = None
    cfg_kv_collect_func: str | None = None
    omni_kv_config: dict[str, Any] | None = None
    # Model subdirectory indirections: for multi-component HF repos where the
    # stage's config/tokenizer lives in a subdirectory (e.g. GLM-Image's AR
    # config is in ``vision_language_encoder/``).  Consumed at stage-init time
    # by ``stage_init_utils._resolve_model_tokenizer_paths``.
    model_subdir: str | None = None
    tokenizer_subdir: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline topology for a model (frozen)."""

    model_type: str
    model_arch: str = ""
    stages: tuple[StagePipelineConfig, ...] = ()
    # HF architecture aliases: used by StageConfigFactory when the model's
    # HF config reports a generic model_type that collides with a different
    # model (e.g. MiMo Audio reports model_type="qwen2"). The factory
    # matches ``hf_config.architectures[*]`` against this tuple to route
    # to the correct pipeline. Leave empty for models with unique model_type.
    hf_architectures: tuple[str, ...] = ()
    # Diffusers pipeline class name: for models that ship a ``model_index.json``
    # (no root ``config.json``), the ``_class_name`` field is matched against
    # this value to auto-detect the pipeline.  Only needed for diffusers-style
    # multi-component repos (e.g. GLM-Image).  ``None`` = not a diffusers model.
    diffusers_class_name: str | None = None

    def get_stage(self, stage_id: int) -> StagePipelineConfig | None:
        """Look up a stage by its ID."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def get_scheduler_cls(self, stage_id: int) -> str | None:
        """Return the inferred scheduler class path for a stage.

        Returns ``None`` for DIFFUSION stages (no vLLM scheduler). Raises
        ``ValueError`` if ``stage_id`` doesn't exist in this pipeline, and
        ``KeyError`` if ``execution_type`` isn't in the scheduler map.
        """
        stage = self.get_stage(stage_id)
        if stage is None:
            raise ValueError(f"Pipeline {self.model_type!r} has no stage with id {stage_id}")
        return _scheduler_path(_EXECUTION_TYPE_TO_SCHEDULER[stage.execution_type])

    def validate(self) -> list[str]:
        """Return list of topology errors (empty if valid)."""
        errors: list[str] = []
        if not self.stages:
            errors.append("Pipeline has no stages defined")
            return errors
        stage_ids = [s.stage_id for s in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            errors.append("Duplicate stage IDs found")
        stage_id_set = set(stage_ids)
        for stage in self.stages:
            for src in stage.input_sources:
                if src not in stage_id_set:
                    errors.append(f"Stage {stage.stage_id} references non-existent input source {src}")
                if src == stage.stage_id:
                    errors.append(f"Stage {stage.stage_id} references itself")
        if not any(not s.input_sources for s in self.stages):
            errors.append("No entry point (stage with empty input_sources)")
        return errors


class _LazyPipelineRegistry:
    """Dict-like registry that lazy-loads pipelines from the central declaration.

    In-tree pipelines are declared once in
    ``vllm_omni/config/pipeline_registry.py`` as
    ``model_type -> (module_path, variable_name)`` entries; the module is
    imported only when the pipeline is first looked up. This mirrors the
    pattern in ``vllm/model_executor/models/registry.py`` and addresses
    https://github.com/vllm-project/vllm-omni/issues/2887 (item 4): having
    every registration in one file makes a missing entry easy to spot.

    Out-of-tree / dynamic registrations via ``register_pipeline()`` are stored
    directly in ``_loaded`` and take precedence over the lazy-map entry with
    the same ``model_type``.

    The class exposes the subset of ``dict`` operations the rest of this
    module relies on (``__contains__``, ``__getitem__``, ``__setitem__``,
    ``get``, ``keys``, ``values``, ``items``, ``__iter__``), so existing call
    sites don't need to change.
    """

    def __init__(self) -> None:
        self._loaded: dict[str, PipelineConfig] = {}
        # Populated lazily to avoid a circular import at module init time.
        self._lazy_map: dict[str, tuple[str, str]] | None = None

    def _get_lazy_map(self) -> dict[str, tuple[str, str]]:
        if self._lazy_map is None:
            from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES

            self._lazy_map = _OMNI_PIPELINES
        return self._lazy_map

    def _load_lazy(self, model_type: str) -> PipelineConfig | None:
        entry = self._get_lazy_map().get(model_type)
        if entry is None:
            return None
        module_path, var_name = entry
        import importlib

        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            logger.error(
                "Failed to import pipeline module %r for %r: %s",
                module_path,
                model_type,
                exc,
            )
            return None
        pipeline = getattr(module, var_name, None)
        if pipeline is None:
            logger.error(
                "Pipeline variable %r not found in module %r (registered for %r)",
                var_name,
                module_path,
                model_type,
            )
            return None
        errors = pipeline.validate()
        if errors:
            logger.warning("Pipeline %s has issues: %s", pipeline.model_type, errors)
        self._loaded[model_type] = pipeline
        return pipeline

    def __contains__(self, model_type: str) -> bool:
        if model_type in self._loaded:
            return True
        return model_type in self._get_lazy_map()

    def __getitem__(self, model_type: str) -> PipelineConfig:
        if model_type in self._loaded:
            return self._loaded[model_type]
        pipeline = self._load_lazy(model_type)
        if pipeline is None:
            raise KeyError(model_type)
        return pipeline

    def get(self, model_type: str, default: PipelineConfig | None = None) -> PipelineConfig | None:
        if model_type in self._loaded:
            return self._loaded[model_type]
        pipeline = self._load_lazy(model_type)
        return pipeline if pipeline is not None else default

    def __setitem__(self, model_type: str, pipeline: PipelineConfig) -> None:
        self._loaded[model_type] = pipeline

    def __delitem__(self, model_type: str) -> None:
        """Remove a dynamically-registered pipeline.

        Only the dynamic-cache side of the registry can be mutated; the
        central declarative registry is immutable at runtime. Calling ``del``
        on a model_type that only exists in the central registry raises
        ``KeyError``.
        """
        if model_type in self._loaded:
            del self._loaded[model_type]
            return
        if model_type in self._get_lazy_map():
            raise KeyError(
                f"{model_type!r} is declared in the central pipeline_registry and "
                "cannot be removed at runtime. Edit "
                "vllm_omni/config/pipeline_registry.py to delete a built-in entry."
            )
        raise KeyError(model_type)

    def keys(self) -> set[str]:
        return set(self._get_lazy_map().keys()) | set(self._loaded.keys())

    def values(self):
        # Iterating values forces load of every lazy pipeline.
        for key in self.keys():
            yield self[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def __iter__(self):
        return iter(self.keys())


_PIPELINE_REGISTRY = _LazyPipelineRegistry()


def register_pipeline(pipeline: PipelineConfig) -> None:
    """Register a pipeline config dynamically.

    In-tree pipelines are declared in ``pipeline_registry._OMNI_PIPELINES``
    and loaded lazily; calling ``register_pipeline`` is only needed for
    out-of-tree plugins or tests that build a ``PipelineConfig`` at runtime.
    A dynamic registration overrides the central-registry entry with the same
    ``model_type``.
    """
    errors = pipeline.validate()
    if errors:
        logger.warning("Pipeline %s has issues: %s", pipeline.model_type, errors)
    _PIPELINE_REGISTRY[pipeline.model_type] = pipeline


_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"


@dataclass
class StageDeployConfig:
    """Per-stage deployment knobs.

    Only fields whose value legitimately varies across stages of the same
    pipeline live here (e.g. ``max_num_seqs`` on thinker vs talker,
    ``devices`` for GPU placement). Pipeline-wide settings
    (``trust_remote_code``, ``distributed_executor_backend``, ``dtype``,
    ``quantization``, prefix/chunked prefill, DP/PP sizes) are declared at
    the top level of ``DeployConfig`` and propagated to every stage.
    """

    stage_id: int
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    max_num_batched_tokens: int = 32768
    max_model_len: int | None = None
    async_scheduling: bool | None = None
    devices: str = "0"
    output_connectors: dict[str, str] | None = None
    input_connectors: dict[str, str] | None = None
    default_sampling_params: dict[str, Any] | None = None
    subtalker_sampling_params: dict[str, Any] | None = None
    engine_extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeployConfig:
    """Loaded from deploy/<model>.yaml — the only config file users edit.

    Top-level fields (``trust_remote_code``, ``distributed_executor_backend``,
    ``dtype``, ``quantization``, ``enable_prefix_caching``,
    ``enable_chunked_prefill``, ``data_parallel_size``,
    ``pipeline_parallel_size``) are pipeline-wide: they apply uniformly to
    every stage. Fields that legitimately vary per stage live in the
    individual ``StageDeployConfig`` entries under ``stages:``.
    """

    async_chunk: bool = True
    connectors: dict[str, Any] | None = None
    edges: list[dict[str, Any]] | None = None
    stages: list[StageDeployConfig] = field(default_factory=list)
    platforms: dict[str, Any] | None = None
    # Overrides the auto-detected pipeline registry key for structural variants.
    pipeline: str | None = None

    # === Pipeline-wide engine settings (applied uniformly to every stage) ===
    trust_remote_code: bool = True
    distributed_executor_backend: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool | None = None
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1


_STAGE_NON_ENGINE_KEYS = frozenset(
    {
        "stage_id",
        "devices",
        "output_connectors",
        "input_connectors",
        "default_sampling_params",
        "engine_extras",
    }
)

# Fields on StageDeployConfig that are populated from engine_args dict
_STAGE_DEPLOY_FIELDS = {f.name: f for f in fields(StageDeployConfig) if f.name not in _STAGE_NON_ENGINE_KEYS}


def _parse_stage_deploy(stage_data: dict[str, Any]) -> StageDeployConfig:
    """Parse a single stage entry from deploy YAML into StageDeployConfig."""
    if "engine_args" in stage_data:
        engine_args = dict(stage_data["engine_args"])
        devices = stage_data.get("runtime", {}).get("devices", stage_data.get("devices", "0"))
    else:
        engine_args = {k: v for k, v in stage_data.items() if k not in _STAGE_NON_ENGINE_KEYS and k != "stage_id"}
        devices = stage_data.get("devices", "0")

    kwargs: dict[str, Any] = {"stage_id": stage_data["stage_id"], "devices": devices}
    for name, f in _STAGE_DEPLOY_FIELDS.items():
        if name in engine_args:
            kwargs[name] = engine_args.pop(name)

    kwargs["output_connectors"] = stage_data.get("output_connectors")
    kwargs["input_connectors"] = stage_data.get("input_connectors")
    kwargs["default_sampling_params"] = stage_data.get("default_sampling_params")
    kwargs["engine_extras"] = engine_args
    return StageDeployConfig(**kwargs)


_DEEP_MERGE_KEYS = frozenset({"default_sampling_params", "subtalker_sampling_params", "engine_extras", "engine_args"})


def _deep_merge_stage(base: dict, overlay: dict) -> dict:
    """Deep-merge ``_DEEP_MERGE_KEYS`` so thin overlays don't drop base keys."""
    merged = dict(base)
    for k, v in overlay.items():
        if k in _DEEP_MERGE_KEYS:
            base_val = merged.get(k)
            if isinstance(v, dict) and isinstance(base_val, dict):
                merged[k] = {**base_val, **v}
                continue
            # Deep-merge key but at least one side isn't a dict: surface the
            # silent clobber so mismatched YAML types don't get past review.
            if base_val is not None:
                logger.warning(
                    "Deep-merge key %r has non-dict value (base=%s, overlay=%s); "
                    "overlay will fully replace base instead of merging.",
                    k,
                    type(base_val).__name__,
                    type(v).__name__,
                )
        merged[k] = v
    return merged


def _merge_stage_lists(
    base_stages: list[dict[str, Any]] | None,
    overlay_stages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge two ``stages:`` lists by ``stage_id`` (overlay wins per field)."""
    by_id: dict[int, dict[str, Any]] = {s["stage_id"]: s for s in (base_stages or [])}
    for overlay_stage in overlay_stages or []:
        sid = overlay_stage["stage_id"]
        if sid in by_id:
            by_id[sid] = _deep_merge_stage(by_id[sid], overlay_stage)
        else:
            by_id[sid] = overlay_stage
    return list(by_id.values())


def _merge_platforms(
    base: dict[str, Any] | None,
    overlay: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Deep-merge two ``platforms:`` blocks per-platform, per-stage_id."""
    if not base and not overlay:
        return None
    base = base or {}
    overlay = overlay or {}
    merged: dict[str, Any] = {}
    for plat in set(base) | set(overlay):
        bp = base.get(plat) or {}
        op = overlay.get(plat) or {}
        merged_plat = {**bp, **{k: v for k, v in op.items() if k != "stages"}}
        merged_plat["stages"] = _merge_stage_lists(bp.get("stages"), op.get("stages"))
        merged[plat] = merged_plat
    return merged


def resolve_deploy_yaml(path: str | Path) -> dict[str, Any]:
    """Load a deploy YAML with optional ``base_config`` inheritance."""
    raw_dict = to_dict(load_yaml_config(path))

    base_path = raw_dict.pop("base_config", None)
    if base_path is None:
        return raw_dict

    # Resolve relative to the overlay file's directory
    base_path = Path(path).parent / base_path
    base_dict = resolve_deploy_yaml(base_path)

    # Merge top-level scalars: overlay wins. ``stages:`` and ``platforms:``
    # are deep-merged below so an overlay can layer on top of the base.
    merged = {
        **base_dict,
        **{k: v for k, v in raw_dict.items() if k not in ("stages", "platforms")},
    }
    merged["stages"] = _merge_stage_lists(base_dict.get("stages"), raw_dict.get("stages"))
    merged_platforms = _merge_platforms(base_dict.get("platforms"), raw_dict.get("platforms"))
    if merged_platforms is not None:
        merged["platforms"] = merged_platforms

    return merged


def load_deploy_config(path: str | Path) -> DeployConfig:
    """Load a deploy YAML (with optional base_config inheritance)."""
    raw_dict = resolve_deploy_yaml(path)

    stages = [_parse_stage_deploy(s) for s in raw_dict.get("stages", [])]

    kwargs: dict[str, Any] = {
        "async_chunk": raw_dict.get("async_chunk", True),
        "connectors": raw_dict.get("connectors", None),
        "edges": raw_dict.get("edges", None),
        "stages": stages,
        "platforms": raw_dict.get("platforms", None),
        "pipeline": raw_dict.get("pipeline", None),
    }
    # Pipeline-wide engine settings: only set if explicitly present in YAML
    # so the DeployConfig dataclass defaults take effect otherwise.
    for name in (
        "trust_remote_code",
        "distributed_executor_backend",
        "dtype",
        "quantization",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "data_parallel_size",
        "pipeline_parallel_size",
    ):
        if name in raw_dict:
            kwargs[name] = raw_dict[name]
    return DeployConfig(**kwargs)


def _extract_platform_overrides(ps: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Return ``(overrides, devices)`` from a platform stage entry.

    Handles both the nested layout (``engine_args:`` / ``runtime.devices``) and
    the flat layout. ``devices`` is ``None`` when no override is set.
    """
    if "engine_args" in ps:
        return dict(ps["engine_args"]), ps.get("runtime", {}).get("devices")
    overrides = {k: v for k, v in ps.items() if k not in ("stage_id", "devices")}
    return overrides, ps.get("devices")


def _apply_platform_overrides(
    deploy: DeployConfig,
    platform: str | None = None,
) -> DeployConfig:
    """Merge platform-specific stage overrides into deploy config."""
    if platform is None:
        from vllm_omni.platforms import current_omni_platform

        platform = current_omni_platform.device_name.lower()
    if platform is None or deploy.platforms is None:
        return deploy
    platform_section = deploy.platforms.get(platform)
    if platform_section is None:
        return deploy

    platform_stages = platform_section.get("stages", [])
    base_by_id = {s.stage_id: s for s in deploy.stages}

    for ps in platform_stages:
        base = base_by_id.get(ps["stage_id"])
        if base is None:
            continue
        overrides, devices = _extract_platform_overrides(ps)
        if devices is not None:
            base.devices = devices
        for key, val in overrides.items():
            if hasattr(base, key):
                setattr(base, key, val)
            else:
                base.engine_extras[key] = val

    return deploy


_EXECUTION_TYPE_TO_STAGE_WORKER: dict[StageExecutionType, tuple[StageType, str | None]] = {
    StageExecutionType.LLM_AR: (StageType.LLM, "ar"),
    StageExecutionType.LLM_GENERATION: (StageType.LLM, "generation"),
    StageExecutionType.DIFFUSION: (StageType.DIFFUSION, None),
}


def _resolve_execution_mode(
    execution_type: StageExecutionType,
) -> tuple[StageType, str | None]:
    """Map ``execution_type`` → ``(stage_type, worker_type)`` legacy tuple."""
    return _EXECUTION_TYPE_TO_STAGE_WORKER.get(execution_type, (StageType.LLM, None))


def _select_processor_funcs(
    ps: StagePipelineConfig,
    async_chunk: bool,
) -> tuple[str | None, str | None]:
    """Pick ``(input_proc, next_stage_proc)`` based on the async_chunk mode."""
    next_stage_proc = ps.custom_process_next_stage_input_func
    input_proc = ps.custom_process_input_func
    if async_chunk and ps.async_chunk_process_next_stage_input_func:
        next_stage_proc = ps.async_chunk_process_next_stage_input_func
    elif not async_chunk and ps.sync_process_input_func:
        input_proc = ps.sync_process_input_func
    return input_proc, next_stage_proc


# Pipeline-wide DeployConfig fields that are propagated to every stage's
# engine args during merge. These live at top level of the deploy YAML.
_PIPELINE_WIDE_ENGINE_FIELDS: tuple[str, ...] = (
    "trust_remote_code",
    "distributed_executor_backend",
    "dtype",
    "quantization",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "data_parallel_size",
    "pipeline_parallel_size",
)


def _build_engine_args(
    ps: StagePipelineConfig,
    ds: StageDeployConfig | None,
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    next_stage_proc: str | None,
) -> dict[str, Any]:
    """Assemble the flat ``yaml_engine_args`` dict for one stage.

    Pipeline-wide DeployConfig fields are applied uniformly to every stage;
    per-stage StageDeployConfig overrides take precedence when present (e.g.
    ``engine_extras`` can still carry a stage-specific ``dtype``).
    """
    engine_args: dict[str, Any] = {"model_arch": ps.model_arch or pipeline.model_arch}
    if ps.engine_output_type:
        engine_args["engine_output_type"] = ps.engine_output_type
    if next_stage_proc:
        engine_args["custom_process_next_stage_input_func"] = next_stage_proc
    # Subdirectory indirections from StagePipelineConfig (structural, not
    # deployment knobs).  Deploy YAML ``engine_extras`` can still override
    # these per-stage if needed.
    if ps.model_subdir:
        engine_args["model_subdir"] = ps.model_subdir
    if ps.tokenizer_subdir:
        engine_args["tokenizer_subdir"] = ps.tokenizer_subdir

    # Pipeline-wide top-level DeployConfig settings, applied to every stage.
    for name in _PIPELINE_WIDE_ENGINE_FIELDS:
        value = getattr(deploy, name)
        if value is not None:
            engine_args[name] = value

    # Per-stage StageDeployConfig values override pipeline-wide settings.
    if ds is not None:
        for k, v in asdict(ds).items():
            if k in _STAGE_NON_ENGINE_KEYS or v is None:
                continue
            engine_args[k] = v
        engine_args.update(ds.engine_extras)
    # Materialize the resolved pipeline-wide async_chunk value into every
    # stage so explicit False overrides do not get lost downstream.
    engine_args["async_chunk"] = bool(deploy.async_chunk)
    if ps.omni_kv_config:
        engine_args["omni_kv_config"] = dict(ps.omni_kv_config)
    return engine_args


def _build_extras(
    ps: StagePipelineConfig,
    ds: StageDeployConfig | None,
) -> dict[str, Any]:
    """Assemble ``yaml_extras`` (sampling + connectors + pipeline extras)."""
    extras: dict[str, Any] = {}
    sampling: dict[str, Any] = {}
    if ds is not None and ds.default_sampling_params:
        sampling.update(ds.default_sampling_params)
    sampling.update(ps.sampling_constraints)
    if sampling:
        extras["default_sampling_params"] = sampling
    if ds is not None and ds.output_connectors:
        extras["output_connectors"] = dict(ds.output_connectors)
    if ds is not None and ds.input_connectors:
        extras["input_connectors"] = dict(ds.input_connectors)
    if ps.prompt_expand_func:
        extras["prompt_expand_func"] = ps.prompt_expand_func
    if ps.cfg_kv_collect_func:
        extras["cfg_kv_collect_func"] = ps.cfg_kv_collect_func
    if ps.extras:
        extras.update(ps.extras)
    return extras


def merge_pipeline_deploy(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    cli_overrides: dict[str, Any] | None = None,
) -> list[StageConfig]:
    """Merge pipeline + deploy + platform overrides → list[StageConfig]."""
    if cli_overrides is None:
        cli_overrides = {}

    deploy = _apply_platform_overrides(deploy)
    deploy_by_id = {s.stage_id: s for s in deploy.stages}

    # A pipeline supports async_chunk if any stage has either an explicit
    # async-chunk-only processor slot OR a custom next-stage processor (some
    # pipelines like qwen3_omni wire async-chunk processing directly through
    # ``custom_process_next_stage_input_func``). Only raise when neither is
    # present — that's the "user enabled async_chunk but pipeline has no
    # inter-stage processing at all" case.
    if deploy.async_chunk and not any(
        ps.async_chunk_process_next_stage_input_func or ps.custom_process_next_stage_input_func
        for ps in pipeline.stages
    ):
        raise ValueError(
            f"Pipeline {pipeline.model_type!r} has async_chunk=True in deploy but no stage "
            "declares a next-stage input processor "
            "(``async_chunk_process_next_stage_input_func`` or ``custom_process_next_stage_input_func``). "
            "Either set async_chunk=False or implement an async-chunk processor on the pipeline."
        )

    result: list[StageConfig] = []
    for ps in pipeline.stages:
        ds = deploy_by_id.get(ps.stage_id)
        stage_type, worker_type = _resolve_execution_mode(ps.execution_type)
        input_proc, next_stage_proc = _select_processor_funcs(ps, deploy.async_chunk)
        engine_args = _build_engine_args(ps, ds, pipeline, deploy, next_stage_proc)
        extras = _build_extras(ps, ds)
        runtime: dict[str, Any] = {"process": True}
        if ds is not None:
            runtime["devices"] = ds.devices

        result.append(
            StageConfig(
                stage_id=ps.stage_id,
                model_stage=ps.model_stage,
                stage_type=stage_type,
                input_sources=list(ps.input_sources),
                custom_process_input_func=input_proc,
                final_output=ps.final_output,
                final_output_type=ps.final_output_type,
                worker_type=worker_type,
                scheduler_cls=_scheduler_path(_EXECUTION_TYPE_TO_SCHEDULER.get(ps.execution_type)),
                hf_config_name=ps.hf_config_name,
                is_comprehension=ps.owns_tokenizer,
                yaml_engine_args=engine_args,
                yaml_runtime=runtime,
                yaml_extras=extras,
            )
        )
    return result


@dataclass
class StageConfig:
    """Per-stage config (legacy path). Used by both new and legacy loaders.

    TODO(@lishunyang12): replace with ResolvedStageConfig once all models are migrated.
    """

    stage_id: int
    model_stage: str
    stage_type: StageType = StageType.LLM
    input_sources: list[int] = field(default_factory=list)
    custom_process_input_func: str | None = None
    final_output: bool = False
    final_output_type: str | None = None
    worker_type: str | None = None
    scheduler_cls: str | None = None
    hf_config_name: str | None = None
    is_comprehension: bool = False
    yaml_engine_args: dict[str, Any] = field(default_factory=dict)
    yaml_runtime: dict[str, Any] = field(default_factory=dict)
    yaml_extras: dict[str, Any] = field(default_factory=dict)
    runtime_overrides: dict[str, Any] = field(default_factory=dict)

    def to_omegaconf(self) -> Any:
        """TODO(@lishunyang12): remove once engine consumes ResolvedStageConfig directly."""
        # Start with YAML engine_args defaults
        engine_args: dict[str, Any] = dict(self.yaml_engine_args)

        # Overlay topology-level fields
        engine_args["model_stage"] = self.model_stage
        if self.worker_type:
            engine_args["worker_type"] = self.worker_type
        if self.scheduler_cls:
            engine_args["scheduler_cls"] = self.scheduler_cls
        if self.hf_config_name:
            engine_args["hf_config_name"] = self.hf_config_name

        # CLI overrides take precedence over YAML defaults
        for key, value in self.runtime_overrides.items():
            if key not in ("devices", "max_batch_size"):
                engine_args[key] = value

        # Build runtime config from YAML defaults + CLI overrides
        runtime: dict[str, Any] = dict(self.yaml_runtime)
        runtime.setdefault("process", True)
        if "devices" in self.runtime_overrides:
            runtime["devices"] = self.runtime_overrides["devices"]

        # Legacy compat: migrate runtime.max_batch_size → engine_args.max_num_seqs
        legacy_mbs = runtime.pop("max_batch_size", None)
        cli_mbs = self.runtime_overrides.get("max_batch_size")
        if legacy_mbs is not None or cli_mbs is not None:
            warnings.warn(
                "runtime.max_batch_size is deprecated and will be removed in a "
                "future release. Use engine_args.max_num_seqs instead.",
                FutureWarning,
                stacklevel=2,
            )
            effective_mbs = int(cli_mbs or legacy_mbs or 1)
            engine_args.setdefault("max_num_seqs", effective_mbs)

        engine_args.setdefault("max_num_seqs", 1)

        # Build full config dict
        config_dict: dict[str, Any] = {
            "stage_id": self.stage_id,
            "stage_type": StageType(self.stage_type).value,
            "engine_args": create_config(engine_args),
            "runtime": create_config(runtime),
            "engine_input_source": self.input_sources,  # Legacy field name
            "final_output": self.final_output,
            "final_output_type": self.final_output_type,
            "is_comprehension": self.is_comprehension,
        }

        if self.custom_process_input_func:
            config_dict["custom_process_input_func"] = self.custom_process_input_func

        # Pass through extra YAML fields (default_sampling_params,
        # output_connectors, input_connectors, tts_args, etc.)
        config_dict.update(self.yaml_extras)

        return create_config(config_dict)


@dataclass
class ModelPipeline:
    """Complete pipeline definition for a multi-stage model (legacy).

    TODO(@lishunyang12): remove once all models migrate to PipelineConfig.
    """

    model_type: str
    stages: list[StageConfig]

    # Pipeline-wide behavior flags
    async_chunk: bool = False

    # Optional distributed configuration
    connectors: dict[str, Any] | None = None
    edges: list[dict[str, Any]] | None = None

    def get_stage(self, stage_id: int) -> StageConfig | None:
        """Look up a stage by its ID.

        Args:
            stage_id: The stage ID to search for.

        Returns:
            The matching StageConfig, or None if not found.
        """
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def validate_pipeline(self) -> list[str]:
        """Validate pipeline topology at model integration time (not runtime).

        Checks:
        - All stage IDs are unique
        - All input_sources reference valid stage IDs
        - At least one entry point (stage with empty input_sources)

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors: list[str] = []

        if not self.stages:
            errors.append("Topology has no stages defined")
            return errors

        # Check for unique stage IDs
        stage_ids = [s.stage_id for s in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            errors.append("Duplicate stage IDs found")

        stage_id_set = set(stage_ids)

        # Check input_sources reference valid stages
        for stage in self.stages:
            for source_id in stage.input_sources:
                if source_id not in stage_id_set:
                    errors.append(f"Stage {stage.stage_id} references non-existent input source {source_id}")
                if source_id == stage.stage_id:
                    errors.append(f"Stage {stage.stage_id} references itself as input source")

        # Check for at least one entry point
        entry_points = [s for s in self.stages if not s.input_sources]
        if not entry_points:
            errors.append("No entry point found (stage with empty input_sources)")

        return errors


class StageConfigFactory:
    """Factory that loads pipeline YAML and merges CLI overrides.

    Handles both single-stage and multi-stage models.

    Pipelines are declared in ``vllm_omni/config/pipeline_registry.py`` and
    loaded lazily via ``_PIPELINE_REGISTRY``; no hardcoded model-type →
    directory mapping is maintained here. Models with generic HF
    ``model_type`` collisions (e.g. MiMo Audio reports ``qwen2``) should
    declare ``hf_architectures=(...)`` on their ``PipelineConfig`` so the
    factory can disambiguate via ``hf_config.architectures``.
    """

    @classmethod
    def create_from_model(
        cls,
        model: str,
        cli_overrides: dict[str, Any] | None = None,
        deploy_config_path: str | None = None,
        **deprecated_kwargs: Any,
    ) -> list[StageConfig] | None:
        """Load pipeline + deploy config, merge with CLI overrides.

        Checks _PIPELINE_REGISTRY first (new path), falls back to legacy YAML.
        """
        _warn_deprecated_kwargs(deprecated_kwargs)

        if cli_overrides is None:
            cli_overrides = {}

        trust_remote_code = cli_overrides.get("trust_remote_code", True)
        if trust_remote_code is None:
            trust_remote_code = False

        # --- New path: check pipeline registry by model_type first ---
        model_type, hf_config = cls._auto_detect_model_type(model, trust_remote_code=trust_remote_code)
        if model_type and model_type in _PIPELINE_REGISTRY:
            return cls._create_from_registry(model_type, cli_overrides, deploy_config_path)

        # --- HF architecture fallback: some models report a generic
        # model_type that collides with another model. Match by the
        # hf_architectures declared on each registered PipelineConfig.
        if hf_config is not None:
            hf_archs = set(getattr(hf_config, "architectures", []) or [])
            if hf_archs:
                for registered in _PIPELINE_REGISTRY.values():
                    if hf_archs.intersection(registered.hf_architectures):
                        return cls._create_from_registry(registered.model_type, cli_overrides, deploy_config_path)

        # --- Legacy path: load from pipeline YAML ---
        pipeline = cls._load_pipeline(model, trust_remote_code=trust_remote_code)

        if pipeline is None:
            return None

        errors = pipeline.validate_pipeline()
        if errors:
            logger.warning(f"Pipeline validation warnings for {model}: {errors}")

        # Materialize the resolved pipeline-wide async_chunk value into every
        # stage so build_engine_args_dict() can inject the stage connector
        # spec and explicit False overrides are preserved.
        resolved_async_chunk = cli_overrides.get("async_chunk")
        if resolved_async_chunk is None:
            resolved_async_chunk = bool(pipeline.async_chunk)
        for stage in pipeline.stages:
            stage.yaml_engine_args["async_chunk"] = bool(resolved_async_chunk)

        # Apply CLI overrides
        result: list[StageConfig] = []
        for stage in pipeline.stages:
            # Merge global CLI overrides
            stage.runtime_overrides = cls._merge_cli_overrides(stage, cli_overrides)
            result.append(stage)

        return result

    @classmethod
    def _create_from_registry(
        cls,
        model_type: str,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None = None,
        **deprecated_kwargs: Any,
    ) -> list[StageConfig]:
        """Create StageConfigs from pipeline registry + deploy YAML.

        Precedence: caller-typed (non-None) value > deploy YAML >
        StageDeployConfig dataclass default.
        """
        _warn_deprecated_kwargs(deprecated_kwargs)

        # Resolve deploy config path
        if deploy_config_path is None:
            deploy_path = _DEPLOY_DIR / f"{model_type}.yaml"
        else:
            deploy_path = Path(deploy_config_path)

        if not deploy_path.exists():
            logger.warning(
                "Deploy config not found: %s — using pipeline defaults only",
                deploy_path,
            )
            deploy_cfg = DeployConfig()
        else:
            deploy_cfg = load_deploy_config(deploy_path)

        cli_async_chunk = cli_overrides.get("async_chunk")
        if cli_async_chunk is not None:
            deploy_cfg.async_chunk = bool(cli_async_chunk)

        pipeline_key = deploy_cfg.pipeline or model_type
        if pipeline_key not in _PIPELINE_REGISTRY:
            raise KeyError(
                f"Pipeline {pipeline_key!r} not in registry "
                f"(resolved from {deploy_path.name!r}). Available: "
                f"{sorted(_PIPELINE_REGISTRY.keys())}"
            )
        pipeline_cfg = _PIPELINE_REGISTRY[pipeline_key]

        stages = merge_pipeline_deploy(pipeline_cfg, deploy_cfg, cli_overrides)

        explicit_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

        for stage in stages:
            stage.runtime_overrides = cls._merge_cli_overrides(stage, explicit_overrides)

        return stages

    @classmethod
    def create_default_diffusion(cls, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Single-stage diffusion - no YAML needed.

        Creates a default diffusion stage configuration for single-stage
        diffusion models. Returns a legacy OmegaConf-compatible dict for
        backward compatibility with OmniStage.

        Args:
            kwargs: Engine arguments from CLI/API.

        Returns:
            List containing a single config dict for the diffusion stage.
        """
        # Calculate devices based on parallel config
        devices = "0"
        if "parallel_config" in kwargs:
            num_devices = kwargs["parallel_config"].world_size
            for i in range(1, num_devices):
                devices += f",{i}"

        engine_args: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in ("parallel_config",):
                continue
            engine_args[key] = value

        # Serialize parallel_config as dict for OmegaConf. Test helpers
        # sometimes pass SimpleNamespace rather than a dataclass instance.
        if "parallel_config" in kwargs:
            parallel_config = kwargs["parallel_config"]
            if dataclasses.is_dataclass(parallel_config) and not isinstance(parallel_config, type):
                engine_args["parallel_config"] = asdict(parallel_config)
            elif hasattr(parallel_config, "__dict__"):
                engine_args["parallel_config"] = dict(vars(parallel_config))
            else:
                engine_args["parallel_config"] = parallel_config

        engine_args.setdefault("cache_backend", "none")
        engine_args["model_stage"] = "diffusion"

        # Convert dtype to string for OmegaConf
        if "dtype" in engine_args:
            engine_args["dtype"] = str(engine_args["dtype"])

        engine_args.setdefault("max_num_seqs", 1)

        config_dict: dict[str, Any] = {
            "stage_id": 0,
            "stage_type": StageType.DIFFUSION.value,
            "runtime": {
                "process": True,
                "devices": devices,
            },
            "engine_args": create_config(engine_args),
            "final_output": True,
            "final_output_type": "image",
        }

        return [config_dict]

    @classmethod
    def _load_pipeline(cls, model: str, trust_remote_code: bool = True) -> ModelPipeline | None:
        """Load a legacy ``pipeline.yaml`` for the model.

        Searches ``model_executor/models/<dir>/pipeline.yaml`` by trying
        (a) the raw ``model_type`` as the directory name, then
        (b) ``model_type`` with hyphens replaced by underscores,
        and finally (c) scanning every ``pipeline.yaml`` for one that
        declares a matching ``model_type`` or ``hf_architectures``.

        Returns None if no pipeline.yaml is found — caller handles the
        ``resolve_model_config_path`` fallback via stage_configs/ YAMLs.
        """
        model_type, hf_config = cls._auto_detect_model_type(model, trust_remote_code=trust_remote_code)
        if model_type is None:
            return None

        # Direct lookups by convention
        candidates = [model_type, model_type.replace("-", "_")]
        for dir_name in candidates:
            pipeline_path = get_pipeline_path(dir_name, "pipeline.yaml")
            if pipeline_path.exists():
                return cls._parse_pipeline_yaml(pipeline_path, model_type)

        # Scan fallback: read every pipeline.yaml and match on declared fields
        hf_archs = set(getattr(hf_config, "architectures", []) or []) if hf_config else set()
        if _MODELS_DIR.exists():
            for subdir in sorted(_MODELS_DIR.iterdir()):
                if not subdir.is_dir():
                    continue
                pipeline_path = subdir / "pipeline.yaml"
                if not pipeline_path.exists():
                    continue
                try:
                    cfg = load_yaml_config(pipeline_path)
                except Exception as exc:
                    logger.debug("Skip %s: %s", pipeline_path, exc)
                    continue
                declared_type = getattr(cfg, "model_type", None)
                declared_archs = set(getattr(cfg, "hf_architectures", None) or [])
                if declared_type == model_type or (hf_archs and hf_archs.intersection(declared_archs)):
                    return cls._parse_pipeline_yaml(pipeline_path, declared_type or model_type)

        logger.debug("No pipeline.yaml found for model_type %s (archs=%s)", model_type, sorted(hf_archs))
        return None

    # Keys consumed as explicit StageConfig fields — everything else is
    # passed through via yaml_extras.
    _KNOWN_STAGE_KEYS: set[str] = {
        "stage_id",
        "model_stage",
        "stage_type",
        "input_sources",
        "engine_input_source",
        "custom_process_input_func",
        "final_output",
        "final_output_type",
        "worker_type",
        "scheduler_cls",
        "hf_config_name",
        "is_comprehension",
        "engine_args",
        "runtime",
    }

    @classmethod
    def _parse_pipeline_yaml(cls, path: Path, model_type: str) -> ModelPipeline:
        """Parse a pipeline YAML file.

        Args:
            path: Path to the YAML file.
            model_type: Model type identifier.

        Returns:
            ModelPipeline object.
        """
        config_data = load_yaml_config(path)

        stages: list[StageConfig] = []
        for stage_data in config_data.stages:
            # Use .get() for optional fields — idiomatic for OmegaConf DictConfig
            stage_type_str = stage_data.get("stage_type", "llm")
            stage_type = StageType(stage_type_str) if stage_type_str else StageType.LLM

            # Handle both 'input_sources' (new) and 'engine_input_source' (legacy)
            input_sources = stage_data.get("input_sources", None)
            if input_sources is None:
                input_sources = stage_data.get("engine_input_source", [])
            if input_sources is None:
                input_sources = []
            input_sources = list(input_sources)

            # Extract per-stage engine_args and runtime dicts
            raw_ea = stage_data.get("engine_args", None)
            yaml_engine_args = to_dict(raw_ea) if raw_ea is not None else {}
            raw_rt = stage_data.get("runtime", None)
            yaml_runtime = to_dict(raw_rt) if raw_rt is not None else {}

            # Migrate legacy runtime.max_batch_size → engine_args.max_num_seqs
            if "max_batch_size" in yaml_runtime:
                mbs = yaml_runtime.pop("max_batch_size")
                yaml_engine_args.setdefault("max_num_seqs", int(mbs))
                logger.debug(
                    "Stage %s: migrated runtime.max_batch_size=%s to engine_args.max_num_seqs",
                    stage_data.get("stage_id", "?"),
                    mbs,
                )

            # Topology-level fields that also live inside engine_args in legacy
            # YAMLs (worker_type, scheduler_cls, etc.) — read from both places.
            worker_type = stage_data.get("worker_type", None) or yaml_engine_args.pop("worker_type", None)
            scheduler_cls = stage_data.get("scheduler_cls", None) or yaml_engine_args.pop("scheduler_cls", None)
            hf_config_name = stage_data.get("hf_config_name", None) or yaml_engine_args.pop("hf_config_name", None)
            model_stage = getattr(stage_data, "model_stage", None) or yaml_engine_args.pop("model_stage", None)

            # Collect pass-through fields (default_sampling_params,
            # output_connectors, input_connectors, tts_args, etc.)
            yaml_extras: dict[str, Any] = {}
            for key in stage_data:
                if key not in cls._KNOWN_STAGE_KEYS:
                    val = stage_data[key]
                    try:
                        yaml_extras[key] = to_dict(val)
                    except ValueError:
                        yaml_extras[key] = val

            stage = StageConfig(
                stage_id=stage_data.stage_id,
                model_stage=model_stage or "",
                stage_type=stage_type,
                input_sources=input_sources,
                custom_process_input_func=stage_data.get("custom_process_input_func", None),
                final_output=stage_data.get("final_output", False),
                final_output_type=stage_data.get("final_output_type", None),
                worker_type=worker_type,
                scheduler_cls=scheduler_cls,
                hf_config_name=hf_config_name,
                is_comprehension=stage_data.get("is_comprehension", False),
                yaml_engine_args=yaml_engine_args,
                yaml_runtime=yaml_runtime,
                yaml_extras=yaml_extras,
            )
            stages.append(stage)

        # Get pipeline-wide flags
        async_chunk = config_data.get("async_chunk", False)

        # Get optional connector config — check both top-level and nested
        # under ``runtime`` (legacy stage_configs format).
        connectors = None
        edges = None
        if hasattr(config_data, "connectors"):
            connectors = to_dict(config_data.connectors)
        if hasattr(config_data, "edges"):
            edges = to_dict(config_data.edges)
        if hasattr(config_data, "runtime") and config_data.runtime is not None:
            top_runtime = config_data.runtime
            if connectors is None and hasattr(top_runtime, "connectors"):
                connectors = to_dict(top_runtime.connectors)
            if edges is None and hasattr(top_runtime, "edges"):
                edges = to_dict(top_runtime.edges)

        return ModelPipeline(
            model_type=getattr(config_data, "model_type", model_type),
            stages=stages,
            async_chunk=async_chunk,
            connectors=connectors,
            edges=edges,
        )

    @classmethod
    def _auto_detect_model_type(cls, model: str, trust_remote_code: bool = True) -> tuple[str | None, Any]:
        """Auto-detect model_type from model directory.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            Tuple of (model_type, hf_config). Both may be None on failure.
        """
        try:
            from vllm.transformers_utils.config import get_config

            hf_config = get_config(model, trust_remote_code=trust_remote_code)
            return hf_config.model_type, hf_config
        except Exception as e:
            logger.debug(f"`get_config` failed for {e}; Falling back to raw config.json path")

        # Fallback: read config.json directly for custom model types that
        # are not registered with transformers (e.g. qwen3_tts).
        try:
            from vllm.transformers_utils.config import get_hf_file_to_dict

            config_dict = get_hf_file_to_dict("config.json", model, revision=None)
            if config_dict:
                if "model_type" in config_dict:
                    return config_dict["model_type"], None
                # VoxCPM2-style configs use singular ``architecture`` rather
                # than HF's standard ``model_type`` / ``architectures``. Accept
                # it as a fallback so the pipeline registry can still match.
                if "architecture" in config_dict and isinstance(config_dict["architecture"], str):
                    return config_dict["architecture"], None
        except Exception as e:
            logger.debug(f"Failed to auto-detect model type for {model}: {e}")

        # Fallback for diffusers-style models: check model_index.json.
        # Some models (e.g. GLM-Image) have no root config.json but ship a
        # model_index.json with _class_name that maps to a pipeline key via
        # PipelineConfig.diffusers_class_name.
        try:
            from vllm.transformers_utils.config import get_hf_file_to_dict

            model_index = get_hf_file_to_dict("model_index.json", model, revision=None)
            if model_index and "_class_name" in model_index:
                class_name = model_index["_class_name"]
                for pipeline_cfg in _PIPELINE_REGISTRY.values():
                    if pipeline_cfg.diffusers_class_name == class_name:
                        logger.info(
                            "Detected pipeline %r from model_index.json (_class_name=%r)",
                            pipeline_cfg.model_type,
                            class_name,
                        )
                        return pipeline_cfg.model_type, None
        except Exception as e:
            logger.debug(f"Failed to detect model type for diffusers-style models: {e}")

        # Final fallback: some models (e.g. CosyVoice3) ship an empty
        # config.json and rely on naming conventions. Match the model path
        # basename against registered pipeline keys — longest match wins
        # so "cosyvoice3" (length 10) beats "cosyvoice" (length 9).
        model_lower = model.lower().replace("-", "").replace("_", "")
        best: str | None = None
        best_len = 0
        for registered_key in _PIPELINE_REGISTRY.keys():
            candidate = registered_key.lower().replace("-", "").replace("_", "")
            if candidate and candidate in model_lower and len(candidate) > best_len:
                best = registered_key
                best_len = len(candidate)
        if best is not None:
            return best, None

        return None, None

    @classmethod
    def _merge_cli_overrides(
        cls,
        stage: StageConfig,
        cli_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge global and per-stage (``stage_N_*``) CLI overrides.

        Orchestrator-owned keys are filtered by ``build_stage_runtime_overrides``
        using ``OrchestratorArgs`` as the single source of truth; unknown
        server/uvicorn keys are dropped downstream by
        ``filter_dataclass_kwargs(OmniEngineArgs, ...)``.
        """
        return build_stage_runtime_overrides(stage.stage_id, cli_overrides)
