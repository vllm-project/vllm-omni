# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage configuration system for vLLM-Omni."""

from __future__ import annotations

import dataclasses
import functools
import re
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler

from vllm_omni.config.endpoint_policy import EndpointRestriction
from vllm_omni.config.yaml_util import create_config, load_yaml_config, to_dict
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

logger = init_logger(__name__)

_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"

_STAGE_OVERRIDE_PATTERN = re.compile(r"^stage_(\d+)_(.+)$")


def pipeline_cfg_resolver(config_type: type[PretrainedConfig]):
    """Wraps a resolver such that we return None if a hf_config of the wrong type is provided."""

    def resolver_builder(func):
        @functools.wraps(func)
        def wrapper(hf_config: PretrainedConfig | None):
            if hf_config is None or not isinstance(hf_config, config_type):
                return None
            return func(hf_config)

        return wrapper

    return resolver_builder


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

        # Some fields are modeled as orchestrator-owned for top-level CLI
        # parsing, but are also legitimate deploy-time stage overrides. Keep
        # the default blacklist for true orchestrator/shared fields while
        # allowing any field explicitly represented by the deploy schema to
        # continue flowing into per-stage overrides.
        internal_keys = (internal_blacklist_keys() | SHARED_FIELDS) - deploy_runtime_override_keys()

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


def _apply_diffusion_parallel_runtime_overrides(
    engine_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> None:
    """Move diffusion parallel overrides into nested ``parallel_config``."""
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    parallel_fields = frozenset(f.name for f in fields(DiffusionParallelConfig))
    parallel_config = engine_args.get("parallel_config")
    parallel_config_dict = dict(parallel_config) if parallel_config is not None else None
    degree_overridden = False
    sequence_parallel_explicit = runtime_overrides.get("sequence_parallel_size") is not None

    for key in list(runtime_overrides.keys()):
        value = runtime_overrides.get(key)
        if value is None or key not in parallel_fields:
            continue
        if parallel_config_dict is None:
            parallel_config_dict = {}
        if key in ("ulysses_degree", "ring_degree"):
            degree_overridden = True
        parallel_config_dict[key] = runtime_overrides.pop(key)

    if parallel_config_dict is not None and degree_overridden and not sequence_parallel_explicit:
        ulysses_degree = parallel_config_dict.get("ulysses_degree") or 1
        ring_degree = parallel_config_dict.get("ring_degree") or 1
        parallel_config_dict["sequence_parallel_size"] = ulysses_degree * ring_degree

    if parallel_config_dict is not None:
        engine_args["parallel_config"] = parallel_config_dict


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


def _resolve_scheduler(
    execution_type: StageExecutionType,
    async_scheduling: bool = True,
) -> type[VLLMScheduler] | None:
    """Return the scheduler class for the given execution_type.

    NOTE: For AutoRegressive stages, we have two schedulers for sync / async
    respectively, and decide which to used based on the value of async_scheduling.
    For other execution types, async_scheduling is not used.
    """
    if execution_type == StageExecutionType.LLM_AR:
        if not async_scheduling:
            return OmniARScheduler
        return OmniARAsyncScheduler
    if execution_type == StageExecutionType.LLM_GENERATION:
        return OmniGenerationScheduler
    # Diffusion currently returns None here.
    return None


def _scheduler_path(cls: type[VLLMScheduler] | None) -> str | None:
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
    scheduler_cls: str | None = None
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
    # Optional second-stage predicate for resolving an arch-name collision
    # between sibling model generations that ship the same
    # ``architectures=[...]`` entry. When the arch-fallback in
    # ``StageConfigFactory.create_from_model`` finds an intersection with
    # ``hf_architectures``, it additionally evaluates this predicate against
    # the loaded ``hf_config`` and only selects this pipeline when it
    # returns ``True``. Leave ``None`` to skip the extra check (default).
    # Example: MiniCPM-o 4.5 and 2.6 both ship ``architectures=["MiniCPMO"]``
    # but differ on the ``version`` field, so the 4.5 pipeline declares
    # ``hf_config_predicate=lambda c: getattr(c, "version", "") == "4.5"``
    # to avoid misrouting 2.6 checkpoints.
    hf_config_predicate: Callable[[Any], bool] | None = None
    # Diffusers pipeline class name: for models that ship a ``model_index.json``
    # (no root ``config.json``), the ``_class_name`` field is matched against
    # this value to auto-detect the pipeline.  Only needed for diffusers-style
    # multi-component repos (e.g. GLM-Image).  ``None`` = not a diffusers model.
    diffusers_class_name: str | None = None
    endpoint_restrictions: tuple[EndpointRestriction, ...] = ()
    # Optional model-owned duplex planner loaded by the stable engine runtime.
    duplex_runtime_extension: str | None = None
    # Optional model-owned Serving adapter loaded only when the Realtime duplex
    # endpoint is enabled. Generic OpenAI modules must not select a model.
    duplex_serving_adapter: str | None = None
    # Explicitly enable the stable duplex control mechanism. This is separate
    # from the optional model extension because turn-commit-only deployments
    # do not require a model planner.
    duplex_control_enabled: bool = False
    # Bundled deploy defaults for this concrete pipeline topology. The file is
    # loaded from vllm_omni/deploy; None uses DeployConfig defaults.
    default_deploy_config_name: str | None = None

    def get_stage(self, stage_id: int) -> StagePipelineConfig | None:
        """Look up a stage by its ID."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

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

    # === Omni stage wrapper fields ===
    # Stage identity and Omni runtime placement.
    stage_id: int
    devices: str | None = None
    num_replicas: int = 1
    env: dict[str, Any] | None = None

    # Inter-stage connector wiring and request defaults.
    output_connectors: dict[str, str] | None = None
    input_connectors: dict[str, str] | None = None
    default_sampling_params: dict[str, Any] | None = None
    subtalker_sampling_params: dict[str, Any] | None = None

    # === Generic stage engine fields ===
    # Parallelism, scheduler, and memory-capacity controls.
    tensor_parallel_size: int | None = None
    enable_expert_parallel: bool | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None

    # Generic execution, scheduling, and KV/cache behavior.
    enforce_eager: bool | None = None
    async_scheduling: bool | None = None
    disable_hybrid_kv_cache_manager: bool | None = None
    mm_processor_cache_gb: float | None = None

    # Generic compilation, profiling, tokenizer/config parsing, and model
    # loading controls.
    compilation_config: dict[str, Any] | None = None
    profiler_config: dict[str, Any] | None = None
    skip_mm_profiling: bool | None = None
    enable_flashinfer_autotune: bool | None = None
    config_format: str | None = None
    load_format: str | None = None
    tokenizer_mode: str | None = None

    # === Diffusion stage runtime fields ===
    # Diffusion parallel_config deploy/runtime override fields.
    ulysses_degree: int | None = None
    ulysses_mode: str | None = None
    ring_degree: int | None = None
    sequence_parallel_size: int | None = None
    cfg_parallel_size: int | None = None
    vae_patch_parallel_size: int | None = None
    vae_parallel_mode: str | None = None
    use_hsdp: bool | None = None
    hsdp_shard_size: int | None = None
    hsdp_replicate_size: int | None = None

    # Diffusion model loading and adapter construction.
    model_class_name: str | None = None
    diffusion_load_format: str | None = None
    diffusers_load_kwargs: dict[str, Any] | None = None
    diffusers_call_kwargs: dict[str, Any] | None = None
    diffusion_quantization_config: str | None = None
    diffusion_attention_backend: str | None = None
    diffusion_attention_config: dict[str, Any] | None = None

    # Diffusion execution, cache, and VAE behavior.
    cache_backend: str | None = None
    cache_config: dict[str, Any] | None = None
    enable_cache_dit_summary: bool | None = None
    step_execution: bool | None = None
    vae_use_slicing: bool | None = None
    vae_use_tiling: bool | None = None
    boundary_ratio: float | None = None
    flow_shift: float | None = None
    diffusion_kv_cache_dtype: str | None = None
    diffusion_kv_cache_skip_steps: str | None = None
    diffusion_kv_cache_skip_layers: str | None = None
    auxiliary_text_encoder: str | None = None

    # Runtime optimizations used by diffusion loading/execution.
    enable_multithread_weight_load: bool | None = None
    num_weight_load_threads: int | None = None
    enable_cpu_offload: bool | None = None
    enable_layerwise_offload: bool | None = None

    # Diffusion-specific debug and observability knobs.
    enable_diffusion_pipeline_profiler: bool | None = None

    # Modality/service constraints consumed outside the core engine config.
    max_generated_image_size: int | None = None
    tts_max_instructions_length: int | None = None

    # === Pass-through stage engine fields ===
    # Pass-through stage engine args that are not represented above.
    engine_extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DuplexSessionRuntimeConfig:
    """Server-owned lifecycle and per-session buffering limits."""

    idle_ttl_s: float | None = 300.0
    disconnect_grace_s: float = 30.0
    reaper_interval_s: float = 5.0
    resume_replay_ttl_s: float = 60.0
    resume_replay_max_bytes_per_session: int = 8 * 1024 * 1024
    max_pending_input_bytes_per_session: int = 16 * 1024 * 1024
    max_pending_turns_per_session: int = 4
    max_sessions: int = 1
    completed_append_cache_size: int = 256

    def __post_init__(self) -> None:
        positive = {
            "disconnect_grace_s": self.disconnect_grace_s,
            "reaper_interval_s": self.reaper_interval_s,
            "resume_replay_ttl_s": self.resume_replay_ttl_s,
            "resume_replay_max_bytes_per_session": self.resume_replay_max_bytes_per_session,
            "max_pending_input_bytes_per_session": self.max_pending_input_bytes_per_session,
            "max_pending_turns_per_session": self.max_pending_turns_per_session,
            "max_sessions": self.max_sessions,
            "completed_append_cache_size": self.completed_append_cache_size,
        }
        if self.idle_ttl_s is not None and self.idle_ttl_s <= 0:
            raise ValueError("duplex_session.idle_ttl_s must be positive or null")
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"duplex_session.{name} must be positive")


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
    session_mode: str = "turn"
    # Stage-1 active stream slots; 0 preserves legacy all-stream cycling.
    active_stream_window: int = 0
    duplex_session: DuplexSessionRuntimeConfig = field(default_factory=DuplexSessionRuntimeConfig)
    connectors: dict[str, Any] | None = None
    edges: list[dict[str, Any]] | None = None
    stages: list[StageDeployConfig] = field(default_factory=list)
    platforms: dict[str, Any] | None = None
    # Overrides the auto-detected pipeline registry key for structural variants.
    pipeline: str | None = None

    # === Pipeline-wide engine settings (applied uniformly to every stage) ===
    trust_remote_code: bool | None = None
    distributed_executor_backend: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    data_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    custom_voice_dir: str | None = None


_STAGE_RESERVED_KEYS = frozenset(
    {
        "stage_id",
        "devices",
        "num_replicas",
        "env",
        "output_connectors",
        "input_connectors",
        "default_sampling_params",
        "engine_extras",
        "engine_args",
        "runtime",
    }
)

# Fields on StageDeployConfig that are populated from engine_args dict
_STAGE_DEPLOY_FIELDS = {f.name: f for f in fields(StageDeployConfig) if f.name not in _STAGE_RESERVED_KEYS}


def deploy_runtime_override_keys() -> frozenset[str]:
    """Return deploy-schema fields that are valid CLI/runtime overrides.

    These keys form the positive contract for stage override propagation:
    stage-scoped deploy knobs plus top-level pipeline-wide engine settings.
    They must remain overridable even if they are also modeled on
    ``OrchestratorArgs`` for top-level CLI parsing.
    """
    return frozenset(_STAGE_DEPLOY_FIELDS) | frozenset(_PIPELINE_WIDE_ENGINE_FIELDS)


def _parse_stage_deploy(stage_data: dict[str, Any]) -> StageDeployConfig:
    """Parse a single stage entry from deploy YAML into StageDeployConfig."""
    # Get the non-reserved keys for this stage
    flat_args = {k: v for k, v in stage_data.items() if k not in _STAGE_RESERVED_KEYS}
    explicit_engine_extras = dict(stage_data.get("engine_extras") or {})
    runtime_cfg = dict(stage_data.get("runtime", {}))
    devices = runtime_cfg.get("devices", stage_data.get("devices"))
    num_replicas = runtime_cfg.get("num_replicas", stage_data.get("num_replicas", 1))
    env = runtime_cfg.get("env", stage_data.get("env"))

    if "engine_args" in stage_data:
        for k, v in stage_data["engine_args"].items():
            existing = flat_args.get(k)
            # If we have multiple dictionaries, merge recursively.
            if isinstance(v, dict) and isinstance(existing, dict):
                flat_args[k] = _get_recursively_merged_dict(existing, v)
            else:
                flat_args[k] = v

    kwargs: dict[str, Any] = {
        "stage_id": stage_data["stage_id"],
        "devices": devices,
        "num_replicas": int(num_replicas),
        "env": env,
    }
    for name, f in _STAGE_DEPLOY_FIELDS.items():
        if name in flat_args:
            kwargs[name] = flat_args.pop(name)

    kwargs["output_connectors"] = stage_data.get("output_connectors")
    kwargs["input_connectors"] = stage_data.get("input_connectors")
    kwargs["default_sampling_params"] = stage_data.get("default_sampling_params")
    kwargs["engine_extras"] = _get_recursively_merged_dict(explicit_engine_extras, flat_args)
    return StageDeployConfig(**kwargs)


_DEEP_MERGE_KEYS = frozenset({"default_sampling_params", "subtalker_sampling_params", "engine_extras", "engine_args"})


def _deep_merge_stage(base: dict, overlay: dict) -> dict:
    """Deep-merge ``_DEEP_MERGE_KEYS`` so thin overlays don't drop base keys."""
    # Deep merge _DEEP_MERGE_KEYS recursively
    base_merge_dict = {k: v for k, v in base.items() if k in _DEEP_MERGE_KEYS}
    overlay_merge_dict = {k: v for k, v in overlay.items() if k in _DEEP_MERGE_KEYS}

    # Get the merge dict; priority is base < overlay < merged sub
    merged_subdict = _get_recursively_merged_dict(original=base_merge_dict, update=overlay_merge_dict)
    merged_dict = {**base, **overlay, **merged_subdict}
    return merged_dict


def _get_recursively_merged_dict(original: dict, update: dict) -> dict:
    """Recursively merge two dicts, returning a new dict."""
    merged = original.copy()
    for k, update_v in update.items():
        orig_v = merged.get(k)
        if isinstance(orig_v, dict) and isinstance(update_v, dict):
            merged[k] = _get_recursively_merged_dict(orig_v, update_v)
        else:
            if orig_v is not None and (isinstance(orig_v, dict) != isinstance(update_v, dict)):
                logger.warning(
                    "Deep-merge key %r has non-dict value (base=%s, overlay=%s); "
                    "overlay will fully replace base instead of merging.",
                    k,
                    type(orig_v).__name__,
                    type(update_v).__name__,
                )

            merged[k] = update_v
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
        "session_mode": raw_dict.get("session_mode", "turn"),
        "active_stream_window": int(raw_dict.get("active_stream_window", 0) or 0),
        "duplex_session": DuplexSessionRuntimeConfig(**(raw_dict.get("duplex_session") or {})),
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
        "custom_voice_dir",
    ):
        if name in raw_dict:
            kwargs[name] = raw_dict[name]
    return DeployConfig(**kwargs)


class PlatformOverrides(NamedTuple):
    overrides: dict[str, Any]
    devices: str | None
    env: dict[str, Any] | None


def _extract_platform_overrides(ps: dict[str, Any]) -> PlatformOverrides:
    """Return overrides, devices, and env from a platform stage entry.

    Handles both the nested layout (``engine_args:`` / ``runtime.devices``) and
    the flat layout. ``devices`` is ``None`` when no override is set.
    """
    if "engine_args" in ps:
        overrides = dict(ps["engine_args"])
        runtime_cfg = ps.get("runtime", {})
        if "num_replicas" in runtime_cfg:
            overrides["num_replicas"] = runtime_cfg["num_replicas"]
        return PlatformOverrides(overrides, runtime_cfg.get("devices"), runtime_cfg.get("env"))
    overrides = {k: v for k, v in ps.items() if k not in ("stage_id", "devices", "env")}
    return PlatformOverrides(overrides, ps.get("devices"), ps.get("env"))


def _apply_platform_overrides(
    deploy: DeployConfig,
    platform: str | None = None,
) -> DeployConfig:
    """Merge platform-specific stage overrides into deploy config."""
    if platform is None:
        from vllm_omni.platforms import current_omni_platform

        device_name = current_omni_platform.device_name
        platform = device_name.lower() if device_name is not None else None
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
        po = _extract_platform_overrides(ps)
        if po.devices is not None:
            base.devices = po.devices
        if po.env is not None:
            if isinstance(base.env, dict) and isinstance(po.env, dict):
                base.env = {**base.env, **po.env}
            else:
                logger.warning(
                    "Stage %s env override replaces base env entirely (base type=%s, override type=%s)",
                    ps["stage_id"],
                    type(base.env).__name__,
                    type(po.env).__name__,
                )
                base.env = po.env
        for key, val in po.overrides.items():
            if hasattr(base, key):
                # Deep-merge dict-valued fields listed in _DEEP_MERGE_KEYS so
                # platform overlays don't silently clobber sibling keys (e.g.
                # setting default_sampling_params={max_tokens: 2048} must not
                # drop temperature / top_p / top_k from the base stage).
                if key in _DEEP_MERGE_KEYS and isinstance(val, dict):
                    base_val = getattr(base, key, None)
                    if isinstance(base_val, dict):
                        setattr(base, key, {**base_val, **val})
                        continue
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
    "active_stream_window",
    "custom_voice_dir",
)
PIPELINE_WIDE_ENGINE_FIELDS = _PIPELINE_WIDE_ENGINE_FIELDS


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
    if ps.execution_type == StageExecutionType.DIFFUSION and ps.model_arch:
        engine_args.setdefault("model_class_name", ps.model_arch)
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
            if k in _STAGE_RESERVED_KEYS or v is None:
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

    # async_chunk is irrelevant for single-stage pipelines, so we always disable it
    if len(pipeline.stages) <= 1:
        deploy.async_chunk = False

    # async_chunk only applies to multi-stage pipelines: a pipeline with no
    # consumer stages (every stage has empty input_sources) has no inter-stage
    # edges, so async_chunk is a no-op and we skip the check entirely.
    # For pipelines that DO have inter-stage edges, require a dedicated per-step
    # async producer (``async_chunk_process_next_stage_input_func``).
    # ``custom_process_next_stage_input_func`` is the full-payload / connector-path
    # producer and does NOT imply async_chunk support — pipelines like qwen2_5_omni
    # and covo_audio have it but removed their consumer-side ``custom_process_input_func``
    # because they don't support async_chunk, so accepting them here would silently
    # miswire the consumer stage instead of raising a clear error.
    _has_inter_stage_edges = any(ps.input_sources for ps in pipeline.stages)
    if (
        deploy.async_chunk
        and _has_inter_stage_edges
        and not any(ps.async_chunk_process_next_stage_input_func for ps in pipeline.stages)
    ):
        raise ValueError(
            f"Pipeline {pipeline.model_type!r} has async_chunk=True in deploy but no stage "
            "declares a dedicated async-chunk next-stage processor "
            "(``async_chunk_process_next_stage_input_func``). "
            "Either set async_chunk=False or implement an async-chunk producer on the pipeline."
        )

    result: list[StageConfig] = []
    for ps in pipeline.stages:
        ds = deploy_by_id.get(ps.stage_id)
        stage_type, worker_type = _resolve_execution_mode(ps.execution_type)
        input_proc, next_stage_proc = _select_processor_funcs(ps, deploy.async_chunk)
        engine_args = _build_engine_args(ps, ds, pipeline, deploy, next_stage_proc)
        sched_cls = _resolve_scheduler(
            ps.execution_type,
            engine_args.get("async_scheduling", True),
        )
        if ps.execution_type == StageExecutionType.LLM_AR:
            engine_args["async_scheduling"] = sched_cls is OmniARAsyncScheduler
        extras = _build_extras(ps, ds)
        runtime: dict[str, Any] = {"process": True}
        if ds is not None:
            if ds.devices is not None:
                runtime["devices"] = ds.devices
            runtime["num_replicas"] = ds.num_replicas
            if ds.env is not None:
                runtime["env"] = ds.env
        runtime["requires_multimodal_data"] = ps.requires_multimodal_data

        result.append(
            StageConfig(
                stage_id=ps.stage_id,
                model_stage=ps.model_stage,
                session_mode=deploy.session_mode,
                stage_type=stage_type,
                input_sources=list(ps.input_sources),
                custom_process_input_func=input_proc,
                final_output=ps.final_output,
                final_output_type=ps.final_output_type,
                worker_type=worker_type,
                scheduler_cls=ps.scheduler_cls or _scheduler_path(sched_cls),
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
    session_mode: str = "turn"
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
        runtime_overrides = dict(self.runtime_overrides)

        # Overlay topology-level fields
        engine_args["model_stage"] = self.model_stage
        if self.worker_type:
            engine_args["worker_type"] = self.worker_type
        if self.scheduler_cls:
            engine_args["scheduler_cls"] = self.scheduler_cls
        if self.hf_config_name:
            engine_args["hf_config_name"] = self.hf_config_name

        if StageType(self.stage_type) == StageType.DIFFUSION:
            _apply_diffusion_parallel_runtime_overrides(engine_args, runtime_overrides)

        # CLI overrides take precedence over YAML defaults
        for key, value in runtime_overrides.items():
            if value is not None and key not in ("devices", "max_batch_size", "num_replicas"):
                engine_args[key] = value

        # Build runtime config from YAML defaults + CLI overrides
        runtime: dict[str, Any] = dict(self.yaml_runtime)
        runtime.setdefault("process", True)
        if runtime_overrides.get("devices") is not None:
            runtime["devices"] = runtime_overrides["devices"]
        if runtime_overrides.get("num_replicas") is not None:
            runtime["num_replicas"] = runtime_overrides["num_replicas"]

        # Legacy compat: migrate runtime.max_batch_size → engine_args.max_num_seqs
        legacy_mbs = runtime.pop("max_batch_size", None)
        cli_mbs = runtime_overrides.get("max_batch_size")
        if legacy_mbs is not None or cli_mbs is not None:
            warnings.warn(
                "runtime.max_batch_size is deprecated and will be removed in a "
                "future release. Use engine_args.max_num_seqs instead.",
                FutureWarning,
                stacklevel=2,
            )
            effective_mbs = int(cli_mbs or legacy_mbs or 1)
            engine_args.setdefault("max_num_seqs", effective_mbs)

        # Build full config dict
        config_dict: dict[str, Any] = {
            "stage_id": self.stage_id,
            "stage_type": StageType(self.stage_type).value,
            "session_mode": self.session_mode,
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
