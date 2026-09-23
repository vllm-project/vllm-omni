# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage configuration system for vLLM-Omni."""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NamedTuple

from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler

from vllm_omni.config.endpoint_policy import EndpointRestriction
from vllm_omni.config.yaml_util import load_yaml_config, to_dict
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
    (``model`` / ``log_stats`` / ``stage_id``) leak
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


def normalize_pipeline_cli_overrides(
    pipeline: PipelineConfig,
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Translate pipeline-owned global CLI aliases into stage-scoped overrides."""
    normalized = dict(cli_overrides)
    for source, (stage_id, target) in pipeline.stage_cli_aliases.items():
        invalid_stage_keys = [
            key
            for key, value in normalized.items()
            if value is not None
            and (match := _STAGE_OVERRIDE_PATTERN.match(key)) is not None
            and match.group(2) == source
            and int(match.group(1)) != stage_id
        ]
        if invalid_stage_keys:
            invalid = ", ".join(sorted(invalid_stage_keys))
            raise ValueError(
                f"{invalid} cannot be set for pipeline {pipeline.model_type!r}; "
                f"{source} belongs to stage {stage_id} as {target}."
            )
        value = normalized.pop(source, None)
        if value is None:
            continue
        stage_key = f"stage_{stage_id}_{target}"
        stage_value = normalized.get(stage_key)
        if stage_value is not None and stage_value != value:
            warnings.warn(
                f"Ignoring {source}={value!r} because {stage_key}={stage_value!r} takes precedence.",
                UserWarning,
                stacklevel=2,
            )
            continue
        normalized[stage_key] = value
    return normalized


def _apply_diffusion_parallel_runtime_overrides(
    engine_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> None:
    """Move diffusion parallel CLI overrides into nested ``parallel_config``."""
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
        if key in ("ulysses_degree", "ring_degree", "allgather_degree"):
            degree_overridden = True
        parallel_config_dict[key] = runtime_overrides.pop(key)

    if parallel_config_dict is not None and degree_overridden and not sequence_parallel_explicit:
        ulysses_degree = parallel_config_dict.get("ulysses_degree") or 1
        ring_degree = parallel_config_dict.get("ring_degree") or 1
        allgather_degree = parallel_config_dict.get("allgather_degree") or 1
        parallel_config_dict["sequence_parallel_size"] = (
            allgather_degree if allgather_degree > 1 else ulysses_degree * ring_degree
        )

    if parallel_config_dict is not None:
        engine_args["parallel_config"] = parallel_config_dict


def reconcile_diffusion_attention_overrides(
    engine_args: dict[str, Any],
    runtime_overrides: Mapping[str, Any],
) -> None:
    """Apply CLI precedence across the two diffusion attention representations."""
    if runtime_overrides.get("diffusion_attention_backend") is not None:
        yaml_config = engine_args.get("diffusion_attention_config")
        if isinstance(yaml_config, Mapping) and yaml_config.get("default") is not None:
            remaining = {k: v for k, v in yaml_config.items() if k != "default"}
            if remaining:
                engine_args["diffusion_attention_config"] = remaining
            else:
                engine_args.pop("diffusion_attention_config")
    if runtime_overrides.get("diffusion_attention_config") is not None:
        engine_args.pop("diffusion_attention_backend", None)


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
    # The model keeps per-request execution state while awaiting the next
    # async chunk, so the parked request continues to consume model capacity.
    retains_state_across_chunks: bool = False
    sampling_constraints: dict[str, Any] = field(default_factory=dict)
    custom_process_input_func: str | None = None
    custom_process_next_stage_input_func: str | None = None
    # Alternates picked by the typed stage resolver based on ``deploy.async_chunk``.
    async_chunk_process_next_stage_input_func: str | None = None
    sync_process_input_func: str | None = None
    supports_native_mrv2_data_plane: bool = False
    # Rewrites the Stage-0 view of a raw prompt before vLLM input processing.
    # The callable receives ``(prompt, sampling_params_list)``; downstream
    # stages continue to receive the original prompt.
    prompt_transform_func: str | None = None
    prompt_expand_func: str | None = None
    cfg_kv_collect_func: str | None = None
    # Payload keys this stage expects to receive from its upstream stage over
    # the omni connector instead of through the orchestrator IPC hop. Declaring
    # them is what enables the worker-side connector receive path, so an empty
    # tuple leaves existing deployments untouched.
    stage_input_payload_keys: tuple[str, ...] = ()
    # Mirror of the above on the producing side: keys this stage hands to the
    # next stage over the connector. Only diffusion producers need this; AR
    # stages already send through ``send_full_payload_outputs``.
    stage_output_payload_keys: tuple[str, ...] = ()
    omni_kv_config: dict[str, Any] | None = None
    scheduler_cls: str | None = None
    # Model subdirectory indirections: for multi-component HF repos where the
    # stage's config/tokenizer lives in a subdirectory (e.g. GLM-Image's AR
    # config is in ``vision_language_encoder/``).  Consumed at stage-init time
    # by ``stage_init_utils._resolve_model_tokenizer_paths``.
    model_subdir: str | None = None
    tokenizer_subdir: str | None = None
    # Model-owned hook that resolves a pipeline root to this stage's checkpoint.
    # Consumed and removed before backend engine args are constructed.
    model_path_resolver: str | None = None
    # Keep a single-replica diffusion stage in the orchestrator process.
    # Disabled by default so existing multi-stage pipelines retain subprocess
    # isolation unless their topology explicitly opts in.
    inline_diffusion: bool = False
    # Whether the non-async path waits for a complete upstream payload from
    # the model-runner connector before scheduling this stage.
    requires_full_payload_input: bool = False
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
    diffusers_class_aliases: tuple[str, ...] = ()
    endpoint_restrictions: tuple[EndpointRestriction, ...] = ()
    # Dotted path of the model's ``DuplexModelPlugin``. Online serving uses
    # DuplexOmni only when the deploy configuration selects session_mode: duplex.
    duplex_plugin: str | None = None
    # Preserve legacy turn deployments when adding an optional duplex plugin.
    default_session_mode: str | None = None
    # Legacy duplex wiring of the models that are not ported to the plugin
    # framework yet (PersonaPlex, Nemotron VoiceChat). Nothing reads them: a
    # pipeline that only declares these is served turn-based. Each field goes
    # away with the follow-up PR that ports its model to ``duplex_plugin``.
    duplex_runtime_extension: str | None = None
    duplex_serving_adapter: str | None = None
    duplex_control_enabled: bool = False
    # Bundled deploy defaults for this concrete pipeline topology. The file is
    # loaded from vllm_omni/deploy; None uses DeployConfig defaults.
    default_deploy_config_name: str | None = None
    # Global CLI spelling -> (stage id, stage-local spelling).
    stage_cli_aliases: dict[str, tuple[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors = self.get_validation_errors()

        if errors:
            # If we have multiple errors, put them on separate indented lines for readability
            multi_sep = "\n\t"
            error_str = multi_sep.join(errors)
            if len(errors) > 1:
                error_str = f"{multi_sep}{error_str}"
            raise ValueError(f"PipelineConfig initialization failed with the following error(s): {error_str}")

    def get_stage(self, stage_id: int) -> StagePipelineConfig | None:
        """Look up a stage by its ID."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def get_validation_errors(self) -> list[str]:
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
        # Request completion and client-visible output are both driven by stages
        # that declare ``final_output`` (see ``Orchestrator._route_output``). A
        # pipeline without one can never emit a result, so every request hangs.
        if not any(s.final_output for s in self.stages):
            errors.append("No terminal stage (stage with final_output=True)")
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

    # False opts this stage out of pipeline-wide async chunking.
    async_chunk: bool | None = None

    # Inter-stage connector wiring and request defaults.
    output_connectors: dict[str, str] | None = None
    input_connectors: dict[str, str] | None = None
    default_sampling_params: dict[str, Any] | None = None
    default_pooling_params: dict[str, Any] | None = None
    subtalker_sampling_params: dict[str, Any] | None = None
    silence_ban_frames: int = 0

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
    # Hybrid-mamba stages (e.g. the NemotronVoiceChat thinker's NemotronH
    # backbone) pin the SSM state dtype; projected onto vLLM CacheConfig.
    mamba_ssm_cache_dtype: str | None = None

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
    ulysses_a2a_permute: bool | None = None
    ring_degree: int | None = None
    allgather_degree: int | None = None
    sequence_parallel_size: int | None = None
    cfg_parallel_size: int | None = None
    vae_patch_parallel_size: int | None = None
    vae_parallel_mode: str | None = None
    text_encoder_tp_size: int | None = None
    use_hsdp: bool | None = None
    hsdp_shard_size: int | None = None
    hsdp_replicate_size: int | None = None

    # Diffusion model loading and adapter construction.
    model_class_name: str | None = None
    diffusion_load_format: str | None = None
    lora_path: str | list[str] | None = None
    lora_backend: str | None = None
    lora_scale: float | None = None
    diffusers_load_kwargs: dict[str, Any] | None = None
    diffusers_call_kwargs: dict[str, Any] | None = None
    diffusion_quantization_config: str | None = None
    diffusion_attention_backend: str | None = None
    fastvideo_vsa_topk: int | None = None
    diffusion_attention_config: dict[str, Any] | None = None

    # Diffusion execution, cache, and VAE behavior.
    diffusion_compile_granularity: str | None = None
    diffusion_compile_dynamic: bool | None = None
    fa_deterministic: bool | None = None
    cache_backend: str | None = None
    cache_config: dict[str, Any] | None = None
    video_output_transport: dict[str, Any] | None = None
    enable_cache_dit_summary: bool | None = None
    step_execution: bool | None = None
    vae_use_slicing: bool | None = None
    vae_use_tiling: bool | None = None
    vae_fast_path: str | None = None
    boundary_ratio: float | None = None
    flow_shift: float | None = None
    diffusion_kv_cache_dtype: str | None = None
    diffusion_kv_cache_skip_steps: str | None = None
    diffusion_kv_cache_skip_layers: str | None = None
    auxiliary_text_encoder: str | None = None

    # Runtime optimizations used by diffusion loading/execution.
    enable_multithread_weight_load: bool | None = None
    enable_broadcast_weight_load: bool | None = None
    num_weight_load_threads: int | None = None
    diffusion_offload_config: dict[str, Any] | None = None
    # Compatibility aliases for existing callers and model-specific stage
    # lifecycles that are broader than the compact dit/text_encoder selector.
    enable_cpu_offload: bool | None = None
    enable_layerwise_offload: bool | None = None

    enable_distributed_layerwise_offload: bool | None = None
    dlo_use_allgather: bool | None = None
    dlo_resident_layers: int | None = None
    host_weight_runtime_mode: str | None = None
    host_weight_runtime_root: str | None = None
    dlo_host_registration_limit_gib: float | None = None
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
    # Unread by the plugin framework. It used to bound the per-session
    # completed-append table that made a retried append RPC submit once; the
    # framework carries appends as one-way commands on the runner's ordered
    # mailbox, so there is no client-visible retry to deduplicate. The field
    # stays so the deploy configs of the models that are not ported yet still
    # parse, and goes away with the PR that ports the last of them.
    completed_append_cache_size: int = 256
    server_vad_model_path: str | None = None
    # Startup warmup, before real ``/v1/realtime`` clients are admitted.
    # Audio-primary models run this many silent frames; 0 disables that path.
    # Video-required models (AURA) still run one short non-silent audio chunk
    # plus one image when this is 0, so ASR, vision, Talker and Code2Wav
    # compile their real shapes. The empty per-stage JIT registry does not.
    # A negative value disables every startup warmup.
    warmup_frames: int = 0

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
        if self.server_vad_model_path is not None and (
            not isinstance(self.server_vad_model_path, str) or not self.server_vad_model_path.strip()
        ):
            raise ValueError("duplex_session.server_vad_model_path must be a non-empty string or null")
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
    model_runner: Literal["v1", "v2"] = "v1"
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
        "async_chunk",
        "stage_id",
        "devices",
        "num_replicas",
        "env",
        "output_connectors",
        "input_connectors",
        "default_sampling_params",
        "default_pooling_params",
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

    kwargs["async_chunk"] = stage_data.get("async_chunk")
    kwargs["output_connectors"] = stage_data.get("output_connectors")
    kwargs["input_connectors"] = stage_data.get("input_connectors")
    kwargs["default_sampling_params"] = stage_data.get("default_sampling_params")
    kwargs["default_pooling_params"] = stage_data.get("default_pooling_params")
    kwargs["engine_extras"] = _get_recursively_merged_dict(explicit_engine_extras, flat_args)
    return StageDeployConfig(**kwargs)


_DEEP_MERGE_KEYS = frozenset(
    {
        "default_sampling_params",
        "default_pooling_params",
        "subtalker_sampling_params",
        "engine_extras",
        "engine_args",
    }
)


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


def _merge_connectors(
    base: dict[str, Any] | None,
    overlay: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Deep-merge named connector definitions from a deploy overlay."""
    if not base and not overlay:
        return None
    return _get_recursively_merged_dict(base or {}, overlay or {})


def resolve_deploy_yaml(path: str | Path) -> dict[str, Any]:
    """Load a deploy YAML with optional ``base_config`` inheritance."""
    raw_dict = to_dict(load_yaml_config(path))

    base_path = raw_dict.pop("base_config", None)
    if base_path is None:
        return raw_dict

    # Resolve relative to the overlay file's directory
    base_path = Path(path).parent / base_path
    base_dict = resolve_deploy_yaml(base_path)

    # Merge top-level scalars: overlay wins. Structured sections are merged
    # below so a thin overlay does not discard inherited runtime contracts.
    merged = {
        **base_dict,
        **{k: v for k, v in raw_dict.items() if k not in ("connectors", "stages", "platforms")},
    }
    merged_connectors = _merge_connectors(base_dict.get("connectors"), raw_dict.get("connectors"))
    if merged_connectors is not None:
        merged["connectors"] = merged_connectors
    merged["stages"] = _merge_stage_lists(base_dict.get("stages"), raw_dict.get("stages"))
    merged_platforms = _merge_platforms(base_dict.get("platforms"), raw_dict.get("platforms"))
    if merged_platforms is not None:
        merged["platforms"] = merged_platforms

    return merged


def load_deploy_config(path: str | Path) -> DeployConfig:
    """Load a deploy YAML (with optional base_config inheritance)."""
    raw_dict = resolve_deploy_yaml(path)
    if "stage_args" in raw_dict:
        raise ValueError(
            f"Deploy config {path} uses the removed `stage_args` schema; "
            "define topology in PipelineConfig and deployment overrides under `stages`."
        )

    stages = [_parse_stage_deploy(s) for s in raw_dict.get("stages", [])]

    model_runner = raw_dict.get("model_runner", "v1")
    if model_runner not in ("v1", "v2"):
        raise ValueError(f"model_runner must be one of ('v1', 'v2'), got {model_runner!r}")

    kwargs: dict[str, Any] = {
        "async_chunk": raw_dict.get("async_chunk", True),
        "session_mode": raw_dict.get("session_mode", "turn"),
        "model_runner": model_runner,
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
    """Merge platform-specific runner and stage overrides into deploy config."""
    if platform is None:
        from vllm_omni.platforms import current_omni_platform

        device_name = current_omni_platform.device_name
        platform = device_name.lower() if device_name is not None else None
    platform_section = (deploy.platforms or {}).get(platform) if platform is not None else None
    if platform_section is not None and "model_runner" in platform_section:
        model_runner = platform_section["model_runner"]
        if model_runner not in ("v1", "v2"):
            raise ValueError(f"platform model_runner must be one of ('v1', 'v2'), got {model_runner!r}")
        deploy.model_runner = model_runner
    if deploy.model_runner == "v2" and platform in {"npu", "xpu"}:
        raise NotImplementedError(
            f"Model Runner V2 is not supported on {platform.upper()}: "
            "the platform worker still uses the legacy chunk-transfer data plane."
        )
    if platform is None or deploy.platforms is None:
        return deploy
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


def resolve_stage_async_chunk(deploy: DeployConfig, stage: StageDeployConfig | None) -> bool:
    return bool(deploy.async_chunk and (stage is None or stage.async_chunk is not False))


def validate_stage_async_chunk_edges(pipeline: PipelineConfig, deploy: DeployConfig) -> None:
    stages = {stage.stage_id: stage for stage in pipeline.stages}
    deploy_by_id = {stage.stage_id: stage for stage in deploy.stages}
    for consumer in pipeline.stages:
        for source_id in consumer.input_sources:
            producer = stages[source_id]
            if not producer.async_chunk_process_next_stage_input_func:
                continue
            producer_async = resolve_stage_async_chunk(deploy, deploy_by_id.get(source_id))
            consumer_async = resolve_stage_async_chunk(deploy, deploy_by_id.get(consumer.stage_id))
            if producer_async != consumer_async:
                raise ValueError(
                    f"Pipeline {pipeline.model_type!r} has incompatible async_chunk settings on "
                    f"connector edge {source_id} -> {consumer.stage_id}. "
                    "Set the same async_chunk mode on both stages or disable pipeline-wide async_chunk."
                )


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


def merge_sampling_constraints(
    sampling_params: Mapping[str, Any] | None,
    constraints: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply scalar constraints and extend stop tokens without mutating inputs."""
    resolved_constraints = dict(constraints)
    if "stop_token_ids" in resolved_constraints:
        caller_stop_ids = (sampling_params or {}).get("stop_token_ids") or []
        required_stop_ids = resolved_constraints["stop_token_ids"] or []
        resolved_constraints["stop_token_ids"] = list(dict.fromkeys([*caller_stop_ids, *required_stop_ids]))
    return {**(sampling_params or {}), **resolved_constraints}
