# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any, cast

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.logger import init_logger

from vllm_omni.config import OmniModelConfig
from vllm_omni.outputs.output_modality import OutputModality
from vllm_omni.platforms import current_omni_platform
from vllm_omni.plugins import load_omni_general_plugins
from vllm_omni.worker.omni_connector_validation import validate_worker_omni_connector

logger = init_logger(__name__)

# Maps model architecture names to their HuggingFace model_type values.
# Used when auto-injecting hf_overrides for models with missing config.json.
_ARCH_TO_MODEL_TYPE: dict[str, str] = {
    "CosyVoice3Model": "cosyvoice3",
    "GLMTTSForConditionalGeneration": "glm_tts",
    "IndexTTS2S2MelDecoder": "indextts2",
    "IndexTTS2TalkerForConditionalGeneration": "indextts2",
    "IndexTTS25S2MelDecoder": "indextts2_5",
    "IndexTTS25TalkerForConditionalGeneration": "indextts2_5",
    "OmniVoiceModel": "omnivoice",
    # PersonaPlex ships an empty config.json, so create_model_config() must patch
    # model_type=personaplex from the arch name or the staged pipeline can't load
    # the HF config without manual overrides.
    "PersonaPlexTalkerForConditionalGeneration": "personaplex",
    "PersonaPlexCode2Wav": "personaplex",
    # NemotronVoiceChat ships a NeMo-style config.json without model_type;
    # create_model_config() patches model_type=nemotron_voicechat from the arch.
    "NemotronVoiceChatThinkerForConditionalGeneration": "nemotron_voicechat",
    "NemotronVoiceChatTalkerForConditionalGeneration": "nemotron_voicechat",
    "NemotronVoiceChatCode2Wav": "nemotron_voicechat",
    "VoxCPM2TalkerForConditionalGeneration": "voxcpm2",
}

# Maps model architecture names to tokenizer subfolder paths within HF repos.
_TOKENIZER_SUBFOLDER_MAP: dict[str, str] = {
    "CosyVoice3Model": "CosyVoice-BlankEN",
    "GLMTTSForConditionalGeneration": "vq32k-phoneme-tokenizer",
}


def _register_omni_hf_configs() -> None:
    try:
        from transformers import AutoConfig

        from vllm_omni.model_executor.models.indextts2.configuration_indextts2 import (
            IndexTTS2Config,
            IndexTTS25Config,
        )
        from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
            MingDenseConfig,
            MingMoeConfig,
        )
        from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
            MossTTSLocalConfig,
            MossTTSRealtimeConfig,
        )
        from vllm_omni.model_executor.models.nemotron_voicechat.configuration_nemotron_voicechat import (
            NemotronVoiceChatConfig,
        )
        from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
            PersonaPlexConfig,
        )
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config
        from vllm_omni.transformers_utils.configs.glm_tts import GLMTTSConfig
        from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig
        from vllm_omni.transformers_utils.configs.voxcpm2 import VoxCPM2Config
    except Exception as exc:  # pragma: no cover - best-effort optional registration
        logger.warning("Skipping omni HF config registration due to import error: %s", exc)
        return

    # Register with both transformers AutoConfig and vLLM's config registry
    # so models with empty/missing config.json (e.g. CosyVoice3) can be
    # resolved when model_type is injected via hf_overrides.
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
    except ImportError:
        _CONFIG_REGISTRY = None

    for model_type, config_cls in [
        ("dense", MingDenseConfig),
        ("bailingmm", MingMoeConfig),
        ("indextts2", IndexTTS2Config),
        ("indextts2_5", IndexTTS25Config),
        ("moss_tts_local", MossTTSLocalConfig),
        ("moss_tts_realtime", MossTTSRealtimeConfig),
        ("qwen3_tts", Qwen3TTSConfig),
        ("personaplex", PersonaPlexConfig),
        ("nemotron_voicechat", NemotronVoiceChatConfig),
        ("cosyvoice3", CosyVoice3Config),
        ("glm_tts", GLMTTSConfig),
        ("omnivoice", OmniVoiceConfig),
        ("voxcpm2", VoxCPM2Config),
    ]:
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError:
            # Already registered elsewhere; ignore.
            pass
        if _CONFIG_REGISTRY is not None and model_type not in _CONFIG_REGISTRY:
            _CONFIG_REGISTRY[model_type] = config_cls


def register_omni_models_to_vllm():
    from vllm.model_executor.models import ModelRegistry

    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    _register_omni_hf_configs()

    # Unconditionally (re)register every omni arch into the upstream global
    # ModelRegistry: vLLM 0.27 added some omni archs (e.g.
    # Qwen3OmniMoeForConditionalGeneration) to its own registry, but resolves
    # them to upstream classes that lack omni features (e.g.
    # buffer_realtime_audio for the realtime endpoint). Registering here
    # overwrites those entries with the vllm-omni classes, which is the
    # behavior entrypoints resolving through the global registry rely on
    # (and matches OmniModelRegistry's omni-wins ordering).
    for arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items():
        ModelRegistry.register_model(arch, f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}:{cls_name}")

    # Register omni-specific reasoning parsers (e.g., step_audio).
    import vllm_omni.reasoning  # noqa: F401


@dataclass
class OmniEngineArgs(EngineArgs):
    """Engine arguments for omni models, extending base EngineArgs.
    Adds omni-specific configuration fields for multi-stage pipeline
    processing and output type specification.
    Args:
        stage_id: Identifier for the stage in a multi-stage pipeline.
            Defaults to 0 for per-stage engine construction. The CLI-level
            single-stage selector remains optional on the parsed argparse
            namespace and should not be forwarded as a nullable per-stage
            engine argument.
        model_stage: Stage type identifier, e.g., "thinker" or "talker"
            (default: "thinker")
        model_arch: Model architecture name
            (default: "Qwen2_5OmniForConditionalGeneration")
        engine_output_type: Optional output type specification for the engine.
            Used to route outputs to appropriate processors (e.g., "image",
            "audio", "latents"). If None, output type is inferred.
        hf_config_name: Optional key for HF config subkey to be extracted
            for this stage, e.g., talker_config; If None, the default
            HF config will be used.
        custom_process_next_stage_input_func: Optional path to a custom function for processing
            inputs from previous stages
            If None, default processing is used.
        stage_connector_spec: Extra configuration for stage connector
        async_chunk: If set to True, perform async chunk
        session_mode: Request lifecycle mode, either turn-based or duplex
        worker_type: Model Type, e.g., "ar" or "generation"
        task_type: Model-defined startup task type. Consumers validate the
            supported values and decide whether it selects request behavior,
            task-specific weights, or both.
        omni_master_address: TCP address that the OmniMasterServer (running
            inside AsyncOmniEngine) listens on for engine core registrations.
            Required when single-stage mode is active.
        omni_master_port: TCP port for the OmniMasterServer registration
            socket.  Required when single-stage mode is active.
        output_modalities: Optional list of output modality names to enable
            (e.g. ["text", "audio"]). If None, all modalities supported by
            the model are used.
        log_stats: Whether to log engine statistics. Defaults to False.
        custom_pipeline_args: Dictionary of arguments for custom pipeline
            initialization (e.g., ``{"pipeline_class": "my.Module"}``).
            Passed through to the diffusion stage engine.
    """

    stage_id: int = 0
    model_stage: str = "thinker"
    model_arch: str | None = None
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    requires_full_payload_input: bool = False
    stage_connector_spec: dict[str, Any] = field(default_factory=dict)
    subtalker_sampling_params: dict[str, Any] | None = None
    silence_ban_frames: int = 0
    async_chunk: bool = False
    session_mode: str = "turn"
    retains_state_across_chunks: bool = False
    # WS-A: Stage-1 active stream slots. 0 = legacy preempt-everything.
    # Must be declared here so engine_args dict propagation does not silently
    # drop the value when constructing OmniEngineArgs from kwargs.
    active_stream_window: int = 0
    # Engine admission is the single source of truth for per-stage pools of
    # model-owned streaming state (codec, decoder, and similar resources).
    duplex_max_sessions: int = 1
    omni_kv_config: dict | None = None
    quantization_config: Any | None = None
    force_cutlass_fp8: bool | None = None
    worker_type: str | None = None
    # Dotted path of a per-stage pooling-output decoder applied worker-side.
    pooling_output_decoder: str | None = None
    task_type: str | None = None
    worker_cls: str = None  # type: ignore[assignment]  # Upstream default is "auto"; omni resolves
    # in __post_init__ based on worker_type (ar/generation), so None is safe here.
    enable_sleep_mode: bool = False
    omni: bool = False
    # Diffusion request-mode batch admission (forwarded to OmniDiffusionConfig).
    request_batch_max_wait_ms: float = 0.0
    fa_deterministic: bool = False

    @classmethod
    def _add_omni_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        try:
            parser.add_argument("--omni", action="store_true", default=False, help="Enable Omni engine features.")
        except argparse.ArgumentError:
            pass
        try:
            parser.add_argument(
                "--enable-sleep-mode", action="store_true", default=False, help="Enable GPU memory pool for sleep mode."
            )
        except argparse.ArgumentError:
            pass
        return parser

    omni_master_address: str | None = None
    omni_master_port: int | None = None
    # OmniCoordinator integration knobs (process-local).
    omni_dp_size_local: int = 1
    omni_lb_policy: str = "random"
    omni_heartbeat_timeout: float = 30.0
    output_modalities: list[str] | None = None
    log_stats: bool = False
    custom_pipeline_args: dict[str, Any] | None = None
    has_sampling_extra_args: bool = False
    sampling_extra_args_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.worker_cls in (None, "auto"):
            if self.worker_type == "ar":
                self.worker_cls = current_omni_platform.get_omni_ar_worker_cls()
            elif self.worker_type == "generation":
                self.worker_cls = current_omni_platform.get_omni_generation_worker_cls()
        load_omni_general_plugins()

        connector_extra = self.stage_connector_spec.get("extra")
        connector_role = connector_extra.get("role") if isinstance(connector_extra, dict) else None
        needs_connector = bool(
            self.requires_full_payload_input or self.custom_process_next_stage_input_func or connector_role is not None
        )
        validate_worker_omni_connector(self.worker_cls, needs_connector)
        super().__post_init__()

    def _ensure_omni_models_registered(self):
        if hasattr(self, "_omni_models_registered"):
            return True
        register_omni_models_to_vllm()
        self._omni_models_registered = True
        return True

    def _patch_empty_hf_config(self, model_type: str) -> None:
        """For models with empty config.json (e.g. CosyVoice3), create a
        patched config in a temp directory with model_type set so that
        transformers AutoConfig.from_pretrained can resolve the config class.
        Sets self.hf_config_path to point to the patched directory."""
        try:
            from transformers import PretrainedConfig

            config_dict, _ = PretrainedConfig.get_config_dict(self.model)
            if config_dict.get("model_type"):
                return  # config.json already has model_type, no patching needed
        except Exception:
            # The official IndexTTS 2.5 bundle has no HuggingFace config.json.
            # Keep this exception model-scoped so other loader failures retain
            # vLLM's normal error path.
            if model_type != "indextts2_5" or not os.path.isdir(self.model):
                return
            config_path = os.path.join(self.model, "config.json")
            if os.path.lexists(config_path):
                return
            config_dict = {}

        # Create a temp dir with a patched config.json
        temp_dir = tempfile.mkdtemp(prefix="omni_hf_config_")
        config_dict["model_type"] = model_type
        config_dict.setdefault("architectures", [self.model_arch])
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        self.hf_config_path = temp_dir
        self._temp_config_dir = temp_dir
        logger.info("Patched empty HF config with model_type=%s at %s", model_type, temp_dir)

    def create_model_config(self) -> OmniModelConfig:
        """Create an OmniModelConfig from these engine arguments.
        Returns:
            OmniModelConfig instance with all configuration fields set
        """
        # register omni models to avoid model not found error
        self._ensure_omni_models_registered()

        # Build stage_connector_config from stage_connector_spec
        stage_connector_config = {
            "name": self.stage_connector_spec.get("name", "SharedMemoryConnector"),
            "extra": self.stage_connector_spec.get("extra", {}).copy(),
        }
        stage_connector_config["extra"]["stage_id"] = self.stage_id

        hf_overrides = cast(dict[str, Any] | Callable[[Any], Any] | None, getattr(self, "hf_overrides", None))
        # If model_arch is specified, inject it into hf_overrides so vLLM can
        # resolve the architecture even when config.json lacks 'architectures'.
        # Also inject model_type so AutoConfig can resolve the correct config
        # class for models with empty or missing config.json (e.g. CosyVoice3).
        if self.model_arch:
            if hf_overrides is None:
                hf_overrides = {}
                self.hf_overrides = hf_overrides
            if isinstance(hf_overrides, dict):
                hf_overrides.setdefault("architectures", [self.model_arch])
                if "model_type" not in hf_overrides:
                    model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                    if model_type is not None:
                        hf_overrides.setdefault("model_type", model_type)

                # Stage wrappers (e.g. Code2Wav) may need max_model_len larger
                # than the base checkpoint's text max_position_embeddings.
                if self.model_arch == "Qwen3TTSCode2Wav" and self.max_model_len is not None:
                    hf_overrides.setdefault("talker_config", {}).setdefault(
                        "max_position_embeddings", int(self.max_model_len)
                    )

            # For models whose HF config.json is empty or lacks model_type
            # (e.g. CosyVoice3), AutoConfig.from_pretrained fails because it
            # cannot determine which config class to use from the empty dict.
            # hf_overrides alone is not enough since transformers reads
            # model_type from config_dict before applying overrides.
            # Workaround: create a patched config.json in a temp directory
            # and point hf_config_path to it so vLLM reads model_type from it.
            if not self.hf_config_path:
                model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                if model_type is not None:
                    self._patch_empty_hf_config(model_type)

        tokenizer = cast(str | None, getattr(self, "tokenizer", None))

        # NemotronVoiceChat: the checkpoint's only tokenizer subfolder is the
        # auxiliary rnnt_tokenizer/ (ASR loss vocab), which the generic
        # subfolder auto-detect below would wrongly pick up. The text tokenizer
        # is the Nemotron backbone's, resolved from a local override or its
        # HF id (cache-friendly).
        if not tokenizer and self.model_arch in (
            "NemotronVoiceChatThinkerForConditionalGeneration",
            "NemotronVoiceChatTalkerForConditionalGeneration",
            "NemotronVoiceChatCode2Wav",
        ):
            self.tokenizer = os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH") or "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
            tokenizer = self.tokenizer
            logger.info("NemotronVoiceChat: using text tokenizer from %s", tokenizer)

        # Auto-detect tokenizer for models that store it in a subdirectory
        # rather than the root (e.g. CosyVoice3 uses CosyVoice-BlankEN/).
        if not tokenizer and self.model:
            model_path = self.model
            if os.path.isdir(model_path) and not os.path.isfile(os.path.join(model_path, "tokenizer_config.json")):
                for subfolder in sorted(os.listdir(model_path)):
                    candidate = os.path.join(model_path, subfolder)
                    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "tokenizer_config.json")):
                        self.tokenizer = candidate
                        tokenizer = candidate
                        logger.info("Auto-detected tokenizer at %s", candidate)
                        break
            elif not os.path.isdir(model_path):
                tokenizer_subfolder = _TOKENIZER_SUBFOLDER_MAP.get(self.model_arch or "")
                if tokenizer_subfolder:
                    # Download just the tokenizer files from the subfolder
                    try:
                        from huggingface_hub import snapshot_download

                        local_dir = snapshot_download(
                            model_path,
                            allow_patterns=[
                                f"{tokenizer_subfolder}/tokenizer*",
                                f"{tokenizer_subfolder}/tokenization_*",
                                f"{tokenizer_subfolder}/special_tokens*",
                                f"{tokenizer_subfolder}/vocab*",
                                f"{tokenizer_subfolder}/merges*",
                                f"{tokenizer_subfolder}/added_tokens*",
                            ],
                        )
                        candidate = os.path.join(local_dir, tokenizer_subfolder)
                        if os.path.isdir(candidate):
                            self.tokenizer = candidate
                            logger.info("Downloaded tokenizer from %s/%s", model_path, subfolder)
                    except Exception as e:
                        logger.warning("Failed to download tokenizer subfolder: %s", e)

        # Build the vLLM config first, then use it to create the Omni config.
        try:
            model_config = super().create_model_config()
        finally:
            # Clean up temp config dir if we created one
            if hasattr(self, "_temp_config_dir"):
                import shutil

                shutil.rmtree(self._temp_config_dir, ignore_errors=True)
                del self._temp_config_dir

        omni_config = OmniModelConfig.from_vllm_model_config(
            model_config=model_config,
            # All kwargs below are Omni specific
            stage_id=self.stage_id,
            async_chunk=self.async_chunk,
            session_mode=self.session_mode,
            retains_state_across_chunks=self.retains_state_across_chunks,
            active_stream_window=self.active_stream_window,
            duplex_max_sessions=self.duplex_max_sessions,
            model_stage=self.model_stage,
            model_arch=self.model_arch,
            worker_type=self.worker_type,
            pooling_output_decoder=self.pooling_output_decoder,
            engine_output_type=self.engine_output_type,
            hf_config_name=self.hf_config_name,
            custom_process_next_stage_input_func=self.custom_process_next_stage_input_func,
            requires_full_payload_input=self.requires_full_payload_input,
            stage_connector_config=stage_connector_config,
            subtalker_sampling_params=self.subtalker_sampling_params,
            silence_ban_frames=self.silence_ban_frames,
            omni_kv_config=self.omni_kv_config,
            task_type=self.task_type,
            has_sampling_extra_args=self.has_sampling_extra_args,
            sampling_extra_args_keys=tuple(self.sampling_extra_args_keys or ()),
        )
        return omni_config


@dataclass
class OmniAsyncEngineArgs(AsyncEngineArgs, OmniEngineArgs):
    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = AsyncEngineArgs.add_cli_args(parser)
        parser = OmniEngineArgs._add_omni_specific_args(parser)
        return parser

    @property
    def output_modality(self) -> OutputModality:
        """Parse engine_output_type into a type-safe OutputModality flag."""
        return OutputModality.from_string(self.engine_output_type)


# ============================================================================
# CLI argument routing
# ============================================================================
#
# vLLM-Omni's CLI flags live in three buckets:
#
#     ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
#     │ OrchestratorArgs │    │  OmniEngineArgs  │    │  (upstream vllm) │
#     │                  │    │                  │    │    server/api    │
#     │  stage_timeout   │    │  max_num_seqs    │    │  host, port      │
#     │  worker_backend  │    │  gpu_mem_util    │    │  ssl_keyfile     │
#     │  deploy_config   │    │  dtype, quant    │    │  api_key         │
#     │     ...          │    │     ...          │    │     ...          │
#     └──────────────────┘    └──────────────────┘    └──────────────────┘
#             │                        │                        │
#             ▼                        ▼                        ▼
#        orchestrator              each stage               uvicorn /
#        consumes                  engine                   FastAPI
#
# Fields in ``SHARED_FIELDS`` (e.g. ``model``, ``log_stats``) flow to BOTH
# orchestrator and engine by design.
#
# Invariants enforced by ``tests/engine/test_arg_utils_shared_fields.py``:
#
#   1. ``OrchestratorArgs`` ∩ ``OmniEngineArgs`` ⊆ ``SHARED_FIELDS``
#   2. Every CLI flag is classifiable into one of the three buckets
#   3. User-typed flags that match none of the above are logged as dropped
#
# Adding a new orchestrator-only flag → add a field to ``OrchestratorArgs``.
# Everything else is automatic.


@dataclass(frozen=True)
class OrchestratorArgs:
    """CLI flags consumed by the orchestrator.

    Contract: every field here is either
      (a) orchestrator-only (never needed by a stage engine), OR
      (b) orchestrator-read-then-redistributed (e.g. ``async_chunk`` is read
          from CLI, written to ``DeployConfig``, then propagated to every
          stage via ``merge_pipeline_deploy`` — not via direct kwargs
          forwarding).

    Fields that BOTH orchestrator and engine genuinely need (e.g. ``model``,
    ``log_stats``) should be listed in ``SHARED_FIELDS`` below.
    """

    # === Lifecycle ===
    stage_init_timeout: int = 300
    init_timeout: int = 600

    # === Cross-stage Communication ===
    batch_timeout: int = 10

    # === Cluster / Backend ===
    worker_backend: str = "multi_process"
    ray_address: str | None = None

    # === Config Files ===
    deploy_config: str | None = None
    stage_overrides: str | None = None  # raw JSON string; parsed downstream
    # Optional composable-parallel strategy.yaml; orchestrator reads it, overlays
    # derived sizing onto merged stages, then drops it before per-stage engine args.
    strategy_config: str | None = None

    # === Mode Switches (orchestrator reads, DeployConfig redistributes) ===
    async_chunk: bool | None = None

    # === Forced aligner (orchestrator injects a pooling stage; never a per-stage knob) ===
    forced_aligner: str | None = None
    forced_aligner_config: str | None = None
    forced_aligner_device: str | None = None

    # === Observability ===
    log_stats: bool = False
    enable_orch_monitor: bool = False

    # === Headless Mode (also forwarded to engine — see SHARED_FIELDS) ===
    stage_id: int | None = None

    # === Pre-built Objects ===
    parallel_config: Any = None

    # === Diffusion model config ===
    num_gpus: int | None = None
    model_class_name: str | None = None
    diffusion_load_format: str | None = None
    lora_path: list[str] | None = None
    lora_backend: str | None = None
    lora_scale: float | None = None
    diffusers_load_kwargs: str = "{}"
    diffusers_call_kwargs: str = "{}"
    ulysses_degree: int | None = None
    ulysses_mode: str = "strict"
    ring_degree: int | None = None
    allgather_degree: int | None = None
    diffusion_quantization_config: str | None = None
    use_hsdp: bool = False
    hsdp_shard_size: int = -1
    hsdp_replicate_size: int = 1
    diffusion_attention_backend: str | None = None
    fastvideo_vsa_topk: int | None = None
    diffusion_attention_config: str | None = None
    diffusion_compile_granularity: str | None = None
    diffusion_compile_dynamic: bool | None = None
    cache_backend: str = "none"
    cache_config: str | None = None
    enable_cache_dit_summary: bool = False
    step_execution: bool = False
    vae_use_slicing: bool = False
    vae_use_tiling: bool = False
    enable_multithread_weight_load: bool = True
    num_weight_load_threads: int = 4
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    enable_distributed_layerwise_offload: bool = False
    dlo_use_allgather: bool = True
    dlo_resident_layers: int = 0
    host_weight_runtime_mode: str = "disabled"
    host_weight_runtime_root: str | None = None
    dlo_host_registration_limit_gib: float = 0.0
    boundary_ratio: float | None = None
    flow_shift: float | None = None
    diffusion_kv_cache_dtype: str | None = None
    diffusion_kv_cache_skip_steps: str | None = None
    diffusion_kv_cache_skip_layers: str | None = None
    cfg_parallel_size: int = 1
    vae_patch_parallel_size: int = 1
    vae_parallel_mode: str = "tile"
    default_sampling_params: str | None = None
    max_generated_image_size: int | None = None
    tts_max_instructions_length: int | None = None
    enable_diffusion_pipeline_profiler: bool = False
    enable_ar_profiler: bool = False
    auxiliary_text_encoder: str | None = None
    log_file: str | None = None
    replica_id: int | None = None
    omni_replica_address: str | None = None

    # === Multi-stage guards ===
    # --tokenizer is captured by the orchestrator and forwarded to stages
    # only when the stage does not define tokenizer/tokenizer_subdir itself.
    # Users wanting a per-stage tokenizer should set it in the deploy YAML.
    tokenizer: str | None = None


# Fields that live in BOTH OrchestratorArgs and OmniEngineArgs by design.
# Changes to this set are a review red flag — revisit the contract.
SHARED_FIELDS: frozenset[str] = frozenset(
    {
        "model",  # orch: detect model_type; engine: load weights
        "stage_id",  # orch: route (headless); engine: identity
        "log_stats",  # both want the flag
        "async_chunk",  # orch: read from CLI, redistribute; engine: per-stage flag
        "tokenizer",  # orch: detect model type; engine: tokenization
    }
)


def orchestrator_field_names() -> frozenset[str]:
    """Return the names of every field on OrchestratorArgs."""
    return frozenset(f.name for f in fields(OrchestratorArgs))


def internal_blacklist_keys() -> frozenset[str]:
    """Return the set of CLI keys that must never be forwarded as per-stage
    engine overrides.

    Derived from ``OrchestratorArgs`` fields minus ``SHARED_FIELDS``, so
    adding a new orchestrator-owned flag is a one-line change to the
    dataclass — this function updates automatically.
    """
    return orchestrator_field_names() - SHARED_FIELDS
