import argparse
import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass, field, fields
from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.logger import init_logger

from vllm_omni.config import OmniModelConfig
from vllm_omni.engine.output_modality import OutputModality
from vllm_omni.platforms import current_omni_platform
from vllm_omni.plugins import load_omni_general_plugins

logger = init_logger(__name__)

# Maps model architecture names to their HuggingFace model_type values.
# Used when auto-injecting hf_overrides for models with missing config.json.
_ARCH_TO_MODEL_TYPE: dict[str, str] = {
    "CosyVoice3Model": "cosyvoice3",
    "OmniVoiceModel": "omnivoice",
    "VoxCPM2TalkerForConditionalGeneration": "voxcpm2",
    "VoxCPMForConditionalGeneration": "voxcpm",
}

# Maps model architecture names to tokenizer subfolder paths within HF repos.
_TOKENIZER_SUBFOLDER_MAP: dict[str, str] = {
    "CosyVoice3Model": "CosyVoice-BlankEN",
}


def _register_omni_hf_configs() -> None:
    try:
        from transformers import AutoConfig

        from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
        from vllm_omni.model_executor.models.omnivoice.config import OmniVoiceConfig
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.voxtral_tts.configuration_voxtral_tts import (
            VoxtralTTSConfig,
        )
        from vllm_omni.transformers_utils.configs.voxcpm import VoxCPMConfig
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
        ("qwen3_tts", Qwen3TTSConfig),
        ("cosyvoice3", CosyVoice3Config),
        ("omnivoice", OmniVoiceConfig),
        ("voxtral_tts", VoxtralTTSConfig),
        ("voxcpm", VoxCPMConfig),
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

    supported_archs = ModelRegistry.get_supported_archs()
    for arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items():
        if arch not in supported_archs:
            ModelRegistry.register_model(arch, f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}:{cls_name}")


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
        worker_type: Model Type, e.g., "ar" or "generation"
        task_type: Default task type for TTS models (CustomVoice, VoiceDesign, or Base).
            If not specified, will be inferred from model path.
        omni_master_address: TCP address that the OmniMasterServer (running
            inside AsyncOmniEngine) listens on for engine core registrations.
            Required when single-stage mode is active.
        omni_master_port: TCP port for the OmniMasterServer registration
            socket.  Required when single-stage mode is active.
        stage_configs_path: Optional path to a JSON/YAML file containing
            stage configurations for the multi-stage pipeline. If None,
            stage configs are resolved from the model's default configuration.
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
    stage_connector_spec: dict[str, Any] = field(default_factory=dict)
    subtalker_sampling_params: dict[str, Any] | None = None
    async_chunk: bool = False
    omni_kv_config: dict | None = None
    quantization_config: Any | None = None
    worker_type: str | None = None
    task_type: str | None = None
    worker_cls: str = None
    enable_sleep_mode: bool = False
    omni: bool = False

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
    stage_configs_path: str | None = None
    output_modalities: list[str] | None = None
    log_stats: bool = False
    custom_pipeline_args: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.worker_cls is None:
            if self.worker_type == "ar":
                self.worker_cls = current_omni_platform.get_omni_ar_worker_cls()
            elif self.worker_type == "generation":
                self.worker_cls = current_omni_platform.get_omni_generation_worker_cls()
        load_omni_general_plugins()
        super().__post_init__()

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "OmniEngineArgs":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)})
        return engine_args

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
            return  # can't load config, let vLLM handle the error

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

        # If model_arch is specified, inject it into hf_overrides so vLLM can
        # resolve the architecture even when config.json lacks 'architectures'.
        # Also inject model_type so AutoConfig can resolve the correct config
        # class for models with empty or missing config.json (e.g. CosyVoice3).
        if self.model_arch:
            if self.hf_overrides is None:
                self.hf_overrides = {}
            if isinstance(self.hf_overrides, dict):
                self.hf_overrides.setdefault("architectures", [self.model_arch])
                if "model_type" not in self.hf_overrides:
                    model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                    if model_type is not None:
                        self.hf_overrides.setdefault("model_type", model_type)

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

        # Auto-detect tokenizer for models that store it in a subdirectory
        # rather than the root (e.g. CosyVoice3 uses CosyVoice-BlankEN/).
        if not self.tokenizer and self.model:
            model_path = self.model
            if os.path.isdir(model_path) and not os.path.isfile(os.path.join(model_path, "tokenizer_config.json")):
                for subfolder in sorted(os.listdir(model_path)):
                    candidate = os.path.join(model_path, subfolder)
                    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "tokenizer_config.json")):
                        self.tokenizer = candidate
                        logger.info("Auto-detected tokenizer at %s", candidate)
                        break
            elif not os.path.isdir(model_path):
                subfolder = _TOKENIZER_SUBFOLDER_MAP.get(self.model_arch)
                if subfolder:
                    # Download just the tokenizer files from the subfolder
                    try:
                        from huggingface_hub import snapshot_download

                        local_dir = snapshot_download(
                            model_path,
                            allow_patterns=[
                                f"{subfolder}/tokenizer*",
                                f"{subfolder}/special_tokens*",
                                f"{subfolder}/vocab*",
                                f"{subfolder}/merges*",
                                f"{subfolder}/added_tokens*",
                            ],
                        )
                        candidate = os.path.join(local_dir, subfolder)
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
            model_stage=self.model_stage,
            model_arch=self.model_arch,
            worker_type=self.worker_type,
            engine_output_type=self.engine_output_type,
            hf_config_name=self.hf_config_name,
            custom_process_next_stage_input_func=self.custom_process_next_stage_input_func,
            stage_connector_config=stage_connector_config,
            subtalker_sampling_params=self.subtalker_sampling_params,
            omni_kv_config=self.omni_kv_config,
            task_type=self.task_type,
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
# Invariants enforced by ``tests/test_arg_utils.py``:
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
    ``log_stats``) should be listed in ``SHARED_FIELDS`` below; ``split_kwargs``
    will copy them to both buckets.
    """

    # === Lifecycle ===
    stage_init_timeout: int = 300
    init_timeout: int = 600

    # === Cross-stage Communication ===
    shm_threshold_bytes: int = 65536
    batch_timeout: int = 10

    # === Cluster / Backend ===
    worker_backend: str = "multi_process"
    ray_address: str | None = None

    # === Config Files ===
    stage_configs_path: str | None = None
    deploy_config: str | None = None
    stage_overrides: str | None = None  # raw JSON string; parsed downstream

    # === Mode Switches (orchestrator reads, DeployConfig redistributes) ===
    async_chunk: bool | None = None

    # === Observability ===
    log_stats: bool = False

    # === Headless Mode (also forwarded to engine — see SHARED_FIELDS) ===
    stage_id: int | None = None

    # === Pre-built Objects ===
    parallel_config: Any = None

    # === Multi-stage guards ===
    # --tokenizer is captured here so it does not propagate to every stage
    # uniformly (different stages often need different tokenizers, e.g.
    # qwen3_omni thinker vs talker). Users wanting a per-stage tokenizer
    # should set it in the deploy YAML.
    tokenizer: str | None = None


# Fields that live in BOTH OrchestratorArgs and OmniEngineArgs by design.
# Changes to this set are a review red flag — revisit the contract.
SHARED_FIELDS: frozenset[str] = frozenset(
    {
        "model",  # orch: detect model_type; engine: load weights
        "stage_id",  # orch: route (headless); engine: identity
        "log_stats",  # both want the flag
        "stage_configs_path",  # orch: load legacy YAML; engine: may reference for validation
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


def split_kwargs(
    kwargs: dict[str, Any],
    *,
    engine_cls: type | None = None,
    user_typed: set[str] | None = None,
    strict: bool = False,
) -> tuple[OrchestratorArgs, dict[str, Any]]:
    """Partition CLI kwargs into (orchestrator, engine) buckets.

    Args:
        kwargs: Raw dict, typically ``vars(args)``.
        engine_cls: Engine dataclass used to whitelist-filter the engine
            bucket. Defaults to ``OmniEngineArgs``. Pass a custom class
            for testing.
        user_typed: Keys the user actually typed on the command line. Used
            to warn when a user-typed flag is unclassifiable.
        strict: If True, raise ``ValueError`` on ambiguous (double-classified
            but not in ``SHARED_FIELDS``) fields. Default False to keep the
            rollout non-breaking; flip to True in tests and CI.

    Returns:
        ``(orchestrator_args, engine_kwargs)``. ``engine_kwargs`` has already
        been whitelist-filtered against ``engine_cls`` — safe to pass directly
        to ``engine_cls(**engine_kwargs)``.
    """
    if engine_cls is None:
        engine_cls = OmniEngineArgs

    orch_fields = orchestrator_field_names()
    engine_fields = {f.name for f in fields(engine_cls)}

    orch_kwargs: dict[str, Any] = {}
    engine_candidate: dict[str, Any] = {}
    shared_values: dict[str, Any] = {}
    unclassified: dict[str, Any] = {}

    for key, value in kwargs.items():
        in_orch = key in orch_fields
        in_engine = key in engine_fields
        is_shared = key in SHARED_FIELDS

        if is_shared:
            shared_values[key] = value
        elif in_orch and in_engine:
            # Declared in both but not marked shared → ambiguous.
            msg = (
                f"Field {key!r} is defined on both OrchestratorArgs and "
                f"{engine_cls.__name__} but is not in SHARED_FIELDS. "
                f"This causes double-routing. Either remove the duplicate or "
                f"add {key!r} to SHARED_FIELDS if the sharing is intentional."
            )
            if strict:
                raise ValueError(msg)
            logger.error(msg)
            # Default: treat as orchestrator-only to preserve existing behavior.
            orch_kwargs[key] = value
        elif in_orch:
            orch_kwargs[key] = value
        elif in_engine:
            engine_candidate[key] = value
        else:
            unclassified[key] = value

    # Warn on user-typed but unclassifiable flags so we don't silently drop
    # something the user cared about (fixes the class of bug that spawned #873).
    if unclassified and user_typed:
        user_typed_unknown = sorted(k for k in unclassified if k in user_typed)
        if user_typed_unknown:
            logger.warning(
                "CLI flags not consumed by vllm-omni and dropped before "
                "per-stage engine construction: %s. If these are vllm "
                "frontend/uvicorn flags (host, port, ssl_*, api_key, …) this "
                "is expected; otherwise check your spelling.",
                user_typed_unknown,
            )

    # Engine bucket: shared + engine-only. We do NOT pass through unclassified
    # fields — that's exactly the server/uvicorn noise we want to shed.
    engine_kwargs = {**shared_values, **engine_candidate}

    # Construct the orchestrator dataclass. Shared fields that OrchestratorArgs
    # also declares get copied into its constructor.
    orch_init: dict[str, Any] = dict(orch_kwargs)
    for key, value in shared_values.items():
        if key in orch_fields:
            orch_init[key] = value
    orch_args = OrchestratorArgs(**orch_init)

    return orch_args, engine_kwargs


def derive_server_dests_from_vllm_parser() -> frozenset[str]:
    """Derive the set of argparse dests that belong to vllm's frontend/server.

    Returns every dest registered by ``make_arg_parser`` that is NOT a field
    of ``OmniEngineArgs`` and NOT a field of ``OrchestratorArgs``. Useful for
    CI tests to assert all CLI flags are classifiable without maintaining
    a hardcoded server list.

    Returns empty frozenset if vllm's parser cannot be built (e.g. in a
    minimal test environment).
    """
    try:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        logger.debug("Cannot import vllm parser — server-dest derivation skipped")
        return frozenset()

    try:
        parser = make_arg_parser(FlexibleArgumentParser())
        all_dests = {a.dest for a in parser._actions if a.dest and a.dest != "help"}
    except Exception as exc:
        logger.debug("Failed to build vllm parser: %s", exc)
        return frozenset()

    engine_fields = {f.name for f in fields(OmniEngineArgs)}
    orch_fields = orchestrator_field_names()

    return frozenset(all_dests - engine_fields - orch_fields - SHARED_FIELDS)


def orchestrator_args_from_argparse(args: Any) -> OrchestratorArgs:
    """Build an ``OrchestratorArgs`` from an ``argparse.Namespace``.

    Only copies attributes that exist on the namespace — missing fields fall
    back to the dataclass default. Useful when the full parser is already
    built and ``vars(args)`` would include noise.
    """
    kwargs: dict[str, Any] = {}
    for f in fields(OrchestratorArgs):
        if hasattr(args, f.name):
            value = getattr(args, f.name)
            if value is not None or f.default is None:
                kwargs[f.name] = value
    return OrchestratorArgs(**kwargs)
