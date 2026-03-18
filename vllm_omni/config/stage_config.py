# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stage Configuration System for vLLM-Omni.

Pipeline structure (stages, types, data-flow) is defined in per-model YAML
files and is set by model developers at integration time.
Runtime parameters (gpu_memory_utilization, tp_size, etc.) come from CLI.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from vllm_omni.config.yaml_util import create_config, load_yaml_config, to_dict

# Pipeline YAMLs live alongside model code in model_executor/models/<model>/
_MODELS_DIR = Path(__file__).resolve().parent.parent / "model_executor" / "models"


def get_pipeline_path(model_dir: str, filename: str) -> Path:
    """Return the full path to a pipeline YAML file.

    Args:
        model_dir: Model subdirectory name (e.g., "qwen3_omni").
        filename: Name of the YAML file (e.g., "pipeline.yaml").

    Returns:
        Absolute path to the file.
    """
    return _MODELS_DIR / model_dir / filename


logger = init_logger(__name__)


class StageType(str, Enum):
    """Type of processing stage in the Omni pipeline."""

    LLM = "llm"
    DIFFUSION = "diffusion"


@dataclass
class StageConfig:
    """Per-stage configuration from pipeline YAML.

    Topology fields (stage_id, input_sources, etc.) define the DAG.
    Engine and runtime defaults come from the YAML; CLI overrides take
    precedence via ``runtime_overrides``.
    """

    # Identity
    stage_id: int
    model_stage: str

    # Stage type
    stage_type: StageType = StageType.LLM

    input_sources: list[int] = field(default_factory=list)
    custom_process_input_func: str | None = None
    final_output: bool = False
    final_output_type: str | None = None  # "text", "audio", "image"
    worker_type: str | None = None  # "ar" or "generation"
    scheduler_cls: str | None = None
    hf_config_name: str | None = None
    is_comprehension: bool = False

    # Per-stage engine args from pipeline YAML (defaults)
    yaml_engine_args: dict[str, Any] = field(default_factory=dict)
    # Per-stage runtime config from pipeline YAML (devices, max_batch_size)
    yaml_runtime: dict[str, Any] = field(default_factory=dict)
    # Pass-through fields from pipeline YAML (default_sampling_params,
    # output_connectors, input_connectors, tts_args, etc.)
    yaml_extras: dict[str, Any] = field(default_factory=dict)

    # Runtime overrides (populated from CLI, not from pipeline YAML)
    runtime_overrides: dict[str, Any] = field(default_factory=dict)

    def to_omegaconf(self) -> Any:
        """Convert to OmegaConf for backward compatibility with OmniStage.

        Returns:
            OmegaConf DictConfig with stage configuration in legacy format.
        """
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
        runtime.setdefault("max_batch_size", self.runtime_overrides.get("max_batch_size", 1))
        if "devices" in self.runtime_overrides:
            runtime["devices"] = self.runtime_overrides["devices"]

        # Inject max_num_seqs from max_batch_size (legacy compat)
        max_batch_size = int(runtime.get("max_batch_size", 1) or 1)
        engine_args.setdefault("max_num_seqs", max_batch_size)

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
    """Complete pipeline definition for a multi-stage model.

    Defined by model developers, bundled with the model, not user-editable.
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
    """

    # Mapping of model types to directories under model_executor/models/.
    PIPELINE_MODELS: dict[str, str] = {
        "qwen3_omni_moe": "qwen3_omni",
        "qwen2_5_omni": "qwen2_5_omni",
        "bagel": "bagel",
        "qwen3_tts": "qwen3_tts",
        "mimo_audio": "mimo_audio",
        "glm-image": "glm_image",
        "cosyvoice3": "cosyvoice3",
        "mammothmoda2": "mammoth_moda2",
    }

    # Fallback: map HF architecture class names to pipeline dirs.
    # Used when model_type collides with another model (e.g. MiMo Audio
    # reports model_type="qwen2" which matches plain Qwen2, not our pipeline).
    _ARCHITECTURE_MODELS: dict[str, str] = {
        "MiMoAudioForConditionalGeneration": "mimo_audio",
        "HunyuanImage3ForCausalMM": "hunyuan_image3",
    }

    @classmethod
    def create_from_model(
        cls,
        model: str,
        cli_overrides: dict[str, Any] | None = None,
    ) -> list[StageConfig] | None:
        """Load pipeline YAML, merge with CLI overrides.

        Args:
            model: Model name or path.
            cli_overrides: CLI overrides from VllmConfig/OmniDiffusionConfig.

        Returns:
            List of StageConfig objects with CLI overrides applied,
            or None if no pipeline definition was found for this model.
        """
        if cli_overrides is None:
            cli_overrides = {}

        trust_remote_code = cli_overrides.get("trust_remote_code", True)
        pipeline = cls._load_pipeline(model, trust_remote_code=trust_remote_code)

        if pipeline is None:
            return None

        errors = pipeline.validate_pipeline()
        if errors:
            logger.warning(f"Pipeline validation warnings for {model}: {errors}")

        # Inject pipeline-wide async_chunk into ALL stages' engine_args.
        # The legacy loader (load_stage_configs_from_yaml) sets async_chunk
        # on every stage so that build_engine_args_dict() can inject the
        # stage_connector_spec.  AsyncOmniEngine.__init__ also reads it
        # from stage_configs[0].engine_args.async_chunk.
        if pipeline.async_chunk:
            for stage in pipeline.stages:
                stage.yaml_engine_args.setdefault("async_chunk", True)

        # Apply CLI overrides
        result: list[StageConfig] = []
        for stage in pipeline.stages:
            # Merge global CLI overrides
            stage.runtime_overrides = cls._merge_cli_overrides(stage, cli_overrides)
            result.append(stage)

        return result

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

        # Serialize parallel_config as dict for OmegaConf compatibility
        if "parallel_config" in kwargs:
            engine_args["parallel_config"] = asdict(kwargs["parallel_config"])

        engine_args.setdefault("cache_backend", "none")
        engine_args["model_stage"] = "diffusion"

        # Convert dtype to string for OmegaConf
        if "dtype" in engine_args:
            engine_args["dtype"] = str(engine_args["dtype"])

        config_dict: dict[str, Any] = {
            "stage_id": 0,
            "stage_type": StageType.DIFFUSION.value,
            "runtime": {
                "process": True,
                "devices": devices,
                "max_batch_size": 1,
            },
            "engine_args": create_config(engine_args),
            "final_output": True,
            "final_output_type": "image",
        }

        return [config_dict]

    @classmethod
    def _load_pipeline(cls, model: str, trust_remote_code: bool = True) -> ModelPipeline | None:
        """Load pipeline YAML for the model.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            ModelPipeline if found, None otherwise.
        """
        model_type, hf_config = cls._auto_detect_model_type(model, trust_remote_code=trust_remote_code)
        if model_type is None:
            return None

        pipeline_dir = cls.PIPELINE_MODELS.get(model_type)

        # Fallback: check HF architectures when model_type doesn't match
        if pipeline_dir is None and hf_config is not None:
            for arch in getattr(hf_config, "architectures", []) or []:
                pipeline_dir = cls._ARCHITECTURE_MODELS.get(arch)
                if pipeline_dir is not None:
                    model_type = pipeline_dir
                    break

        if pipeline_dir is None:
            logger.debug(f"No pipeline mapping for model_type: {model_type}")
            return None

        pipeline_path = get_pipeline_path(pipeline_dir, "pipeline.yaml")

        if not pipeline_path.exists():
            logger.debug(f"Pipeline file not found: {pipeline_path}")
            return None

        return cls._parse_pipeline_yaml(pipeline_path, model_type)

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
        except Exception:
            pass

        # Fallback: read config.json directly for custom model types that
        # are not registered with transformers (e.g. qwen3_tts).
        try:
            from vllm.transformers_utils.config import get_hf_file_to_dict

            config_dict = get_hf_file_to_dict("config.json", model, revision=None)
            if config_dict and "model_type" in config_dict:
                return config_dict["model_type"], None
        except Exception as e:
            logger.debug(f"Failed to auto-detect model type for {model}: {e}")

        return None, None

    # Keys that should never be forwarded as engine overrides (internal /
    # orchestrator-only knobs, complex objects, etc.).
    _INTERNAL_KEYS: set[str] = {
        "model",
        "stage_configs_path",
        "stage_id",
        "stage_init_timeout",
        "init_timeout",
        "shm_threshold_bytes",
        "worker_backend",
        "ray_address",
        "batch_timeout",
        "log_stats",
        "tokenizer",
        "parallel_config",
    }

    @classmethod
    def _merge_cli_overrides(
        cls,
        stage: StageConfig,
        cli_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge CLI overrides into stage runtime config.

        All CLI arguments registered by engine config classes (e.g.
        EngineArgs / OmniDiffusionConfig) are accepted as overrides
        unless they appear in ``_INTERNAL_KEYS``.

        Handles:
        - Global overrides (apply to all stages)
        - Per-stage overrides (--stage-N-* format, take precedence)

        Args:
            stage: The stage to merge overrides into.
            cli_overrides: CLI arguments from VllmConfig/OmniDiffusionConfig.

        Returns:
            Dict of runtime overrides for this stage.
        """
        result: dict[str, Any] = {}

        # Apply global overrides – any key not in the internal blocklist
        # is forwarded so that engine-registered params work out of the box.
        for key, value in cli_overrides.items():
            if key in cls._INTERNAL_KEYS:
                continue
            if re.match(r"stage_\d+_", key):
                # Per-stage keys handled below
                continue
            if value is not None:
                result[key] = value

        # Apply per-stage overrides (--stage-N-* format, take precedence)
        stage_prefix = f"stage_{stage.stage_id}_"
        for key, value in cli_overrides.items():
            if key.startswith(stage_prefix) and value is not None:
                param_name = key[len(stage_prefix) :]
                if param_name in cls._INTERNAL_KEYS:
                    continue
                result[param_name] = value

        return result
