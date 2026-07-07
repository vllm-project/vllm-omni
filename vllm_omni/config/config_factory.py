# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config factories for vllm-omni, e.g., StageConfigFactory."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    _DEPLOY_DIR,
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageType,
    build_stage_runtime_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.config.yaml_util import create_config
from vllm_omni.diffusion.utils.hf_utils import _looks_like_dreamzero

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class PipelineResolution:
    """Result of resolving a model/deploy pair against ``OMNI_PIPELINES``."""

    registry_key: str
    pipeline_config: PipelineConfig
    hf_config: PretrainedConfig | None


class StageConfigFactory:
    """Factory that loads pipeline YAML and merges CLI overrides.

    Handles both single-stage and multi-stage models.

    Pipelines are declared in ``vllm_omni/config/pipeline_registry.py`` and
    where keys in OMNI_PIPELINES map to either a PipelineConfig, or a callable
    which accepts a Transformers config as an arg & resolves to a PipelineConfig.

    NOTE: Models with generic HF ``model_type`` collisions (e.g. MiMo Audio
    reports ``qwen2``) should declare ``hf_architectures=(...)`` on their
    ``PipelineConfig`` so the factory can disambiguate via ``hf_config.architectures``.
    """

    @classmethod
    def create_from_model(
        cls,
        model: str,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None,
    ) -> VllmOmniConfig | None:
        """Load pipeline + deploy config, merge with CLI overrides.

        Checks OMNI_PIPELINES first, since supported models should be explicitly
        registered. If a model is not registered in OMNI_PIPELINES, tries to fall
        back to using the Transformers config & finding pipelines that have overlapping
        supported architectures.
        """
        resolution = cls.resolve_pipeline_from_model(
            model=model,
            trust_remote_code=cls._get_trust_remote_code(cli_overrides),
            deploy_config_path=deploy_config_path,
        )
        if resolution is None:
            return None

        # Preserve the explicit create_from_model() argument for structured
        # diffusion stages. It intentionally overrides any stale
        # cli_overrides["model"], since auto-detection above uses this model.
        registry_cli_overrides = {**cli_overrides, "model": model}
        return VllmOmniConfig.from_registry(
            resolution.registry_key,
            hf_config=resolution.hf_config,
            deploy_config_path=deploy_config_path,
            cli_overrides=registry_cli_overrides,
        )

    @classmethod
    def create_legacy_stage_configs_from_model(
        cls,
        model: str,
        cli_overrides: dict[str, Any] | None = None,
        deploy_config_path: str | None = None,
        **deprecated_kwargs: Any,
    ) -> list[StageConfig] | None:
        """Resolve current runtime stage configs without consuming VllmOmniConfig.

        ``VllmOmniConfig`` is the structured config object for registry-backed
        models, but the engine startup/runtime path still reads the legacy
        ``StageConfig``/OmegaConf shape (``stage.engine_args``,
        ``stage.runtime``, etc.). Keep that runtime contract isolated here so
        follow-up RFC #4021 changes can replace consumers incrementally with direct
        ``VllmOmniConfig`` access instead of relying on a fake typed-to-legacy
        projection.
        """
        if cli_overrides is None:
            cli_overrides = {}

        resolution = cls.resolve_pipeline_from_model(
            model=model,
            trust_remote_code=cls._get_trust_remote_code(cli_overrides),
            deploy_config_path=deploy_config_path,
        )
        if resolution is None:
            return None

        return cls._create_legacy_from_registry(
            resolution.registry_key,
            resolution.pipeline_config,
            cli_overrides,
            deploy_config_path,
        )

    @classmethod
    def resolve_pipeline_from_model(
        cls,
        model: str,
        trust_remote_code: bool,
        deploy_config_path: str | None,
    ) -> PipelineResolution | None:
        """Resolve a model/deploy pair once for structured and legacy consumers."""
        model_type, hf_config = cls._detect_registry_model_type(model, trust_remote_code=trust_remote_code)

        explicit_pipeline = cls._get_deploy_pipeline(deploy_config_path)
        pipeline_cfg = cls.resolve_pipeline_config(explicit_pipeline, hf_config)
        if pipeline_cfg is not None:
            return PipelineResolution(explicit_pipeline, pipeline_cfg, hf_config)
        logger.warning(
            "Deploy config %s requested pipeline %r which is not in OMNI_PIPELINES; falling back to auto-detection.",
            deploy_config_path,
            explicit_pipeline,
        )

        pipeline_cfg = cls.resolve_pipeline_config(model_type, hf_config)
        if pipeline_cfg is not None:
            return PipelineResolution(model_type, pipeline_cfg, hf_config)

        logger.warning("Inferred model type %s is not registered to an Omni pipeline", model_type)
        if hf_config is not None:
            hf_archs = set(getattr(hf_config, "architectures", []) or [])
            if hf_archs:
                for registered_key, registered in OMNI_PIPELINES.items():
                    pipeline_cfg = registered if isinstance(registered, PipelineConfig) else registered(hf_config)
                    if pipeline_cfg is None:
                        continue
                    predicate = pipeline_cfg.hf_config_predicate
                    if predicate is not None:
                        try:
                            if not predicate(hf_config):
                                logger.debug(
                                    "Pipeline %r matched on architectures %s but its "
                                    "hf_config_predicate rejected the loaded config; "
                                    "continuing fallback search.",
                                    pipeline_cfg.model_type,
                                    sorted(hf_archs.intersection(pipeline_cfg.hf_architectures)),
                                )
                                continue
                        except Exception:
                            logger.exception(
                                "Pipeline %r hf_config_predicate raised; skipping.",
                                pipeline_cfg.model_type,
                            )
                            continue

                    if isinstance(pipeline_cfg, PipelineConfig) and hf_archs.intersection(
                        pipeline_cfg.hf_architectures
                    ):
                        return PipelineResolution(registered_key, pipeline_cfg, hf_config)
        return None

    @classmethod
    def _detect_registry_model_type(cls, model: str, trust_remote_code: bool = True) -> tuple[str | None, Any]:
        """Detect the vllm-omni registry key for a model."""
        model_type, hf_config = cls._auto_detect_model_type(model, trust_remote_code=trust_remote_code)
        return cls._normalize_model_type_for_registry(model, model_type), hf_config

    @staticmethod
    def _normalize_model_type_for_registry(model: str, model_type: str | None) -> str | None:
        """Map generic HF model_type values to vllm-omni registry keys."""
        if model_type == "vla" and _looks_like_dreamzero(model):
            return "dreamzero"
        return model_type

    @staticmethod
    def _get_trust_remote_code(cli_overrides: dict[str, Any]) -> bool:
        trust_remote_code = cli_overrides.get("trust_remote_code", True)
        return False if trust_remote_code is None else bool(trust_remote_code)

    @classmethod
    def _get_deploy_pipeline(cls, deploy_config_path: str | None) -> str | None:
        if not deploy_config_path:
            return None
        deploy_path = Path(deploy_config_path)
        if not deploy_path.exists():
            return None
        try:
            return load_deploy_config(deploy_path).pipeline
        except Exception:
            logger.exception("Failed to read 'pipeline' key from deploy config %s", deploy_config_path)
            return None

    @classmethod
    def _create_legacy_from_registry(
        cls,
        model_type: str,
        pipeline_cfg: PipelineConfig,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None = None,
        **deprecated_kwargs: Any,
    ) -> list[StageConfig]:
        """Create current runtime StageConfigs from registry + deploy YAML.

        This is intentionally separate from ``create_from_model``:
        ``create_from_model`` returns ``VllmOmniConfig`` for the new config API,
        while current engine consumers still need legacy stage configs until
        future RFC #4021 changes migrate the runtime chain.

        Once the engine startup path consumes ``VllmOmniConfig`` directly, this
        transitional helper should disappear rather than become a second
        long-term registry implementation.
        """
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
            # Fallback to using the deploy config pipeline class if it's a mismatch
            if deploy_cfg.pipeline and deploy_cfg.pipeline != model_type:
                resolved = cls.resolve_pipeline_config(deploy_cfg.pipeline)
                if resolved is None:
                    raise KeyError(
                        f"Pipeline {deploy_cfg.pipeline!r} from {deploy_path.name!r} "
                        f"not found in OMNI_PIPELINES. Available: "
                        f"{sorted(OMNI_PIPELINES.keys())}"
                    )
                pipeline_cfg = resolved

        cli_async_chunk = cli_overrides.get("async_chunk")
        if cli_async_chunk is not None:
            deploy_cfg.async_chunk = bool(cli_async_chunk)

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
    def _auto_detect_model_type(cls, model: str, trust_remote_code: bool = True) -> tuple[str | None, Any]:
        """Auto-detect model_type from model directory.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            Tuple of (model_type, hf_config). Both may be None on failure.
        """
        hf_config = None

        try:
            hf_config = get_config(model, trust_remote_code=trust_remote_code)
            return hf_config.model_type, hf_config
        except Exception as e:
            logger.debug(f"`get_config` failed for {e}; Falling back to raw config.json path")

        # Fallback: read config.json directly for custom model types that
        # are not registered with transformers (e.g. qwen3_tts).
        try:
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
            model_index = get_hf_file_to_dict("model_index.json", model, revision=None)
            if model_index and "_class_name" in model_index:
                class_name = model_index["_class_name"]
                for obj in OMNI_PIPELINES.values():
                    # If we have a resolver, call it with the optional hf_config
                    # to get the default pipeline config for this key
                    pipeline_cfg = obj(hf_config) if callable(obj) else obj
                    if pipeline_cfg is not None and pipeline_cfg.diffusers_class_name == class_name:
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
        for registered_key in OMNI_PIPELINES.keys():
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

    @staticmethod
    def resolve_pipeline_config(
        model_type: str | None,
        hf_config: PretrainedConfig | None = None,
    ) -> PipelineConfig | None:
        """Given a model type, resolve to the pipeline to be used. If the pipeline
        maps to a callable we resolve based on the HF config."""
        if model_type not in OMNI_PIPELINES:
            return None
        obj = OMNI_PIPELINES[model_type]
        return obj(hf_config) if callable(obj) else obj
