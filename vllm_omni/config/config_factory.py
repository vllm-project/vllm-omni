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
        cli_overrides: dict[str, Any] | None = None,
        deploy_config_path: str | None = None,
        **deprecated_kwargs: Any,
    ) -> list[StageConfig] | None:
        """Load pipeline + deploy config, merge with CLI overrides.

        Checks OMNI_PIPELINES first, since supported models should be explicitly
        registered. If a model is not registered in OMNI_PIPELINES, tries to fall
        back to using the Transformers config & finding pipelines that have overlapping
        supported architectures.
        """
        if cli_overrides is None:
            cli_overrides = {}

        trust_remote_code = cli_overrides.get("trust_remote_code", True)
        if trust_remote_code is None:
            trust_remote_code = False

        # --- New path: check pipeline registry by model_type first ---
        model_type, hf_config = cls._auto_detect_model_type(model, trust_remote_code=trust_remote_code)
        if model_type == "vla":
            if _looks_like_dreamzero(model):
                model_type = "dreamzero"

        if model_type and model_type in OMNI_PIPELINES:
            pipeline_cfg = cls.resolve_pipeline_config(model_type, hf_config)
            if pipeline_cfg is not None:
                return cls._create_from_registry(
                    model_type,
                    pipeline_cfg,
                    cli_overrides,
                    deploy_config_path,
                )

        # --- HF architecture fallback: some models report a generic
        # model_type that collides with another model. Match by the
        # hf_architectures declared on each registered PipelineConfig.
        if hf_config is not None:
            logger.warning("Inferred model type %s is not registered to an Omni pipeline", model_type)
            hf_archs = set(getattr(hf_config, "architectures", []) or [])
            if hf_archs:
                for registered in OMNI_PIPELINES.values():
                    pipeline_cfg = registered if isinstance(registered, PipelineConfig) else registered(hf_config)
                    # Resolvers that get configs of the incorrect type should return None
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
                        return cls._create_from_registry(
                            pipeline_cfg.model_type,
                            pipeline_cfg,
                            cli_overrides,
                            deploy_config_path,
                        )

        # --- Explicit deploy-config pipeline ---
        # When auto-detection above resolves nothing (generic/missing HF model_type
        # and no matching architecture), honor an explicit pipeline key in the deploy config
        if deploy_config_path is not None:
            deploy_path = Path(deploy_config_path)
            if deploy_path.exists():
                deploy_cfg = load_deploy_config(deploy_path)
                if deploy_cfg.pipeline:
                    pipeline_cfg = cls.resolve_pipeline_config(deploy_cfg.pipeline, hf_config)
                    if pipeline_cfg is not None:
                        return cls._create_from_registry(
                            pipeline_cfg.model_type,
                            pipeline_cfg,
                            cli_overrides,
                            deploy_config_path,
                        )

        # Not in the pipeline registry — let the caller fall back to the
        # legacy ``stage_configs/*.yaml`` path (resolve_model_config_path).
        return None

    @classmethod
    def _create_from_registry(
        cls,
        model_type: str,
        pipeline_cfg: PipelineConfig,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None = None,
        **deprecated_kwargs: Any,
    ) -> list[StageConfig]:
        """Create StageConfigs from pipeline registry + deploy YAML.

        Precedence: caller-typed (non-None) value > deploy YAML >
        StageDeployConfig dataclass default.
        """
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
    def resolve_pipeline_config(model_type: str, hf_config: PretrainedConfig | None = None) -> PipelineConfig | None:
        """Given a model type, resolve to the pipeline to be used. If the pipeline
        maps to a callable we resolve based on the HF config."""
        if model_type not in OMNI_PIPELINES:
            return None
        obj = OMNI_PIPELINES[model_type]
        return obj(hf_config) if callable(obj) else obj
