# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config factories for vllm-omni, e.g., StageConfigFactory."""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from vllm_omni.config.endpoint_policy import EndpointRestriction
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
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
    def get_pipeline_endpoint_restrictions(
        cls,
        model: str,
        trust_remote_code: bool,
        deploy_config_path: str | None,
    ) -> tuple[EndpointRestriction, ...]:
        """Given a model string, determine the corresponding endpoint restrictions.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.
            deploy_config_path: Optional path to the deploy config for the pipeline.

        Returns:
            A tuple of model specific endpoint restrictions.
        """
        pipeline_cfg = StageConfigFactory.get_pipeline_config(
            model=model,
            trust_remote_code=trust_remote_code,
            deploy_config_path=deploy_config_path,
        )
        return pipeline_cfg.endpoint_restrictions if pipeline_cfg else ()

    @classmethod
    @functools.cache
    def get_hf_config(cls, model: str, trust_remote_code: bool) -> PretrainedConfig | None:
        """Fetch the HF config (if it exists) from the model directory.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            the model's config or None.
        """
        hf_config = None
        try:
            return get_config(model, trust_remote_code=trust_remote_code)
        except Exception as e:
            logger.debug(f"`get_config` failed with exception {e}; inferred HF config is None")
        return hf_config

    @classmethod
    @functools.cache
    def try_infer_model_type(cls, model: str, trust_remote_code: bool) -> str | None:
        """Auto-detect model_type from model directory and apply any model
        specific patches to get the correct model_type str. If we are unable
        to infer it from the model directory, we fall back to the PipelineConfig.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            model_type as a string; may be None on failure.
        """
        model_type = cls._try_infer_model_type(
            model=model,
            trust_remote_code=trust_remote_code,
        )
        if model_type == "vla":
            if _looks_like_dreamzero(model):
                model_type = "dreamzero"
        return model_type

    @classmethod
    def _try_infer_model_type(cls, model: str, trust_remote_code: bool) -> str | None:
        """Auto-detect model_type from model directory.

        Args:
            model: Model name or path.
            trust_remote_code: Whether to trust remote code for HF config loading.

        Returns:
            model_type as a string; may be None on failure.
        """
        hf_config = cls.get_hf_config(
            model=model,
            trust_remote_code=trust_remote_code,
        )
        if hf_config is not None:
            return hf_config.model_type

        # Fallback: read config.json directly for custom model types that
        # are not registered with transformers (e.g. qwen3_tts).
        try:
            config_dict = get_hf_file_to_dict("config.json", model, revision=None)
            if config_dict:
                if "model_type" in config_dict:
                    return config_dict["model_type"]
                # VoxCPM2-style configs use singular ``architecture`` rather
                # than HF's standard ``model_type`` / ``architectures``. Accept
                # it as a fallback so the pipeline registry can still match.
                if "architecture" in config_dict and isinstance(config_dict["architecture"], str):
                    return config_dict["architecture"]
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
                        return pipeline_cfg.model_type
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
            return best

        return None

    @classmethod
    @functools.cache
    def get_pipeline_config(
        cls,
        model: str,
        trust_remote_code: bool,
        deploy_config_path: str | None = None,
    ) -> PipelineConfig | None:
        """Resolve the PipelineConfig for a model path/name."""
        model_type = cls.try_infer_model_type(model=model, trust_remote_code=trust_remote_code)
        hf_config = cls.get_hf_config(model=model, trust_remote_code=trust_remote_code)

        # Resolve the deploy config & check if the user set the pipeline;
        # If the pipeline is explicitly set, it takes highest priority
        deploy_config_pipe = cls._get_deploy_override_pipe_config(hf_config, deploy_config_path)
        if deploy_config_pipe is not None:
            return deploy_config_pipe

        # Pipeline isn't set in the yaml spec, so we need infer it ourselves.
        if model_type and model_type in OMNI_PIPELINES:
            pipeline_cfg = resolve_pipeline_config(model_type, hf_config)
            if pipeline_cfg is not None:
                return pipeline_cfg

        if hf_config is not None:
            if model_type is not None:
                logger.warning("Inferred model type %s is not registered to an Omni pipeline", model_type)
            hf_archs = set(getattr(hf_config, "architectures", []) or [])
            if hf_archs:
                for registered in OMNI_PIPELINES.values():
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
                        return pipeline_cfg
        return None

    @classmethod
    def _get_deploy_override_pipe_config(
        cls,
        hf_config: PretrainedConfig | None,
        deploy_config_path: str | None,
    ) -> PipelineConfig | None:
        """Load the deploy config and extract + resolve its pipeline field."""
        if deploy_config_path is None:
            return None

        deploy_path = Path(deploy_config_path)
        if deploy_path.exists():
            deploy_cfg = load_deploy_config(deploy_path)
            if deploy_cfg.pipeline is not None:
                return resolve_pipeline_config(deploy_cfg.pipeline, hf_config)
        return None

    @staticmethod
    def _get_trust_remote_code(cli_overrides: dict[str, Any]) -> bool:
        trust_remote_code = cli_overrides.get("trust_remote_code", True)
        return False if trust_remote_code is None else bool(trust_remote_code)

    @staticmethod
    def _get_default_deploy_key(model_type: str | None, pipeline_cfg: PipelineConfig) -> str:
        """Use the resolved pipeline key when HF reports an unregistered type.

        For example, MiMo Audio reports ``model_type="qwen2"`` but resolves
        through its HF architecture to the ``mimo_audio`` pipeline and deploy
        config.
        """
        if model_type in OMNI_PIPELINES:
            return model_type
        return pipeline_cfg.model_type

    @classmethod
    def create_from_model(
        cls,
        model: str,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None,
    ) -> VllmOmniConfig | None:
        """Build the structured Omni config for a model/deploy pair."""
        trust_remote_code = cls._get_trust_remote_code(cli_overrides)
        pipeline_cfg = cls.get_pipeline_config(
            model=model,
            trust_remote_code=trust_remote_code,
            deploy_config_path=deploy_config_path,
        )
        if pipeline_cfg is None:
            return None

        model_type = cls.try_infer_model_type(model=model, trust_remote_code=trust_remote_code)
        deploy_key = cls._get_default_deploy_key(model_type, pipeline_cfg)
        registry_cli_overrides = {**cli_overrides, "model": model}
        return VllmOmniConfig.from_registry(
            deploy_key,
            resolved_pipeline=pipeline_cfg,
            deploy_config_path=deploy_config_path,
            cli_overrides=registry_cli_overrides,
        )

    @classmethod
    def create_legacy_stage_configs_from_model(
        cls,
        model: str,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None,
    ) -> list[StageConfig] | None:
        """Build current runtime stage configs from the shared resolution.

        The engine still consumes the legacy StageConfig/OmegaConf shape.
        RFC #4021 will replace this transitional path as runtime consumers move
        to VllmOmniConfig.
        """
        trust_remote_code = cls._get_trust_remote_code(cli_overrides)
        pipeline_cfg = cls.get_pipeline_config(
            model=model,
            trust_remote_code=trust_remote_code,
            deploy_config_path=deploy_config_path,
        )
        if pipeline_cfg is None:
            return None

        model_type = cls.try_infer_model_type(model=model, trust_remote_code=trust_remote_code)
        deploy_key = cls._get_default_deploy_key(model_type, pipeline_cfg)
        return cls._create_legacy_from_registry(
            deploy_key,
            pipeline_cfg,
            cli_overrides,
            deploy_config_path,
        )

    @classmethod
    def _create_legacy_from_registry(
        cls,
        registry_key: str,
        pipeline_cfg: PipelineConfig,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None = None,
    ) -> list[StageConfig]:
        """Create current runtime StageConfigs from registry + deploy YAML.

        Precedence: caller-typed (non-None) value > deploy YAML >
        StageDeployConfig dataclass default.
        """
        # Resolve deploy config path
        if deploy_config_path is None:
            deploy_path = _DEPLOY_DIR / f"{registry_key}.yaml"
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
