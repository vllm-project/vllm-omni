# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config factories for vllm-omni, e.g., StageConfigFactory."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
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


# Default degree for any parallel axis / replica count that isn't set anywhere
# (CLI, deploy YAML, or pipeline default): a single, un-parallelized rank.
# TODO(composable_parallel): this "1" default and the per-axis device-layout
# fallbacks are currently re-derived in the merge, apply, and reconcile layers.
# Centralize them in one schema so the pre-spawn device guard and the engine-args
# defaults can't drift apart. This is the light slice; the full device-layout
# centralization is tracked as a follow-up.
_DEFAULT_PARALLEL_DEGREE = 1


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
    def get_pipeline_config(
        cls,
        model: str,
        trust_remote_code: bool,
        deploy_config_path: str | None = None,
        user_deploy_config: DeployConfig | None = None,
    ) -> PipelineConfig | None:
        """Resolve the PipelineConfig for a model path/name."""
        model_type = cls.try_infer_model_type(model=model, trust_remote_code=trust_remote_code)
        hf_config = cls.get_hf_config(model=model, trust_remote_code=trust_remote_code)

        # Resolve the deploy config & check if the user set the pipeline;
        # If the pipeline is explicitly set, it takes highest priority
        if user_deploy_config is None:
            user_deploy_config = cls._load_user_deploy_config(deploy_config_path)
        deploy_config_pipe = cls._get_deploy_override_pipe_config(hf_config, user_deploy_config)
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
        deploy_config: DeployConfig | None,
    ) -> PipelineConfig | None:
        """Resolve an explicit pipeline override from a loaded deploy config."""
        if deploy_config is None or deploy_config.pipeline is None:
            return None

        pipeline_cfg = resolve_pipeline_config(deploy_config.pipeline, hf_config)
        if pipeline_cfg is None:
            raise KeyError(
                f"Pipeline {deploy_config.pipeline!r} from deploy config is not registered "
                f"to OMNI_PIPELINES. Available: {sorted(OMNI_PIPELINES)}"
            )
        return pipeline_cfg

    @staticmethod
    def _load_user_deploy_config(deploy_config_path: str | None) -> DeployConfig | None:
        """Load an explicit deploy YAML once for resolution and construction."""
        if deploy_config_path is None:
            return None
        deploy_path = Path(deploy_config_path)
        if not deploy_path.exists() and deploy_path.parent == Path("."):
            candidate = _DEPLOY_DIR / deploy_path
            if candidate.exists():
                deploy_path = candidate
        if not deploy_path.exists():
            raise FileNotFoundError(f"Deploy config not found: {deploy_path}")
        return load_deploy_config(deploy_path)

    @classmethod
    def create_from_model(
        cls,
        model: str,
        *,
        trust_remote_code: bool,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None,
    ) -> VllmOmniConfig | None:
        """Build the structured Omni config for a model/deploy pair."""
        user_deploy_config = cls._load_user_deploy_config(deploy_config_path)
        pipeline_cfg = cls.get_pipeline_config(
            model=model,
            trust_remote_code=trust_remote_code,
            deploy_config_path=deploy_config_path,
            user_deploy_config=user_deploy_config,
        )
        if pipeline_cfg is None:
            return None

        registry_cli_overrides = {
            **cli_overrides,
            "trust_remote_code": trust_remote_code,
            "model": model,
        }
        return VllmOmniConfig.from_pipeline_config(
            pipeline_cfg,
            user_deploy_config=user_deploy_config,
            deploy_config_path=deploy_config_path,
            cli_overrides=registry_cli_overrides,
        )

    @classmethod
    def create_legacy_stage_configs_from_model(
        cls,
        model: str,
        *,
        trust_remote_code: bool,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None,
        strategy_specs: Mapping[Any, Any] | None = None,
    ) -> tuple[list[StageConfig] | None, str | None]:
        """Build current runtime stage configs from the shared resolution.

        The engine still consumes the legacy StageConfig/OmegaConf shape.
        RFC #4021 will replace this transitional path as runtime consumers move
        to VllmOmniConfig.
        """
        user_deploy_config = cls._load_user_deploy_config(deploy_config_path)
        pipeline_cfg = cls.get_pipeline_config(
            model=model,
            trust_remote_code=trust_remote_code,
            deploy_config_path=deploy_config_path,
            user_deploy_config=user_deploy_config,
        )
        if pipeline_cfg is None:
            return None, None

        legacy_cli_overrides = {
            **cli_overrides,
            "trust_remote_code": trust_remote_code,
        }
        return cls._create_legacy_from_registry(
            pipeline_cfg,
            legacy_cli_overrides,
            deploy_config_path,
            user_deploy_config=user_deploy_config,
            strategy_specs=strategy_specs,
        )

    @classmethod
    def _create_legacy_from_registry(
        cls,
        pipeline_cfg: PipelineConfig,
        cli_overrides: dict[str, Any],
        deploy_config_path: str | None = None,
        user_deploy_config: DeployConfig | None = None,
        strategy_specs: Mapping[Any, Any] | None = None,
    ) -> tuple[list[StageConfig], str | None]:
        """Create current runtime StageConfigs from registry + deploy YAML.

        Precedence: caller-typed (non-None) value > deploy YAML >
        StageDeployConfig dataclass default.

        Returns ``(stages, omni_lb_policy)`` — the strategy-derived pipeline-wide
        load-balancer policy (``None`` when no strategy set one) travels with the
        stages instead of through a mutable out-param.
        """
        if user_deploy_config is not None:
            deploy_cfg = user_deploy_config
        elif deploy_config_path is not None:
            deploy_cfg = cls._load_user_deploy_config(deploy_config_path)
            assert deploy_cfg is not None
        elif pipeline_cfg.default_deploy_config_name is not None:
            deploy_cfg = load_deploy_config(_DEPLOY_DIR / pipeline_cfg.default_deploy_config_name)
        else:
            deploy_cfg = DeployConfig()

        cli_async_chunk = cli_overrides.get("async_chunk")
        if cli_async_chunk is not None:
            deploy_cfg.async_chunk = bool(cli_async_chunk)

        stages = merge_pipeline_deploy(pipeline_cfg, deploy_cfg, cli_overrides)

        # Overlay declarative parallel strategies (opt-in) before CLI overrides.
        applied = cls._apply_strategy_specs(stages, strategy_specs)

        explicit_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

        for stage in stages:
            stage.runtime_overrides = cls._merge_cli_overrides(stage, explicit_overrides)

        # Re-validate the resolved layout now that CLI overrides are on top.
        cls._reconcile_strategy_with_cli(stages, applied)

        omni_lb_policy = applied.omni_lb_policy if applied is not None else None
        return stages, omni_lb_policy

    @staticmethod
    def _apply_strategy_specs(
        stages: list[StageConfig],
        strategy_specs: Mapping[Any, Any] | None,
    ) -> Any:
        """Overlay derived parallel sizing onto merged stages (opt-in).

        ``omni_lb_policy`` cannot be set from stage configs (omni reads it once
        at orchestrator construction), so a derived non-default policy is logged
        here and carried on the returned ``StrategyApplyResult`` for the caller
        to hand to the orchestrator.

        Returns the ``StrategyApplyResult`` (or ``None`` when no strategy was
        supplied) so the caller can re-validate the resolved layout once CLI
        overrides have been merged on top, and read its ``omni_lb_policy``.
        """
        if not strategy_specs:
            return None
        from vllm_omni.config.composable_parallel import apply_strategy_specs

        applied = apply_strategy_specs(stages, strategy_specs)
        if applied.omni_lb_policy is not None:
            logger.info(
                "[composable_parallel] strategy derived omni_lb_policy=%r; it will be applied "
                "to the orchestrator unless an explicit --omni-lb-policy was given.",
                applied.omni_lb_policy,
            )
        return applied

    @staticmethod
    def _reconcile_strategy_with_cli(
        stages: list[StageConfig],
        applied: Any,
    ) -> None:
        """Reconcile CLI overrides applied *after* a strategy overlay.

        CLI overrides are applied last (in ``to_omegaconf``), so for any stage a
        strategy touched we must (a) warn loudly when a CLI arg overrides a
        strategy-declared axis — the strategy is meant to be the single writer
        for the axes it declares, so a silent CLI win is surprising — and
        (b) re-run the device-count check against the *effective* (post-CLI)
        ``tp``/``dp``/``pp``/``num_replicas``/``devices`` so a CLI
        ``--devices``/``--tensor-parallel-size``/``--num-replicas`` cannot slip
        past the pre-spawn guard that exists to prevent silent OOMs.
        """
        if applied is None:
            return
        from vllm_omni.config.composable_parallel import check_device_layout

        # Axis kind -> (engine-arg field, strategy-derived attribute on the cfg).
        axis_fields = {
            "tp": ("tensor_parallel_size", "tensor_parallel_size"),
            "dp": ("data_parallel_size", "data_parallel_size"),
            "pp": ("pipeline_parallel_size", "pipeline_parallel_size"),
        }

        for stage in stages:
            cfg = applied.per_stage_config.get(stage.stage_id)
            if cfg is None:
                continue
            overrides = stage.runtime_overrides or {}
            declared = set(cfg.l1_owners.keys())

            for kind, (field_name, attr) in axis_fields.items():
                if kind in declared and overrides.get(field_name) is not None:
                    cli_val = overrides[field_name]
                    derived = getattr(cfg, attr)
                    if cli_val != derived:
                        logger.warning(
                            "[composable_parallel] stage %s: CLI %s=%s overrides the "
                            "strategy-derived %s=%s. The CLI value wins; remove one to avoid ambiguity.",
                            stage.stage_id,
                            field_name,
                            cli_val,
                            field_name,
                            derived,
                        )
            if "stage_replica" in declared and overrides.get("num_replicas") is not None:
                cli_val = overrides["num_replicas"]
                if cli_val != cfg.stage_replica_size:
                    logger.warning(
                        "[composable_parallel] stage %s: CLI num_replicas=%s overrides the "
                        "strategy-derived num_replicas=%s. The CLI value wins; remove one to avoid ambiguity.",
                        stage.stage_id,
                        cli_val,
                        cfg.stage_replica_size,
                    )

            def _eff(field_name: str, fallback: Any) -> Any:
                val = overrides.get(field_name)
                return val if val is not None else fallback

            def _eff_degree(field_name: str, source: dict[str, Any]) -> int:
                # Single place the per-axis "default to 1" fallback is applied,
                # for both the override and the YAML-default sides (see the
                # _DEFAULT_PARALLEL_DEGREE TODO).
                value = _eff(field_name, source.get(field_name, _DEFAULT_PARALLEL_DEGREE))
                return int(value or _DEFAULT_PARALLEL_DEGREE)

            check_device_layout(
                _eff("devices", stage.yaml_runtime.get("devices")),
                tensor_parallel_size=_eff_degree("tensor_parallel_size", stage.yaml_engine_args),
                data_parallel_size=_eff_degree("data_parallel_size", stage.yaml_engine_args),
                pipeline_parallel_size=_eff_degree("pipeline_parallel_size", stage.yaml_engine_args),
                num_replicas=_eff_degree("num_replicas", stage.yaml_runtime),
                role=stage.model_stage,
            )

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
