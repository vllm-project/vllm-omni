# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import logging
from pathlib import Path
from typing import Any

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = logging.getLogger(__name__)

LTX25_ORIGINAL_MODEL_ID = "Lightricks/LTX-2.5"
LTX25_DIFFUSERS_MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
LTX25_DIFFUSERS_COMMIT = "7564fb016dabda0c943416190fc92398c50b1b20"

# Diffusers' public LTX-2.5 schedule intentionally excludes the scheduler's
# terminal zero. FlowMatchEulerDiscreteScheduler appends that internally.
LTX25_DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]

_LTX25_DISTILLED_GUIDANCE_DEFAULTS: dict[str, float] = {
    "guidance_scale": 1.0,
    "audio_guidance_scale": 1.0,
    "stg_scale": 0.0,
    "audio_stg_scale": 0.0,
    "modality_scale": 1.0,
    "audio_modality_scale": 1.0,
    "guidance_rescale": 0.0,
    "audio_guidance_rescale": 0.0,
}


def _supports_ltx25_diffusers() -> bool:
    """Check the standard-pipeline capabilities added with LTX-2.5."""
    try:
        import diffusers
        import transformers

        pipeline_class = diffusers.LTX2Pipeline
        init_parameters = inspect.signature(pipeline_class.__init__).parameters
        return (
            "duration_head" in init_parameters
            and "prompt_enhancer" in init_parameters
            and hasattr(transformers, "Gemma4UnifiedForConditionalGeneration")
        )
    except (AttributeError, ImportError, RuntimeError):
        return False


def is_ltx25_diffusers_model(od_config: OmniDiffusionConfig) -> bool:
    """Return whether config targets the official converted LTX-2.5 repo."""
    normalized_model = str(od_config.model).rstrip("/\\").replace("\\", "/").lower()
    if normalized_model == LTX25_DIFFUSERS_MODEL_ID.lower():
        return True

    # A downloaded checkpoint may be served from any directory name. The
    # duration-head component is the pipeline-level marker introduced for
    # LTX-2.5; older LTX-2.0/2.3 indexes do not contain it.
    model_path = Path(str(od_config.model)).expanduser()
    subfolder = od_config.diffusers_load_kwargs.get("subfolder")
    if subfolder is not None:
        model_path /= str(subfolder)
    try:
        with (model_path / "model_index.json").open(encoding="utf-8") as model_index_file:
            model_index = json.load(model_index_file)
    except (OSError, TypeError, ValueError):
        return False
    if not isinstance(model_index, dict):
        return False

    duration_head = model_index.get("duration_head")
    return (
        model_index.get("_class_name") == "LTX2Pipeline"
        and isinstance(duration_head, list)
        and bool(duration_head)
        and duration_head[0] is not None
    )


class BasePipelineUtils:
    """No-op hooks for pipeline-specific diffusers adapter behavior."""

    def update_load_kwargs(self, od_config: OmniDiffusionConfig, load_kwargs: dict[str, Any]) -> None:
        pass

    def apply_post_load_updates(self, pipeline: DiffusionPipeline, od_config: OmniDiffusionConfig) -> None:
        pass

    def enable_vae_optimization(self, pipeline: DiffusionPipeline, optimization: str) -> None:
        getattr(pipeline, f"enable_vae_{optimization}")()

    def validate_runtime_sampling_params(self, sampling: OmniDiffusionSamplingParams) -> None:
        pass

    def update_call_kwargs(
        self,
        od_config: OmniDiffusionConfig,
        call_kwargs: dict[str, Any],
    ) -> None:
        pass

    def normalize_output(
        self,
        pipeline: DiffusionPipeline,
        od_config: OmniDiffusionConfig,
        output: Any,
    ) -> Any:
        return output


def validate_model_compatibility(od_config: OmniDiffusionConfig) -> None:
    """Reject known model/dependency combinations before loading weights."""
    normalized_model = str(od_config.model).rstrip("/\\").replace("\\", "/").lower()
    if normalized_model == LTX25_ORIGINAL_MODEL_ID.lower():
        raise ValueError(
            f"{LTX25_ORIGINAL_MODEL_ID} is a component-weight repository without a Diffusers "
            f"model_index.json. Use the official converted repository {LTX25_DIFFUSERS_MODEL_ID} "
            "with --diffusion-load-format diffusers."
        )

    if is_ltx25_diffusers_model(od_config) and not _supports_ltx25_diffusers():
        raise ImportError(
            f"{LTX25_DIFFUSERS_MODEL_ID} requires unreleased LTX-2.5 support from Diffusers and "
            "Gemma 4 Unified support from Transformers. Install the source-preview dependencies with "
            "`python -m pip install --upgrade 'transformers>=5.10.1' "
            f"'diffusers @ git+https://github.com/huggingface/diffusers.git@{LTX25_DIFFUSERS_COMMIT}'`."
        )


class LTX2PipelineUtils(BasePipelineUtils):
    """Diffusers call policy for the official default LTX-2.5 checkpoint."""

    def enable_vae_optimization(self, pipeline: DiffusionPipeline, optimization: str) -> None:
        pipeline_method = getattr(pipeline, f"enable_vae_{optimization}", None)
        if callable(pipeline_method):
            pipeline_method()
        else:
            getattr(pipeline.vae, f"enable_{optimization}")()

    def update_call_kwargs(
        self,
        od_config: OmniDiffusionConfig,
        call_kwargs: dict[str, Any],
    ) -> None:
        if not is_ltx25_diffusers_model(od_config):
            return

        has_sigmas = call_kwargs.get("sigmas") is not None
        has_timesteps = call_kwargs.get("timesteps") is not None
        has_step_count = call_kwargs.get("num_inference_steps") is not None
        if not has_sigmas and not has_timesteps:
            if has_step_count:
                logger.warning(
                    "LTX-2.5 distilled inference received num_inference_steps without sigmas. "
                    "Diffusers will use a generic schedule; omit num_inference_steps to use the "
                    "official distilled schedule."
                )
            else:
                call_kwargs["sigmas"] = list(LTX25_DISTILLED_SIGMAS)
                call_kwargs["num_inference_steps"] = len(LTX25_DISTILLED_SIGMAS)

        for key, value in _LTX25_DISTILLED_GUIDANCE_DEFAULTS.items():
            if call_kwargs.get(key) is None:
                call_kwargs[key] = value

    def normalize_output(
        self,
        pipeline: DiffusionPipeline,
        od_config: OmniDiffusionConfig,
        output: Any,
    ) -> Any:
        if not is_ltx25_diffusers_model(od_config):
            return output

        frames = getattr(output, "frames", None)
        audio = getattr(output, "audio", None)
        if frames is None or audio is None:
            return output

        envelope: dict[str, Any] = {"payload": {"video": frames, "audio": audio}}
        config = getattr(getattr(pipeline, "vocoder", None), "config", None)
        value = (
            config.get("output_sampling_rate")
            if hasattr(config, "get")
            else getattr(config, "output_sampling_rate", None)
        )
        try:
            sample_rate = None if isinstance(value, bool) else int(value)
        except (TypeError, ValueError):
            sample_rate = None
        if sample_rate is not None and sample_rate > 0:
            envelope["metadata"] = {"audio": {"sample_rate": sample_rate}}
        return envelope


class WanPipelineUtils(BasePipelineUtils):
    def update_load_kwargs(self, od_config: OmniDiffusionConfig, load_kwargs: dict[str, Any]) -> None:
        if od_config.boundary_ratio is not None:
            load_kwargs["boundary_ratio"] = od_config.boundary_ratio

    def apply_post_load_updates(self, pipeline: DiffusionPipeline, od_config: OmniDiffusionConfig) -> None:
        if od_config.flow_shift is not None:
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

            pipeline.scheduler = UniPCMultistepScheduler.from_config(
                pipeline.scheduler.config, flow_shift=od_config.flow_shift
            )

    def validate_runtime_sampling_params(self, sampling: OmniDiffusionSamplingParams) -> None:
        if sampling.boundary_ratio is not None:
            raise ValueError(
                "Boundary ratio is not supported at runtime with the diffusers backend for Wan models. Please set "
                "it at model loading time using the `boundary_ratio` kwarg or `--diffusers-load-kwargs` JSON."
            )
        if sampling.extra_args.get("flow_shift") is not None:
            raise ValueError(
                "Flow shift is not supported at runtime with the diffusers backend for Wan models. Please set "
                "it at model loading time using the `flow_shift` kwarg."
            )


PIPELINE_UTILS_REGISTRY: dict[str, type[BasePipelineUtils]] = {
    "LTX2Pipeline": LTX2PipelineUtils,
    "WanPipeline": WanPipelineUtils,
    "WanImageToVideoPipeline": WanPipelineUtils,
    "WanVACEPipeline": WanPipelineUtils,
    "WanVideoToVideoPipeline": WanPipelineUtils,
    "WanAnimatePipeline": WanPipelineUtils,
}


def get_pipeline_utils(pipeline_class_name: str | None) -> BasePipelineUtils:
    if pipeline_class_name is None:
        return BasePipelineUtils()
    pipeline_utils_cls = PIPELINE_UTILS_REGISTRY.get(pipeline_class_name, BasePipelineUtils)
    return pipeline_utils_cls()
