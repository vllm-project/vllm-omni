# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class BasePipelineUtils:
    """No-op hooks for pipeline-specific diffusers adapter behavior."""

    def update_load_kwargs(self, od_config: OmniDiffusionConfig, load_kwargs: dict[str, Any]) -> None:
        pass

    def apply_post_load_updates(self, pipeline: DiffusionPipeline, od_config: OmniDiffusionConfig) -> None:
        pass

    def validate_runtime_sampling_params(self, sampling: OmniDiffusionSamplingParams) -> None:
        pass


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
