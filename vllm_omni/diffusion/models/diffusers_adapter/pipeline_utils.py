# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import resolve_video_num_frames
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class BasePipelineUtils:
    """No-op hooks for pipeline-specific diffusers adapter behavior."""

    def resolve_pipeline_class(
        self,
        od_config: OmniDiffusionConfig,
        configured_pipeline_class: type[DiffusionPipeline] | None,
    ) -> type[DiffusionPipeline] | None:
        """Resolve the concrete Diffusers class used to load a checkpoint."""
        return configured_pipeline_class

    def update_load_kwargs(self, od_config: OmniDiffusionConfig, load_kwargs: dict[str, Any]) -> None:
        pass

    def apply_post_load_updates(self, pipeline: DiffusionPipeline, od_config: OmniDiffusionConfig) -> None:
        pass

    def validate_runtime_sampling_params(self, sampling: OmniDiffusionSamplingParams) -> None:
        if sampling.enable_frame_interpolation:
            raise ValueError(
                "Frame interpolation is not supported with the Diffusers backend because "
                "Diffusers returns already postprocessed frames."
            )

    def update_call_kwargs(
        self,
        req: DiffusionRequestBatch,
        sampling: OmniDiffusionSamplingParams,
        accepted_call_kwargs: set[str] | None,
        call_kwargs: dict[str, Any],
    ) -> None:
        pass


class SanaVideoPipelineUtils(BasePipelineUtils):
    def resolve_pipeline_class(
        self,
        od_config: OmniDiffusionConfig,
        configured_pipeline_class: type[DiffusionPipeline] | None,
    ) -> type[DiffusionPipeline] | None:
        # Both released SANA-Video checkpoints declare SanaVideoPipeline in
        # model_index.json. Honour an explicit Omni I2V class selection by
        # loading the same components into Diffusers' I2V pipeline instead.
        if od_config.model_class_name == "SanaImageToVideoPipeline":
            from diffusers import SanaImageToVideoPipeline

            return SanaImageToVideoPipeline
        return configured_pipeline_class

    def update_call_kwargs(
        self,
        req: DiffusionRequestBatch,
        sampling: OmniDiffusionSamplingParams,
        accepted_call_kwargs: set[str] | None,
        call_kwargs: dict[str, Any],
    ) -> None:
        # Diffusers SANA-Video calls the shared request's `num_frames`
        # parameter `frames`.
        if (
            accepted_call_kwargs is not None
            and "frames" in accepted_call_kwargs
            and "num_frames" not in accepted_call_kwargs
        ):
            configured_default = call_kwargs.get("frames", 81)
            if configured_default is None:
                configured_default = 81
            call_kwargs["frames"] = resolve_video_num_frames(
                sampling.num_frames,
                default_num_frames=configured_default,
                is_dummy_run=req.is_dummy_run(),
            )

        # SANA's startup dummy dimensions can map to a resolution bucket that
        # conflicts with the upstream pipeline's divisibility validation.
        if req.is_dummy_run() and accepted_call_kwargs is not None and "use_resolution_binning" in accepted_call_kwargs:
            call_kwargs["use_resolution_binning"] = False


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
        super().validate_runtime_sampling_params(sampling)
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
    "SanaVideoPipeline": SanaVideoPipelineUtils,
    "SanaImageToVideoPipeline": SanaVideoPipelineUtils,
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


def get_pipeline_utils_for_config(od_config: OmniDiffusionConfig) -> BasePipelineUtils:
    """Return model-specific adapter hooks for the requested or configured class."""
    requested_class_name = od_config.model_class_name
    if requested_class_name in PIPELINE_UTILS_REGISTRY:
        return get_pipeline_utils(requested_class_name)

    configured_pipeline_class = od_config.diffusers_pipeline_cls
    configured_class_name = configured_pipeline_class.__name__ if configured_pipeline_class is not None else None
    return get_pipeline_utils(configured_class_name)


def resolve_diffusers_pipeline_class(od_config: OmniDiffusionConfig) -> type[DiffusionPipeline] | None:
    """Resolve the concrete Diffusers pipeline used for loading and capability checks."""
    pipeline_utils = get_pipeline_utils_for_config(od_config)
    return pipeline_utils.resolve_pipeline_class(od_config, od_config.diffusers_pipeline_cls)
