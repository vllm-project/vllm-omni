# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.experimental.ar_diffusion.cell_schedule import CellPlan, CellSchedule, Ordering, build_cell_plan
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def validate_physis_request(
    params: OmniDiffusionSamplingParams,
    *,
    num_frames: int,
    frame_rate: float | None = None,
) -> None:
    expected = {
        "height": 480,
        "width": 832,
        "num_frames": num_frames,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "guidance_scale_2": 1.0,
        "num_outputs_per_prompt": 1,
    }
    if frame_rate is not None:
        expected["frame_rate"] = frame_rate
    for name, required in expected.items():
        value = getattr(params, name)
        if value is not None and value != required:
            raise ValueError(f"Physis requires {name}={required}; received {value}")
    if params.do_classifier_free_guidance:
        raise ValueError("Physis checkpoints are guidance distilled and do not support CFG")
    if params.timesteps is not None or params.sigmas is not None:
        raise ValueError("Physis uses its trained four-step timestep and sigma schedule")


def build_physis_cell_plan(
    *, chunks: int, step_replicas: int, layer_groups: int, blocks: int, blocks_per_rank: int
) -> CellPlan:
    return build_cell_plan(
        CellSchedule(
            chunks=chunks - 1,
            num_denoise_steps=4,
            step_replicas=step_replicas,
            layer_groups=layer_groups,
            ordering=Ordering.SAME_STEP,
            kv_history_chunks=18,
            blocks=blocks,
            blocks_per_rank=blocks_per_rank,
        )
    )


def prepare_video_output(output: DiffusionOutput, *, output_type: str) -> DiffusionOutput:
    if output.media is not None:
        video = output.media.video
        if output_type == "latent":
            return replace(output, output=video.tensor, media=None)
        output.media = output.media.with_video(replace(video, spec=replace(video.spec, preserve_input_dtype=True)))
    return output
