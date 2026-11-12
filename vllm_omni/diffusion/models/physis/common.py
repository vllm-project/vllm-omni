# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionOutput, DiffusionParallelConfig
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.batch import ComputeBatch
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import CellAdapter
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import (
    CellPlan,
    CellSchedule,
    Ordering,
    build_cell_plan,
)
from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class PhysisCell(WanCell):
    """Physis four-step backend; consumes executor ``ComputeBatch`` (R=1 today)."""

    def forward(
        self,
        batch: ComputeBatch,
        members: Sequence[CellAdapter],
        *,
        task_count: int,
        member_inputs: Sequence[Any | None],
    ) -> dict[str, torch.Tensor]:
        del task_count
        if batch.size != 1:
            raise ValueError(f"PhysisCell currently supports batch size 1, got {batch.size}")
        member = members[0]
        if not isinstance(member, PhysisCell):
            raise TypeError(f"expected PhysisCell member, got {type(member).__name__}")
        member_ref = batch.members[0]
        req = member_ref.request_id
        chunk, step, block = member_ref.chunk, member_ref.step, member_ref.block
        if not 0 <= step < 4:
            raise ValueError(f"Physis expects denoise step in [0, 4), got {step}")
        live = member._live.setdefault(req, {})
        latent, tokens = member._resolve_member_input(
            req=req,
            chunk=chunk,
            member_input=member_inputs[0] if member_inputs else None,
            live=live,
        )
        tf = member.transformer
        timestep = member._timesteps[step]
        if block == tf.start_layer and tf.is_layer_group_first:
            tokens = tf.embed(latent)
        else:
            if tokens is None:
                raise RuntimeError(f"cell {(chunk, step, block)} missing hidden_states")
        tokens = tf.block(
            block,
            tokens,
            chunk=chunk + 1,
            step=step,
            latent=latent,
            timestep=timestep,
            kv_ctx=batch.kv_views[0],
        )
        if not (block == tf.end_layer - 1 and tf.is_layer_group_last):
            live[chunk] = latent
            return {"latent": latent, "hidden_states": tokens}
        prediction = tf.unembed(tokens, chunk=chunk + 1, step=step, latent=latent, timestep=timestep)
        latent = member.sampler.advance(latent, prediction, step=step, seed=member.seed, chunk=chunk + 1)
        live[chunk] = latent
        if step == 3:
            member.finished.setdefault(req, {})[chunk] = latent.detach()
            if member.on_finished is not None:
                member.on_finished(chunk + 1, latent)
        return {"latent": latent}


def validate_physis_parallel_config(parallel: DiffusionParallelConfig) -> None:
    for name in ("tensor_parallel_size", "ulysses_degree", "ring_degree", "allgather_degree", "cfg_parallel_size"):
        value = getattr(parallel, name)
        if value != 1:
            raise ValueError(f"Physis Cell execution requires TP1/SP1/CFG1; received {name}={value}")


def validate_physis_request(
    params: OmniDiffusionSamplingParams,
    *,
    num_frames: int,
    frame_rate: float | None = None,
) -> None:
    if params.num_frames is None:
        params.num_frames = num_frames
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
