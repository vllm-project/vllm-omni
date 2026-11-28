# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the execution summary (``--enable-execution-summary``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.denoise_progress import denoise_progress_scope
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.metrics import execution, stats
from vllm_omni.metrics.stats import OrchestratorAggregator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request() -> OmniDiffusionRequest:
    params = OmniDiffusionSamplingParams(seed=42, num_inference_steps=50, fps=24, extra_args={"flow_shift": 5.0})
    return OmniDiffusionRequest(prompt="a cat", sampling_params=params, request_id="abc123-0a1b2c3d")


def _video_output() -> DiffusionOutput:
    video = VideoMediaOutput(
        tensor=torch.zeros(1, 3, 9, 16, 24),
        spec=VideoTensorSpec(
            layout=VideoTensorLayout.BCTHW,
            encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
            value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
        ),
    )
    return DiffusionOutput(media=DiffusionMediaOutput(video=video), num_scheduled_steps=50, num_completed_steps=50)


def test_diffusion_facts_report_generated_tensor_and_counted_steps():
    # num_frames still holds the image-model sentinel (1) the video pipeline replaced.
    facts = execution.diffusion_facts(_request(), _video_output())

    assert {key: facts[key] for key in ("width", "height", "num_frames", "size_source", "fps", "seed")} == {
        "width": 24,
        "height": 16,
        "num_frames": 9,
        "size_source": "tensor",
        "fps": 24,
        "seed": 42,
    }
    assert (facts["num_inference_steps"], facts["num_scheduled_steps"], facts["num_completed_steps"]) == (50, 50, 50)
    assert (facts["guidance_scale"], facts["flow_shift"]) == (None, 5.0)


def test_diffusion_facts_read_frame_counts_of_encoded_video():
    # Pipelines that encode MP4 in the worker return bytes, not images.
    formatted = [
        SimpleNamespace(images=[b"mp4"], multimodal_output={"metadata": {"video": {"fps": 24, "frame_counts": [124]}}})
    ]

    facts = execution.diffusion_facts(_request(), DiffusionOutput(), formatted)

    assert (facts["num_frames"], facts["size_source"]) == (124, "metadata")


@pytest.mark.parametrize(
    ("completed", "reason", "status", "failure_reason"),
    [(True, None, "completed", None), (False, "client_disconnect", "failed", "client_disconnect")],
)
def test_request_event(monkeypatch, completed, reason, status, failure_reason):
    lines = []
    monkeypatch.setattr(stats, "log_execution", lambda event, payload: lines.append((event, payload)))
    agg = OrchestratorAggregator(num_stages=1, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e=0)
    rid = "abc123-0a1b2c3d"
    agg.stage_events[rid] = [
        SimpleNamespace(
            stage_id=0,
            replica_id=0,
            final_output_type="video",
            batch_size=1,
            num_tokens_in=0,
            num_tokens_out=0,
            audio_duration_s=0.0,
        )
    ]
    agg.accumulate_diffusion_metrics(
        "diffusion", rid, SimpleNamespace(metrics={"execution": {"num_completed_steps": 50}}), stage_id=0
    )
    if completed:
        agg.e2e_done.add(rid)

    agg.log_execution_summary(rid, "abc123", reason)

    [(event, payload)] = lines
    assert event == "request"
    assert (payload["request_id"], payload["external_request_id"]) == (rid, "abc123")
    assert (payload["status"], payload["failure_reason"]) == (status, failure_reason)
    assert payload["stages"][0]["diffusion"] == {"num_completed_steps": 50}


class _Pipeline(ProgressBarMixin):
    pass


def test_counts_progress_bar_steps():
    pipeline = _Pipeline()
    pipeline.set_progress_bar_config(disable=True)

    with denoise_progress_scope() as progress:
        with pipeline.progress_bar(total=4) as bar:
            for _ in range(3):
                bar.update()

    assert progress.result() == (3, 4)
