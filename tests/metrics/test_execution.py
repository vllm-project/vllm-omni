# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the execution summary (``--enable-execution-summary``)."""

from __future__ import annotations

import io
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
    return DiffusionOutput(media=DiffusionMediaOutput(video=video), num_completed_steps=50)


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
    assert (facts["num_inference_steps"], facts["completed_steps"]) == (50, 50)
    assert (facts["guidance_scale"], facts["flow_shift"]) == (None, 5.0)


def test_diffusion_facts_read_frame_counts_of_encoded_video():
    # Pipelines that encode MP4 in the worker return bytes, not images.
    formatted = [
        SimpleNamespace(images=[b"mp4"], multimodal_output={"metadata": {"video": {"fps": 24, "frame_counts": [124]}}})
    ]

    facts = execution.diffusion_facts(_request(), DiffusionOutput(), formatted)

    assert (facts["num_frames"], facts["frames_source"]) == (124, "metadata")
    assert facts["size_source"] == "request"


def test_diffusion_facts_prefer_values_the_pipeline_resolved():
    # The request left size and shift to the model defaults; the pipeline encoded
    # the video in the worker and reported what it ran with.
    request = OmniDiffusionRequest(
        prompt="a cat", sampling_params=OmniDiffusionSamplingParams(seed=42), request_id="defaults-0a1b2c3d"
    )
    formatted = [
        SimpleNamespace(images=[b"mp4"], multimodal_output={"metadata": {"video": {"fps": 24, "frame_counts": [209]}}})
    ]
    output = DiffusionOutput(
        num_completed_steps=50,
        execution_facts={"width": 1344, "height": 768, "flow_shift": 12.0, "audio_flow_shift": 5.0},
    )

    facts = execution.diffusion_facts(request, output, formatted)

    assert (facts["width"], facts["height"], facts["size_source"]) == (1344, 768, "pipeline")
    assert (facts["num_frames"], facts["frames_source"]) == (209, "metadata")
    assert (facts["flow_shift"], facts["audio_flow_shift"]) == (12.0, 5.0)
    assert (facts["num_inference_steps"], facts["completed_steps"]) == (None, 50)


def test_model_identity_resolves_hub_snapshot_behind_symlink_view(tmp_path):
    commit = "42ed227ee7df40d41602854ae760620d6eb651fe"
    snapshot = tmp_path / "hub" / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}")
    view = tmp_path / "model-view"
    view.mkdir()
    (view / "model_index.json").symlink_to(snapshot / "model_index.json")

    assert execution._model_identity(str(view)) == ("MiniMaxAI/MiniMax-H3", commit)
    assert execution._model_identity(str(tmp_path / "missing")) == (str(tmp_path / "missing"), None)


def test_engine_runtime_reports_config():
    od_config = SimpleNamespace(
        model="org/name",
        revision="abc",
        model_class_name="Pipe",
        dtype=torch.bfloat16,
        quantization_config=None,
        cache_backend="none",
        cache_config={"rel_l1_thresh": 0.2},
        parallel_config=SimpleNamespace(ulysses_degree=2),
        max_num_seqs=1,
    )

    runtime = execution.engine_runtime(od_config)

    assert {key: runtime[key] for key in ("model", "model_revision", "dtype", "quantization", "cache_backend")} == {
        "model": "org/name",
        "model_revision": "abc",
        "dtype": "bfloat16",
        "quantization": None,
        "cache_backend": "none",
    }
    assert runtime["parallel_config"]["ulysses_degree"] == 2
    # Cache settings only matter when a cache backend runs.
    assert runtime["cache_config"] is None
    od_config.cache_backend = "tea_cache"
    assert execution.engine_runtime(od_config)["cache_config"] == {"rel_l1_thresh": 0.2}


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
        "diffusion",
        rid,
        SimpleNamespace(
            metrics={"execution": {"diffusion": {"completed_steps": 50}, "runtime": {"dtype": "bfloat16"}}}
        ),
        stage_id=0,
    )
    if completed:
        agg.e2e_done.add(rid)

    agg.log_execution_summary(rid, "abc123", reason)

    [(event, payload)] = lines
    assert event == "diffusion_request_completed"
    assert (payload["request_id"], payload["engine_request_id"]) == ("abc123", rid)
    assert (payload["status"], payload["failure_reason"]) == (status, failure_reason)
    [stage] = payload["stages"]
    assert stage["diffusion"] == {"completed_steps": 50}
    assert stage["runtime"] == {"dtype": "bfloat16"}
    assert "num_tokens_in" not in stage


class _Pipeline(ProgressBarMixin):
    pass


@pytest.mark.parametrize("disable", [True, False])
def test_counts_each_iterated_step_once(disable):
    # An enabled bar refreshes on every item (mininterval=0, miniters=1), so
    # tqdm calls update() for items the iteration already counted.
    pipeline = _Pipeline()
    pipeline.set_progress_bar_config(disable=disable, mininterval=0, miniters=1, file=io.StringIO())

    with denoise_progress_scope() as progress:
        for _ in pipeline.progress_bar(range(5)):
            pass

    assert progress.result() == 5


@pytest.mark.parametrize("disable", [True, False])
def test_counts_a_partially_iterated_loop(disable):
    pipeline = _Pipeline()
    pipeline.set_progress_bar_config(disable=disable, mininterval=0, miniters=1, file=io.StringIO())

    with denoise_progress_scope() as progress:
        for index, _ in enumerate(pipeline.progress_bar(range(5))):
            if index == 2:
                break

    # The break leaves the third step's body unfinished.
    assert progress.result() == 2


@pytest.mark.parametrize("disable", [True, False])
def test_counts_explicit_updates(disable):
    pipeline = _Pipeline()
    pipeline.set_progress_bar_config(disable=disable, mininterval=0, miniters=1, file=io.StringIO())

    with denoise_progress_scope() as progress:
        with pipeline.progress_bar(total=4) as bar:
            bar.update(2)
            bar.update()

    assert progress.result() == 3


def test_counts_progress_bar_steps():
    pipeline = _Pipeline()
    pipeline.set_progress_bar_config(disable=True)

    with denoise_progress_scope() as progress:
        with pipeline.progress_bar(total=4) as bar:
            for _ in range(3):
                bar.update()

    assert progress.result() == 3
