# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the execution summary (``--enable-execution-summary``)."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, count_denoise_steps
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.metrics import execution, stats
from vllm_omni.metrics.stats import OrchestratorAggregator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(**params) -> OmniDiffusionRequest:
    params = {"seed": 42, "num_inference_steps": 50, "fps": 24, **params}
    return OmniDiffusionRequest(prompt="a cat", sampling_params=OmniDiffusionSamplingParams(**params), request_id="r-1")


def _video(frames: int, height: int, width: int) -> DiffusionMediaOutput:
    spec = VideoTensorSpec(
        layout=VideoTensorLayout.BCTHW,
        encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
        value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
    )
    return DiffusionMediaOutput(video=VideoMediaOutput(tensor=torch.zeros(1, 3, frames, height, width), spec=spec))


def _encoded(frame_counts=None):
    metadata = {"metadata": {"video": {"fps": 24, "frame_counts": frame_counts}}} if frame_counts else {}
    return [SimpleNamespace(images=[b"mp4"], multimodal_output=metadata)]


@pytest.mark.parametrize(
    ("output", "formatted", "frames"),
    [
        (DiffusionOutput(media=_video(9, 16, 24)), None, 9),  # generated tensor
        (DiffusionOutput(), _encoded([124]), 124),  # encoder's frame counts
        (DiffusionOutput(), _encoded(), 209),  # the encoded MP4 itself
    ],
)
def test_diffusion_facts_report_generated_frames(monkeypatch, output, formatted, frames):
    from vllm_omni.diffusion.utils import media_utils

    monkeypatch.setattr(media_utils, "count_mp4_frames", lambda data: 209)

    assert execution.diffusion_facts(_request(), output, formatted)["num_frames"] == frames


def test_diffusion_facts_read_fps_from_the_output_when_the_request_omits_it():
    assert execution.diffusion_facts(_request(fps=None), DiffusionOutput(), _encoded([209]))["fps"] == 24


def test_diffusion_facts_prefer_values_the_pipeline_resolved():
    output = DiffusionOutput(
        num_completed_steps=50,
        num_cached_steps=20,
        execution_facts={"width": 1344, "height": 768, "flow_shift": 12.0, "audio_flow_shift": 3.0, "cfg": False},
    )

    facts = execution.diffusion_facts(_request(num_inference_steps=None, guidance_scale=5.0), output)

    assert (facts["width"], facts["height"], facts["flow_shift"], facts["audio_flow_shift"]) == (1344, 768, 12.0, 3.0)
    assert (facts["num_inference_steps"], facts["completed_steps"], facts["cached_steps"]) == (None, 50, 20)
    # The pipeline runs without CFG, so the request's guidance did not apply.
    assert (facts["guidance_scale"], facts["cfg"]) == (None, False)
    assert execution.diffusion_facts(_request(guidance_scale=5.0), DiffusionOutput())["guidance_scale"] == 5.0


def test_diffusion_facts_list_request_and_static_lora():
    request = _request()
    request.sampling_params.lora_request = SimpleNamespace(lora_name="style")
    od_config = SimpleNamespace(lora_path="/loras/base.safetensors", flow_shift=None)

    assert execution.diffusion_facts(request, DiffusionOutput(), None, od_config)["lora"] == [
        "style",
        "/loras/base.safetensors",
    ]
    assert execution.diffusion_facts(_request(), DiffusionOutput())["lora"] is None


def test_model_identity_resolves_hub_snapshot_behind_symlink_view(tmp_path):
    commit = "42ed227ee7df40d41602854ae760620d6eb651fe"
    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}")
    view = tmp_path / "model-view"
    view.mkdir()
    (view / "model_index.json").symlink_to(snapshot / "model_index.json")

    assert execution._model_identity(str(view)) == ("MiniMaxAI/MiniMax-H3", commit)
    assert execution._model_identity("org/name") == ("org/name", None)


def test_model_identity_survives_an_unreadable_directory(tmp_path, monkeypatch):
    def unreadable(path):
        raise PermissionError(path)

    monkeypatch.setattr(execution.os, "listdir", unreadable)

    assert execution._model_identity(str(tmp_path)) == (str(tmp_path), None)


def test_engine_runtime_reports_cache_config_only_with_a_cache_backend():
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

    assert (runtime["model_revision"], runtime["configured_dtype"], runtime["cache_config"]) == (
        "abc",
        "bfloat16",
        None,
    )
    assert runtime["parallel_config"]["ulysses_degree"] == 2
    od_config.cache_backend = "tea_cache"
    assert execution.engine_runtime(od_config)["cache_config"] == {"rel_l1_thresh": 0.2}


def test_weight_dtypes_count_parameters_and_packed_buffers_per_component():
    class Packed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("packed_weight", torch.zeros(4, 2, dtype=torch.uint8))
            self.register_buffer("cache", torch.zeros(100), persistent=False)

    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    pipeline.video_vae = torch.nn.Sequential(torch.nn.Linear(2, 2), Packed())
    pipeline.vae = pipeline.video_vae  # alias, counted once

    assert execution._weight_dtypes(pipeline) == {
        "transformer": {"bfloat16": 20},
        "video_vae": {"float32": 6, "uint8": 8},
    }


def test_cached_step_count_sums_cached_transformers(monkeypatch):
    from vllm_omni.diffusion.cache.cachedit import backend

    cached, plain = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    monkeypatch.setattr(backend.BlockAdapter, "is_cached", staticmethod(lambda module: module is cached))
    monkeypatch.setattr(backend.cache_dit, "summary", lambda module, **kw: [SimpleNamespace(cached_steps=[3, 5, 9])])

    assert backend.cached_step_count(SimpleNamespace(_cache_dit_targets=(cached, plain))) == 3
    assert backend.cached_step_count(SimpleNamespace(_cache_dit_targets=(plain,))) is None


@pytest.mark.parametrize(
    ("completed", "status", "reason"), [(True, "completed", None), (False, "failed", "client_disconnect")]
)
def test_request_event(monkeypatch, completed, status, reason):
    lines = []
    monkeypatch.setattr(stats, "log_execution", lambda event, payload: lines.append((event, payload)))
    agg = OrchestratorAggregator(num_stages=1, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e=0)
    rid = "abc123-0a1b2c3d"
    agg.stage_events[rid] = [SimpleNamespace(stage_id=0, replica_id=0, final_output_type="video", batch_size=1)]
    execution_metrics = {"execution": {"diffusion": {"completed_steps": 50}, "runtime": {"world_size": 1}}}
    agg.accumulate_diffusion_metrics("diffusion", rid, SimpleNamespace(metrics=execution_metrics), stage_id=0)
    if completed:
        agg.e2e_done.add(rid)

    agg.log_execution_summary(rid, "abc123", reason)

    [(event, payload)] = lines
    assert event == "diffusion_request_completed"
    assert (payload["request_id"], payload["engine_request_id"], payload["status"]) == ("abc123", rid, status)
    assert payload["failure_reason"] == reason
    [stage] = payload["stages"]
    assert (stage["diffusion"], stage["runtime"]) == ({"completed_steps": 50}, {"world_size": 1})


@pytest.mark.parametrize("disable", [True, False])
@pytest.mark.parametrize(("mode", "expected"), [("iterate", 5), ("break", 2), ("update", 3)])
def test_progress_bar_counts_each_completed_step_once(disable, mode, expected):
    # mininterval=0/miniters=1 makes an enabled tqdm refresh (and call update()) on every item.
    pipeline = ProgressBarMixin()
    pipeline.set_progress_bar_config(disable=disable, mininterval=0, miniters=1, file=io.StringIO())

    with count_denoise_steps() as steps:
        if mode == "update":
            with pipeline.progress_bar(total=4) as bar:
                bar.update(2)
                bar.update()
        else:
            for index, _ in enumerate(pipeline.progress_bar(range(5))):
                if mode == "break" and index == 2:
                    break  # the third step's body did not finish

    assert steps[0] == expected
