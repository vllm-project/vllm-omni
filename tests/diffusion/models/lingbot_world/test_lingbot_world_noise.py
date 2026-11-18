# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native finite-horizon RNG order, with legacy unbounded ticks kept separate."""

import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.models.lingbot_world import test_pipeline_lingbot_world as base
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def pipeline_dependencies(monkeypatch):
    base._stub_pipeline_dependencies.__wrapped__(monkeypatch)


def _reference(seed, chunks, *, finite):
    """Native C,T,H,W draw order, independent of pipeline noise helpers."""
    generator = torch.Generator().manual_seed(seed)
    full = torch.randn(16, chunks * 3, 2, 2, generator=generator) if finite else None
    fixture = Path(__file__).with_name("fixtures") / "lingbot_world_official_dmd_schedule.json.fixture"
    schedule = json.loads(fixture.read_text())["schedules"]["5"]
    initial, outputs = [], []
    for index in range(chunks):
        current = full[:, index * 3 : (index + 1) * 3] if finite else torch.randn(16, 3, 2, 2, generator=generator)
        initial.append(current.clone().unsqueeze(0))
        for step_index, step in enumerate(schedule):
            # The recording transformer predicts constant flow=1.
            current = (current.double() - step["x0_sigma_first_match"]).float()
            if step_index != len(schedule) - 1:
                sigma = schedule[step_index + 1]["renoise_sigma_second_match"]
                current = (1 - sigma) * current + sigma * torch.randn(current.shape, generator=generator)
        outputs.append(current.unsqueeze(0))
    return initial, torch.cat(outputs, 2), generator.get_state()


def _sampling(chunks, seed):
    frames = chunks * 12 - 3
    sampling = base._SamplingParams(num_frames=frames, seed=seed)
    sampling.extra_args["_lingbot_camera_trajectory"] = base._CameraTrajectory(
        poses=torch.eye(4).repeat(frames, 1, 1),
        intrinsics=torch.tensor([[100.0, 100.0, 8.0, 8.0]]).repeat(frames, 1),
    )
    return sampling


def _pipeline():
    transformer = base._RecordingTransformer()
    pipeline = base._pipeline(base.lingbot_pipeline, transformer=transformer)
    pipeline._ar_height = pipeline._ar_width = 16
    return pipeline, transformer


@pytest.mark.parametrize("chunks", [1, 2, 17])
@pytest.mark.parametrize("seed", [17, 42])
def test_finite_request_and_stepwise_match_native_initial_renoise_and_next_draw(chunks, seed):
    expected_initial, expected_output, expected_state = _reference(seed, chunks, finite=True)
    request_pipeline, request_transformer = _pipeline()
    sampling = _sampling(chunks, seed)
    result = request_pipeline(base._request(sampling=sampling))
    torch.testing.assert_close(result.output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(sampling.generator.get_state(), expected_state, rtol=0, atol=0)
    for index, initial in enumerate(expected_initial):
        torch.testing.assert_close(
            request_transformer.calls[index * 5]["hidden_states"][:, :16], initial, rtol=0, atol=0
        )

    step_pipeline, step_transformer = _pipeline()
    state = base._stepwise_state(num_frames=chunks * 12 - 3, seed=seed)
    with step_pipeline.bind_ar_diffusion_state(state.request_id, base._FakeARState(state.request_id)):
        step_pipeline.prepare_encode(state)
        buffer = weakref.ref(state.extra["initial_noise"])
        step_pipeline.prepare_next_chunk(state)
        outputs = []
        while not state.request_denoise_completed:
            flow = step_pipeline.denoise_step(None, states=[state])
            step_pipeline.step_scheduler(state, flow)
            if state.chunk_denoise_completed:
                outputs.append(step_pipeline.post_decode(state).output["payload"]["latents"])
                if not state.request_denoise_completed:
                    step_pipeline.prepare_next_chunk(state)
    torch.testing.assert_close(torch.cat(outputs, 2), expected_output, rtol=0, atol=0)
    torch.testing.assert_close(state.sampling.generator.get_state(), expected_state, rtol=0, atol=0)
    for index, initial in enumerate(expected_initial):
        torch.testing.assert_close(step_transformer.calls[index * 5]["hidden_states"][:, :16], initial, rtol=0, atol=0)
    assert "initial_noise" not in state.extra
    gc.collect()
    assert buffer() is None  # Last slice stopped retaining the full-horizon allocation.
    # Equality of the final RNG state also establishes equality of the next draw.
    next_generator = torch.Generator().set_state(expected_state)
    torch.testing.assert_close(
        torch.randn(37, generator=state.sampling.generator), torch.randn(37, generator=next_generator), rtol=0, atol=0
    )


@pytest.mark.parametrize("seed", [17, 42])
def test_two_unknown_horizon_ticks_keep_legacy_incremental_generator_state(seed):
    expected_initial, expected_output, expected_state = _reference(seed, 2, finite=False)
    pipeline, transformer = _pipeline()
    ar_state = base._FakeARState("world-1")
    outputs = []
    with pipeline.bind_ar_diffusion_state("world-1", ar_state):
        for index in range(2):
            sampling = base._SamplingParams(extra_args=base._tick_extra_args(chunk_index=index), seed=seed)
            result = pipeline(base._request(sampling=sampling))
            outputs.append(result.output["payload"]["latents"])
    torch.testing.assert_close(torch.cat(outputs, 2), expected_output, rtol=0, atol=0)
    torch.testing.assert_close(pipeline._ar_sessions["world-1"].generator_state, expected_state, rtol=0, atol=0)
    for index, initial in enumerate(expected_initial):
        torch.testing.assert_close(transformer.calls[index * 5]["hidden_states"][:, :16], initial, rtol=0, atol=0)


def test_independent_requests_own_distinct_noise_and_abort_releases_retired_batch():
    pipeline, _ = _pipeline()
    states = [base._stepwise_state(request_id=name, num_frames=21, seed=42) for name in ("a", "b")]
    for state in states:
        with pipeline.bind_ar_diffusion_state(state.request_id, base._FakeARState(state.request_id)):
            pipeline.prepare_encode(state)
            pipeline.prepare_next_chunk(state)
    a, b = states
    assert a.extra["initial_noise"].data_ptr() != b.extra["initial_noise"].data_ptr()
    torch.testing.assert_close(a.extra["initial_noise"], b.extra["initial_noise"], rtol=0, atol=0)
    a_buffer = weakref.ref(a.extra["initial_noise"])
    b_buffer = weakref.ref(b.extra["initial_noise"])
    runner = SimpleNamespace(
        state_cache={"a": a, "b": b},
        pipeline=pipeline,
        input_batch=SimpleNamespace(request_ids=["a", "b"], states=states),
    )
    DiffusionModelRunner.release_step_requests(runner, ["a"])
    assert runner.input_batch is None and set(runner.state_cache) == {"b"}
    del a, states
    gc.collect()
    assert a_buffer() is None
    assert b_buffer() is not None
    # Advancing/resetting an unrelated request did not change B's RNG state.
    expected = torch.Generator().manual_seed(42)
    torch.randn(16, 6, 2, 2, generator=expected)
    torch.testing.assert_close(b.sampling.generator.get_state(), expected.get_state(), rtol=0, atol=0)
