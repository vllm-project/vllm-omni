"""Verify production early-stop against the checked-out Framework sampler."""

import ast
import json
import os
import runpy
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_has_pipeline import Harness, denoise
from test_policy_http import request_item

from vllm_omni.entrypoints.cosmos3.action_ready_stream import ActionReadyPublisher
from vllm_omni.entrypoints.cosmos3.remote_action_stream import ActionStreamChannel
from vllm_omni.entrypoints.cosmos3.vllm_omni_action import VllmOmniActionTransport
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo


@pytest.fixture(scope="module")
def native_sampler():
    reference = os.environ.get("COSMOS3_FRAMEWORK_ROOT")
    if not reference:
        pytest.skip("Optional native numerical alignment requires explicit COSMOS3_FRAMEWORK_ROOT")
    root = Path(reference)
    path = root / "cosmos_framework/utils/action_euler.py"
    if not path.exists():
        pytest.skip("Set COSMOS3_FRAMEWORK_ROOT to run native numerical alignment")
    # Execute the original helpers/classes, excluding imports of training deps.
    namespace = runpy.run_path(str(root / "cosmos_framework/utils/has_schedule.py"))
    namespace.update(
        dataclass=dataclass,
        Callable=Callable,
        ActionTimeSchedule=str,
        progress_bar=lambda iterator, **kwargs: iterator,
    )
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef | ast.FunctionDef)]
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ActionEulerSampler"](), namespace["ActionLayout"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("schedule", ["has_euler", "constant_euler"])
@pytest.mark.parametrize("count", [1, 4, 16])
def test_native_numerics_and_real_forward_count(native_sampler, device, schedule, count):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sampler, layout = native_sampler
    calls = []

    def velocity(samples, global_time, action_times):
        times = action_times if action_times is not None else global_time.expand(1, 16)
        actions = samples[0][2:].reshape(16, 64)
        prediction = actions + 1 + times[0, :, None] / 1000
        prediction[:, 10:] = 0
        calls.append(times.clone())
        return [torch.cat([torch.ones(2, device=device), prediction.flatten()])]

    expected = sampler(
        velocity,
        [torch.zeros(2 + 16 * 64, device=device)],
        num_steps=10,
        action_layouts=[layout(offset=2, horizon=16, action_dim=64)],
        schedule=schedule,
        early_stop_actions=count,
    )[0]
    harness, metrics = Harness(), {}
    video, actual = denoise(harness, device=device, schedule=schedule, early_stop=count, metrics=metrics)
    steps = {1: 1, 4: 5, 16: 10}[count] if schedule == "has_euler" else 10
    assert len(harness.calls) == len(calls) == steps
    assert metrics == {"denoise_steps": steps, "requested_denoise_steps": 10}
    torch.testing.assert_close(video.flatten(), expected[:2], rtol=0, atol=0)
    torch.testing.assert_close(actual.flatten(), expected[2:], rtol=0, atol=0)
    _, full = denoise(Harness(), device=device, schedule=schedule)
    torch.testing.assert_close(actual[:, :count], full[:, :count], rtol=0, atol=0)
    assert harness.reset_count == 1


@pytest.mark.parametrize("count", [1, 4, 16])
def test_stream_prefix_is_complete_before_early_return(monkeypatch, count):
    harness, events = Harness(), []
    original = ActionReadyPublisher.__init__

    def initialize(self, *args):
        original(self, *args)
        self.session.close()

        def post(url, *, json, timeout):
            events.append(json)
            return SimpleNamespace(raise_for_status=lambda: None)

        self.session = SimpleNamespace(post=post, close=lambda: None)

    monkeypatch.setattr(ActionReadyPublisher, "__init__", initialize)
    _, actual = denoise(
        harness,
        early_stop=count,
        stream={"version": 1, "callback_url": "http://127.0.0.1/internal/action_events/test"},
    )
    channel = ActionStreamChannel(lambda event: None, count)
    for event in events:
        channel.accept(event)
    channel.verify_final(actual[0, :count, :10].numpy())
    assert [index for event in events for index in event["action_indices"]] == list(range(count))
    assert events[-1]["step"] == len(harness.calls)


@pytest.mark.parametrize("count", [0, -1, 17, True, 1.5, "4"])
def test_invalid_counts_do_not_run_model(count):
    harness = Harness()
    with pytest.raises(ValueError, match="early_stop_actions"):
        denoise(harness, early_stop=count)
    assert not harness.calls
    with pytest.raises(ValueError, match="early_stop_actions"):
        VllmOmniActionTransport("http://localhost", early_stop_actions=count)


def test_default_scheduler_rejects_early_stop():
    with pytest.raises(ValueError, match="Euler"):
        denoise(Harness(), early_stop=4, schedule="default")


@pytest.mark.parametrize("schedule,stream", [("has_euler", False), ("has_euler", True), ("constant_euler", False)])
def test_http_prefix_reaches_worker_and_final_response(bridge, schedule, stream):
    url, state, session, _ = bridge
    transport = state.service.transport
    transport.early_stop_actions = 4
    transport.action_time_schedule = schedule
    if schedule == "constant_euler":
        transport.policy_profile = "libero10_native_constant_v1"
    if stream:
        state.service.cfg = replace(state.service.cfg, streaming_enabled=True, callback_base_url=url)
    response = session.post(
        url + ("/predict_stream" if stream else "/predict_batch"),
        json=request_item() if stream else {"items": [request_item(), request_item(90)]},
        timeout=5,
    )
    response.raise_for_status()
    result = json.loads(response.text.splitlines()[-1]) if stream else response.json()
    chunks = [result["action"]] if stream else result["actions"]
    assert all(np.shape(chunk) == (4, 10) for chunk in chunks)
    for fields, _ in state.posts:
        extra = json.loads(fields["extra_params"])
        assert extra["early_stop_actions"] == 4 and extra["action_chunk_size"] == 16
    expected_steps = 5 if schedule == "has_euler" else 10
    assert all(item["denoise_steps"] == expected_steps for item in result["performance"]["worker_metrics"])
    assert result["performance"]["generated_actions"] == 4 * len(chunks)
    assert session.get(url + "/info", timeout=3).json()["early_stop_actions"] == 4
    assert len(state.deleted) == len(chunks) and not state.service.streams.channels


def test_action_metadata_survives_video_serialization():
    action = OmniOpenAIServingVideo._make_video_action(
        torch.zeros(1, 4, 10),
        {
            "early_stop_actions": 4,
            "denoise_steps": 5,
            "requested_denoise_steps": 10,
            "policy_timing": {"infer_ms": 7.5},
            "timing_contract": "moh_sample_actions_v1",
        },
    )
    assert action.shape == [1, 4, 10]
    assert action.early_stop_actions == 4 and action.denoise_steps == 5 and action.requested_denoise_steps == 10
    assert action.model_dump()["policy_timing"] == {"infer_ms": 7.5}
    assert action.timing_contract == "moh_sample_actions_v1"


def test_old_worker_cannot_silently_ignore_early_stop(bridge):
    url, state, session, _ = bridge
    state.service.transport.early_stop_actions = 4
    state.confirm_early_stop = False
    response = session.post(url + "/predict", json=request_item(), timeout=5)
    assert response.status_code == 400
    assert "did not confirm early-stop" in response.json()["error"]
    assert len(state.deleted) == 1
