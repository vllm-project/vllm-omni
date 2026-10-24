# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace

import pytest

from tests.scheduling.helpers import controller_context, drain
from vllm_omni.scheduling import controller as controller_module
from vllm_omni.scheduling import estimation as estimation_module
from vllm_omni.scheduling.estimation import estimate_service_time_s

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


def params(size=512, steps=20):
    return SimpleNamespace(width=size, height=size, num_inference_steps=steps, num_frames=1, num_outputs_per_prompt=1)


def admit(policy, request_id, sampling):
    return asyncio.create_task(policy.acquire(request_id, sampling, "QwenImagePipeline"))


@pytest.mark.asyncio
async def test_calibrated_service_estimates_drive_actual_dispatch_order(monkeypatch):
    monkeypatch.setattr(controller_module.time, "perf_counter", lambda: 100.0)
    async with controller_context([0]) as (policy, waiting):
        await policy.acquire("running", params(), "QwenImagePipeline")
        for request_id, sampling in (("short", params()), ("medium", params(1024, 25)), ("long", params(1536, 35))):
            waiting[request_id] = admit(policy, request_id, sampling)
        await drain()
        assert not any(task.done() for task in waiting.values())
        # Equal wait: the real 910B2 estimates are 8.60, 14.22 and 43.22 seconds.
        previous = "running"
        for expected in ("long", "medium", "short"):
            policy.complete(previous)
            await drain()
            assert [name for name, task in waiting.items() if task.done()] == [expected]
            assert (await waiting.pop(expected)).replica_id == 0
            previous = expected


@pytest.mark.parametrize("band,expected", [((3, 3), "newer"), ((2, 2), "older")])
@pytest.mark.asyncio
async def test_waiting_time_and_queue_depth_beta_change_next_dispatch(monkeypatch, band, expected):
    now = [100.0]
    monkeypatch.setattr(controller_module.time, "perf_counter", lambda: now[0])
    async with controller_context([0], band_risk_beta=0.2, band_min_pending=band[0], band_max_pending=band[1]) as (
        policy,
        waiting,
    ):
        await policy.acquire("running", params(), "QwenImagePipeline")
        waiting["older"] = admit(policy, "older", params())
        await drain()
        now[0] += 10.0
        waiting["newer"] = admit(policy, "newer", params(1536, 35))
        await drain()
        assert not any(task.done() for task in waiting.values())
        policy.complete("running")
        await drain()
        # At depth two, beta=.2 favors the older short request; beta=.85
        # outside the configured band favors the newer, longer request.
        assert [name for name, task in waiting.items() if task.done()] == [expected]
        assert (await waiting[expected]).replica_id == 0


@pytest.mark.parametrize("success,expected_replica", [(True, 0), (False, 1)])
@pytest.mark.asyncio
async def test_completion_feedback_changes_replica_choice_without_learning_failures(
    monkeypatch, success, expected_replica
):
    now = [0.0]
    monkeypatch.setattr(controller_module.time, "perf_counter", lambda: now[0])
    async with controller_context([0, 1]) as (policy, _):
        assert (await policy.acquire("slow", params(), "QwenImagePipeline")).replica_id == 0
        assert (await policy.acquire("fast", params(), "QwenImagePipeline")).replica_id == 1
        now[0] = 10.0
        policy.complete("fast")
        now[0] = 20.0
        policy.complete("slow")
        # Both slots are idle: successful latency feedback now beats ID order.
        assert (await policy.acquire("probe", params(), "QwenImagePipeline")).replica_id == 1
        now[0] = 220.0
        policy.complete("probe", success=success)
        # A slow success raises replica 1's EMA above replica 0; failure must
        # release capacity without teaching the scheduler that failed latency.
        assert (await policy.acquire("next", params(), "QwenImagePipeline")).replica_id == expected_replica


@pytest.mark.parametrize("model_class", ["QwenImagePipeline", "WanPipeline", "Wan22Pipeline"])
@pytest.mark.parametrize("profile", ["910B2", "910B3"])
def test_model_profile_exact_and_scaled_estimates(model_class, profile):
    qwen = model_class == "QwenImagePipeline"
    sampling = (
        params(1536, 35) if qwen else SimpleNamespace(width=1280, height=720, num_inference_steps=6, num_frames=80)
    )
    expected = (43.22 if profile == "910B2" else 49.34) if qwen else 119.71
    assert estimate_service_time_s(sampling, model_class, profile) == pytest.approx(expected)
    sampling.num_inference_steps *= 2
    sampling.num_frames = 1 if qwen else 81
    assert estimate_service_time_s(sampling, model_class, profile) == pytest.approx(expected * 2)


@pytest.mark.parametrize("profile,other_profile", [("910B2", "910B3"), ("910B3", "910B2")])
def test_wan_uses_selected_profile_for_exact_and_scaled_estimates(monkeypatch, profile, other_profile):
    sampling = SimpleNamespace(width=1280, height=720, num_inference_steps=6, num_frames=80)
    # Synthetic test data proves profile selection without claiming new calibration.
    anchors = dict(estimation_module._WAN_ANCHORS[profile])
    anchors[(1280, 720, 6, 81)] = 200.0
    monkeypatch.setitem(estimation_module._WAN_ANCHORS, profile, anchors)
    assert estimate_service_time_s(sampling, "Wan22Pipeline", profile) == pytest.approx(200.0)
    assert estimate_service_time_s(sampling, "Wan22Pipeline", other_profile) == pytest.approx(119.71)
    sampling.num_inference_steps = 12
    sampling.num_frames = 81
    assert estimate_service_time_s(sampling, "Wan22Pipeline", profile) == pytest.approx(200.0 * 2)
    assert estimate_service_time_s(sampling, "Wan22Pipeline", other_profile) == pytest.approx(119.71 * 2)


@pytest.mark.parametrize("profile", ["910B2", "910B3"])
@pytest.mark.parametrize(
    "width,steps,frames,expected", [(854, 3, 80, 38.07), (854, 4, 120, 71.34), (1280, 6, 80, 119.71)]
)
def test_wan_equivalent_execution_shapes_have_equal_estimates(profile, width, steps, frames, expected):
    raw = SimpleNamespace(
        width=width, height=480 if width == 854 else 720, num_inference_steps=steps, num_frames=frames
    )
    normalized = SimpleNamespace(**vars(raw))
    normalized.width = width // 16 * 16
    normalized.num_frames = frames // 4 * 4 + 1
    for sampling in (raw, normalized):
        assert estimate_service_time_s(sampling, "Wan22Pipeline", profile) == pytest.approx(expected)
    raw.num_inference_steps += 1
    normalized.num_inference_steps += 1
    assert estimate_service_time_s(raw, "Wan22Pipeline", profile) == pytest.approx(
        estimate_service_time_s(normalized, "Wan22Pipeline", profile)
    )


@pytest.mark.parametrize("model_class", ["QwenImagePipeline", "WanPipeline", "Wan22Pipeline"])
@pytest.mark.parametrize("profile", ["910B2", "910B3"])
@pytest.mark.asyncio
async def test_controller_uses_selected_model_profile_for_dispatch(monkeypatch, mocker, model_class, profile):
    monkeypatch.setattr(controller_module.time, "perf_counter", lambda: 100.0)
    if model_class == "QwenImagePipeline":
        preferred, other = params(), params(1536, 35)
        tables = estimation_module._QWEN_ANCHORS
        changed_anchor = {(512, 512): (20, 200.0)}
    else:
        preferred = SimpleNamespace(width=854, height=480, num_inference_steps=3, num_frames=80)
        other = SimpleNamespace(width=1280, height=720, num_inference_steps=6, num_frames=80)
        tables = estimation_module._WAN_ANCHORS
        changed_anchor = {(848, 480, 3, 81): 200.0}
    # Synthetic data reverses risk order only on the selected model/profile.
    # A default profile/model or an ignored estimate selects "other" instead.
    monkeypatch.setitem(tables, profile, tables[profile] | changed_anchor)
    estimate = mocker.spy(controller_module, "estimate_service_time_s")
    async with controller_context([0], hardware_profile=profile) as (policy, waiting):
        await policy.acquire("running", other, model_class)
        waiting["other"] = asyncio.create_task(policy.acquire("other", other, model_class))
        waiting["preferred"] = asyncio.create_task(policy.acquire("preferred", preferred, model_class))
        await drain()
        assert not any(task.done() for task in waiting.values())
        estimate.assert_any_call(preferred, model_class, profile)
        policy.complete("running")
        await drain()
        assert waiting["preferred"].done() and not waiting["other"].done()
        assert (await waiting["preferred"]).replica_id == 0
