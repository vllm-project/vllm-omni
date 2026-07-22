# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ServingRLRollout (RFC #3747, P0).

The DreamZero engine is mocked. Tests verify session lifecycle, step_id
semantics, committed_step_id invariants, and error paths without a GPU.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.protocol.rollout import (
    Action,
    CreateSessionRequest,
    Observation,
    RolloutStepRequest,
)
from vllm_omni.entrypoints.openai.serving_rl_rollout import ServingRLRollout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_result(video: Any = None) -> MagicMock:
    result = MagicMock()
    if video is None:
        video = np.zeros((1, 3, 64, 64), dtype=np.float32)
    result.multimodal_output = {"video": video}
    return result


def _make_openpi(video: Any = None, raises: Exception | None = None):
    openpi = MagicMock()
    openpi.engine_client = MagicMock()

    async def _generate(*_, **__):
        if raises:
            raise raises
        yield _make_fake_result(video)

    openpi.engine_client.generate = _generate
    openpi.build_request = MagicMock(
        return_value=MagicMock(
            prompts=[""],
            request_id="test-req",
            sampling_params=MagicMock(),
        )
    )
    return openpi


@pytest.fixture
def serving() -> ServingRLRollout:
    return ServingRLRollout(_make_openpi())


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session(serving):
    resp = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    assert resp.session_id
    assert resp.mode == "world_model_env"


@pytest.mark.asyncio
async def test_get_status_after_create(serving):
    resp = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    status = await serving.get_status(resp.session_id)
    assert status.committed_step_id == -1
    assert status.context_length == 0
    assert not status.closed


@pytest.mark.asyncio
async def test_close_session(serving):
    resp = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    await serving.close_session(resp.session_id)
    # closed session state is tested at the session-store level;
    # HTTP 410 behavior is verified via route integration tests


# ---------------------------------------------------------------------------
# Step - success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_step_advances_committed_step_id(serving):
    sess = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    req = RolloutStepRequest(
        step_id=0,
        observation=Observation(state=[0.1, 0.2]),
        action=Action(joint_positions=[0.0] * 7),
    )
    resp = await serving.step(sess.session_id, req)
    assert resp.error is None
    assert resp.model_metadata.committed_step_id == 0

    status = await serving.get_status(sess.session_id)
    assert status.committed_step_id == 0
    assert status.context_length == 1


@pytest.mark.asyncio
async def test_repeated_steps_advance_monotonically(serving):
    sess = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    for i in range(3):
        req = RolloutStepRequest(
            step_id=i,
            observation=Observation(state=[float(i)]),
            action=Action(joint_positions=[0.0] * 7),
        )
        resp = await serving.step(sess.session_id, req)
        assert resp.error is None
        assert resp.model_metadata.committed_step_id == i

    status = await serving.get_status(sess.session_id)
    assert status.committed_step_id == 2
    assert status.context_length == 3


@pytest.mark.asyncio
async def test_next_observation_present_on_success(serving):
    sess = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    req = RolloutStepRequest(
        step_id=0,
        observation=Observation(),
        action=Action(),
    )
    resp = await serving.step(sess.session_id, req)
    assert resp.next_observation is not None
    assert "video" in resp.next_observation
    assert resp.next_observation["dtype"] == "float32"


@pytest.mark.asyncio
async def test_duplicate_step_does_not_rerun_or_advance(serving):
    sess = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    req = RolloutStepRequest(step_id=0, observation=Observation(), action=Action())
    assert (await serving.step(sess.session_id, req)).error is None

    resp = await serving.step(sess.session_id, req)

    assert resp.error is not None
    assert resp.error.code == "step_already_committed"
    assert resp.model_metadata.committed_step_id == 0
    assert resp.model_metadata.context_length == 1
    serving._openpi.build_request.assert_called_once()


@pytest.mark.asyncio
async def test_out_of_order_step_does_not_rerun_or_advance(serving):
    sess = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    req = RolloutStepRequest(step_id=2, observation=Observation(), action=Action())

    resp = await serving.step(sess.session_id, req)

    assert resp.error is not None
    assert resp.error.code == "step_out_of_order"
    assert resp.model_metadata.committed_step_id == -1
    assert resp.model_metadata.context_length == 0
    serving._openpi.build_request.assert_not_called()


@pytest.mark.asyncio
async def test_stateless_step_does_not_commit_context(serving):
    sess = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    req = RolloutStepRequest(
        step_id=7,
        observation=Observation(),
        action=Action(),
        use_session_context=False,
    )

    resp = await serving.step(sess.session_id, req)

    assert resp.error is None
    assert resp.model_metadata.committed_step_id == -1
    assert resp.model_metadata.context_length == 0
    serving._openpi.build_request.assert_called_once()
    kwargs = serving._openpi.build_request.call_args.kwargs
    assert kwargs["reset"] is True
    assert kwargs["session_id"] != sess.session_id
    assert kwargs["session_id"].startswith(f"{sess.session_id}:stateless:")


# ---------------------------------------------------------------------------
# Step - failure path: committed_step_id must NOT advance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_step_does_not_advance_committed_step_id():
    serving = ServingRLRollout(_make_openpi(raises=RuntimeError("engine dead")))
    sess = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    req = RolloutStepRequest(step_id=0, observation=Observation(), action=Action())
    resp = await serving.step(sess.session_id, req)

    assert resp.error is not None
    assert resp.error.code == "inference_error"
    assert resp.model_metadata.committed_step_id == -1  # must not have advanced

    status = await serving.get_status(sess.session_id)
    assert status.committed_step_id == -1


@pytest.mark.asyncio
async def test_failed_step_after_success_preserves_last_committed():
    good_serving = ServingRLRollout(_make_openpi())
    sess = await good_serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    # Step 0 succeeds
    await good_serving.step(sess.session_id, RolloutStepRequest(step_id=0, observation=Observation(), action=Action()))

    # Swap engine to failing one - reuse the same store via internal reference
    good_serving._openpi = _make_openpi(raises=RuntimeError("oops"))
    resp = await good_serving.step(
        sess.session_id, RolloutStepRequest(step_id=1, observation=Observation(), action=Action())
    )
    assert resp.error is not None
    assert resp.model_metadata.committed_step_id == 0  # still at last good step
    assert resp.model_metadata.context_length == 1


# ---------------------------------------------------------------------------
# Step - unknown session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_unknown_session_returns_error(serving):
    req = RolloutStepRequest(step_id=0, observation=Observation(), action=Action())
    resp = await serving.step("no-such-session", req)
    assert resp.error is not None
    assert resp.error.code == "session_not_found"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_clears_committed_step_id(serving):
    sess = await serving.create_session(CreateSessionRequest(model="dreamzero", mode="world_model_env"))
    await serving.step(sess.session_id, RolloutStepRequest(step_id=0, observation=Observation(), action=Action()))
    await serving.reset_session(sess.session_id)
    status = await serving.get_status(sess.session_id)
    assert status.committed_step_id == -1
    assert status.context_length == 0


# ---------------------------------------------------------------------------
# action -> obs folding (module assumption)
# ---------------------------------------------------------------------------


def test_merge_action_into_obs_concatenates_state():
    from vllm_omni.entrypoints.openai.serving_rl_rollout import _merge_action_into_obs

    obs = Observation(state=[1.0, 2.0], prompt="test")
    action = Action(joint_positions=[3.0, 4.0, 5.0])
    result = _merge_action_into_obs(obs, action)

    assert result["prompt"] == "test"
    np.testing.assert_array_almost_equal(result["state"], [1.0, 2.0, 3.0, 4.0, 5.0])


def test_merge_action_none_uses_obs_state_only():
    from vllm_omni.entrypoints.openai.serving_rl_rollout import _merge_action_into_obs

    obs = Observation(state=[1.0, 2.0])
    result = _merge_action_into_obs(obs, None)
    np.testing.assert_array_almost_equal(result["state"], [1.0, 2.0])


def test_merge_no_state_no_action_omits_state_key():
    from vllm_omni.entrypoints.openai.serving_rl_rollout import _merge_action_into_obs

    obs = Observation()
    result = _merge_action_into_obs(obs, None)
    assert "state" not in result
