# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest
import torch

from vllm_omni.diffusion.models.lingbot_world.actions import (
    LINGBOT_CAMERA_ACTION_SCHEMA,
    LINGBOT_CAMERA_TRAJECTORY_SCHEMA,
    LingBotCameraControlReducer,
    integrate_lingbot_camera_actions,
    parse_lingbot_camera_action_frames,
)
from vllm_omni.experimental.ar_diffusion.session import (
    ARDiffusionSession,
    ARDiffusionSessionEvent,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _action_control(mode: str, **data) -> ARDiffusionControlInput:
    return ARDiffusionControlInput(
        track="camera",
        schema=LINGBOT_CAMERA_ACTION_SCHEMA,
        data={"mode": mode, **data},
    )


def _prepared_frames(prepared) -> tuple[tuple[str, ...], ...]:
    assert len(prepared.controls) == 1
    return prepared.controls[0].data["frames"]


def test_state_controls_hold_release_and_preserve_short_pulse() -> None:
    reducer = LingBotCameraControlReducer()
    pressed = reducer.prepare(
        current_controls={},
        events=(
            ARDiffusionSessionEvent(
                event_id=1,
                controls=(
                    _action_control(
                        "state",
                        transitions=[{"actions": ["W"], "client_ts_ms": 10}],
                    ),
                ),
            ),
        ),
        chunk_index=0,
    )
    assert _prepared_frames(pressed) == (("w",), ("w",), ("w",))
    assert pressed.controls[0].to_dict()["data"]["frames"] == [["w"], ["w"], ["w"]]
    reducer.commit(pressed)

    held = reducer.prepare(
        current_controls={"camera": pressed.controls[0]},
        events=(),
        chunk_index=1,
    )
    assert _prepared_frames(held) == (("w",), ("w",), ("w",))
    reducer.commit(held)

    pulse = reducer.prepare(
        current_controls={"camera": held.controls[0]},
        events=(
            ARDiffusionSessionEvent(
                event_id=2,
                controls=(
                    _action_control(
                        "state",
                        transitions=[
                            {"actions": ["j"], "client_ts_ms": 20},
                            {"actions": [], "client_ts_ms": 21},
                        ],
                    ),
                ),
            ),
        ),
        chunk_index=2,
    )
    assert _prepared_frames(pulse) == (("j",), (), ())
    reducer.commit(pulse)

    released = reducer.prepare(
        current_controls={"camera": pulse.controls[0]},
        events=(),
        chunk_index=3,
    )
    assert _prepared_frames(released) == ((), (), ())


def test_script_is_transactional_consumed_once_and_neutral_padded() -> None:
    reducer = LingBotCameraControlReducer()
    event = ARDiffusionSessionEvent(
        event_id=1,
        controls=(
            _action_control(
                "script",
                frames=[["w"], ["d"], ["l"], ["i"]],
            ),
        ),
    )
    first_attempt = reducer.prepare(
        current_controls={},
        events=(event,),
        chunk_index=0,
    )
    assert _prepared_frames(first_attempt) == (("w",), ("d",), ("l",))

    # An uncommitted failed attempt must not consume the reducer's script.
    second_attempt = reducer.prepare(
        current_controls={},
        events=(event,),
        chunk_index=0,
    )
    assert _prepared_frames(second_attempt) == (("w",), ("d",), ("l",))
    reducer.commit(second_attempt)

    tail = reducer.prepare(
        current_controls={"camera": second_attempt.controls[0]},
        events=(),
        chunk_index=1,
    )
    assert _prepared_frames(tail) == (("i",), (), ())
    reducer.commit(tail)
    exhausted = reducer.prepare(
        current_controls={"camera": tail.controls[0]},
        events=(),
        chunk_index=2,
    )
    assert _prepared_frames(exhausted) == ((), (), ())


def test_trajectory_control_remains_an_expert_override() -> None:
    reducer = LingBotCameraControlReducer()
    trajectory = ARDiffusionControlInput(
        track="camera",
        schema=LINGBOT_CAMERA_TRAJECTORY_SCHEMA,
        data={"poses": [[[1.0]]], "intrinsics": [[1.0]]},
    )
    prepared = reducer.prepare(
        current_controls={},
        events=(
            ARDiffusionSessionEvent(
                event_id=1,
                controls=(trajectory,),
            ),
        ),
        chunk_index=0,
    )
    assert prepared.controls == (trajectory,)


def test_action_integrator_matches_lingbot_motion_and_calibration() -> None:
    trajectory, pitch = integrate_lingbot_camera_actions(
        [["w"], ["d"], ["l"]],
        width=832,
        height=480,
    )

    assert pitch == 0.0
    torch.testing.assert_close(
        trajectory.poses[0, :3, 3],
        torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        trajectory.poses[1, :3, 3],
        torch.tensor([0.05, 0.0, 0.05], dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )
    expected_yaw = torch.tensor(
        [
            [0.9945219, 0.0, 0.10452846],
            [0.0, 1.0, 0.0],
            [-0.10452846, 0.0, 0.9945219],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        trajectory.poses[2, :3, :3],
        expected_yaw,
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        trajectory.intrinsics[0],
        torch.tensor([500.0, 500.0, 416.0, 240.0]),
    )

    resized, _ = integrate_lingbot_camera_actions(
        [[]],
        width=416,
        height=240,
    )
    scaled = resized.intrinsics[0] * torch.tensor([416 / 832, 240 / 480, 416 / 832, 240 / 480])
    torch.testing.assert_close(
        scaled,
        torch.tensor([500.0, 500.0, 208.0, 120.0]),
    )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"mode": "frames", "frames": [["x"], [], []]}, "W/A/S/D/I/J/K/L"),
        ({"mode": "frames", "frames": [["w"]]}, "exactly one action list"),
        ({"mode": "state", "frames": [[], [], []]}, "mode='frames'"),
    ],
)
def test_model_tick_action_payload_validation(data, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_lingbot_camera_action_frames(data, expected_frames=3)


class _TickConsumer:
    def __init__(self) -> None:
        self.ticks: list[ARDiffusionTickRequest] = []

    async def execute_tick(self, tick: ARDiffusionTickRequest) -> ARDiffusionChunkMetadata:
        self.ticks.append(tick)
        return ARDiffusionChunkMetadata.from_tick(tick)

    def chunk_metadata(self, output: ARDiffusionChunkMetadata) -> ARDiffusionChunkMetadata:
        return output


class _Lifecycle:
    async def reset_session(self, session_id: str) -> None:
        del session_id

    async def close_session(self, session_id: str) -> None:
        del session_id


def test_session_reducer_keeps_prompt_and_actions_atomic_and_resets() -> None:
    async def exercise() -> None:
        consumer = _TickConsumer()
        reducer = LingBotCameraControlReducer()
        session = ARDiffusionSession(
            "world-actions",
            tick_consumer=consumer,
            lifecycle=_Lifecycle(),
            max_pending_events=8,
            request_id_factory=lambda session_id, chunk_index: f"{session_id}-{chunk_index}",
            control_reducer=reducer,
        )
        await session.accept_event(
            ARDiffusionSessionEvent(
                event_id=1,
                prompt="turn toward the window",
                controls=(
                    _action_control(
                        "state",
                        transitions=[{"actions": ["d"]}],
                    ),
                ),
            )
        )
        await session.next_chunk()

        tick = consumer.ticks[0]
        assert tick.prompt == "turn toward the window"
        assert tick.applied_event_ids == (1,)
        assert tick.controls[0].data["frames"] == (("d",), ("d",), ("d",))

        await session.reset()
        await session.accept_event(
            ARDiffusionSessionEvent(
                event_id=2,
                controls=(
                    _action_control(
                        "state",
                        transitions=[{"actions": []}],
                    ),
                ),
            )
        )
        await session.next_chunk()
        assert consumer.ticks[1].chunk_index == 0
        assert consumer.ticks[1].controls[0].data["frames"] == ((), (), ())

    asyncio.run(exercise())
