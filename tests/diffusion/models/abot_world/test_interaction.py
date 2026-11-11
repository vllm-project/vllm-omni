# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ABot camera interaction on the stepwise path."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.interaction.registry import STRUCTURED_HANDLER_REGISTRY
from vllm_omni.diffusion.interaction.types import MAX_PENDING_INTERACTION_EVENTS, ChunkMediaSpec
from vllm_omni.diffusion.models.abot_world.interaction import ABotCameraInteractionHandler
from vllm_omni.diffusion.models.abot_world.pipeline_abot_world import _ABotRequestInputs
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _state(**extra) -> StepRequestState:
    return StepRequestState(request_id="r", sampling=OmniDiffusionSamplingParams(), extra=dict(extra))


def _hold(*transitions):
    return {
        "mode": "state",
        "transitions": [t if isinstance(t, dict) else {"actions": list(t)} for t in transitions],
    }


def _script(*frames):
    return {"mode": "script", "frames": [list(frame) for frame in frames]}


def _f(*frames: str) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(frame) for frame in frames)


def _run(chunks):
    """Enqueue each chunk's updates before its boundary; return actions and acks per chunk."""
    handler = ABotCameraInteractionHandler()
    state = _state()
    clock = 0.0
    event_index = 0
    actions, metadata = [], []
    for chunk_index, payloads in enumerate(chunks):
        for payload in payloads:
            clock += 1.0
            event_index += 1
            handler.enqueue(
                state,
                event_id=f"e{event_index}",
                received_at=clock,
                payload=payload,
                transition_chunks=None,
            )
        clock += 1.0
        metadata.append(
            handler.apply_at_chunk_boundary(state, boundary_at=clock, chunk_index=chunk_index, num_latent_frames=3)
        )
        actions.append(state.interaction_sessions["camera"].chunk_actions)
    return actions, metadata


# Expected actions were recorded from ABotCameraControlReducer, the typed-tick
# reducer this handler replaces, for the same update sequences.
REDUCER_PARITY = {
    "hold": ([[_hold("w")], [], []], [_f("w", "w", "w")] * 3),
    "pulse_in_one_event": ([[_hold("w", "")], []], [_f("w", "", ""), _f("", "", "")]),
    "pulse_across_events": ([[_hold("w"), _hold("")], []], [_f("w", "", ""), _f("", "", "")]),
    "hold_then_change": (
        [[_hold("w")], [_hold("wj")], [_hold("")], []],
        [_f("w", "w", "w"), _f("wj", "wj", "wj"), _f("", "", ""), _f("", "", "")],
    ),
    "script_fifo": (
        [[_script("w", "w", "a", "d", "j")], [], []],
        [_f("w", "w", "a"), _f("d", "j", ""), _f("", "", "")],
    ),
    "script_then_state_same_window": (
        [[_script("w", "w", "w", "w"), _hold("a")], []],
        [_f("a", "a", "a"), _f("a", "a", "a")],
    ),
    "state_then_script_same_window": (
        [[_hold("a"), _script("j", "k")], []],
        [_f("j", "k", ""), _f("", "", "")],
    ),
    "script_resets_held_keys": (
        [[_hold("w")], [_script("a")], [], [_hold("j")], []],
        [_f("w", "w", "w"), _f("a", "", ""), _f("", "", ""), _f("j", "j", "j"), _f("j", "j", "j")],
    ),
    "normalized_key_order": ([[_hold(["D", "w", "w", "a"])]], [_f("wad", "wad", "wad")]),
    "no_updates": ([[], []], [None, None]),
    "timestamped_transitions": (
        [
            [
                _hold(
                    {"actions": ["w"], "client_ts_ms": 10},
                    {"actions": ["w", "l"], "client_ts_ms": 20},
                    {"actions": [], "client_ts_ms": 30},
                )
            ],
            [],
        ],
        [_f("wl", "", ""), _f("", "", "")],
    ),
    "pulse_equal_to_final": ([[_hold("w", "a", "a")], []], [_f("a", "a", "a"), _f("a", "a", "a")]),
    "release_then_press": (
        [[_hold("w")], [_hold("", "s")], []],
        [_f("w", "w", "w"), _f("s", "s", "s"), _f("s", "s", "s")],
    ),
}


@pytest.mark.parametrize("name", sorted(REDUCER_PARITY))
def test_actions_match_the_typed_tick_reducer(name: str) -> None:
    chunks, expected = REDUCER_PARITY[name]
    actions, _ = _run(chunks)
    assert actions == expected


def test_state_update_is_acknowledged_at_its_boundary() -> None:
    _, metadata = _run([[_hold("w")], []])
    first, second = metadata
    assert first is not None
    assert first.started_event_ids == first.active_event_ids == first.completed_event_ids == ["e1"]
    assert first.coalesced_event_ids == []
    assert second is None


def test_script_stays_active_until_its_frames_are_consumed() -> None:
    _, metadata = _run([[_script("w", "w", "a", "d", "j")], [], []])
    first, second, third = metadata
    assert (first.started_event_ids, first.active_event_ids, first.completed_event_ids) == (["e1"], ["e1"], [])
    assert (second.started_event_ids, second.active_event_ids, second.completed_event_ids) == ([], ["e1"], ["e1"])
    assert third is None


def test_update_replaced_before_its_boundary_is_coalesced() -> None:
    _, (meta, _) = _run([[_hold("a"), _script("j", "k")], []])
    assert meta.coalesced_event_ids == ["e1"]
    assert meta.started_event_ids == meta.active_event_ids == ["e2"]


def test_active_script_replaced_mid_way_is_completed() -> None:
    _, (_, meta) = _run([[_script("w", "w", "w", "w")], [_hold("a")]])
    assert meta.completed_event_ids == ["e1", "e2"]
    assert meta.started_event_ids == meta.active_event_ids == ["e2"]
    assert meta.coalesced_event_ids == []


def test_update_after_the_boundary_waits_for_the_next_one() -> None:
    handler = ABotCameraInteractionHandler()
    state = _state()
    handler.enqueue(state, event_id="late", received_at=5.0, payload=_hold("w"), transition_chunks=None)

    assert handler.apply_at_chunk_boundary(state, boundary_at=4.0, num_latent_frames=3) is None
    assert state.interaction_sessions["camera"].chunk_actions is None

    meta = handler.apply_at_chunk_boundary(state, boundary_at=6.0, num_latent_frames=3)
    assert meta.completed_event_ids == ["late"]
    assert state.interaction_sessions["camera"].chunk_actions == _f("w", "w", "w")


@pytest.mark.parametrize(
    ("payload", "transition_chunks", "match"),
    [
        ({"mode": "hold", "transitions": []}, None, "mode='state' or mode='script'"),
        ({**_hold("w"), "extra": 1}, None, "take exactly mode and transitions"),
        ({"mode": "state", "transitions": []}, None, "non-empty transitions"),
        (_hold(["x"]), None, "W/A/S/D/I/J/K/L"),
        (
            _hold({"actions": ["w"], "client_ts_ms": 20}, {"actions": [], "client_ts_ms": 10}),
            None,
            "monotonic",
        ),
        ({"mode": "script", "frames": [["w"], "w"]}, None, "sequence of ABot action keys"),
        (_hold("w"), 2, "transition_chunks must be 0"),
    ],
)
def test_invalid_updates_are_rejected_without_queueing(payload, transition_chunks, match) -> None:
    handler = ABotCameraInteractionHandler()
    state = _state()
    with pytest.raises(ValueError, match=match):
        handler.validate_payload(state, event_id="e", payload=payload, transition_chunks=transition_chunks)
    with pytest.raises(ValueError, match=match):
        handler.enqueue(state, event_id="e", received_at=1.0, payload=payload, transition_chunks=transition_chunks)
    assert "camera" not in state.interaction_sessions


def test_live_updates_are_rejected_for_a_scripted_request() -> None:
    handler = ABotCameraInteractionHandler()
    state = _state(
        inputs=_ABotRequestInputs(
            prompt="p",
            image=torch.zeros(3, 512, 832),
            reference_images=None,
            camera_action_script=(_f("w", "", ""),),
            height=512,
            width=832,
            num_frames=9,
            num_latent_frames=3,
            num_frame_per_block=3,
            output_type="pil",
            max_sequence_length=512,
            flow_shift=5.0,
            generator=torch.Generator(),
        )
    )
    with pytest.raises(ValueError, match="camera_action_script"):
        handler.validate_payload(state, event_id="e", payload=_hold("w"), transition_chunks=None)


def test_pending_updates_are_bounded() -> None:
    handler = ABotCameraInteractionHandler()
    state = _state()
    for index in range(MAX_PENDING_INTERACTION_EVENTS):
        handler.enqueue(state, event_id=f"e{index}", received_at=10.0, payload=_hold("w"), transition_chunks=None)
    with pytest.raises(ValueError, match="Too many pending ABot camera interactions"):
        handler.validate_payload(state, event_id="extra", payload=_hold("w"), transition_chunks=None)


class _OpeningChunkPipeline:
    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        return ChunkMediaSpec(num_media_frames=9, fps=16.0, num_latent_frames=3)

    def apply_interaction_at_chunk_boundary(self, state: StepRequestState) -> None:
        pass

    def prepare_next_chunk(self, state: StepRequestState) -> None:
        pass


def test_registered_handler_applies_an_early_update_to_the_first_chunk() -> None:
    assert STRUCTURED_HANDLER_REGISTRY["ABotWorldCausalPipeline"] == {"camera": ABotCameraInteractionHandler}
    pipeline = _OpeningChunkPipeline()
    coordinator = InteractionCoordinator.build(
        pipeline, OmniDiffusionConfig(model_class_name="ABotWorldCausalPipeline")
    )
    assert coordinator.has_modality("camera") and coordinator.needs_chunk_media
    state = _state()

    coordinator.enqueue_parts(
        state,
        parts=[("camera", _hold("d"))],
        event_id="early",
        received_at=0.0,
        transition_chunks=None,
    )
    meta = coordinator.maybe_prepare_initial_session(state, pipeline)

    assert meta.completed_event_ids == ["early"]
    assert state.interaction_sessions["camera"].chunk_actions == _f("d", "d", "d")
