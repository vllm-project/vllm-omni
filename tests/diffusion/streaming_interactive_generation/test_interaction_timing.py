# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for engine-clock control windows and continuous camera motion."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.modality_handlers.camera import CameraSession, SE3DeltaCameraHandler
from vllm_omni.diffusion.interaction.modality_handlers.prompt import PromptInteractionHandler, prompt_update_versions
from vllm_omni.diffusion.interaction.types import (
    ChunkMediaSpec,
    InteractionBoundaryContext,
    InteractionChunkMetadata,
    merge_interaction_metadata,
)
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def state() -> StepRequestState:
    return StepRequestState(
        request_id="paced",
        sampling=OmniDiffusionSamplingParams(streaming_buffer_seconds=1.25),
        interaction_boundary=InteractionBoundaryContext(boundary_at=1.0, window_opened_at=0.0, window_end_at=1.0),
        chunk_index=1,
    )


def enqueue(handler, state, event_id, at, *, mode="velocity", z=0.05, transition=0):
    handler.enqueue(
        state,
        event_id=event_id,
        received_at=at,
        payload={"mode": mode, "data": {"translation": [0, 0, z], "rotation": [0, 0, 0, 1]}},
        transition_chunks=transition,
    )


def apply(handler, state, *, frames=3, fps=3.0):
    return handler.apply_at_chunk_boundary(
        state,
        boundary_at=state.interaction_boundary.boundary_at,
        num_media_frames=frames,
        num_latent_frames=3,
        fps=fps,
    )


def translations(state):
    return state.interaction_sessions["camera"].last_absolute_poses[:, 2, 3]


def _enqueue_rotation(handler, state, event_id, at, rotation, translation):
    handler.enqueue(
        state,
        event_id=event_id,
        received_at=at,
        payload={"mode": "velocity", "data": {"translation": translation, "rotation": rotation}},
        transition_chunks=0,
    )


@pytest.mark.parametrize("translation", [[0, 0, 0], [0.1, -0.2, 0.3]])
@pytest.mark.parametrize("rotation", [[0.0, 0.0, 0.5, 0.8660254037844386], [0.0, 1.0, 0.0, 0.0]])
def test_paced_rotation_velocity_is_invariant_to_event_segmentation(state, rotation, translation):
    single = SE3DeltaCameraHandler()
    _enqueue_rotation(single, state, "single", 0.0, rotation, translation)
    apply(single, state)
    single_matrix = state.interaction_sessions["camera"].last_absolute_poses.clone()

    split_state = StepRequestState(
        request_id="split",
        sampling=OmniDiffusionSamplingParams(streaming_buffer_seconds=1.25),
        interaction_boundary=InteractionBoundaryContext(boundary_at=1.0, window_opened_at=0.0, window_end_at=1.0),
        chunk_index=1,
    )
    split = SE3DeltaCameraHandler()
    _enqueue_rotation(split, split_state, "first", 0.0, rotation, translation)
    _enqueue_rotation(split, split_state, "second", 0.15, rotation, translation)
    apply(split, split_state)

    torch.testing.assert_close(
        single_matrix,
        split_state.interaction_sessions["camera"].last_absolute_poses,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("with_target", [False, True])
def test_short_press_survives_pure_and_mixed_control_windows(state, with_target):
    handler = SE3DeltaCameraHandler()
    if with_target:
        enqueue(handler, state, "target", 0.0, mode="target", z=0)
    enqueue(handler, state, "press", 0.1)
    enqueue(handler, state, "release", 0.2, z=0)
    meta = apply(handler, state)
    torch.testing.assert_close(translations(state), torch.full((3,), 0.015, dtype=torch.float64))
    assert meta.started_event_ids == (["target"] if with_target else []) + ["press", "release"]


@pytest.mark.parametrize("at,eligible", [(0.999, True), (1.0, False), (1.001, False)])
def test_velocity_cutoff_is_exclusive(state, at, eligible):
    handler = SE3DeltaCameraHandler()
    enqueue(handler, state, "velocity", at)
    meta = apply(handler, state)
    assert meta.started_event_ids == (["velocity"] if eligible else [])
    assert len(state.interaction_sessions["camera"].pending_events) == (0 if eligible else 1)


def test_target_updates_coalesce_without_dropping_intervening_velocity(state):
    handler = SE3DeltaCameraHandler()
    enqueue(handler, state, "obsolete", 0.0, mode="target", z=100)
    enqueue(handler, state, "target", 0.01, mode="target", z=1)
    enqueue(handler, state, "press", 0.1)
    enqueue(handler, state, "release", 0.2, z=0)
    meta = apply(handler, state)
    assert meta.coalesced_event_ids == ["obsolete"]
    assert meta.started_event_ids == ["target", "press", "release"]
    torch.testing.assert_close(translations(state), torch.full((3,), 1.015, dtype=torch.float64))


def test_long_pause_does_not_integrate_wall_time(state):
    handler = SE3DeltaCameraHandler()
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=100, window_opened_at=0, window_end_at=1)
    enqueue(handler, state, "press", 0.0)
    apply(handler, state)
    assert translations(state)[-1].item() == pytest.approx(0.15)


def test_short_press_deferred_across_pause_preserves_its_duration(state):
    handler = SE3DeltaCameraHandler()
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=10, window_opened_at=0, window_end_at=1)
    enqueue(handler, state, "press", 2.0)
    enqueue(handler, state, "release", 2.1, z=0)
    assert apply(handler, state).started_event_ids == []
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=11, window_opened_at=10, window_end_at=11)
    assert apply(handler, state).started_event_ids == ["press", "release"]
    torch.testing.assert_close(translations(state), torch.full((3,), 0.015, dtype=torch.float64))
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=12, window_opened_at=11, window_end_at=12)
    assert apply(handler, state).started_event_ids == []
    assert translations(state)[-1].item() == pytest.approx(0.015)


@pytest.mark.parametrize("fresh_at", [10.05, 10.2])
def test_deferred_and_fresh_short_presses_keep_both_intervals(state, fresh_at):
    handler = SE3DeltaCameraHandler()
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=10, window_opened_at=0, window_end_at=1)
    enqueue(handler, state, "deferred-press", 2)
    enqueue(handler, state, "deferred-release", 2.1, z=0)
    apply(handler, state)
    enqueue(handler, state, "fresh-press", fresh_at)
    enqueue(handler, state, "fresh-release", fresh_at + 0.1, z=0)
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=11, window_opened_at=10, window_end_at=11)
    meta = apply(handler, state)
    assert meta.started_event_ids == ["deferred-press", "deferred-release", "fresh-press", "fresh-release"]
    assert translations(state)[-1].item() == pytest.approx(0.03)


def test_overflow_preserves_the_fresh_press_for_following_window(state):
    handler = SE3DeltaCameraHandler()
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=10, window_opened_at=0, window_end_at=1)
    enqueue(handler, state, "deferred-press", 2)
    enqueue(handler, state, "deferred-release", 3, z=0)
    apply(handler, state)
    enqueue(handler, state, "fresh-press", 10.2)
    enqueue(handler, state, "fresh-release", 10.3, z=0)
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=11, window_opened_at=10, window_end_at=11)
    assert apply(handler, state).started_event_ids == ["deferred-press"]
    assert translations(state)[-1].item() == pytest.approx(0.15)
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=12, window_opened_at=11, window_end_at=12)
    assert apply(handler, state).started_event_ids == ["deferred-release", "fresh-press", "fresh-release"]
    assert translations(state)[-1].item() == pytest.approx(0.165)


def test_late_target_applies_on_resume_and_supersedes_older_deferred_velocity(state):
    handler = SE3DeltaCameraHandler()
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=10, window_opened_at=0, window_end_at=1)
    enqueue(handler, state, "stale-velocity", 2)
    enqueue(handler, state, "late-target", 5, mode="target", z=2)
    meta = apply(handler, state)
    assert meta.started_event_ids == ["late-target"]
    assert meta.coalesced_event_ids == ["stale-velocity"]
    assert not state.interaction_sessions["camera"].pending_events
    assert translations(state)[-1].item() == pytest.approx(2)


def test_target_at_resume_cutoff_is_deferred(state):
    handler = SE3DeltaCameraHandler()
    enqueue(handler, state, "target", 1.0, mode="target", z=2)
    assert apply(handler, state).started_event_ids == []
    assert [event.event_id for event in state.interaction_sessions["camera"].pending_events] == ["target"]


def test_first_short_chunk_scales_held_velocity_by_actual_media_duration(state):
    handler = SE3DeltaCameraHandler()
    state.interaction_boundary = InteractionBoundaryContext(
        boundary_at=9 / 16, window_opened_at=0, window_end_at=9 / 16
    )
    enqueue(handler, state, "hold", 0)
    apply(handler, state, frames=12, fps=16)
    assert translations(state)[-1].item() == pytest.approx(0.05 * (9 / 16) / (12 / 16 / 3))


def test_more_than_three_events_integrate_instead_of_displacing_events(state):
    handler = SE3DeltaCameraHandler()
    for index in range(8):
        enqueue(handler, state, str(index), index * 0.05, z=0.05 if index % 2 == 0 else 0)
    meta = apply(handler, state)
    assert meta.started_event_ids == [str(index) for index in range(8)]
    assert translations(state)[-1].item() == pytest.approx(0.03)


def test_coordinator_uses_engine_boundary_without_sampling_worker_clock(state, monkeypatch):
    import vllm_omni.diffusion.interaction.coordinator as coordinator_module
    import vllm_omni.diffusion.interaction.mixin as mixin_module

    def forbidden(*_args):
        raise AssertionError("worker-local boundary clock sampled")

    monkeypatch.setattr(mixin_module, "synchronized_monotonic_time", forbidden)
    monkeypatch.setattr(coordinator_module, "synchronized_monotonic_time", forbidden)

    class Pipeline(InteractionMixin):
        def peek_chunk_media(self, state):
            return ChunkMediaSpec(3, 3.0, 3)

    pipeline = Pipeline()
    pipeline._interaction_coordinator = InteractionCoordinator({"camera": SE3DeltaCameraHandler()})
    pipeline._interaction_coordinator.maybe_prepare_initial_session(state, pipeline)
    pipeline.apply_interaction_at_chunk_boundary(state)
    assert isinstance(state.interaction_sessions["camera"], CameraSession)
    assert state.interaction_sessions["camera"].last_boundary_at == 1.0


@pytest.mark.parametrize(
    "context", [(-1, None, None), (float("nan"), None, None), (1, 0, None), (0.5, 0, 1), (1, 1, 1)]
)
def test_invalid_boundary_context_is_rejected(context):
    with pytest.raises(ValueError):
        InteractionBoundaryContext(*context)


def test_coalesced_metadata_merges_with_deduplication():
    result = merge_interaction_metadata(
        [
            InteractionChunkMetadata(coalesced_event_ids=["a", "b"]),
            InteractionChunkMetadata(coalesced_event_ids=["b", "c"]),
        ]
    )
    assert result.as_dict()["coalesced_event_ids"] == ["a", "b", "c"]


def test_target_transition_completes_once_on_media_grid(state):
    handler = SE3DeltaCameraHandler()
    enqueue(handler, state, "target", 0, mode="target", z=1, transition=1)
    meta = apply(handler, state)
    torch.testing.assert_close(translations(state), torch.tensor([1 / 3, 2 / 3, 1], dtype=torch.float64))
    assert meta.completed_event_ids == ["target"]
    assert meta.active_event_ids == []
    state.interaction_boundary = InteractionBoundaryContext(boundary_at=2, window_opened_at=1, window_end_at=2)
    assert apply(handler, state).completed_event_ids == []


def test_prompt_last_write_uses_ingress_order_and_keeps_revision_updates(state):
    def encode_prompt(*, prompt, **kwargs):
        return torch.full((1, 2, 3), float(prompt)), None

    handler = PromptInteractionHandler(encode_prompt=encode_prompt, device="cpu", dtype=torch.float32)
    state.prompt_embeds = torch.zeros(1, 2, 3)
    for event_id, stamp, prompt in (("first", 0.1, "1"), ("latest", 0.3, "3"), ("reordered", 0.2, "2")):
        handler.enqueue(state, event_id=event_id, received_at=stamp, payload={"prompt": prompt}, transition_chunks=0)
    assert prompt_update_versions([state]) == (0,)
    meta = handler.apply_at_chunk_boundary(state, boundary_at=1)
    assert meta.coalesced_event_ids == ["first", "reordered"]
    assert meta.started_event_ids == ["latest"]
    assert meta.completed_event_ids == ["latest"]
    assert prompt_update_versions([state]) == (1,)
    torch.testing.assert_close(state.prompt_embeds, torch.full((1, 2, 3), 3.0))
    again = handler.apply_at_chunk_boundary(state, boundary_at=2)
    assert again.coalesced_event_ids == []
    assert again.completed_event_ids == []
    assert prompt_update_versions([state]) == (1,)


@pytest.mark.parametrize("mode", ["target", "velocity"])
def test_camera_pending_controls_are_bounded_until_boundary(state, mode):
    handler = SE3DeltaCameraHandler()
    for index in range(256):
        enqueue(handler, state, str(index), index / 1000, mode=mode)
    with pytest.raises(ValueError, match="pending camera interactions"):
        enqueue(handler, state, "overflow", 0.5, mode=mode)
    session = state.interaction_sessions["camera"]
    assert len(session.pending_events) == 256
    meta = apply(handler, state)
    assert set(meta.started_event_ids + meta.coalesced_event_ids) == {str(index) for index in range(256)}
    assert not session.pending_events
    enqueue(handler, state, "after-boundary", 1.1, mode=mode)
    assert [event.event_id for event in session.pending_events] == ["after-boundary"]


def test_prompt_pending_metadata_is_bounded_before_encoding(state, mocker):
    encode_prompt = mocker.MagicMock(return_value=(torch.ones(1, 2, 3), None))
    handler = PromptInteractionHandler(encode_prompt=encode_prompt, device="cpu", dtype=torch.float32)
    state.prompt_embeds = torch.zeros(1, 2, 3)
    for index in range(256):
        handler.enqueue(
            state, event_id=str(index), received_at=index / 1000, payload={"prompt": str(index)}, transition_chunks=0
        )
    with pytest.raises(ValueError, match="pending prompt interactions"):
        handler.enqueue(
            state, event_id="overflow", received_at=0.5, payload={"prompt": "overflow"}, transition_chunks=0
        )
    assert encode_prompt.call_count == 256
    session = state.interaction_sessions["prompt"]
    assert len(session.coalesced_event_ids) == 255
    assert session.pending_event.event_id == "255"
    meta = handler.apply_at_chunk_boundary(state, boundary_at=1)
    assert meta.started_event_ids == ["255"]
    assert meta.coalesced_event_ids == [str(index) for index in range(255)]
    handler.enqueue(state, event_id="after-boundary", received_at=1.1, payload={"prompt": "after"}, transition_chunks=0)
    assert session.pending_event.event_id == "after-boundary"


def test_full_camera_queue_rejects_composite_before_prompt_encoding(state, mocker):
    camera = SE3DeltaCameraHandler()
    for index in range(256):
        enqueue(camera, state, str(index), index / 1000)
    encode_prompt = mocker.MagicMock(return_value=(torch.ones(1, 2, 3), None))
    prompt = PromptInteractionHandler(encode_prompt=encode_prompt, device="cpu", dtype=torch.float32)
    state.prompt_embeds = torch.zeros(1, 2, 3)
    coordinator = InteractionCoordinator({"prompt": prompt, "camera": camera})
    with pytest.raises(ValueError, match="pending camera interactions"):
        coordinator.enqueue_parts(
            state,
            event_id="overflow",
            received_at=0.5,
            transition_chunks=0,
            parts=[
                ("prompt", {"prompt": "must not encode"}),
                ("camera", {"mode": "velocity", "data": {"translation": [0, 0, 0]}}),
            ],
        )
    encode_prompt.assert_not_called()
    assert "prompt" not in state.interaction_sessions
    assert len(state.interaction_sessions["camera"].pending_events) == 256
