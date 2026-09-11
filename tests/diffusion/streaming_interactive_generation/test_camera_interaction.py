# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for camera interaction handlers and coordinator registration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.modality_handlers.camera import CameraSession, SE3DeltaCameraHandler
from vllm_omni.diffusion.interaction.modality_handlers.prompt import PromptSession
from vllm_omni.diffusion.interaction.registry import STRUCTURED_HANDLER_REGISTRY
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec, resolve_event_frame_offset
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_FORWARD_VELOCITY = {
    "mode": "velocity",
    "data": {"translation": [0.0, 0.0, 0.05], "rotation": [0.0, 0.0, 0.0, 1.0]},
}


class _FakePromptPipeline(InteractionMixin):
    """Minimal pipeline stub with prompt-update support."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.transformer = SimpleNamespace(dtype=torch.float32)
        self.encode_prompt = MagicMock(return_value=(torch.full((1, 4, 2), 2.0), None))
        self._interaction_coordinator = None

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        del state
        return ChunkMediaSpec(num_frames=8, fps=16.0)


class _FakeLingBotPipeline(InteractionMixin):
    """Minimal pipeline stub for LingBot camera registration."""

    def __init__(self) -> None:
        self._interaction_coordinator = None

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        del state
        return ChunkMediaSpec(num_frames=3, fps=3.0)


def _make_state(*, request_id: str = "req-1") -> StepRequestState:
    state = StepRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(num_outputs_per_prompt=1, max_sequence_length=226),  # pyright: ignore[reportArgumentType]
        prompt="hello",
    )
    state.prompt_embeds = torch.zeros(1, 4, 2)
    state.extra = {}
    return state


def _boundary_at(previous_boundary_at: float | None, num_frames: int, fps: float) -> float:
    if previous_boundary_at is None:
        return 0.0
    if fps <= 0:
        return previous_boundary_at
    return previous_boundary_at + num_frames / fps


class TestCoordinatorResolution:
    def test_helios_has_prompt_but_no_camera(self) -> None:
        pipeline = _FakePromptPipeline()
        od_config = SimpleNamespace(model_class_name="HeliosPipeline")
        coordinator = InteractionCoordinator.build(pipeline, od_config)

        assert coordinator.has_modality("prompt")
        assert not coordinator.has_modality("camera")
        assert "camera" not in STRUCTURED_HANDLER_REGISTRY["HeliosPipeline"]

    def test_lingbot_registers_se3_camera_handler(self) -> None:
        pipeline = _FakeLingBotPipeline()
        od_config = SimpleNamespace(model_class_name="LingBotWorldCausalDMDPipeline")
        coordinator = InteractionCoordinator.build(pipeline, od_config)

        assert coordinator.has_modality("camera")
        handler = coordinator.get_handler("camera")
        assert isinstance(handler, SE3DeltaCameraHandler)
        assert handler.lazy_initialize_session is False

    def test_non_lazy_camera_apply_creates_session_without_enqueue(self) -> None:
        """Camera apply must run before any enqueue to materialize identity poses."""
        pipeline = _FakeLingBotPipeline()
        od_config = SimpleNamespace(model_class_name="LingBotWorldCausalDMDPipeline")
        coordinator = InteractionCoordinator.build(pipeline, od_config)
        state = _make_state()

        assert "camera" not in state.interaction_sessions
        meta = coordinator.maybe_prepare_initial_session(state, pipeline)
        session = state.interaction_sessions["camera"]
        assert isinstance(session, CameraSession)
        assert session.last_absolute_poses is not None
        assert session.last_absolute_poses.shape == (3, 4, 4)
        assert state.conditioning["camera"].shape == (3, 4, 4)
        assert meta.started_event_ids == []
        assert meta.active_event_ids == []

    def test_lazy_prompt_apply_skips_without_enqueue(self) -> None:
        pipeline = _FakePromptPipeline()
        od_config = SimpleNamespace(model_class_name="HeliosPipeline")
        coordinator = InteractionCoordinator.build(pipeline, od_config)
        state = _make_state()

        meta = coordinator.apply_at_chunk_boundary(state, boundary_at=0.0, chunk_index=0)
        assert "prompt" not in state.interaction_sessions
        assert meta.started_event_ids == []
        assert meta.active_event_ids == []
        assert meta.completed_event_ids == []

        initial = coordinator.maybe_prepare_initial_session(state, pipeline)
        assert "prompt" not in state.interaction_sessions
        assert initial.started_event_ids == []

    def test_unsupported_modality_includes_model_context(self) -> None:
        pipeline = _FakePromptPipeline()
        od_config = SimpleNamespace(model_class_name="HeliosPipeline")
        coordinator = InteractionCoordinator.build(pipeline, od_config)

        with pytest.raises(ValueError, match="HeliosPipeline"):
            coordinator.enqueue(
                _make_state(),
                modality="camera",
                event_id="cam-1",
                received_at=0.0,
                payload=_FORWARD_VELOCITY,
                transition_chunks=None,
            )

    def test_enqueue_parts_rejects_unsupported_modality_atomically(self) -> None:
        """Unsupported tracks in a composite event must not mutate any queues."""
        pipeline = _FakePromptPipeline()
        od_config = SimpleNamespace(model_class_name="HeliosPipeline")
        coordinator = InteractionCoordinator.build(pipeline, od_config)
        state = _make_state()
        coordinator.enqueue(
            state,
            modality="prompt",
            event_id="prior",
            received_at=0.0,
            payload={"prompt": "already-queued"},
            transition_chunks=1,
        )
        prior_session = state.interaction_sessions["prompt"]
        assert isinstance(prior_session, PromptSession)
        prior = prior_session.pending_event
        assert prior is not None
        pipeline.encode_prompt.reset_mock()

        with pytest.raises(ValueError, match="camera"):
            coordinator.enqueue_parts(
                state,
                parts=[
                    ("prompt", {"prompt": "should-not-replace"}),
                    ("camera", _FORWARD_VELOCITY),
                ],
                event_id="composite-bad",
                received_at=1.0,
                transition_chunks=2,
            )

        pipeline.encode_prompt.assert_not_called()
        pending = prior_session.pending_event
        assert pending is not None
        assert pending.event_id == "prior"
        assert pending.prompt == "already-queued"
        assert "camera" not in state.interaction_sessions


class TestCameraHandlers:
    def test_rejects_wasd_actions_payload(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        with pytest.raises(ValueError, match="actions"):
            handler.enqueue(
                state,
                event_id="cam-wasd",
                received_at=0.0,
                payload={"mode": "velocity", "data": {"actions": ["w"]}},
                transition_chunks=None,
            )

    def test_target_transition_progress_and_se3_projection(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="cam-1",
            received_at=0.0,
            payload={
                "mode": "target",
                "data": {"translation": [0.0, 0.0, 3.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=2,
        )

        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=4,
            fps=16.0,
            boundary_at=0.25,
        )
        tensor = state.conditioning["camera"]
        assert meta is not None
        assert meta.started_event_ids == ["cam-1"]
        assert meta.active_event_ids == ["cam-1"]
        assert meta.completed_event_ids == []
        assert tensor.shape == (4, 4, 4)

        meta2 = handler.apply_at_chunk_boundary(
            state,
            chunk_index=1,
            num_frames=4,
            fps=16.0,
            boundary_at=0.5,
        )
        assert meta2 is not None
        assert meta2.started_event_ids == []
        assert meta2.active_event_ids == []
        assert meta2.completed_event_ids == ["cam-1"]
        session = state.interaction_sessions["camera"]
        assert isinstance(session, CameraSession)
        assert session.active_event is None
        assert session.current_pose.translation[2] == pytest.approx(3.0)
        assert session.last_absolute_poses is not None
        assert session.last_absolute_poses.shape == (4, 4, 4)

    def test_single_chunk_target_completes_once(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="cam-fast",
            received_at=0.0,
            payload={
                "mode": "target",
                "data": {"translation": [1.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=1,
        )

        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=8,
            fps=16.0,
            boundary_at=0.5,
        )
        assert meta is not None
        assert meta.started_event_ids == ["cam-fast"]
        assert meta.active_event_ids == []
        assert meta.completed_event_ids == ["cam-fast"]

        meta2 = handler.apply_at_chunk_boundary(
            state,
            chunk_index=1,
            num_frames=8,
            fps=16.0,
            boundary_at=1.0,
        )
        assert meta2 is not None
        assert meta2.started_event_ids == []
        assert meta2.active_event_ids == []
        assert meta2.completed_event_ids == []

    def test_velocity_composes_structural_se3_delta(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="vel-1",
            received_at=0.0,
            payload=_FORWARD_VELOCITY,
            transition_chunks=None,
        )

        meta0 = handler.apply_at_chunk_boundary(state, chunk_index=0, num_frames=3, fps=16.0, boundary_at=0.0)
        assert meta0 is not None
        assert meta0.started_event_ids == ["vel-1"]
        assert meta0.active_event_ids == ["vel-1"]
        session = state.interaction_sessions["camera"]
        assert isinstance(session, CameraSession)
        assert session.current_pose.translation[2] == pytest.approx(0.15)
        assert session.last_absolute_poses is not None
        assert session.last_absolute_poses.shape == (3, 4, 4)
        conditioning = state.conditioning["camera"]
        assert conditioning.shape == (3, 4, 4)
        torch.testing.assert_close(conditioning[0], torch.eye(4), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            conditioning[1, :3, 3],
            torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64),
            atol=1e-6,
            rtol=0,
        )

        meta1 = handler.apply_at_chunk_boundary(state, chunk_index=1, num_frames=2, fps=16.0, boundary_at=0.125)
        assert meta1 is not None
        assert meta1.active_event_ids == ["vel-1"]
        assert session.current_pose.translation[2] == pytest.approx(0.25)

    def test_velocity_completed_when_replaced_by_target(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        handler.enqueue(
            state,
            event_id="vel-1",
            received_at=0.0,
            payload=_FORWARD_VELOCITY,
            transition_chunks=None,
        )
        meta0 = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=2,
            fps=16.0,
            boundary_at=0.0,
        )
        assert meta0 is not None
        assert meta0.active_event_ids == ["vel-1"]

        handler.enqueue(
            state,
            event_id="tgt-1",
            received_at=0.1,
            payload={
                "mode": "target",
                "data": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=0,
        )
        meta1 = handler.apply_at_chunk_boundary(
            state,
            chunk_index=1,
            num_frames=2,
            fps=16.0,
            boundary_at=0.125,
        )
        assert meta1 is not None
        assert meta1.started_event_ids == ["tgt-1"]
        assert "vel-1" in meta1.completed_event_ids
        assert meta1.active_event_ids == []

    def test_same_chunk_velocity_then_target_by_frame_offset(self) -> None:
        handler = SE3DeltaCameraHandler()
        state = _make_state()
        fps = 10.0
        boundary = 0.0
        handler.enqueue(
            state,
            event_id="vel",
            received_at=0.0,
            payload=_FORWARD_VELOCITY,
            transition_chunks=None,
        )
        handler.enqueue(
            state,
            event_id="tgt",
            received_at=0.5,
            payload={
                "mode": "target",
                "data": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
            },
            transition_chunks=0,
        )
        state.interaction_sessions["camera"].last_boundary_at = boundary
        meta = handler.apply_at_chunk_boundary(
            state,
            chunk_index=0,
            num_frames=10,
            fps=fps,
            boundary_at=_boundary_at(boundary, 10, fps),
        )
        assert meta is not None
        assert meta.started_event_ids == ["vel", "tgt"]
        assert meta.completed_event_ids == ["vel", "tgt"]
        assert meta.active_event_ids == []
        session = state.interaction_sessions["camera"]
        assert isinstance(session, CameraSession)
        assert session.active_event is None
        assert session.current_pose.translation[2] == pytest.approx(0.0)


class TestResolveEventFrameOffset:
    def test_clamps_and_floors(self) -> None:
        assert resolve_event_frame_offset(received_at=0.0, previous_boundary_at=None, num_frames=8, fps=16.0) == 0
        assert resolve_event_frame_offset(received_at=0.1, previous_boundary_at=0.0, num_frames=8, fps=16.0) == 1
        assert resolve_event_frame_offset(received_at=10.0, previous_boundary_at=0.0, num_frames=8, fps=16.0) == 7
        assert resolve_event_frame_offset(received_at=-1.0, previous_boundary_at=0.0, num_frames=8, fps=16.0) == 0
