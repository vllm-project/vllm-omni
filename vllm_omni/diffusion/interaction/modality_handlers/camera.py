# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Camera modality interaction handlers.

Payload is structural SE(3): ``translation`` (xyz) + unit
quaternion ``rotation`` (x, y, z, w). WASD-style key tokens belong in
clients, not in the engine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, cast

import torch
from typing_extensions import Self, override

from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler
from vllm_omni.diffusion.interaction.types import (
    InteractionChunkMetadata,
    InteractionEvent,
    InteractionMode,
    InteractionPayload,
    InteractionSession,
    resolve_event_frame_offset,
)
from vllm_omni.diffusion.worker.utils import StepRequestState

Vec3 = tuple[float, float, float]
# Unit quaternion in ``(x, y, z, w)`` order. Identity is ``(0, 0, 0, 1)``.
Quat = tuple[float, float, float, float]

_IDENTITY_QUAT: Quat = (0.0, 0.0, 0.0, 1.0)


@dataclass
class CameraPose:
    """Canonical absolute camera pose used by the shared timeline.

    Coordinates follow Unity semantics: ``+X`` right, ``+Y`` up,
    ``+Z`` forward. Session start is the identity transform;
    ``target`` poses are relative to that origin.
    """

    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quat = _IDENTITY_QUAT

    @classmethod
    def identity(cls) -> CameraPose:
        return cls()

    def clone(self) -> CameraPose:
        return CameraPose(translation=self.translation, rotation=self.rotation)

    def as_matrix(self) -> torch.Tensor:
        """Return a 4x4 rigid transform from translation + unit quaternion."""
        mat = torch.eye(4, dtype=torch.float64)
        mat[:3, :3] = _quat_to_rotmat(self.rotation)
        mat[0, 3] = float(self.translation[0])
        mat[1, 3] = float(self.translation[1])
        mat[2, 3] = float(self.translation[2])
        return mat

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> CameraPose:
        if matrix.shape != (4, 4):
            raise ValueError("camera pose matrix must be 4x4")
        rotation = _rotmat_to_quat(matrix[:3, :3].detach().cpu().double())
        translation = (
            float(matrix[0, 3]),
            float(matrix[1, 3]),
            float(matrix[2, 3]),
        )
        return cls(translation=translation, rotation=rotation)


@dataclass(kw_only=True)
class QueuedCameraEvent(InteractionEvent):
    """Timestamped camera command waiting for the next chunk boundary."""

    # Absolute pose for ``target``; per-frame SE3 delta for ``velocity``.
    pose: CameraPose = field(default_factory=CameraPose.identity)


@dataclass
class CameraSession(InteractionSession):
    """Request-local camera timeline under ``state.interaction_sessions['camera']``."""

    current_pose: CameraPose = field(default_factory=CameraPose.identity)
    pending_events: list[QueuedCameraEvent] = field(default_factory=list)
    # In-flight command; ``elapsed_transition_chunks`` lives on the event.
    active_event: QueuedCameraEvent | None = None
    target_source: CameraPose | None = None
    # Absolute C2W poses ``[T, 4, 4]`` for the most recent chunk (pipeline may
    # rebuild model-native conditioning from these without double-relativizing).
    last_absolute_poses: torch.Tensor | None = None


class SE3DeltaCameraHandler(InteractionHandler):
    """Generic camera timeline that projects absolute poses to f2f SE3 deltas.

    * ``mode=target``: lerp toward an absolute pose (relative to session start).
    * ``mode=velocity``: hold a structural per-frame SE3 delta until replaced.

    ``state.conditioning['camera']`` receives dense frame-to-frame 4x4 deltas
    ``[T, 4, 4]``. Absolute poses for the chunk are stored on
    ``CameraSession.last_absolute_poses`` for pipelines that need them (e.g.
    LingBot plucker embedding via ``prepare_next_chunk``).
    """

    modality: ClassVar[str] = "camera"
    needs_chunk_media: ClassVar[bool] = True
    # Camera conditioning must exist every chunk; apply creates the session and
    # materializes identity/hold poses before any client enqueue.
    lazy_initialize_session: ClassVar[bool] = False
    default_transition_chunks: ClassVar[int] = 1

    @classmethod
    @override
    def from_pipeline(cls, pipeline: object) -> Self:
        """It doesn't need anything from the pipeline (e.g., `encode_prompt` method)"""
        del pipeline
        return cls()

    @override
    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        if not event_id:
            raise ValueError("event_id must be non-empty")

        mode = payload.get("mode", "target")
        if mode not in ("target", "velocity"):
            raise ValueError("camera mode must be 'target' or 'velocity'")
        camera_mode = cast(InteractionMode, mode)
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("camera data must be an object")
        if "actions" in data:
            raise ValueError(
                "camera data.actions (WASD keys) is not accepted by the engine; "
                "send structural translation/rotation instead"
            )

        pose = _parse_pose(data)
        if camera_mode == "target":
            duration = self.default_transition_chunks if transition_chunks is None else int(transition_chunks)
            if duration < 0:
                raise ValueError("transition_chunks must be >= 0")
        else:
            duration = 0

        session = _get_camera_session(state)
        with session.lock:
            session.pending_events.append(
                QueuedCameraEvent(
                    event_id=event_id,
                    received_at=received_at,
                    mode=camera_mode,
                    transition_chunks=duration,
                    pose=pose,
                )
            )

    @override
    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        boundary_at: float,
        chunk_index: int | None = None,
        num_frames: int | None = None,
        fps: float | None = None,
    ) -> InteractionChunkMetadata | None:
        del chunk_index
        if num_frames is None or fps is None:
            raise ValueError("SE3DeltaCameraHandler requires chunk num_frames and fps")
        session = _get_camera_session(state)
        with session.lock:
            samples, started, active, completed = self._step_one_chunk(
                session,
                num_frames=num_frames,
                fps=fps,
                boundary_at=boundary_at,
            )
            if samples:
                session.last_absolute_poses = torch.stack([p.as_matrix() for p in samples], dim=0)
            else:
                session.last_absolute_poses = torch.zeros((0, 4, 4), dtype=torch.float64)
            state.conditioning[self.modality] = self._project(samples)

        return InteractionChunkMetadata(
            started_event_ids=started,
            active_event_ids=active,
            completed_event_ids=completed,
        )

    def _step_one_chunk(
        self,
        session: CameraSession,
        *,
        num_frames: int,
        fps: float,
        boundary_at: float,
    ) -> tuple[list[CameraPose], list[str], list[str], list[str]]:
        """Sample absolute poses for this chunk under target/velocity semantics."""

        num_frames = max(int(num_frames), 1)
        pending = list(session.pending_events)
        session.pending_events.clear()

        resolved: list[tuple[int, QueuedCameraEvent]] = []
        for event in pending:
            frame = resolve_event_frame_offset(
                received_at=event.received_at,
                previous_boundary_at=session.last_boundary_at,
                num_frames=num_frames,
                fps=fps,
            )
            resolved.append((frame, event))
        resolved.sort(key=lambda item: item[0])

        by_frame: dict[int, list[QueuedCameraEvent]] = {}
        for frame, event in resolved:
            by_frame.setdefault(frame, []).append(event)

        started: list[str] = []
        completed: list[str] = []
        seen_completed: set[str] = set()
        poses: list[CameraPose] = []

        def _mark_completed(event_id: str) -> None:
            if event_id not in seen_completed:
                seen_completed.add(event_id)
                completed.append(event_id)

        for frame_idx in range(num_frames):
            for event in by_frame.get(frame_idx, []):
                started.append(event.event_id)
                cancelled_id = self._activate_event(session, event)
                if cancelled_id is not None:
                    _mark_completed(cancelled_id)
            just_completed = self._step_one_frame(session, num_frames)
            poses.append(session.current_pose.clone())
            if just_completed is not None:
                _mark_completed(just_completed)

        active: list[str] = []
        if session.active_event is not None:
            active.append(session.active_event.event_id)

        session.last_boundary_at = boundary_at
        return poses, started, active, completed

    def _activate_event(self, session: CameraSession, event: QueuedCameraEvent) -> str | None:
        """Activate ``event``, returning the event_id cancelled by this replacement if any."""
        cancelled_id: str | None = None
        if session.active_event is not None and session.active_event.event_id != event.event_id:
            cancelled_id = session.active_event.event_id

        event.elapsed_transition_chunks = 0.0
        session.active_event = event
        if event.mode == "target":
            session.target_source = session.current_pose.clone()
        else:
            session.target_source = None
        return cancelled_id

    def _clear_active_target(self, session: CameraSession) -> str:
        """Snap to the target pose, clear the active target track, and return its event_id."""
        assert session.active_event is not None and session.active_event.mode == "target"
        event = session.active_event
        event_id = event.event_id
        session.current_pose = event.pose.clone()
        session.active_event = None
        session.target_source = None
        return event_id

    def _step_one_frame(self, session: CameraSession, total_num_frames_this_chunk: int) -> str | None:
        """Advance the active camera command by one output frame."""
        event = session.active_event
        if event is None:
            return None

        if event.mode == "target":
            duration = event.transition_chunks
            source = session.target_source or session.current_pose
            target = event.pose
            if duration <= 0:
                return self._clear_active_target(session)

            event.elapsed_transition_chunks += 1.0 / float(total_num_frames_this_chunk)
            alpha = min(1.0, event.elapsed_transition_chunks / float(duration))
            session.current_pose = _lerp_pose(source, target, alpha)
            if event.elapsed_transition_chunks >= duration:
                return self._clear_active_target(session)
            return None

        # Velocity: apply the held structural SE3 delta once per output frame.
        session.current_pose = _compose_pose(session.current_pose, event.pose)
        return None

    def _project(self, poses: list[CameraPose]) -> torch.Tensor:
        """Pack absolute poses into dense frame-to-frame 4x4 SE3 deltas."""
        if not poses:
            return torch.zeros((0, 4, 4), dtype=torch.float64)
        mats = [p.as_matrix() for p in poses]
        deltas: list[torch.Tensor] = [torch.eye(4, dtype=torch.float64)]
        prev = mats[0]
        for mat in mats[1:]:
            # Relative rigid transform: T_delta = inv(T_prev) @ T_curr.
            deltas.append(torch.linalg.inv(prev) @ mat)
            prev = mat
        return torch.stack(deltas, dim=0)


def _get_camera_session(state: StepRequestState) -> CameraSession:
    session = state.interaction_sessions.get("camera")
    if isinstance(session, CameraSession):
        return session
    session = CameraSession()
    state.interaction_sessions["camera"] = session
    return session


def _as_xyz(value: object, *, name: str) -> Vec3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"camera {name} must be a length-3 list/tuple")
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_quat(value: object, *, name: str) -> Quat:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"camera {name} must be a length-4 list/tuple (x, y, z, w)")
    return _quat_normalize((float(value[0]), float(value[1]), float(value[2]), float(value[3])))


def _parse_pose(data: Mapping[str, object]) -> CameraPose:
    translation = _as_xyz(data.get("translation", (0.0, 0.0, 0.0)), name="translation")
    rotation = _as_quat(data.get("rotation", _IDENTITY_QUAT), name="rotation")
    return CameraPose(translation=translation, rotation=rotation)


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def _lerp_pose(source: CameraPose, target: CameraPose, alpha: float) -> CameraPose:
    alpha = min(1.0, max(0.0, alpha))
    return CameraPose(
        translation=cast(
            Vec3,
            tuple(_lerp(s, t, alpha) for s, t in zip(source.translation, target.translation)),
        ),
        rotation=_quat_nlerp(source.rotation, target.rotation, alpha),
    )


def _compose_pose(base: CameraPose, delta: CameraPose) -> CameraPose:
    """Apply ``delta`` after ``base`` in the session frame (T_new = T_base @ T_delta)."""
    return CameraPose.from_matrix(base.as_matrix() @ delta.as_matrix())


def _quat_normalize(q: Quat) -> Quat:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("camera rotation quaternion must be non-zero")
    return (x / norm, y / norm, z / norm, w / norm)


def _quat_nlerp(a: Quat, b: Quat, alpha: float) -> Quat:
    """Normalized linear interpolation; flips ``b`` when needed for shortest path."""
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if dot < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
    return _quat_normalize(
        (
            _lerp(a[0], b[0], alpha),
            _lerp(a[1], b[1], alpha),
            _lerp(a[2], b[2], alpha),
            _lerp(a[3], b[3], alpha),
        )
    )


def _quat_to_rotmat(q: Quat) -> torch.Tensor:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.tensor(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=torch.float64,
    )


def _rotmat_to_quat(rotation: torch.Tensor) -> Quat:
    """Convert a 3x3 rotation matrix to a unit quaternion ``(x, y, z, w)``."""
    m = rotation
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (float(m[2, 1]) - float(m[1, 2])) / s
        y = (float(m[0, 2]) - float(m[2, 0])) / s
        z = (float(m[1, 0]) - float(m[0, 1])) / s
    elif float(m[0, 0]) > float(m[1, 1]) and float(m[0, 0]) > float(m[2, 2]):
        s = math.sqrt(1.0 + float(m[0, 0]) - float(m[1, 1]) - float(m[2, 2])) * 2.0
        w = (float(m[2, 1]) - float(m[1, 2])) / s
        x = 0.25 * s
        y = (float(m[0, 1]) + float(m[1, 0])) / s
        z = (float(m[0, 2]) + float(m[2, 0])) / s
    elif float(m[1, 1]) > float(m[2, 2]):
        s = math.sqrt(1.0 + float(m[1, 1]) - float(m[0, 0]) - float(m[2, 2])) * 2.0
        w = (float(m[0, 2]) - float(m[2, 0])) / s
        x = (float(m[0, 1]) + float(m[1, 0])) / s
        y = 0.25 * s
        z = (float(m[1, 2]) + float(m[2, 1])) / s
    else:
        s = math.sqrt(1.0 + float(m[2, 2]) - float(m[0, 0]) - float(m[1, 1])) * 2.0
        w = (float(m[1, 0]) - float(m[0, 1])) / s
        x = (float(m[0, 2]) + float(m[2, 0])) / s
        y = (float(m[1, 2]) + float(m[2, 1])) / s
        z = 0.25 * s
    return _quat_normalize((x, y, z, w))


__all__ = [
    "CameraPose",
    "CameraSession",
    "QueuedCameraEvent",
    "SE3DeltaCameraHandler",
]
