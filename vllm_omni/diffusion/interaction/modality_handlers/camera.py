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
    # Integer frame progress for ``target`` lerps (avoids float reciprocal drift).
    elapsed_transition_frames: int = 0


@dataclass
class CameraSession(InteractionSession):
    """Request-local camera timeline under ``state.interaction_sessions['camera']``."""

    current_pose: CameraPose = field(default_factory=CameraPose.identity)
    pending_events: list[QueuedCameraEvent] = field(default_factory=list)
    # In-flight command; ``elapsed_transition_chunks`` lives on the event.
    active_event: QueuedCameraEvent | None = None
    target_source: CameraPose | None = None
    # Absolute C2W poses ``[num_latent_frames, 4, 4]`` for the most recent chunk.
    last_absolute_poses: torch.Tensor | None = None
    # Since a camera session is not lazily initialized, use this field to mark that a client enqueue is ever received.
    has_received_input: bool = False


class SE3DeltaCameraHandler(InteractionHandler):
    """Generic camera timeline that samples absolute poses per frame.

    * ``mode=target``: lerp toward an absolute pose (relative to session start).
    * ``mode=velocity``: hold a structural SE3 delta until replaced.

    ``received_at`` is resolved on the media grid (``num_media_frames``/``fps``), then mapped onto latent steps.
    Later integration and ``last_absolute_poses`` aligns with ``num_latent_frames``.
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
    def validate_payload(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        if not event_id:
            raise ValueError("event_id must be non-empty")

        mode = payload.get("mode", "target")
        if mode not in ("target", "velocity"):
            raise ValueError("camera mode must be 'target' or 'velocity'")
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("camera data must be an object")
        if "actions" in data:
            raise ValueError(
                "camera data.actions (WASD keys) is not accepted by the engine; "
                "send structural translation/rotation instead"
            )
        _parse_pose(data)
        if mode == "target":
            duration = self.default_transition_chunks if transition_chunks is None else int(transition_chunks)
            if duration < 0:
                raise ValueError("transition_chunks must be >= 0")
        del state

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
        self.validate_payload(
            state,
            event_id=event_id,
            payload=payload,
            transition_chunks=transition_chunks,
        )
        mode = payload.get("mode", "target")
        camera_mode = cast(InteractionMode, mode)
        data = cast(Mapping[str, object], payload.get("data"))
        pose = _parse_pose(data)
        if camera_mode == "target":
            duration = self.default_transition_chunks if transition_chunks is None else int(transition_chunks)
        else:
            duration = 0

        session = state.interaction_sessions.setdefault("camera", CameraSession())
        assert isinstance(session, CameraSession)
        with session.lock:
            session.has_received_input = True
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
        num_media_frames: int | None = None,
        fps: float | None = None,
        num_latent_frames: int | None = None,
    ) -> InteractionChunkMetadata | None:
        """Advance camera state for one chunk and store latent-aligned poses.

        .. note::
            Hardcoded the following policy of per-frame alignment:

            - Resolve ``received_at`` on the media grid, then map each media index onto a latent step.
            - Integrate one velocity step per **latent frame**, so holding speed ``0.05`` for a chunk with
              ``num_latent_frames=3`` yields total travel ``0.15``.
            - ``last_absolute_poses.shape[0] == num_latent_frames``: interpolation is latent-aligned.

            If future models need different scale/alignment policies, can refactor interaction registry
            to be able to mark a policy.
        """
        del chunk_index
        if num_media_frames is None or fps is None:
            raise ValueError("SE3DeltaCameraHandler requires chunk num_media_frames and fps")
        if num_latent_frames is None:
            num_latent_frames = num_media_frames
        if int(num_latent_frames) <= 0:
            raise ValueError(f"num_latent_frames must be > 0, got {num_latent_frames}")

        session = state.interaction_sessions.setdefault("camera", CameraSession())
        assert isinstance(session, CameraSession)
        with session.lock:
            poses, started, active, completed = self._step_one_chunk(
                session,
                num_media_frames=num_media_frames,
                fps=fps,
                num_latent_frames=num_latent_frames,
                boundary_at=boundary_at,
            )
            if poses:
                session.last_absolute_poses = torch.stack([p.as_matrix() for p in poses], dim=0)
            else:
                session.last_absolute_poses = torch.zeros((0, 4, 4), dtype=torch.float64)

        return InteractionChunkMetadata(
            started_event_ids=started,
            active_event_ids=active,
            completed_event_ids=completed,
        )

    def _step_one_chunk(
        self,
        session: CameraSession,
        *,
        num_media_frames: int,
        fps: float,
        num_latent_frames: int,
        boundary_at: float,
    ) -> tuple[list[CameraPose], list[str], list[str], list[str]]:
        """Sample absolute poses for this chunk under target/velocity semantics."""
        num_media_frames = max(int(num_media_frames), 1)
        num_latent_frames = max(int(num_latent_frames), 1)
        pending = list(session.pending_events)
        session.pending_events.clear()

        by_latent: dict[int, list[QueuedCameraEvent]] = {}
        for event in pending:
            media_frame = resolve_event_frame_offset(
                received_at=event.received_at,
                previous_boundary_at=session.last_boundary_at,
                num_frames=num_media_frames,
                fps=fps,
            )
            latent_idx = _media_frame_to_latent(
                media_frame,
                num_media_frames=num_media_frames,
                num_latent_frames=num_latent_frames,
            )
            by_latent.setdefault(latent_idx, []).append(event)
        for events in by_latent.values():
            events.sort(key=lambda item: item.received_at)

        started: list[str] = []
        completed: list[str] = []
        seen_completed: set[str] = set()
        poses: list[CameraPose] = []

        def _mark_completed(event_id: str) -> None:
            if event_id not in seen_completed:
                seen_completed.add(event_id)
                completed.append(event_id)

        for latent_idx in range(num_latent_frames):
            for event in by_latent.get(latent_idx, []):
                started.append(event.event_id)
                cancelled_id = self._activate_event(session, event)
                if cancelled_id is not None:
                    _mark_completed(cancelled_id)
            just_completed = self._step_one_frame(session, num_latent_frames)
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
        event.elapsed_transition_frames = 0
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
        """Advance the active camera command by one frame."""
        event = session.active_event
        if event is None:
            return None

        if event.mode == "target":
            duration = event.transition_chunks
            source = session.target_source or session.current_pose
            target = event.pose
            if duration <= 0:
                return self._clear_active_target(session)

            total_frames = max(1, int(round(duration * float(total_num_frames_this_chunk))))
            event.elapsed_transition_frames += 1
            event.elapsed_transition_chunks = event.elapsed_transition_frames / float(total_num_frames_this_chunk)
            alpha = min(1.0, event.elapsed_transition_frames / float(total_frames))
            session.current_pose = _lerp_pose(source, target, alpha)
            if event.elapsed_transition_frames >= total_frames:
                return self._clear_active_target(session)
            return None

        # Velocity: apply the held structural SE3 delta once per frame.
        session.current_pose = _compose_pose(session.current_pose, event.pose)
        return None


def _media_frame_to_latent(
    media_frame: int,
    *,
    num_media_frames: int,
    num_latent_frames: int,
) -> int:
    """Map a media-timeline frame index onto the latent control grid."""
    if num_latent_frames <= 1 or num_media_frames <= 1:
        return 0
    return min(int(media_frame) * int(num_latent_frames) // int(num_media_frames), num_latent_frames - 1)


def _as_xyz(value: object, *, name: str) -> Vec3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"camera {name} must be a length-3 list/tuple")
    if any(math.isnan(v) or math.isinf(v) for v in value):
        raise ValueError(f"camera {name} must be non-nan and non-inf")
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_quat(value: object, *, name: str) -> Quat:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"camera {name} must be a length-4 list/tuple (x, y, z, w)")
    if any(math.isnan(v) or math.isinf(v) for v in value):
        raise ValueError(f"camera {name} must be non-nan and non-inf")
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
