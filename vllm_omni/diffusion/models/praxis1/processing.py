# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Architecture adapted from https://github.com/runwayml/praxis-1-oss (Praxis-1).
"""Shared observation history and DROID camera/pose conversion."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

FPS = 15.0

ACTION_CHUNK = 24


FRAMES_PER_TOKEN = 4

ASPECT_TOL = 0.03
NATIVE_VIEW = (720, 1280)


def contiguous_chunk_geometry(action_chunk: int, n_chunks: int) -> list[tuple[int, int]]:
    """Return ``(raw_offset, raw_len)`` per chunk of one contiguous window."""
    lat = (action_chunk - 1) // 4 + 1
    out: list[tuple[int, int]] = []
    off = 0
    for k in range(n_chunks):
        ln = 4 * lat - 3 if k == 0 else 4 * lat
        out.append((off, ln))
        off += ln
    return out


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` wxyz quaternions to ``(..., 3, 3)`` matrices."""
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-08)
    w, x, y, z = (q[..., 0], q[..., 1], q[..., 2], q[..., 3])
    r = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return r.reshape(*q.shape[:-1], 3, 3)


def rot6d(r: np.ndarray) -> np.ndarray:
    """Return the first two columns of ``(..., 3, 3)`` matrices as ``(..., 6)``."""
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Convert one rot6d ``[c0(3), c1(3)]`` to a float64 3x3 matrix."""
    a0 = r6[:3].astype(np.float64)
    a1 = r6[3:6].astype(np.float64)
    c0 = a0 / (np.linalg.norm(a0) + 1e-08)
    a1 = a1 - np.dot(c0, a1) * c0
    c1 = a1 / (np.linalg.norm(a1) + 1e-08)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=1)


def pose8_to_10d(pose8: np.ndarray) -> np.ndarray:
    """Convert ``[xyz3, quat_wxyz4, grip1]`` to ``[xyz3, rot6d6, grip1]`` (float32)."""
    xyz = pose8[..., :3]
    r = quat_wxyz_to_matrix(pose8[..., 3:7])
    grip = pose8[..., 7:8]
    return np.concatenate([xyz, rot6d(r), grip], axis=-1).astype(np.float32)


@dataclass(frozen=True)
class NormStats:
    """Normalization statistics of the actions and the proprio (float32)."""

    action_q01: np.ndarray
    action_q99: np.ndarray
    proprio_mean: np.ndarray
    proprio_std: np.ndarray


def aspect_crop_box(
    src_h: int, src_w: int, tile_h: int, tile_w: int, tol: float = ASPECT_TOL
) -> tuple[int, int, int, int]:
    """Return the centre crop ``(y0, y1, x0, x1)`` to the tile aspect."""
    src_ar, tile_ar = (src_w / src_h, tile_w / tile_h)
    if abs(tile_ar / src_ar - 1.0) <= tol:
        return (0, src_h, 0, src_w)
    if src_ar > tile_ar:
        new_w = round(src_h * tile_ar)
        x0 = (src_w - new_w) // 2
        return (0, src_h, x0, x0 + new_w)
    new_h = round(src_w / tile_ar)
    y0 = (src_h - new_h) // 2
    return (y0, y0 + new_h, 0, src_w)


def aspect_crop(fr: torch.Tensor, tile_h: int, tile_w: int, tol: float = ASPECT_TOL) -> torch.Tensor:
    """Apply :func:`aspect_crop_box` to a ``(..., H, W)`` frame stack."""
    y0, y1, x0, x1 = aspect_crop_box(fr.shape[-2], fr.shape[-1], tile_h, tile_w, tol)
    if (y0, y1, x0, x1) == (0, fr.shape[-2], 0, fr.shape[-1]):
        return fr
    return fr[..., y0:y1, x0:x1]


class ContiguousStreamContext:
    """Raw frame and pose streams since the start of the context window."""

    def __init__(self, action_chunk: int = ACTION_CHUNK, max_context_chunks: int = 3):
        geom = contiguous_chunk_geometry(action_chunk, max_context_chunks + 1)
        self.first_len = geom[0][1]
        self.cont_len = geom[1][1] if len(geom) > 1 else 0
        self.lat_per_chunk = (action_chunk - 1) // 4 + 1
        self.max_context_chunks = int(max_context_chunks)
        self.max_raws = self.first_len + self.cont_len * (self.max_context_chunks - 1)
        self._clip: torch.Tensor | None = None
        self._poses: np.ndarray | None = None

    def reset(self) -> None:
        """Drop all retained frames and poses."""
        self._clip = None
        self._poses = None

    @property
    def n_raws(self) -> int:
        """Return the number of retained raw frames."""
        return 0 if self._clip is None else int(self._clip.shape[2])

    @property
    def depth(self) -> int:
        """Return the number of complete clean chunks in the stream."""
        t = self.n_raws
        if t < self.first_len:
            return 0
        return min(1 + (t - self.first_len) // self.cont_len, self.max_context_chunks)

    @property
    def expected_next_rows(self) -> int:
        """Return the raw count of the next executed chunk (21 first, then 24)."""
        return self.first_len if self.n_raws == 0 else self.cont_len

    def boundary_offsets(self) -> list[int]:
        """Return the window offsets of the retained clean chunk starts."""
        return [off for off, _ln in contiguous_chunk_geometry(self.cont_len, self.max_context_chunks)][: self.depth]

    def append_executed(self, clip: torch.Tensor, poses: np.ndarray | None = None) -> None:
        """Append the executed chunk and slide the window when it overflows."""
        t = int(clip.shape[2])
        if t != self.expected_next_rows:
            raise ValueError(f"executed chunk spans {t} raws, expected {self.expected_next_rows}")
        was_empty = self._clip is None
        self._clip = clip if self._clip is None else torch.cat([self._clip, clip], dim=2)
        aligned = self._poses is not None or was_empty
        if poses is not None and len(poses) == t and aligned:
            p = np.asarray(poses, dtype=np.float32)
            self._poses = p if self._poses is None else np.concatenate([self._poses, p], axis=0)
        else:
            self._poses = None
        if self.n_raws > self.max_raws:
            drop = self.n_raws - self.max_raws
            self._clip = self._clip[:, :, drop:]
            if self._poses is not None:
                self._poses = self._poses[drop:]

    def cleans(self, encode: Callable[[torch.Tensor], torch.Tensor]) -> list[torch.Tensor]:
        """Encode the retained prefix once and return the clean chunks (oldest first)."""
        d = self.depth
        if d == 0:
            return []
        assert self._clip is not None
        n_raws = self.first_len + self.cont_len * (d - 1)
        lat = encode(self._clip[:, :, :n_raws])
        n = self.lat_per_chunk
        return [lat.narrow(-3, k * n, n).contiguous() for k in range(d)]

    def anchor_frame(self) -> torch.Tensor | None:
        """Return the window start frame ``(1, 3, 1, H, W)``, or ``None`` on cold start."""
        return None if self._clip is None else self._clip[:, :, :1]

    def boundary_poses(self) -> np.ndarray | None:
        """Return the pose rows at the retained clean chunk starts, or ``None``."""
        if self._poses is None or self.depth == 0:
            return None
        if len(self._poses) != self.n_raws:
            return None
        offs = self.boundary_offsets()
        if not offs or offs[-1] >= len(self._poses):
            return None
        return np.stack([self._poses[o] for o in offs], axis=0)


def action_rows(pred_action: torch.Tensor, depth: int, frames_per_token: int, action_dim: int) -> np.ndarray:
    """Unpack the predicted ``(1, L, F * D)`` tokens to the executed rows (float32)."""
    a = pred_action.detach().cpu().numpy().astype(np.float32)
    if a.ndim == 3:
        a = a[0]
    rows = a.reshape(a.shape[0], frames_per_token, -1)[..., :action_dim].reshape(-1, action_dim)
    return np.concatenate([rows[:1], rows[frames_per_token:]]) if depth == 0 else rows


def text_encoder_prompt(prompt: str) -> str:
    """Return the string for the text encoder. A blank prompt becomes one space."""
    return prompt if prompt.strip() else " "


ACTION_DIM = 10

N_CHUNKS = 4

MAX_CONTEXT_CHUNKS = 3

TILE_SIZE: tuple[int, int] = (224, 416)

TILE_CHUNK_FRAMES = 32

_C, _S = (np.cos(-np.pi / 4), np.sin(-np.pi / 4))

FLANGE_TURN = np.array([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]])

NO_TURN = np.eye(3)


@dataclass(frozen=True)
class RigFrame:
    """The frame of the poses of one client, relative to the Franka flange."""

    tool_offset: float
    turn: tuple[tuple[float, ...], ...]
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _rows(m: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in m)


TURNED_FLANGE = RigFrame(0.0, _rows(FLANGE_TURN))

FR3_HAND = RigFrame(0.1034, _rows(FLANGE_TURN))

RIG_FRAMES: dict[str, RigFrame] = {
    "": FR3_HAND,
    "remote.robolab.droid": FR3_HAND,
    "remote.molmo_spaces.droid": RigFrame(0.155, _rows(NO_TURN), (0.0, 0.0, 0.58)),
    "praxis.sim": TURNED_FLANGE,
}

FK_TOLERANCE = 0.005

FRANKA_DH: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)

FRANKA_FLANGE_D = 0.107

Q99_DIMS = np.array([0, 1, 2, 9])


POSE_KEYS: tuple[str, ...] = ("ee_pose", "robot_state.ee_pose")

GRIP_KEYS: tuple[str, ...] = ("grip", "robot_state.grip", "gripper")

VIEW_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "exterior": ("exterior_image", "image.exterior", "observation.images.side"),
    "wrist": ("wrist_image", "image.wrist", "observation.images.left"),
}

DROID_IMPEDANCE: dict[str, object] = {
    "type": "impedance",
    "kq": [40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0],
    "kqd": [4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0],
    "kx": [750.0, 750.0, 750.0, 15.0, 15.0, 15.0],
    "kxd": [37.0, 37.0, 37.0, 2.0, 2.0, 2.0],
}

CONTROL_MODES: dict[str, dict[str, object] | None] = {"none": None, "droid_impedance": DROID_IMPEDANCE}


def control_mode_payload(control_mode: str) -> dict[str, object] | None:
    """Return the ``robot_command["mode"]`` value of a control mode (a copy)."""
    if control_mode not in CONTROL_MODES:
        raise ValueError(f"Unknown control_mode {control_mode!r}. The known modes are {sorted(CONTROL_MODES)}.")
    payload = CONTROL_MODES[control_mode]
    return None if payload is None else copy.deepcopy(payload)


def _pick(obs: dict, candidates: tuple[str, ...]) -> object:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(f"obs missing any of {candidates}; got keys={list(obs)}")


def extract_pose8(obs: dict) -> np.ndarray:
    """Return the current pose8 ``(8,)``: the last row of the pose and grip stacks."""
    ee = np.asarray(_pick(obs, POSE_KEYS), dtype=np.float32)
    if ee.ndim >= 2:
        ee = ee[-1]
    ee = ee.ravel()
    if ee.shape[0] < 7:
        raise ValueError(f"ee_pose must be (>=7,), got {ee.shape}")
    grip = np.asarray(_pick(obs, GRIP_KEYS), dtype=np.float32)
    if grip.ndim >= 2:
        grip = grip[-1]
    elif grip.ndim == 1 and grip.shape[0] > 1:
        grip = grip[-1:]
    grip = grip.ravel()
    return np.concatenate([ee[:7], grip[:1]]).astype(np.float32)


def pose8_history(obs: dict, need: int) -> np.ndarray | None:
    """Return the ``(need, 8)`` pose8 rows of the executed frames, or ``None``."""
    try:
        ee = np.asarray(_pick(obs, POSE_KEYS), dtype=np.float32)
        grip = np.asarray(_pick(obs, GRIP_KEYS), dtype=np.float32)
    except KeyError:
        return None
    if ee.ndim != 2 or ee.shape[0] < need + 1 or ee.shape[1] < 7:
        return None
    g = grip.reshape(grip.shape[0], -1) if grip.ndim >= 1 else None
    if g is None or g.shape[0] < need + 1:
        return None
    rows = np.concatenate([ee[-(need + 1) : -1, :7], g[-(need + 1) : -1, :1]], axis=1)
    return rows.astype(np.float32)


def latest_frame(obs: dict, view: str) -> np.ndarray:
    """Return the current frame ``(H, W, 3)`` uint8 of one view."""
    img = np.asarray(_pick(obs, VIEW_KEY_CANDIDATES[view]))
    if img.ndim == 4:
        img = img[-1]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"image must be (H,W,3), got {img.shape}")
    return img.astype(np.uint8)


def fit_resize_size(tile_w: int) -> tuple[int, int]:
    """Return the native view fitted to the tile width (234x416)."""
    nh, nw = NATIVE_VIEW
    return min(nh, tile_w, nh * tile_w // nw), min(nw, tile_w)


def _tile_view(frames: np.ndarray, tile_h: int, tile_w: int, tol: float) -> torch.Tensor:
    """Fit a view to the native aspect and trim to the checkpoint tile."""
    a = np.asarray(frames, np.uint8)
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3), got {a.shape}")
    fr = torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)
    return _fit_view(fr, tile_h, tile_w, tol)


def _resize_view(fr: torch.Tensor, rh: int, rw: int) -> torch.Tensor:
    if tuple(fr.shape[-2:]) != (rh, rw):
        return torch.cat(
            [
                torch.nn.functional.interpolate(
                    chunk.float(),
                    size=(rh, rw),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                for chunk in fr.split(TILE_CHUNK_FRAMES, dim=0)
            ]
        )
    return fr


def _fit_view(fr: torch.Tensor, tile_h: int, tile_w: int, tol: float) -> torch.Tensor:
    """Return the fit-rule tile ``(T, 3, tile_h, tile_w)`` of ``fr`` (uint8)."""
    rh, rw = fit_resize_size(tile_w)
    if rh < tile_h or rw < tile_w:
        raise ValueError(f"the fit size {rh}x{rw} is below the tile {tile_h}x{tile_w}")
    cropped = aspect_crop(fr, *NATIVE_VIEW, tol)
    cropped = _resize_view(cropped, rh, rw)
    y0, x0 = (rh - tile_h) // 2, (rw - tile_w) // 2
    return cropped[..., y0 : y0 + tile_h, x0 : x0 + tile_w].contiguous()


def tile_clip(
    exterior: np.ndarray,
    wrist: np.ndarray | None,
    tile_h: int = TILE_SIZE[0],
    tile_w: int = TILE_SIZE[1],
    tol: float = ASPECT_TOL,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert both views to the checkpoint canvas (1, 3, T, H, 2W)."""
    if wrist is None:
        raise ValueError("The tile canvas needs the exterior and the wrist view.")
    tiles = [_tile_view(v, tile_h, tile_w, tol) for v in (exterior, wrist)]
    if tiles[0].shape[0] != tiles[1].shape[0]:
        raise ValueError(f"exterior has {tiles[0].shape[0]} frames, wrist has {tiles[1].shape[0]}")
    canvas = torch.cat(tiles, dim=-1).float()
    return canvas.permute(1, 0, 2, 3).unsqueeze(0).to(device)


def _tile_args(tile_size: tuple[int, int] | None, aspect_tol: float | None) -> tuple[int, int, float]:
    th, tw = TILE_SIZE if tile_size is None else tile_size
    return (int(th), int(tw), ASPECT_TOL if aspect_tol is None else float(aspect_tol))


def current_clip(
    obs: dict,
    device: torch.device | str = "cpu",
    tile_size: tuple[int, int] | None = None,
    aspect_tol: float | None = None,
) -> torch.Tensor:
    """Return the tile clip ``(1, 3, 1, H, W)`` of the current frames."""
    exterior = latest_frame(obs, "exterior")
    try:
        wrist = latest_frame(obs, "wrist")
    except KeyError:
        wrist = None
    th, tw, tol = _tile_args(tile_size, aspect_tol)
    return tile_clip(exterior[None], None if wrist is None else wrist[None], th, tw, tol, device)


def executed_clip(
    obs: dict,
    need: int,
    device: torch.device | str = "cpu",
    tile_size: tuple[int, int] | None = None,
    aspect_tol: float | None = None,
) -> tuple[torch.Tensor, np.ndarray | None] | None:
    """Return the clip and the pose8 rows of the ``need`` executed raws, or ``None``."""
    ext = np.asarray(_pick(obs, VIEW_KEY_CANDIDATES["exterior"]))
    if ext.ndim != 4 or ext.shape[0] < need + 1:
        return None
    ext_chunk = ext[-(need + 1) : -1].astype(np.uint8)
    wrist_chunk = None
    try:
        wrist = np.asarray(_pick(obs, VIEW_KEY_CANDIDATES["wrist"]))
        if wrist.ndim == 4 and wrist.shape[0] >= need + 1:
            wrist_chunk = wrist[-(need + 1) : -1].astype(np.uint8)
    except KeyError:
        wrist_chunk = None
    if wrist_chunk is None:
        return None
    th, tw, tol = _tile_args(tile_size, aspect_tol)
    clip = tile_clip(ext_chunk, wrist_chunk, th, tw, tol, device)
    return (clip, pose8_history(obs, need))


def pose8_to_model(pose8: np.ndarray, frame: RigFrame = TURNED_FLANGE) -> np.ndarray:
    """Convert client pose8 rows ``(..., 8)`` in ``frame`` to the flange frame of the model."""
    out = np.array(pose8, dtype=np.float64, copy=True)
    rot = Rotation.from_quat(out[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix().reshape(*out.shape[:-1], 3, 3)
    base = np.asarray(frame.base_offset, dtype=np.float64)
    out[..., :3] = out[..., :3] - base - frame.tool_offset * rot[..., :, 2]
    turn = np.asarray(frame.turn).T @ NO_TURN
    quat = Rotation.from_matrix((rot @ turn).reshape(-1, 3, 3)).as_quat().reshape(*out.shape[:-1], 4)
    out[..., 3:7] = quat[..., [3, 0, 1, 2]]
    out[..., 7] = np.clip(out[..., 7], 0.0, 1.0)
    return out.astype(np.float32)


def pose_to_rig(xyz: np.ndarray, rot: np.ndarray, frame: RigFrame = TURNED_FLANGE) -> tuple[np.ndarray, np.ndarray]:
    """Convert a flange-frame target to ``frame``, the frame of the client."""
    xyz = xyz + frame.tool_offset * rot[:, 2] + np.asarray(frame.base_offset)
    return (xyz, rot @ NO_TURN.T @ np.asarray(frame.turn))


def franka_flange_pose(q: np.ndarray) -> np.ndarray:
    """Return the flange pose ``(4, 4)`` in the robot base frame for the joints ``q (7,)``."""
    m = np.eye(4)
    for (a, d, alpha), theta in zip(FRANKA_DH, np.asarray(q, dtype=np.float64), strict=True):
        ca, sa, ct, st = (np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta))
        m = m @ np.array(
            [[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0.0, 0.0, 0.0, 1.0]]
        )
    m[:3, 3] += FRANKA_FLANGE_D * m[:3, 2]
    return m


def rig_frame(obs: dict) -> RigFrame:
    """Return the pose frame of the client of ``obs`` (its ``descriptor``)."""
    descriptor = obs.get("descriptor") or ""
    if isinstance(descriptor, bytes):
        descriptor = descriptor.decode("utf-8", "replace")
    frame = RIG_FRAMES.get(str(descriptor))
    if frame is None:
        raise ValueError(
            f"The client {descriptor!r} reports poses in an unknown frame. The known clients are {sorted(RIG_FRAMES)}."
        )
    return frame


def check_flange(obs: dict, pose8_model: np.ndarray) -> None:
    """Check a converted pose against the flange forward kinematics."""
    if "robot_state.q" not in obs:
        return
    q = np.asarray(obs["robot_state.q"])
    if q.ndim == 2:
        q = q[-1]
    if q.shape != (7,) or q.dtype.kind not in "iuf":
        raise ValueError(f"robot_state.q must be 7 joint angles, got {q.shape}")
    xyz = np.asarray(pose8_model[:3], dtype=np.float64)
    err = float(np.linalg.norm(xyz - franka_flange_pose(q)[:3, 3]))
    if not np.isfinite(err) or err > FK_TOLERANCE:
        raise ValueError(
            f"Converted ee_pose differs from robot_state.q flange FK by {err * 1000.0:.1f} mm; check the rig frame."
        )


def proprio_tensors(poses: np.ndarray, stats: NormStats, device: torch.device | str) -> torch.Tensor:
    """Return normalized proprio for every boundary pose (float32)."""
    p = np.concatenate([pose8_to_10d(poses[..., i : i + 8]) for i in range(0, poses.shape[-1], 8)], axis=-1).astype(
        np.float32
    )
    std = np.maximum(stats.proprio_std.astype(np.float32), np.float32(1e-6))
    p = ((p - stats.proprio_mean.astype(np.float32)) / std).astype(np.float32)
    p = np.asarray(p).reshape(-1, p.shape[-1])
    every = torch.from_numpy(np.ascontiguousarray(p)).reshape(1, p.shape[0], -1).to(device, torch.float32)
    return every


def rel10_to_wire(
    rel: np.ndarray,
    pose8_now: np.ndarray,
    dt: float = 1.0 / FPS,
    mode: dict[str, object] | None = None,
    frame: RigFrame = TURNED_FLANGE,
) -> list[dict]:
    """Convert denormalized ``(T, 10)`` relative actions to command dicts."""
    rel = np.asarray(rel, dtype=np.float32)
    if rel.ndim != 2 or rel.shape[1] < ACTION_DIM:
        raise ValueError(f"rel actions must be (T, >={ACTION_DIM}), got {rel.shape}")
    anchor = np.asarray(pose8_now)
    anchor_xyz = anchor[:3].astype(np.float64)
    r_anchor = Rotation.from_quat(anchor[[4, 5, 6, 3]]).as_matrix()
    out: list[dict] = []
    for t, r in enumerate(rel):
        xyz_abs = r[:3].astype(np.float64) + anchor_xyz
        r_abs = rot6d_to_matrix(r[3:9]) @ r_anchor
        xyz_abs, r_abs = pose_to_rig(xyz_abs, r_abs, frame)
        pose12 = np.concatenate([xyz_abs, r_abs.reshape(-1)]).astype(np.float32)
        cmd: dict[str, object] = {"type": "cartesian_pos", "pose": pose12}
        if mode is not None:
            cmd["mode"] = mode
        out.append({"robot_command": cmd, "target_grip": float(r[9]), "timestamp": float(t * dt)})
    return out


def decode_actions(
    pred_action: torch.Tensor, stats: NormStats, depth: int, dims: np.ndarray = Q99_DIMS
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized rows and the relative actions of the commands of a chunk."""
    rows = action_rows(pred_action, depth, FRAMES_PER_TOKEN, len(stats.action_q01))
    rel = rows.astype(np.float32).copy()
    lo, hi = stats.action_q01[dims].astype(np.float32), stats.action_q99[dims].astype(np.float32)
    rel[..., dims] = (np.clip(rel[..., dims], -1, 1) + 1.0) / 2.0 * (hi - lo) + lo
    rel[..., 9::10] = np.clip(rel[..., 9::10], 0.0, 1.0)
    return rows, rel


@dataclass
class ChunkInputs:
    """Model inputs of one served chunk."""

    anchor: torch.Tensor
    cleans: list[torch.Tensor]
    proprio_all: torch.Tensor
    depth: int
    pose_now: np.ndarray
    chunk_index: int
    prompt: str


class ServeSession:
    """Shared observed-context state machine; default geometry is the DROID rig."""

    def __init__(
        self,
        encode: Callable[[torch.Tensor], torch.Tensor],
        stats: NormStats,
        device: torch.device | str,
        prompt: str = "",
        max_context_chunks: int = MAX_CONTEXT_CHUNKS,
        fps: float = FPS,
        dims: np.ndarray | list[int] | None = None,
        control_mode: str = "none",
        exec_horizon: int | None = None,
        n_chunks: int = N_CHUNKS,
        tile_size: tuple[int, int] | None = None,
        aspect_tol: float | None = None,
    ):
        self.tile_size = None if tile_size is None else (int(tile_size[0]), int(tile_size[1]))
        self.aspect_tol = None if aspect_tol is None else float(aspect_tol)
        self.encode = encode
        self.stats = stats
        self.device = device
        self.prompt = prompt
        self.fps = float(fps)
        self.dims = np.asarray(
            dims
            if dims is not None
            else [i + offset for offset in range(0, len(stats.action_q01), 10) for i in (0, 1, 2, 9)],
            dtype=np.int64,
        )
        self.control_mode = str(control_mode)
        self.mode = control_mode_payload(self.control_mode)
        self.exec_horizon = None if exec_horizon is None else int(exec_horizon)
        self.max_context_chunks = max(0, min(int(max_context_chunks), int(n_chunks) - 1))
        self.stateless = self.max_context_chunks == 0 or (
            self.exec_horizon is not None and self.exec_horizon < ACTION_CHUNK
        )
        self.stream = ContiguousStreamContext(
            action_chunk=ACTION_CHUNK, max_context_chunks=max(self.max_context_chunks, 1)
        )
        self.chunk_index = 0
        self._boundary_poses: list[np.ndarray] = []
        self._cap = self.max_context_chunks + 1
        self.frame = TURNED_FLANGE

    def restart(self) -> None:
        """Drop the context. The next call is a cold start at depth 0."""
        self._boundary_poses = []
        self.stream.reset()

    def _current(self, obs: dict) -> tuple[np.ndarray, torch.Tensor]:
        self.frame = rig_frame(obs)
        pose = pose8_to_model(extract_pose8(obs), self.frame)
        check_flange(obs, pose)
        return pose, current_clip(obs, self.device, self.tile_size, self.aspect_tol)

    def _executed(self, obs: dict, need: int) -> tuple[torch.Tensor, np.ndarray | None] | None:
        got = executed_clip(obs, need, self.device, self.tile_size, self.aspect_tol)
        if got is None:
            return None
        clip, rows = got
        return clip, None if rows is None else pose8_to_model(rows, self.frame)

    def rel_to_wire(self, rel: np.ndarray, pose: np.ndarray) -> list[dict]:
        return rel10_to_wire(rel, pose, 1.0 / max(self.fps, 1e-6), self.mode, self.frame)

    def prepare(self, obs: dict) -> ChunkInputs:
        """Update the context from ``obs`` and return the inputs of this chunk."""
        if bool(obs.get("reset_context") or obs.get("__reset_context__")):
            self.restart()
        override = obs.get("task") or obs.get("prompt")
        if override:
            if isinstance(override, bytes):
                override = override.decode("utf-8", "replace")
            new = str(override).strip()
            if new and new != self.prompt:
                self.prompt = new
        pose8, cur_clip = self._current(obs)
        if self.stateless:
            self.restart()
        elif self._boundary_poses:
            need = self.stream.expected_next_rows
            got = self._executed(obs, need)
            if got is None:
                self.restart()
            else:
                clip, rows = got
                self.stream.append_executed(clip, rows)
        pose = np.asarray(pose8, dtype=np.float32)
        self._boundary_poses = [*self._boundary_poses, pose][-self._cap :]
        cleans = self.stream.cleans(self.encode)
        d = len(cleans)
        af = self.stream.anchor_frame()
        anchor = self.encode(cur_clip if af is None else af)
        bp = self.stream.boundary_poses() if d else None
        if d == 0:
            poses = pose8[None].astype(np.float32)
        elif bp is not None:
            poses = np.concatenate([bp, pose8[None]], axis=0).astype(np.float32)
        else:
            n = min(d + 1, len(self._boundary_poses))
            poses = np.stack(self._boundary_poses[-n:], axis=0).astype(np.float32)
        proprio_all = proprio_tensors(poses, self.stats, self.device)
        return ChunkInputs(
            anchor=anchor,
            cleans=cleans,
            proprio_all=proprio_all,
            depth=d,
            pose_now=pose8,
            chunk_index=self.chunk_index,
            prompt=text_encoder_prompt(self.prompt),
        )

    def finish(self, inputs: ChunkInputs, pred_action: torch.Tensor) -> tuple[list[dict], np.ndarray, np.ndarray]:
        """Decode actions once, produce commands and advance the episode."""
        rows, rel = decode_actions(pred_action, self.stats, inputs.depth, self.dims)
        wire = self.rel_to_wire(rel, inputs.pose_now)
        self.chunk_index += 1
        return wire if self.exec_horizon is None else wire[: self.exec_horizon], rows, rel
