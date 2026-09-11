# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client-side helpers that map WASD/IJKL keys to structural SE3 camera payloads.
Convenience for demos.
"""

from __future__ import annotations

import math
from typing import Any

# Unity-frame (+X right, +Y up, +Z forward) step sizes used by the optional
# WASD helper. Magnitudes mirror LingBot's script integrator; the engine never
# sees key tokens — only structural SE3 payloads.
_WASD_STEP = 0.05
_WASD_PITCH_DEG = 4.0
_WASD_YAW_DEG = 6.0


def _quat_from_axis_angle(axis: tuple[float, float, float], degrees: float) -> list[float]:
    """Return a unit quaternion ``[x, y, z, w]`` for a right-handed axis-angle."""
    radians = math.radians(degrees)
    half = 0.5 * radians
    sine = math.sin(half)
    return [axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(half)]


def wasd_to_camera_payload(actions: list[str], *, mode: str = "velocity") -> dict[str, Any]:
    """Map WASD/IJKL key tokens to a structural camera interaction payload.

    This is a client convenience only. Servers reject ``data.actions``.
    """
    keys = {str(action).lower() for action in actions}
    dx = _WASD_STEP * (("d" in keys) - ("a" in keys))
    dy = 0.0
    dz = _WASD_STEP * (("w" in keys) - ("s" in keys))
    pitch = _WASD_PITCH_DEG * (("i" in keys) - ("k" in keys))
    yaw = _WASD_YAW_DEG * (("l" in keys) - ("j" in keys))

    # Compose yaw (around +Y up) then pitch (around +X) into one delta quaternion.
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    if abs(yaw) > 0.0:
        yx, yy, yz, yw = _quat_from_axis_angle((0.0, 1.0, 0.0), yaw)
        qx, qy, qz, qw = yx, yy, yz, yw
    if abs(pitch) > 0.0:
        px, py, pz, pw = _quat_from_axis_angle((1.0, 0.0, 0.0), pitch)
        # q = pitch ⊗ yaw (apply yaw first in the world frame).
        qx, qy, qz, qw = (
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
            pw * qw - px * qx - py * qy - pz * qz,
        )

    return {
        "mode": mode,
        "data": {
            "translation": [dx, dy, dz],
            "rotation": [qx, qy, qz, qw],
        },
    }
