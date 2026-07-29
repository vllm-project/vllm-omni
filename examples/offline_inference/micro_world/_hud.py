# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HUD (keyboard + mouse cursor) overlay drawn on top of generated video frames.

Mirrors the reference Micro-World pipeline (`with_ui=True` path in
`pipeline_wan_action_t2w.py`). The keys/cursor are NOT rendered by the
diffusion model — they are drawn by this OpenCV post-processing step on
the decoded RGB frames before mp4 encoding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_MOUSE_ICON = ASSETS_DIR / "mouse.png"


def parse_config(
    mouse_actions: np.ndarray,
    keyboard_actions: np.ndarray,
    initial_mouse_position: tuple[int, int] = (320, 176),
    mouse_scale: tuple[float, float] = (0.4, 1.6),
) -> tuple[dict[int, dict[str, bool]], dict[int, tuple[float, float]]]:
    """Convert per-frame action arrays into key state + cumulative cursor position.

    Args:
        mouse_actions: Shape ``(num_frames, 2)``, columns ``(mouse_y, mouse_x)``.
        keyboard_actions: Shape ``(num_frames, 7)``, columns
            ``(W, S, A, D, Space, Shift, Ctrl)``.
        initial_mouse_position: Starting cursor pixel in frame 0.
        mouse_scale: ``(scale_x, scale_y)`` applied to mouse deltas. Default
            matches reference (0.4, 4 * 0.4 = 1.6).

    Returns:
        ``(key_data, mouse_data)``. ``key_data[i]`` is a 7-key dict of bools;
        ``mouse_data[i]`` is the ``(x, y)`` cursor position for frame ``i``.
    """
    if hasattr(mouse_actions, "cpu"):
        mouse_actions = mouse_actions.detach().cpu().float().numpy()
    elif not isinstance(mouse_actions, np.ndarray):
        mouse_actions = np.asarray(mouse_actions, dtype=np.float32)

    if hasattr(keyboard_actions, "cpu"):
        keyboard_actions = keyboard_actions.detach().cpu().float().numpy()
    elif not isinstance(keyboard_actions, np.ndarray):
        keyboard_actions = np.asarray(keyboard_actions, dtype=np.float32)

    if mouse_actions.ndim == 3:
        mouse_actions = mouse_actions[0]
    if keyboard_actions.ndim == 3:
        keyboard_actions = keyboard_actions[0]

    scale_x, scale_y = mouse_scale
    key_data: dict[int, dict[str, bool]] = {}
    mouse_data: dict[int, tuple[float, float]] = {}

    for i in range(len(mouse_actions)):
        w, s, a, d, space, shift, ctrl = keyboard_actions[i]
        mouse_y, mouse_x = mouse_actions[i]

        key_data[i] = {
            "W": bool(w),
            "A": bool(a),
            "S": bool(s),
            "D": bool(d),
            "Space": bool(space),
            "Shift": bool(shift),
            "Ctrl": bool(ctrl),
        }
        if i == 0:
            mouse_data[i] = (float(initial_mouse_position[0]), float(initial_mouse_position[1]))
        else:
            prev_x, prev_y = mouse_data[i - 1]
            mouse_data[i] = (
                prev_x + float(mouse_x) * scale_x,
                prev_y + float(mouse_y) * scale_y,
            )

    return key_data, mouse_data


def _draw_rounded_rectangle(image, top_left, bottom_right, color, radius=10, alpha=0.5):
    import cv2

    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_keys_on_frame(
    frame: np.ndarray,
    keys: dict[str, bool],
    key_size: tuple[int, int] = (50, 50),
    spacing: int = 10,
    bottom_margin: int = 20,
) -> None:
    """Draw the WASD/Space/Shift/Ctrl HUD onto ``frame`` in place.

    Pressed keys are filled green, idle keys gray. Layout matches the
    reference Micro-World UI.
    """
    import cv2

    h, w, _ = frame.shape
    horizon_shift = 90
    vertical_shift = -20
    horizon_shift_all = 50
    key_positions = {
        "W": (
            w // 2 - key_size[0] // 2 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] * 2 + vertical_shift - 20,
        ),
        "A": (
            w // 2 - key_size[0] * 2 + 5 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] + vertical_shift,
        ),
        "S": (
            w // 2 - key_size[0] // 2 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] + vertical_shift,
        ),
        "D": (
            w // 2 + key_size[0] - 5 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] + vertical_shift,
        ),
        "Space": (
            w // 2 + key_size[0] * 2 + spacing * 2 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] + vertical_shift,
        ),
        "Shift": (
            w // 2 + key_size[0] * 3 + spacing * 7 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] + vertical_shift,
        ),
        "Ctrl": (
            w // 2 + key_size[0] * 4 + spacing * 12 - horizon_shift - horizon_shift_all,
            h - bottom_margin - key_size[1] + vertical_shift,
        ),
    }

    wide = {"Space", "Shift", "Ctrl"}
    for key, (x, y) in key_positions.items():
        is_pressed = keys.get(key, False)
        top_left = (x, y)
        bottom_right = (x + key_size[0] + (40 if key in wide else 0), y + key_size[1])

        color = (0, 255, 0) if is_pressed else (200, 200, 200)
        alpha = 0.8 if is_pressed else 0.5
        _draw_rounded_rectangle(frame, top_left, bottom_right, color, radius=10, alpha=alpha)

        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        if key in wide:
            text_x = x + (key_size[0] + 40 - text_size[0]) // 2
        else:
            text_x = x + (key_size[0] - text_size[0]) // 2
        text_y = y + (key_size[1] + text_size[1]) // 2
        cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def overlay_icon(
    frame: np.ndarray,
    icon: np.ndarray,
    position: tuple[float, float],
    scale: float = 0.2,
    rotation: float = -20,
) -> None:
    """Alpha-composite ``icon`` (RGBA) onto ``frame`` at ``position`` in place."""
    import cv2

    x, y = position
    h, w, _ = icon.shape

    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    icon_resized = cv2.resize(icon, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    center = (scaled_w // 2, scaled_h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    icon_rotated = cv2.warpAffine(
        icon_resized,
        rotation_matrix,
        (scaled_w, scaled_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    h, w, _ = icon_rotated.shape
    fh, fw, _ = frame.shape

    top_left_x = max(0, int(x - w // 2))
    top_left_y = max(0, int(y - h // 2))
    bottom_right_x = min(fw, int(x + w // 2))
    bottom_right_y = min(fh, int(y + h // 2))
    if bottom_right_x <= top_left_x or bottom_right_y <= top_left_y:
        return  # entirely off-frame

    icon_x_start = max(0, int(-x + w // 2))
    icon_y_start = max(0, int(-y + h // 2))
    icon_x_end = icon_x_start + (bottom_right_x - top_left_x)
    icon_y_end = icon_y_start + (bottom_right_y - top_left_y)

    icon_region = icon_rotated[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
    if icon_region.shape[2] == 4:
        alpha = icon_region[:, :, 3:4] / 255.0
        icon_rgb = icon_region[:, :, :3]
    else:
        alpha = np.ones((*icon_region.shape[:2], 1), dtype=np.float32)
        icon_rgb = icon_region

    frame_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x].astype(np.float32)
    frame_region = (1 - alpha) * frame_region + alpha * icon_rgb.astype(np.float32)
    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = frame_region.astype(frame.dtype)


def draw_hud(
    video: np.ndarray,
    mouse_actions: np.ndarray,
    keyboard_actions: np.ndarray,
    mouse_icon_path: str | Path | None = None,
) -> np.ndarray:
    """Apply keyboard + cursor HUD to every frame of an RGB video.

    Args:
        video: ``(num_frames, H, W, 3)`` array of RGB frames. Accepts
            uint8 ``[0, 255]`` or float ``[0, 1]``; the returned array
            preserves the input dtype/range.
        mouse_actions: ``(num_frames, 2)`` mouse_y, mouse_x deltas.
        keyboard_actions: ``(num_frames, 7)`` W S A D Space Shift Ctrl flags.
        mouse_icon_path: Optional override for the cursor PNG. Defaults to
            ``examples/offline_inference/micro_world/assets/mouse.png``.

    Returns:
        ``(num_frames, H, W, 3)`` array with the HUD drawn, same dtype as
        the input.
    """
    import cv2

    icon_path = Path(mouse_icon_path) if mouse_icon_path else DEFAULT_MOUSE_ICON
    # ``cv2.imread`` returns BGRA. Convert to RGBA so the alpha-composite
    # mixes against an RGB frame correctly.
    mouse_icon = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
    if mouse_icon is None:
        raise FileNotFoundError(f"mouse icon not found at {icon_path}")
    if mouse_icon.shape[2] == 4:
        mouse_icon = cv2.cvtColor(mouse_icon, cv2.COLOR_BGRA2RGBA)
    else:
        mouse_icon = cv2.cvtColor(mouse_icon, cv2.COLOR_BGR2RGB)

    original_dtype = video.dtype
    is_float = np.issubdtype(original_dtype, np.floating)
    if is_float:
        video = (np.clip(video, 0.0, 1.0) * 255).astype(np.uint8)
    video = np.ascontiguousarray(video)

    key_data, mouse_data = parse_config(mouse_actions, keyboard_actions)
    num_frames, height, width, _ = video.shape
    default_keys = {k: False for k in ("W", "A", "S", "D", "Space", "Shift", "Ctrl")}

    for frame_idx in range(num_frames):
        keys = key_data.get(frame_idx, default_keys)
        cursor_pos = mouse_data.get(frame_idx, (width / 2, height / 2))
        frame = video[frame_idx]
        draw_keys_on_frame(frame, keys)
        overlay_icon(frame, mouse_icon, cursor_pos)

    if is_float:
        video = video.astype(np.float32) / 255.0
    return video
