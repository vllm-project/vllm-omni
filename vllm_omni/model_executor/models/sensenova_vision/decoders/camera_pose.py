"""Parser for SenseNova-Vision camera-pose structured-text output.

The model emits camera pose estimates as a sequence of tagged frames, for
example::

    <frame>
    <quat>[x,y,z,w]</quat>
    <offset>[x,y,z]</offset>
    <scale>value</scale>
    </frame>

Ports ``inference/utils_3d/camera_pose_parser.py`` from the official
SenseNova-Vision repository. Values inside the tags are integers (milli-units)
and are converted to floating point by dividing by 1000; the final translation
is ``offset * scale / 100.0``.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["parse_camera_pose"]

_NUMBER_RE = re.compile(r"-?\d+")


def _extract_tag_values(input_str: str, tag: str, expected_count: int) -> np.ndarray:
    """Extract integer rows from repeated ``<tag>...</tag>`` blocks.

    Invalid blocks (wrong value count) are skipped. Values are converted from
    the model's milli-units by dividing by 1000.
    """
    if not isinstance(input_str, str):
        raise TypeError(f"Input must be a string, got {type(input_str)}")

    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", flags=re.DOTALL)
    contents = pattern.findall(input_str)

    if not contents:
        return np.array([])

    values = []
    for content in contents:
        numbers = _NUMBER_RE.findall(content)
        if len(numbers) != expected_count:
            continue
        values.append([int(x) / 1000 for x in numbers])

    return np.array(values) if values else np.array([])


def _extract_scales(input_str: str) -> np.ndarray:
    """Extract integer scale values from ``<scale>...</scale>`` blocks."""
    if not isinstance(input_str, str):
        raise TypeError(f"Input must be a string, got {type(input_str)}")

    contents = re.findall(r"<scale>(.*?)</scale>", input_str, flags=re.DOTALL)
    scales = []
    for content in contents:
        numbers = _NUMBER_RE.findall(content)
        if not numbers:
            continue
        scales.append(int(numbers[0]))
    return np.array(scales) if scales else np.array([])


def parse_camera_pose(pose_str: str) -> dict | None:
    """Parse a camera pose string into quaternion rotations and translations.

    Args:
        pose_str: String containing ``<quat>``, ``<offset>`` and ``<scale>``
            tags, optionally wrapped in ``<frame>...</frame>`` blocks. When
            ``<frame>`` tags are present, each frame is parsed independently
            and its records are concatenated.

    Returns:
        A dict with ``rotation`` (N x 4 quaternion ``[x, y, z, w]``) and
        ``translation`` (N x 3) lists of floats, or ``None`` when the string
        cannot be parsed (missing/invalid tags or count mismatches).
    """
    if not isinstance(pose_str, str):
        raise TypeError(f"Input must be a string, got {type(pose_str)}")

    frames = re.findall(r"<frame>(.*?)</frame>", pose_str, flags=re.DOTALL)
    if not frames:
        frames = [pose_str]

    quat_rows: list[list[float]] = []
    offset_rows: list[list[float]] = []
    scale_rows: list[float] = []

    for frame in frames:
        quat_block = _extract_tag_values(frame, "quat", 4)
        offset_block = _extract_tag_values(frame, "offset", 3)
        scale_block = _extract_scales(frame)

        if offset_block.size == 0 or scale_block.size == 0:
            continue
        if offset_block.shape[0] != scale_block.shape[0]:
            continue
        if quat_block.size != 0 and quat_block.shape[0] != offset_block.shape[0]:
            continue

        count = offset_block.shape[0]
        if quat_block.size == 0:
            quat_block = np.zeros((count, 4), dtype=np.float32)

        quat_rows.extend(quat_block.tolist())
        offset_rows.extend(offset_block.tolist())
        scale_rows.extend(scale_block.tolist())

    if not offset_rows or not scale_rows:
        return None

    offset_array = np.asarray(offset_rows, dtype=np.float32)
    scale_array = np.asarray(scale_rows, dtype=np.float32)
    quat_array = np.asarray(quat_rows, dtype=np.float32)

    valid_mask = np.isfinite(offset_array).all(axis=1) & np.isfinite(scale_array)
    valid_mask = valid_mask & np.isfinite(quat_array).all(axis=1)

    offset_array = offset_array[valid_mask]
    scale_array = scale_array[valid_mask]
    quat_array = quat_array[valid_mask]

    if offset_array.size == 0 or scale_array.size == 0:
        return None

    translation_array = offset_array * scale_array[:, None] / 100.0

    return {
        "rotation": quat_array.tolist(),
        "translation": translation_array.tolist(),
    }
