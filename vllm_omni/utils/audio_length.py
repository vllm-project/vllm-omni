"""Audio length helpers (pure-Python).

These utilities are used to keep code2wav (codec -> mel -> wav) lengths aligned
when applying a mel-frame cap.
"""

from __future__ import annotations


def resolve_max_mel_frames(max_mel_frames: int | None, *, default: int = 30000) -> int:
    """Resolve max mel frames from an explicit value or default.

    Args:
        max_mel_frames: Explicit value to use. If None, uses `default`.
        default: Default value to use when `max_mel_frames` is None.

    Returns:
        The resolved max mel frames value.
    """
    if max_mel_frames is not None:
        return int(max_mel_frames)
    return int(default)


def cap_and_align_mel_length(
    *,
    code_len: int,
    repeats: int,
    max_mel_frames: int | None,
    default_max_mel_frames: int = 30000,
) -> tuple[int, int]:
    """Compute a (target_code_len, target_mel_len) pair.

    - `mel_len` is always a multiple of `repeats` (codec expansion factor).
    - If `max_mel_frames` is None, uses `default_max_mel_frames`.
    - If `max_mel_frames` <= 0, no cap is applied (mel_len == code_len * repeats).
    - If `max_mel_frames` is smaller than `repeats` and code_len > 0, we still
      return at least one codec token worth of mel frames (mel_len == repeats)
      so downstream repeat-interleave stays valid.
    """
    code_len = int(code_len)
    repeats = int(repeats)
    if repeats <= 0:
        raise ValueError(f"repeats must be > 0, got {repeats}")
    if code_len <= 0:
        return 0, 0

    if max_mel_frames is None:
        max_mel_frames = int(default_max_mel_frames)
    else:
        max_mel_frames = int(max_mel_frames)

    maximum_duration = int(code_len * repeats)
    if max_mel_frames > 0:
        target_duration = min(maximum_duration, max_mel_frames)
    else:
        target_duration = maximum_duration

    # Align down to repeats; then ensure we keep at least one codec token.
    target_duration = (target_duration // repeats) * repeats
    if target_duration <= 0:
        target_duration = min(maximum_duration, repeats)

    target_code_len = target_duration // repeats
    if target_code_len <= 0:
        target_code_len = 1
        target_duration = repeats

    return int(target_code_len), int(target_duration)
