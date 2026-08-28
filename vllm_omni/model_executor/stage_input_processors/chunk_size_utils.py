# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import torch

logger = logging.getLogger(__name__)


def max_ic_for_chunk_size(chunk_size: int) -> int:
    """Largest power of 2 strictly less than chunk_size."""
    if chunk_size <= 2:
        return 1
    return 1 << ((chunk_size - 1).bit_length() - 1)


def compute_dynamic_initial_chunk_size(
    active_requests: int,
    max_num_seqs: int,
    max_ic: int,
) -> int:
    """Select IC from power-of-2 steps [2, 4, ..., max_ic] based on load factor.

    - Low load: small IC (faster TTFA).
    - High load: large IC (amortise decode cost).
    """
    steps: list[int] = []
    v = 2
    while v <= max_ic:
        steps.append(v)
        v <<= 1
    if not steps:
        return max(1, max_ic)
    if max_num_seqs <= 0:
        return steps[0]
    load_factor = min(active_requests / max_num_seqs, 1.0)
    idx = int(round(load_factor * (len(steps) - 1)))
    return steps[idx]


def parse_chunk_ramp(cfg: dict, steady: int | None = None) -> list[int] | None:
    """Parse ``codec_chunk_ramp`` from connector extra config.

    Accepts a list of positive ints or a comma-separated string.
    Returns ``None`` when the key is absent or invalid (ramp disabled).
    Requires >= 2 entries; all entries must be > 0.

    When ``steady`` is provided, warns if ``ramp[-1] != steady`` (the last
    ramp entry should match ``codec_chunk_frames`` to avoid reintroducing
    the cliff the ramp is meant to eliminate).
    """
    raw = cfg.get("codec_chunk_ramp")
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            raw = [int(x.strip()) for x in raw.split(",")]
        ramp = [int(x) for x in raw]
    except (TypeError, ValueError):
        logger.warning("codec_chunk_ramp parse error for %r; disabling ramp.", raw)
        return None
    if len(ramp) < 2:
        logger.warning("codec_chunk_ramp needs >= 2 entries, got %d; disabling.", len(ramp))
        return None
    if any(x <= 0 for x in ramp):
        logger.warning("codec_chunk_ramp entries must be positive; disabling.")
        return None
    if steady is not None and ramp[-1] != steady:
        logger.warning(
            "codec_chunk_ramp[-1]=%d != codec_chunk_frames=%d; this reintroduces the chunk-size cliff at index %d.",
            ramp[-1],
            steady,
            len(ramp) - 1,
        )
    return ramp


def ramp_chunk_size(index: int, ramp: list[int], steady: int) -> int:
    """Return the target chunk size for chunk ``index`` under the ramp schedule.

    Indices within the ramp table use ``ramp[index]``; indices past the table
    fall back to ``steady`` (typically ``codec_chunk_frames``).
    """
    if index < len(ramp):
        return ramp[index]
    return steady


def ramp_cumulative(index: int, ramp: list[int], steady: int) -> int:
    """Cumulative frame count through chunk ``index`` (inclusive).

    O(len(ramp)) closed form: sum the ramp table once, then add
    ``(index + 1 - len(ramp)) * steady`` for indices past the table.
    """
    ramp_len = len(ramp)
    if index < ramp_len:
        return sum(ramp[: index + 1])
    return sum(ramp) + (index + 1 - ramp_len) * steady


def compute_ramp_emit(
    length: int,
    chunk_index: int,
    ramp: list[int],
    steady: int,
    finished: bool,
) -> tuple[bool, int]:
    """Decide whether to emit a chunk under the ramp schedule.

    Uses cumulative thresholds to determine emission boundaries.  Each chunk
    ``i`` covers frames ``[ramp_cumulative(i-1), ramp_cumulative(i))``.

    Returns:
        ``(emit, context_length)``

        - ``emit=False, context_length=0``: not enough frames yet, hold.
        - ``emit=True, context_length>0``: emit with this many new frames.
        - ``emit=True, context_length=0``: finished with no new frames
            (caller should emit an empty-finished sentinel).
    """
    threshold = ramp_cumulative(chunk_index, ramp, steady)
    prev_threshold = ramp_cumulative(chunk_index - 1, ramp, steady) if chunk_index > 0 else 0

    if not finished and length < threshold:
        return False, 0

    if finished and length <= prev_threshold:
        return True, 0

    return True, length - prev_threshold


def emit_threshold(
    length: int,
    *,
    ramp: list[int] | None,
    chunk_index: int,
    chunk_size: int,
    initial_chunk_size: int,
) -> int:
    """Smallest committed-frame count at which a chunk could be emitted.

    Pure host arithmetic mirroring the emission gate below, used only to decide
    whether the pending frames are worth resolving yet. Returning a value that
    is too small costs an extra sync; returning one that is too large would
    delay a chunk, so every branch here has to match the gate.
    """
    if ramp is not None:
        return max(1, ramp_cumulative(chunk_index, ramp, chunk_size))
    use_first_chunk = 0 < initial_chunk_size < chunk_size
    if use_first_chunk and length < initial_chunk_size:
        return initial_chunk_size
    initial_coverage = initial_chunk_size if use_first_chunk else 0
    remainder = max(0, length - initial_coverage) % chunk_size
    return max(1, length + (chunk_size - remainder if remainder else 0))


def stack_frames(frames: list) -> torch.Tensor:
    """Stack frames into ``[num_frames, num_quantizers]`` on one device.

    Frames arriving from the talker are device tensors, but callers may seed
    ``code_prompt_token_ids`` with plain lists, so coerce rather than assume.
    A seeded host prefix followed by device frames would otherwise reach
    ``torch.stack`` as a mixed-device list and raise, and the adapter swallows
    processor exceptions, so that surfaces as a silently dropped chunk. Pick the
    first tensor's device and normalize everything onto it.
    """
    device = next((f.device for f in frames if isinstance(f, torch.Tensor)), None)
    return torch.stack([torch.as_tensor(f, dtype=torch.long, device=device) for f in frames])


def commit_pending_frames(committed: list, pending: list) -> None:
    """Move real frames from ``pending`` into ``committed``, dropping padding.

    One ``.tolist()`` for the whole pending batch, so the emptiness test costs a
    single device sync per resolve instead of one per decode step. Frames stay
    on-device; only the boolean mask crosses to the host.

    Padding is dropped *here*, not at emit, so ``len(committed)`` keeps meaning
    "real frames so far". The gate below is driven by that count, and roughly
    2% of mid-stream frames are all-zero padding (the talker zeroes any frame
    whose layer-0 id is out of codebook range, which covers EOS and the prefill
    span), so counting padding would move every chunk boundary.
    """
    if not pending:
        return
    keep = stack_frames(pending).any(dim=1).tolist()
    committed.extend(frame for frame, ok in zip(pending, keep) if ok)
    pending.clear()
