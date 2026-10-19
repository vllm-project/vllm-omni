# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure-function helpers for warmup-time diffusion batch budget calibration.

These functions contain no torch / distributed dependencies so they can be
unit-tested on CPU in isolation. The engine orchestrates the ladder sweep and
calls into these helpers to aggregate step records, decide early stop, and
pick the sweet-spot batch size.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from typing import Any

# Bump when the cached payload schema changes so old files are ignored.
CACHE_FORMAT_VERSION = 1


def generate_sparse_ladder(b_max: int) -> list[int]:
    """Return a sparse geometric ladder up to ``b_max``.

    Levels: powers of two (1, 2, 4, ...) plus ``round(0.75 * b_max)`` and
    ``b_max`` itself, ascending and de-duplicated. For 128 this yields about
    10 levels instead of 128.
    """
    b_max = max(1, int(b_max))
    levels: set[int] = set()

    level = 1
    while level <= b_max:
        levels.add(level)
        level *= 2

    levels.add(round(0.75 * b_max))
    levels.add(b_max)
    return sorted(levels)


def aggregate_step_records(
    records: list[tuple[int, int, float]],
    discard_steps: int,
) -> dict[int, float]:
    """Group per-step records by actual batch and return median ms per batch.

    Args:
        records: ``[(batch, step_index, ms)]`` collected by the runner.
        discard_steps: Number of leading steps (per batch) to drop; absorbs
            per-level compile / lazy-init noise.

    Returns:
        Mapping ``batch -> median step ms`` (only for batches with at least
        one surviving step).
    """
    by_batch: dict[int, list[float]] = defaultdict(list)
    for batch, _step_index, ms in records:
        by_batch[int(batch)].append(float(ms))

    discard = max(0, int(discard_steps))
    out: dict[int, float] = {}
    for batch, times in by_batch.items():
        kept = sorted(times)[discard:]
        if kept:
            out[batch] = statistics.median(kept)
    return out


def should_stop_early(
    prev: dict[str, float] | None,
    cur: dict[str, float],
    max_step_ms: float | None,
    min_marginal_gain: float,
) -> tuple[bool, str | None]:
    """Decide whether to stop the ladder sweep after measuring ``cur``.

    Budget condition takes priority over saturation.

    Args:
        prev: Previous ladder point ``{"b", "t", "thr"}`` or None.
        cur: Current ladder point ``{"b", "t", "thr"}``.
        max_step_ms: Per-step wall-clock budget in ms; None disables.
        min_marginal_gain: Minimum relative throughput gain to continue;
            ``<= 0`` disables the saturation check.

    Returns:
        ``(should_stop, reason_label)`` where reason is one of
        ``"budget"`` / ``"saturation"``.
    """
    if max_step_ms is not None and max_step_ms > 0:
        if cur["t"] > max_step_ms:
            return True, "budget"

    if min_marginal_gain > 0 and prev is not None:
        prev_thr = prev["thr"]
        if prev_thr > 0:
            gain = (cur["thr"] - prev_thr) / prev_thr
            if gain < min_marginal_gain:
                return True, "saturation"

    return False, None


def choose_sweet_spot_batch(
    points: list[dict[str, float]],
    max_step_ms: float | None,
    min_marginal_gain: float,
) -> tuple[int, dict[str, Any]]:
    """Pick the sweet-spot batch size from measured ladder points.

    Args:
        points: Measured points ``{"b", "t", "thr"}`` (ascending by ``b``).
        max_step_ms: Per-step budget; points with ``t > max_step_ms`` are
            filtered out when set.
        min_marginal_gain: Accepted for API compatibility; saturation already
            stopped the sweep, so it does not influence the winner.

    Returns:
        ``(b_star, analysis)`` where analysis carries method / points /
        throughput for logging and tests.
    """
    del min_marginal_gain  # used only during sweep; winner is max-throughput

    if not points:
        raise ValueError("choose_sweet_spot_batch requires at least one measured point.")

    if max_step_ms is not None and max_step_ms > 0:
        legal = [p for p in points if p["t"] <= max_step_ms]
        if not legal:
            fallback = min(points, key=lambda p: p["b"])
            return int(fallback["b"]), {
                "method": "fallback_min_batch",
                "points": points,
                "throughput": {int(p["b"]): p["thr"] for p in points},
            }
    else:
        legal = points

    best = max(legal, key=lambda p: (p["thr"], -p["b"]))
    return int(best["b"]), {
        "method": "max_throughput",
        "points": points,
        "throughput": {int(p["b"]): p["thr"] for p in points},
    }


def build_profile_cache_key(fields: dict[str, Any]) -> str:
    """Return a stable short digest of the calibration-relevant environment.

    Any change in the fields (model, GPU, reference resolution, sweep
    parameters, ...) yields a different key, so stale profiles are never
    silently reused across configurations.
    """
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_cached_profile(
    cache_dir: str,
    key: str,
    ttl_s: float = 0.0,
) -> dict[str, Any] | None:
    """Load a previously saved sweet-spot profile, or None on any problem.

    A missing/corrupt file, a schema-version mismatch or an expired entry
    (when ``ttl_s > 0``) all return None so the caller simply re-runs the
    sweep. ``b_star`` is validated because every downstream consumer keys
    off it.
    """
    path = os.path.join(cache_dir, f"{key}.json")
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("format_version") != CACHE_FORMAT_VERSION:
        return None
    if ttl_s > 0 and time.time() - float(payload.get("saved_at", 0.0)) > ttl_s:
        return None
    b_star = payload.get("b_star")
    if not isinstance(b_star, int) or b_star < 1:
        return None
    return payload


def save_cached_profile(
    cache_dir: str,
    key: str,
    payload: dict[str, Any],
) -> str | None:
    """Persist a sweet-spot profile atomically; returns the path or None.

    Never raises: the cache is best-effort and must not fail startup when
    the cache directory is unwritable.
    """
    full = dict(payload)
    full["format_version"] = CACHE_FORMAT_VERSION
    full["saved_at"] = time.time()
    path = os.path.join(cache_dir, f"{key}.json")
    tmp = os.path.join(cache_dir, f".{key}.{os.getpid()}.tmp")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(full, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return path
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
