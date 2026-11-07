# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Ported from the Apache-2.0 SeedVR2 reference implementation:
#   https://github.com/ByteDance-Seed/SeedVR (seedvr/models/...)
#   https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler (src/models/dit_3b/window.py)
"""SeedVR2 window geometry.

The SeedVR2 DiT (``NaDiT``) never attends globally: every layer partitions the
video tokens into 3D windows and attends only inside a window.  Window sizes are
derived from a *fixed 720p reference area* (``45 * 80``) rather than from the
requested resolution, so raising the output resolution increases the number of
windows, not the size of a window.

Two methods are used, alternating per layer:

``720pwin_by_size_bysize``
    Regular (non-shifted) tiling.
``720pswin_by_size_bysize``
    Swin-style shifted tiling.  The shift is realised with *boundary-clipped*
    slices (``st = sh = sw = 0.5`` window), **not** a cyclic ``torch.roll``.

Both methods produce an exact partition of the ``(T, H, W)`` post-patch token
grid: every token appears in exactly one window of a given layout.  The shifted
layout simply moves the cut positions by half a window, which is why a regular
layer followed by a shifted layer needs a re-shard.

The canonical video token id used throughout this package is the same one the
reference implementation obtains by windowing ``torch.arange`` of the flattened
sequence::

    token_id(t, h, w) = (t * H + h) * W + w

Windows are enumerated ``for iw ... for ih ... for it`` (w outermost, t
innermost) and tokens inside a window are flattened in ``(t, h, w)`` row-major
order, exactly matching the reference window functions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from math import ceil, sqrt

import torch

#: 720p reference area used to normalise window sizes (``BYTEDANCE_720P_REF_AREA``).
REFERENCE_AREA = 45 * 80

#: Temporal windows are capped at this many latent frames
#: (``BYTEDANCE_MAX_TEMPORAL_WINDOW``).
MAX_TEMPORAL_WINDOW = 30

#: Bumped whenever the semantics of the geometry below change in a way that
#: invalidates cached layouts / redistributions.
GEOMETRY_VERSION = 1

WindowSlices = list[tuple[slice, slice, slice]]
Size3 = tuple[int, int, int]


def _window_size(size: Size3, num_windows: Size3) -> tuple[int, int, int]:
    """Return ``(wt, wh, ww)`` for the 720p-normalised window grid."""
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    scale = sqrt(REFERENCE_AREA / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)
    wt = ceil(min(t, MAX_TEMPORAL_WINDOW) / resized_nt)
    return wt, wh, ww


def make_720p_windows(size: Size3, num_windows: Size3) -> WindowSlices:
    """Regular (non-shifted) window slices, reference-faithful."""
    t, h, w = size
    wt, wh, ww = _window_size(size, num_windows)
    nt, nh, nw = ceil(t / wt), ceil(h / wh), ceil(w / ww)
    return [
        (
            slice(it * wt, min((it + 1) * wt, t)),
            slice(ih * wh, min((ih + 1) * wh, h)),
            slice(iw * ww, min((iw + 1) * ww, w)),
        )
        for iw in range(nw)
        if min((iw + 1) * ww, w) > iw * ww
        for ih in range(nh)
        if min((ih + 1) * wh, h) > ih * wh
        for it in range(nt)
        if min((it + 1) * wt, t) > it * wt
    ]


def make_720p_shifted_windows(size: Size3, num_windows: Size3) -> WindowSlices:
    """Shifted window slices with boundary clipping, reference-faithful."""
    t, h, w = size
    wt, wh, ww = _window_size(size, num_windows)

    st, sh, sw = (
        0.5 if wt < t else 0,
        0.5 if wh < h else 0,
        0.5 if ww < w else 0,
    )
    nt, nh, nw = ceil((t - st) / wt), ceil((h - sh) / wh), ceil((w - sw) / ww)
    nt, nh, nw = (
        nt + 1 if st > 0 else 1,
        nh + 1 if sh > 0 else 1,
        nw + 1 if sw > 0 else 1,
    )
    return [
        (
            slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
            slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
            slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)),
        )
        for iw in range(nw)
        if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
        for ih in range(nh)
        if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
        for it in range(nt)
        if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
    ]


WINDOW_METHODS: dict[str, Callable[[Size3, Size3], WindowSlices]] = {
    "720pwin_by_size_bysize": make_720p_windows,
    "720pswin_by_size_bysize": make_720p_shifted_windows,
}

#: Default per-layer schedule of the 3B checkpoint: regular/shifted alternating.
DEFAULT_WINDOW_METHODS: tuple[str, ...] = ("720pwin_by_size_bysize", "720pswin_by_size_bysize")

#: ``window = num_layers * [(4, 3, 3)]`` in the released 3B config.
DEFAULT_WINDOW: Size3 = (4, 3, 3)


def get_window_op(name: str) -> Callable[[Size3, Size3], WindowSlices]:
    """Return the window function for a reference ``window_method`` name."""
    try:
        return WINDOW_METHODS[name]
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(f"Unknown windowing method: {name}") from None


def window_layout_geometry(size: Size3, window_method: str, window: Size3 = DEFAULT_WINDOW):
    """Build the CPU geometry (offsets, canonical ids, shapes) of one layout.

    Returns ``(window_offsets, window_token_ids, window_shapes)`` where

    * ``window_offsets`` is int64 ``[W + 1]`` (start row of every window in
      window-packed order),
    * ``window_token_ids`` is int64 ``[N]`` with the canonical token id of every
      packed row, and
    * ``window_shapes`` is int64 ``[W, 3]`` with the ``(t, h, w)`` extent of
      every window.

    The ids satisfy ``window_token_ids[offsets[i]:offsets[i + 1]]`` = canonical
    ids of window ``i``, in reference window order and reference in-window
    flatten order.
    """
    t, h, w = size
    if t <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"token grid must be positive, got {size}")
    window_slices = get_window_op(window_method)(size, window)

    ids: list[torch.Tensor] = []
    shapes: list[tuple[int, int, int]] = []
    offsets = [0]
    total = 0
    for st, sh, sw in window_slices:
        nt, nh, nw = st.stop - st.start, sh.stop - sh.start, sw.stop - sw.start
        if nt <= 0 or nh <= 0 or nw <= 0:
            continue
        tt = torch.arange(st.start, st.stop, dtype=torch.int64)
        hh = torch.arange(sh.start, sh.stop, dtype=torch.int64)
        ww = torch.arange(sw.start, sw.stop, dtype=torch.int64)
        grid = (tt[:, None, None] * h + hh[None, :, None]) * w + ww[None, None, :]
        ids.append(grid.reshape(-1))
        shapes.append((nt, nh, nw))
        total += nt * nh * nw
        offsets.append(total)

    if not ids:
        raise ValueError(f"window geometry produced no windows for size={size}, method={window_method}")

    window_token_ids = torch.cat(ids)
    window_shapes = torch.tensor(shapes, dtype=torch.int64)
    window_offsets = torch.tensor(offsets, dtype=torch.int64)

    # Every layout must be an exact partition of the token grid: this is the
    # invariant the whole redistribution design relies on.
    num_tokens = t * h * w
    if total != num_tokens:
        raise ValueError(
            f"window layout is not a partition of the grid: {total} packed rows for {num_tokens} tokens "
            f"(size={size}, method={window_method}, window={window})"
        )
    if not torch.equal(torch.sort(window_token_ids).values, torch.arange(num_tokens, dtype=torch.int64)):
        raise ValueError(f"window layout covers tokens with duplicates or holes (size={size}, method={window_method})")

    return window_offsets, window_token_ids, window_shapes


def geometry_fingerprint(
    size: Size3, window_method: str, window: Size3, geometry_version: int = GEOMETRY_VERSION
) -> str:
    """Deterministic fingerprint of a layout geometry.

    Uses SHA-1 over the parameters and the materialised window boundaries so the
    value is stable across processes (unlike Python's randomised ``hash()``).
    """
    hasher = hashlib.sha1()
    hasher.update(repr((tuple(size), window_method, tuple(window), int(geometry_version))).encode())
    window_slices = get_window_op(window_method)(size, window)
    for st, sh, sw in window_slices:
        hasher.update(f"{st.start}:{st.stop},{sh.start}:{sh.stop},{sw.start}:{sw.stop};".encode())
    return hasher.hexdigest()[:16]
