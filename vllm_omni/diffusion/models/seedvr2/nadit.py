# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SeedVR2 DiT: window geometry, sequence parallelism, rotary embeddings and NaDiT."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import ceil, sqrt

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group


def validate_seedvr2_parallel_config(parallel: DiffusionParallelConfig) -> None:
    if (
        parallel.sequence_parallel_size != parallel.ulysses_degree
        or parallel.ring_degree != 1
        or parallel.allgather_degree != 1
        or parallel.ulysses_mode != "strict"
        or parallel.ulysses_a2a_permute
    ):
        raise ValueError(
            "SeedVR2 window SP requires pure ulysses_degree with ring_degree=allgather_degree=1, "
            "ulysses_mode='strict', and ulysses_a2a_permute=False. "
            "The degree selects the SP group; SeedVR2 keeps all attention heads on each rank."
        )


# Ported from the Apache-2.0 SeedVR2 reference implementation:
#   https://github.com/ByteDance-Seed/SeedVR (seedvr/models/...)
#   https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler (src/models/dit_3b/window.py)


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


# Vendored from the Apache-2.0 SeedVR2 reference implementation
# (`src/models/dit_3b/rope.py`) and the MIT-licensed `rotary_embedding_torch`
# primitives it builds on, so the port does not add a runtime dependency.


#: The reference always builds the axial table for the *first* ``dim // 2``
#: frequency slots of a ``lang`` frequency bank.
LANG_THETA = 10000.0


def lang_freqs(axis_dim: int, theta: float = LANG_THETA) -> torch.Tensor:
    """``1 / theta ** (arange(0, axis_dim, 2) / axis_dim)`` (``rotary_embedding_torch``)."""
    return 1.0 / (theta ** (torch.arange(0, axis_dim, 2)[: axis_dim // 2].float() / axis_dim))


def _axis_table(positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Outer product of positions with the frequency bank, each freq duplicated."""
    angles = positions.float().unsqueeze(-1) * freqs.float().unsqueeze(0)
    return torch.repeat_interleave(angles, 2, dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Rotate ``t`` with precomputed ``freqs`` (fp32 math, original dtype out).

    Mirrors ``rotary_embedding_torch.apply_rotary_emb`` for the 2-D frequency
    tables and 3-D ``(heads, seq, dim)`` inputs used here: the rotated span is
    ``freqs.shape[-1]`` channels and any remaining channels are untouched.
    """
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    if rot_dim > t.shape[-1]:
        raise ValueError(f"rotary dim {rot_dim} exceeds the feature dim {t.shape[-1]}")
    if freqs.shape[0] < t.shape[-2]:
        raise ValueError(f"freqs cover {freqs.shape[0]} positions but the sequence has {t.shape[-2]}")
    freqs = freqs[-t.shape[-2] :].to(torch.float32)

    t_middle = t[..., :rot_dim].float()
    rotated = t_middle * freqs.cos() + rotate_half(t_middle) * freqs.sin()
    if rot_dim == t.shape[-1]:
        return rotated.to(dtype)
    return torch.cat((rotated, t[..., rot_dim:]), dim=-1).to(dtype)


class NaMMRotaryEmbedding3d(nn.Module):
    """Reference ``mmrope3d``: axial 3-D RoPE for windows plus 1-D RoPE for text."""

    mm = True

    def __init__(self, rotary_dim: int, num_axes: int = 3, theta: float = LANG_THETA) -> None:
        super().__init__()
        if num_axes <= 0 or rotary_dim < num_axes:
            raise ValueError(f"invalid rotary_dim={rotary_dim} for num_axes={num_axes}")
        self.rotary_dim = int(rotary_dim)
        self.num_axes = int(num_axes)
        # Reference: ``RotaryEmbedding(dim=rope_dim // 3, freqs_for="lang")`` and
        # the axial table concatenates one such bank per axis.
        self.axis_dim = self.rotary_dim // self.num_axes
        self.rot_dim = self.axis_dim * self.num_axes
        if self.rot_dim % 2:
            raise ValueError(
                f"rotary_dim={rotary_dim} with num_axes={num_axes} gives an odd rotary span "
                f"({self.rot_dim}); the reference pairs channels two at a time"
            )
        # Registered as `freqs`.  The released checkpoint stores this table one
        # module deeper (`...attn.rope.rope.freqs`) than this port registers it, so
        # the validation loader normalizes exactly that suffix; every other key
        # must already match the port's parameter layout.
        self.register_buffer("freqs", lang_freqs(self.axis_dim, theta), persistent=True)
        self._axis_cache: dict[tuple, torch.Tensor] = {}

    # -- frequency construction -------------------------------------------
    def _axis_angles(self, positions: torch.Tensor, *, device, dtype) -> torch.Tensor:
        key = (positions.numel(), int(positions[0]), float(positions[-1]), str(device), str(dtype))
        cached = self._axis_cache.get(key)
        if cached is None:
            cached = _axis_table(positions.to(torch.float32), self.freqs.float()).to(device=device, dtype=dtype)
            if len(self._axis_cache) > 256:
                self._axis_cache.clear()
            self._axis_cache[key] = cached
        return cached

    def _table(self, dims: tuple[int, ...], *, device, dtype) -> torch.Tensor:
        """Axial table of shape ``(*dims, rot_dim)`` (``get_axial_freqs``)."""
        axes = []
        ref = self.freqs
        for axis, dim in enumerate(dims):
            positions = torch.arange(dim, device=ref.device, dtype=torch.float32)
            angles = self._axis_angles(positions, device=device, dtype=dtype)
            shape = [1] * len(dims) + [angles.shape[-1]]
            shape[axis] = dim
            axes.append(angles.reshape(shape))
        return torch.cat(torch.broadcast_tensors(*axes), dim=-1)

    # -- per-window / per-text positions ----------------------------------
    def window_freqs(self, window_shape: tuple[int, int, int], text_len: int, *, device, dtype) -> torch.Tensor:
        """``[f * h * w, rot_dim]`` frequencies of one window, window-local."""
        f, h, w = (int(v) for v in window_shape)
        table = self._table((text_len + f, h, w), device=device, dtype=dtype)
        return table[text_len : text_len + f].reshape(-1, table.shape[-1])

    def window_freqs_batch(
        self,
        window_shapes: torch.Tensor | list[tuple[int, int, int]],
        text_len: int,
        *,
        device,
        dtype,
    ) -> torch.Tensor:
        """Concatenated per-window video frequencies for a window batch."""
        if isinstance(window_shapes, torch.Tensor):
            shapes = [tuple(int(v) for v in row) for row in window_shapes.tolist()]
        else:
            shapes = [(int(a), int(b), int(c)) for a, b, c in window_shapes]
        if not shapes:
            return torch.empty((0, self.rot_dim), device=device, dtype=dtype)
        return torch.cat([self.window_freqs(s, text_len, device=device, dtype=dtype) for s in shapes], dim=0)

    def text_freqs(self, text_len: int, *, device, dtype) -> torch.Tensor:
        """``[text_len, rot_dim]`` text frequencies (one axis repeated ``rope_dim`` times)."""
        if text_len <= 0:
            return torch.empty((0, self.rot_dim), device=device, dtype=dtype)
        angles = self._axis_angles(
            torch.arange(text_len, device=self.freqs.device, dtype=torch.float32), device=device, dtype=dtype
        )
        return angles.repeat(1, self.num_axes)

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        vid_q: torch.Tensor,  # (L h d)
        vid_k: torch.Tensor,
        vid_freqs: torch.Tensor,  # (L rot_dim)
        txt_q: torch.Tensor,  # (l h d)
        txt_k: torch.Tensor,
        txt_freqs: torch.Tensor,  # (l rot_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vid_q = apply_rotary_emb(vid_freqs, vid_q.transpose(0, 1)).transpose(0, 1)
        vid_k = apply_rotary_emb(vid_freqs, vid_k.transpose(0, 1)).transpose(0, 1)
        txt_q = apply_rotary_emb(txt_freqs, txt_q.transpose(0, 1)).transpose(0, 1)
        txt_k = apply_rotary_emb(txt_freqs, txt_k.transpose(0, 1)).transpose(0, 1)
        return vid_q, vid_k, txt_q, txt_k


#: Bumped whenever assignment or routing semantics change.
PLANNER_VERSION = 1

#: int32 ``cu_seqlens`` bounds (torch's varlen kernels take int32 offsets).
INT32_MAX = 2**31 - 1


# =============================================================================
# Layout / assignment / plan data structures (CPU metadata)
# =============================================================================


@dataclass(frozen=True)
class WindowLayoutKey:
    """Identity of one window layout (shape + geometry, not ownership)."""

    token_grid: tuple[int, int, int]
    window_method: str
    window_shape: tuple[int, int, int]
    geometry_version: int
    geometry_fingerprint: str


@dataclass(frozen=True, eq=False)
class WindowLayout:
    """A partition of the video tokens into windows, in window-packed order.

    ``window_token_ids`` holds canonical token ids (``(t * H + h) * W + w``) of
    every packed row; ``window_offsets`` delimits the windows inside it.
    """

    key: WindowLayoutKey
    num_tokens: int
    window_offsets: torch.Tensor  # CPU int64 [W + 1]
    window_token_ids: torch.Tensor  # CPU int64 [N]
    window_shapes: torch.Tensor  # CPU int64 [W, 3]

    @property
    def num_windows(self) -> int:
        return int(self.window_offsets.numel() - 1)

    def window_lengths(self) -> torch.Tensor:
        return (self.window_offsets[1:] - self.window_offsets[:-1]).to(torch.int64)

    def window_token_ids_of(self, window_ids: torch.Tensor) -> torch.Tensor:
        """Canonical ids of the given windows, concatenated in that order."""
        if window_ids.numel() == 0:
            return torch.empty(0, dtype=torch.int64)
        ids = window_ids.tolist()
        pieces = [self.window_token_ids[int(self.window_offsets[w]) : int(self.window_offsets[w + 1])] for w in ids]
        return torch.cat(pieces)


@dataclass(frozen=True, eq=False)
class WindowAssignment:
    """Which rank owns which window (a whole window belongs to one rank)."""

    layout_key: WindowLayoutKey
    world_size: int
    planner_version: int
    owner_rank: torch.Tensor  # CPU int64 [W]
    rank_window_ids: tuple[torch.Tensor, ...]
    rank_token_counts: tuple[int, ...]
    rank_window_counts: tuple[int, ...]
    max_window_tokens: int = 0

    @property
    def max_rank_tokens(self) -> int:
        return max(self.rank_token_counts) if self.rank_token_counts else 0

    @property
    def ideal_lower_bound(self) -> int:
        """Largest window, or perfectly balanced load, whichever is bigger."""
        total = sum(self.rank_token_counts)
        return max(self.max_window_tokens, math.ceil(total / self.world_size))

    def load_imbalance(self) -> float:
        total = sum(self.rank_token_counts)
        if total == 0:
            return 1.0
        return self.max_rank_tokens / (total / self.world_size)


@dataclass(frozen=True, eq=False)
class RankWindowPlan:
    """Rank-local view of a layout: what this rank owns and how it is packed."""

    layout_key: WindowLayoutKey
    rank: int
    window_ids: torch.Tensor  # CPU int64 [W_r]
    global_token_ids: torch.Tensor  # CPU int64 [N_r], canonical ids in local row order
    video_cu_seqlens: torch.Tensor  # CPU int32 [W_r + 1]
    window_shapes: torch.Tensor  # CPU int64 [W_r, 3]

    @property
    def num_local_tokens(self) -> int:
        return int(self.global_token_ids.numel())

    @property
    def num_local_windows(self) -> int:
        return int(self.window_ids.numel())


@dataclass(frozen=True, eq=False)
class WindowRedistributionPlan:
    """Plan A routing contract for one rank of one layout transition.

    ``input_split_sizes[d]`` is the number of rows this rank sends to rank ``d``
    and ``output_split_sizes[s]`` the number of rows it receives from rank
    ``s`` (feature rows, not bytes).  ``send_indices`` selects rows of the
    local source tensor in send order; ``recv_to_dst_indices`` permutes the
    received rows into destination-local order.
    """

    src_key: WindowLayoutKey
    dst_key: WindowLayoutKey
    world_size: int
    rank: int
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    send_indices: torch.Tensor  # CPU int64 [N_src]
    recv_to_dst_indices: torch.Tensor  # CPU int64 [N_dst]
    network_exchange_required: bool
    planner_version: int = PLANNER_VERSION

    @property
    def num_send_rows(self) -> int:
        return int(self.send_indices.numel())

    @property
    def num_recv_rows(self) -> int:
        return int(self.recv_to_dst_indices.numel())


@dataclass(frozen=True)
class DeviceWindowRedistributionPlan:
    """Device-side materialisation of :class:`WindowRedistributionPlan`."""

    send_indices: torch.Tensor  # int64 [N_src]
    recv_to_dst_indices: torch.Tensor  # int64 [N_dst]
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    num_recv_rows: int
    network_exchange_required: bool


# =============================================================================
# Layouts and assignment
# =============================================================================


def build_window_layout(
    token_grid: tuple[int, int, int],
    window_method: str,
    *,
    window: tuple[int, int, int] = DEFAULT_WINDOW,
    geometry_version: int = GEOMETRY_VERSION,
) -> WindowLayout:
    """Build the CPU layout of one layer's window partition."""

    window_offsets, window_token_ids, window_shapes = window_layout_geometry(token_grid, window_method, window)
    key = WindowLayoutKey(
        token_grid=(int(token_grid[0]), int(token_grid[1]), int(token_grid[2])),
        window_method=window_method,
        window_shape=(int(window[0]), int(window[1]), int(window[2])),
        geometry_version=int(geometry_version),
        geometry_fingerprint=geometry_fingerprint(token_grid, window_method, window, geometry_version),
    )
    return WindowLayout(
        key=key,
        num_tokens=int(window_token_ids.numel()),
        window_offsets=window_offsets,
        window_token_ids=window_token_ids,
        window_shapes=window_shapes,
    )


def build_window_assignment(
    layout: WindowLayout,
    world_size: int,
    *,
    planner_version: int = PLANNER_VERSION,
) -> WindowAssignment:
    """Deterministic token-balanced window -> rank assignment (LPT).

    Windows are considered in ``(-video_token_count, window_id)`` order and each
    is placed on the rank with the smallest ``(assigned_tokens, rank_id)``.  The
    result is independent of container iteration order.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    lengths = [int(v) for v in layout.window_lengths().tolist()]
    if not lengths:
        raise ValueError("layout has no windows")
    if any(n <= 0 for n in lengths):
        raise ValueError(f"every window must have a positive token count, got {lengths}")

    loads = [0] * world_size
    buckets: list[list[int]] = [[] for _ in range(world_size)]
    for w in sorted(range(len(lengths)), key=lambda w: (-lengths[w], w)):
        r = min(range(world_size), key=lambda r: (loads[r], r))
        buckets[r].append(w)
        loads[r] += lengths[w]

    rank_window_ids = tuple(torch.tensor(sorted(b), dtype=torch.int64) for b in buckets)
    owner_rank = torch.empty(len(lengths), dtype=torch.int64)
    for r, ids in enumerate(rank_window_ids):
        if ids.numel():
            owner_rank[ids] = r

    return WindowAssignment(
        layout_key=layout.key,
        world_size=world_size,
        planner_version=planner_version,
        owner_rank=owner_rank,
        rank_window_ids=rank_window_ids,
        rank_token_counts=tuple(loads),
        rank_window_counts=tuple(int(ids.numel()) for ids in rank_window_ids),
        max_window_tokens=max(lengths),
    )


def build_rank_window_plan(layout: WindowLayout, assignment: WindowAssignment, rank: int) -> RankWindowPlan:
    """Rank-local metadata for one layout (used for windows/attention packing)."""
    _check_same_layout(layout, assignment.layout_key)
    if not 0 <= rank < assignment.world_size:
        raise ValueError(f"rank {rank} out of range for world_size {assignment.world_size}")

    window_ids = assignment.rank_window_ids[rank]
    lengths = layout.window_lengths()
    if window_ids.numel():
        local_lengths = lengths[window_ids]
        local_counts = torch.cumsum(local_lengths, dim=0)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), local_counts])
        global_token_ids = layout.window_token_ids_of(window_ids)
        window_shapes = layout.window_shapes[window_ids]
    else:
        offsets = torch.zeros(1, dtype=torch.int64)
        global_token_ids = torch.empty(0, dtype=torch.int64)
        window_shapes = torch.empty((0, 3), dtype=torch.int64)

    video_cu_seqlens = to_cu_seqlens(offsets, context=f"rank {rank} video packing")
    return RankWindowPlan(
        layout_key=layout.key,
        rank=rank,
        window_ids=window_ids,
        global_token_ids=global_token_ids,
        video_cu_seqlens=video_cu_seqlens,
        window_shapes=window_shapes,
    )


def to_cu_seqlens(offsets: torch.Tensor, *, context: str = "") -> torch.Tensor:
    """Convert int64 offsets to int32 ``cu_seqlens`` with an explicit overflow check."""
    offsets = offsets.to(torch.int64)
    if offsets.numel() == 0:
        raise ValueError("cu_seqlens must contain at least the leading zero")
    if int(offsets[0]) != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {int(offsets[0])} ({context})")
    if int(offsets[-1]) > INT32_MAX:
        raise ValueError(f"sequence length {int(offsets[-1])} exceeds int32 cu_seqlens range ({context})")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError(f"cu_seqlens must be non-decreasing ({context})")
    return offsets.to(torch.int32)


def joint_cu_seqlens(video_cu_seqlens: torch.Tensor, text_len: int) -> torch.Tensor:
    """``[W + 1]`` int32 joint (video + replicated text) offsets for local attention.

    ``video_cu_seqlens`` describes video packing only and must not be used
    directly as joint Q/K offsets.  The offsets are accumulated in int64 on the
    input's device and converted to int32 only after validation, so CUDA
    metadata never round-trips through the host and an overflow is caught before
    a wrapped offset can be produced.  ``text_len`` may be 0; an input of ``[0]``
    (a rank without windows) yields ``[0]``.
    """
    if text_len < 0:
        raise ValueError(f"text_len must be >= 0, got {text_len}")
    video = video_cu_seqlens.to(torch.int64)
    if video.numel() == 0:
        raise ValueError("video_cu_seqlens must contain at least the leading zero")
    if bool((video[0] != 0).item()):
        raise ValueError(f"video_cu_seqlens must start at 0, got {int(video[0])}")
    if bool((video[1:] < video[:-1]).any().item()):
        raise ValueError("video_cu_seqlens must be non-decreasing")
    windows = video.numel() - 1
    text_offsets = torch.arange(windows + 1, device=video.device, dtype=torch.int64) * int(text_len)
    return to_cu_seqlens(video + text_offsets, context="joint video+text packing")


# =============================================================================
# Plan A routing
# =============================================================================


def _check_same_layout(layout: WindowLayout, key: WindowLayoutKey) -> None:
    if layout.key != key:
        raise ValueError(
            "layout/assignment mismatch: "
            f"layout={layout.key.token_grid}/{layout.key.window_method}, "
            f"assignment={key.token_grid}/{key.window_method}"
        )


def _rank_global_token_ids(layout: WindowLayout, assignment: WindowAssignment) -> list[torch.Tensor]:
    return [layout.window_token_ids_of(assignment.rank_window_ids[r]) for r in range(assignment.world_size)]


def build_redistribution_plans(
    src_layout: WindowLayout,
    src_assignment: WindowAssignment,
    dst_layout: WindowLayout,
    dst_assignment: WindowAssignment,
    *,
    planner_version: int = PLANNER_VERSION,
) -> tuple[WindowRedistributionPlan, ...]:
    """Build the bidirectional Plan A routing plans for every rank.

    Returns one :class:`WindowRedistributionPlan` per rank, indexed by SP-local
    rank.  Both directions (regular -> shifted and shifted -> regular) are plain
    calls to this function; neither is derived from the other by transposing.
    """
    _check_same_layout(src_layout, src_assignment.layout_key)
    _check_same_layout(dst_layout, dst_assignment.layout_key)
    if src_assignment.world_size != dst_assignment.world_size:
        raise ValueError(
            f"source world_size {src_assignment.world_size} != destination world_size {dst_assignment.world_size}"
        )
    if src_layout.num_tokens != dst_layout.num_tokens:
        raise ValueError(f"layouts cover different token counts: {src_layout.num_tokens} != {dst_layout.num_tokens}")
    world_size = src_assignment.world_size

    src_ids = _rank_global_token_ids(src_layout, src_assignment)
    dst_ids = _rank_global_token_ids(dst_layout, dst_assignment)
    num_tokens = src_layout.num_tokens

    # global token id -> (source owner rank, local row)
    src_owner = torch.full((num_tokens,), -1, dtype=torch.int64)
    src_local_pos = torch.full((num_tokens,), -1, dtype=torch.int64)
    for r, ids in enumerate(src_ids):
        if ids.numel():
            src_owner[ids] = r
            src_local_pos[ids] = torch.arange(ids.numel(), dtype=torch.int64)
    if bool((src_owner < 0).any()):
        raise ValueError("source assignment does not cover every token exactly once")

    counts = torch.zeros((world_size, world_size), dtype=torch.int64)  # counts[src, dst]
    send_rows: dict[tuple[int, int], torch.Tensor] = {}
    dst_positions: dict[tuple[int, int], torch.Tensor] = {}

    for d in range(world_size):
        z = dst_ids[d]
        if z.numel():
            owners = src_owner[z]
            if bool((owners < 0).any()):
                raise ValueError(f"destination rank {d} requires tokens missing from the source layouts")
            order = torch.argsort(owners, stable=True)
            offsets = torch.cumsum(torch.bincount(owners, minlength=world_size), dim=0)
        else:
            order = torch.empty(0, dtype=torch.int64)
            offsets = torch.zeros(world_size, dtype=torch.int64)

        start = 0
        for r in range(world_size):
            end = int(offsets[r])
            seg = order[start:end]
            start = end
            counts[r, d] = end - (int(offsets[r - 1]) if r else 0)
            if seg.numel():
                send_rows[(r, d)] = src_local_pos[z[seg]]
                dst_positions[(r, d)] = seg
            else:
                send_rows[(r, d)] = torch.empty(0, dtype=torch.int64)
                dst_positions[(r, d)] = torch.empty(0, dtype=torch.int64)

    # Group consistency: computed from the global count matrix, so every rank
    # reaches the same decision and no rank skips a collective alone.
    off_diagonal = counts.clone()
    off_diagonal.fill_diagonal_(0)
    network_exchange_required = bool(int(off_diagonal.sum()) > 0)

    plans: list[WindowRedistributionPlan] = []
    for rank in range(world_size):
        send_indices = torch.cat([send_rows[(rank, d)] for d in range(world_size)])
        received_dst_positions = torch.cat([dst_positions[(s, rank)] for s in range(world_size)])
        num_src_rows = int(src_ids[rank].numel())
        num_dst_rows = int(dst_ids[rank].numel())
        if send_indices.numel() != num_src_rows:
            raise ValueError(f"rank {rank} routes {send_indices.numel()} rows but owns {num_src_rows} source rows")
        if received_dst_positions.numel() != num_dst_rows:
            raise ValueError(
                f"rank {rank} receives {received_dst_positions.numel()} rows but needs {num_dst_rows} destination rows"
            )
        if num_dst_rows and not torch.equal(
            torch.sort(received_dst_positions).values, torch.arange(num_dst_rows, dtype=torch.int64)
        ):
            raise ValueError(f"rank {rank} received positions are not a permutation of its destination rows")
        recv_to_dst_indices = torch.argsort(received_dst_positions)

        input_split_sizes = tuple(int(v) for v in counts[rank].tolist())
        output_split_sizes = tuple(int(v) for v in counts[:, rank].tolist())
        if sum(input_split_sizes) != num_src_rows or sum(output_split_sizes) != num_dst_rows:
            raise ValueError(
                f"rank {rank} split sizes are inconsistent with its row counts "
                f"(send {sum(input_split_sizes)}/{num_src_rows}, recv {sum(output_split_sizes)}/{num_dst_rows})"
            )

        plans.append(
            WindowRedistributionPlan(
                src_key=src_layout.key,
                dst_key=dst_layout.key,
                world_size=world_size,
                rank=rank,
                input_split_sizes=input_split_sizes,
                output_split_sizes=output_split_sizes,
                send_indices=send_indices,
                recv_to_dst_indices=recv_to_dst_indices,
                network_exchange_required=network_exchange_required,
                planner_version=planner_version,
            )
        )
    return tuple(plans)


# =============================================================================
# Cache
# =============================================================================


class WindowPlanCache:
    """Small LRU cache for layouts, assignments, rank plans and routes.

    The assignment / routing identity includes the SP world size, the planner
    version and the geometry fingerprint, so a change in any of them cannot hit
    a stale entry.  Runtime (device) objects are intentionally *not* cached
    here: they are owned by the caller and must be released when the process
    group or device changes.
    """

    def __init__(self, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._layouts: OrderedDict[WindowLayoutKey, WindowLayout] = OrderedDict()
        self._assignments: OrderedDict[tuple, WindowAssignment] = OrderedDict()
        self._rank_plans: OrderedDict[tuple, RankWindowPlan] = OrderedDict()
        self._routes: OrderedDict[tuple, tuple[WindowRedistributionPlan, ...]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _touch(self, cache: OrderedDict, key):
        value = cache.pop(key)
        cache[key] = value
        return value

    def _store(self, cache: OrderedDict, key, value):
        cache[key] = value
        while len(cache) > self._capacity:
            cache.popitem(last=False)
        return value

    def clear(self) -> None:
        self._layouts.clear()
        self._assignments.clear()
        self._rank_plans.clear()
        self._routes.clear()

    def layout(self, token_grid, window_method, window=DEFAULT_WINDOW, geometry_version=GEOMETRY_VERSION):
        key = (tuple(token_grid), window_method, tuple(window), int(geometry_version))
        if key in self._layouts:
            self.hits += 1
            return self._touch(self._layouts, key)
        self.misses += 1
        layout = build_window_layout(token_grid, window_method, window=window, geometry_version=geometry_version)
        return self._store(self._layouts, key, layout)

    def assignment(self, layout: WindowLayout, world_size: int, planner_version: int = PLANNER_VERSION):
        key = (layout.key, int(world_size), int(planner_version))
        if key in self._assignments:
            self.hits += 1
            return self._touch(self._assignments, key)
        self.misses += 1
        return self._store(
            self._assignments, key, build_window_assignment(layout, world_size, planner_version=planner_version)
        )

    def rank_plan(self, layout: WindowLayout, assignment: WindowAssignment, rank: int):
        key = (
            layout.key,
            assignment.layout_key,
            int(assignment.world_size),
            int(assignment.planner_version),
            int(rank),
        )
        if key in self._rank_plans:
            self.hits += 1
            return self._touch(self._rank_plans, key)
        self.misses += 1
        return self._store(self._rank_plans, key, build_rank_window_plan(layout, assignment, rank))

    def route(
        self,
        src_layout: WindowLayout,
        src_assignment: WindowAssignment,
        dst_layout: WindowLayout,
        dst_assignment: WindowAssignment,
    ):
        key = (
            src_layout.key,
            src_assignment.world_size,
            src_assignment.planner_version,
            dst_layout.key,
            dst_assignment.world_size,
            dst_assignment.planner_version,
        )
        if key in self._routes:
            self.hits += 1
            return self._touch(self._routes, key)
        self.misses += 1
        plans = build_redistribution_plans(src_layout, src_assignment, dst_layout, dst_assignment)
        return self._store(self._routes, key, plans)


# =============================================================================
# Runtime (device)
# =============================================================================


def materialize_redistribution_plan(
    plan: WindowRedistributionPlan, *, device: torch.device | str
) -> DeviceWindowRedistributionPlan:
    """Move one CPU redistribution plan to the device (index tensors only)."""
    device = torch.device(device)
    return DeviceWindowRedistributionPlan(
        send_indices=plan.send_indices.to(device=device, dtype=torch.int64, non_blocking=True),
        recv_to_dst_indices=plan.recv_to_dst_indices.to(device=device, dtype=torch.int64, non_blocking=True),
        input_split_sizes=plan.input_split_sizes,
        output_split_sizes=plan.output_split_sizes,
        num_recv_rows=plan.num_recv_rows,
        network_exchange_required=plan.network_exchange_required,
    )


def redistribute_window_rows(
    hidden: torch.Tensor,
    plan: DeviceWindowRedistributionPlan,
    *,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Re-shard local video rows from one window layout to another (Plan A).

    ``hidden`` is ``[N_src, C]`` (contiguous trailing feature dimension) in the
    source layout's rank-local row order; the result is ``[N_dst, C]`` in the
    destination layout's rank-local row order.  Only row placement and order
    change -- values are moved bit-exactly.

    The first version is deliberately synchronous: no extra streams, no
    ``cuda.synchronize()`` / barrier "fixups", and no hidden catch-up all-gather.
    """
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be 2D [rows, features], got shape {tuple(hidden.shape)}")
    expected_src = sum(plan.input_split_sizes)
    if hidden.shape[0] != expected_src:
        raise ValueError(f"hidden has {hidden.shape[0]} rows but the plan sends {expected_src}")

    if not plan.network_exchange_required:
        # No rank crosses a boundary: every token stays local, only the packed
        # order changes (e.g. SP = 1, or identical layouts).
        if plan.send_indices.numel() == hidden.shape[0]:
            return hidden.index_select(0, plan.send_indices)
        return hidden

    send = hidden.index_select(0, plan.send_indices).contiguous()
    recv = torch.empty(
        (plan.num_recv_rows, hidden.shape[1]),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=list(plan.output_split_sizes),
        input_split_sizes=list(plan.input_split_sizes),
        group=group,
        async_op=False,
    )
    if plan.recv_to_dst_indices.numel() == plan.num_recv_rows:
        return recv.index_select(0, plan.recv_to_dst_indices)
    return recv


def reduction_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulator dtype for a window reduction (fp16/bf16 reduce in fp32)."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def global_window_mean(
    local_window_sum: torch.Tensor,
    global_window_count: int,
    *,
    group: dist.ProcessGroup | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reference text reduction: ``(sum over all windows) / global_window_count``.

    Every rank contributes the *sum* of its own windows' text attention output
    (a zero tensor on ranks without windows) and all ranks divide by the global
    window count.  This is neither a token-weighted mean nor a mean of rank
    means.

    ``group=None`` means *local* reduction: no collective is issued, and ``None``
    is never forwarded to ``all_reduce`` (which would silently use the default
    process group).  fp16/bf16 inputs are accumulated in fp32, fp32/fp64 keep
    their own precision, the caller's tensor is not modified, and ``dtype`` is
    applied only after the reduction and the division.
    """
    if global_window_count <= 0:
        raise ValueError(f"global_window_count must be positive, got {global_window_count}")
    accumulate = reduction_dtype(local_window_sum.dtype)
    if group is None:
        total = local_window_sum.to(accumulate)
    else:
        total = local_window_sum.to(accumulate).clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    total = total / float(global_window_count)
    return total.to(dtype) if dtype is not None else total


def transition_count(methods: Sequence[str], num_layers: int | None = None) -> int:
    """Number of layout transitions for a layer schedule (diagnostic only).

    The 3B schedule alternates, so every consecutive layer pair is a transition;
    the entry distribution and the final reconstruction are counted separately
    by the caller.
    """
    schedule = [methods[i % len(methods)] for i in range(num_layers if num_layers is not None else len(methods))]
    return sum(1 for a, b in zip(schedule, schedule[1:]) if a != b)


def schedule_methods(num_layers: int, methods: Iterable[str] = DEFAULT_WINDOW_METHODS) -> tuple[str, ...]:
    methods = tuple(methods)
    return tuple(methods[i % len(methods)] for i in range(num_layers))


# Every key here carries the geometry fingerprint, the SP world size and the
# planner version, and nothing device-owned is stored, so one cache can serve
# every request in the process. Rebuilding it per request costs seconds on a
# large token grid, which is pure host time with the accelerators idle.
_SHARED_PLAN_CACHE = WindowPlanCache(capacity=64)


def shared_plan_cache() -> WindowPlanCache:
    """The process-wide plan cache that requests reuse across identical geometry."""
    return _SHARED_PLAN_CACHE


class WindowLayoutManager:
    """Drives ``ensure_layout`` across a model's per-layer window schedule.

    One manager per request / per model instance.  It owns the device plans of
    the current layout pair and is the only place that decides whether a
    transition is needed and in which direction.  The CPU plan cache is shared
    process-wide, so a second request with the same geometry does not rebuild
    layouts and routing plans that cost seconds on a large token grid.
    """

    def __init__(
        self,
        token_grid: tuple[int, int, int],
        *,
        group: dist.ProcessGroup | None,
        world_size: int,
        rank: int,
        window: tuple[int, int, int] = DEFAULT_WINDOW,
        methods: Sequence[str] = DEFAULT_WINDOW_METHODS,
        num_layers: int = 32,
        cache: WindowPlanCache | None = None,
        planner_version: int = PLANNER_VERSION,
    ) -> None:
        if group is not None and world_size != dist.get_world_size(group):
            raise ValueError(
                f"world_size {world_size} does not match the window-SP group size {dist.get_world_size(group)}"
            )
        if group is None and world_size != 1:
            raise ValueError("a window-SP group is required when world_size > 1")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self.token_grid = (int(token_grid[0]), int(token_grid[1]), int(token_grid[2]))
        self.group = group
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.window = tuple(window)
        self.methods = tuple(methods)
        self.num_layers = int(num_layers)
        self.planner_version = int(planner_version)
        self.cache = shared_plan_cache() if cache is None else cache
        self.transitions = 0
        self._device: torch.device | None = None
        self._device_plans: dict[tuple[WindowLayoutKey, WindowLayoutKey], DeviceWindowRedistributionPlan] = {}

    # -- layouts -----------------------------------------------------------
    def layer_method(self, layer_index: int) -> str:
        return self.methods[layer_index % len(self.methods)]

    def layout(self, window_method: str) -> WindowLayout:
        return self.cache.layout(self.token_grid, window_method, self.window)

    def layer_layout(self, layer_index: int) -> WindowLayout:
        return self.layout(self.layer_method(layer_index))

    def layout_for_key(self, key: WindowLayoutKey) -> WindowLayout:
        return self.cache.layout(key.token_grid, key.window_method, key.window_shape)

    def assignment(self, layout: WindowLayout) -> WindowAssignment:
        return self.cache.assignment(layout, self.world_size, self.planner_version)

    def rank_plan(self, layout: WindowLayout, assignment: WindowAssignment | None = None) -> RankWindowPlan:
        return self.rank_plan_for(layout, self.rank, assignment)

    def rank_plan_for(
        self, layout: WindowLayout, rank: int, assignment: WindowAssignment | None = None
    ) -> RankWindowPlan:
        assignment = assignment or self.assignment(layout)
        return self.cache.rank_plan(layout, assignment, rank)

    # -- transitions -------------------------------------------------------
    def device_plan(self, src_layout: WindowLayout, dst_layout: WindowLayout) -> DeviceWindowRedistributionPlan:
        key = (src_layout.key, dst_layout.key)
        plan = self._device_plans.get(key)
        if plan is None:
            src_assignment = self.assignment(src_layout)
            dst_assignment = self.assignment(dst_layout)
            routes = self.cache.route(src_layout, src_assignment, dst_layout, dst_assignment)
            plan = materialize_redistribution_plan(routes[self.rank], device=self.device)
            self._device_plans[key] = plan
        return plan

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("WindowLayoutManager device is unset; call set_device() first")
        return self._device

    def set_device(self, device: torch.device | str) -> None:
        device = torch.device(device)
        if self._device != device:
            self._device = device
            self._device_plans.clear()

    def ensure_layout(
        self,
        hidden: torch.Tensor,
        src_layout_key: WindowLayoutKey,
        dst_layout_key: WindowLayoutKey,
    ) -> torch.Tensor:
        """Return ``hidden`` re-sharded into the destination layout (or as-is)."""
        if src_layout_key == dst_layout_key:
            return hidden
        self.set_device(hidden.device)
        src_layout = self.cache.layout(
            src_layout_key.token_grid, src_layout_key.window_method, src_layout_key.window_shape
        )
        dst_layout = self.cache.layout(
            dst_layout_key.token_grid, dst_layout_key.window_method, dst_layout_key.window_shape
        )
        plan = self.device_plan(src_layout, dst_layout)
        if not plan.network_exchange_required and plan.send_indices.numel() == hidden.shape[0]:
            if torch.equal(plan.send_indices, torch.arange(hidden.shape[0], device=hidden.device)):
                return hidden
        self.transitions += 1
        return redistribute_window_rows(hidden, plan, group=self.group)


@dataclass
class LocalWindowContext:
    """Everything one rank needs to run local window attention for one layout."""

    layout_key: WindowLayoutKey
    window_shapes: torch.Tensor  # int64 [W_r, 3]
    video_cu_seqlens: torch.Tensor  # int32 [W_r + 1]
    joint_cu_seqlens: torch.Tensor  # int32 [W_r + 1]
    joint_order: torch.Tensor  # int64 [N_r + W_r * L]
    vid_src: torch.Tensor  # int64 [N_r] (positions in the attention output)
    txt_src: torch.Tensor  # int64 [W_r * L] (positions in the attention output)
    text_len: int
    local_windows: int
    global_windows: int

    @property
    def num_video_tokens(self) -> int:
        return int(self.vid_src.numel())

    @property
    def joint_len(self) -> int:
        return int(self.joint_order.numel())

    @property
    def joint_lengths(self) -> torch.Tensor:
        return (self.joint_cu_seqlens[1:] - self.joint_cu_seqlens[:-1]).to(torch.int64)

    @property
    def max_joint_len(self) -> int:
        return int(self.joint_lengths.max()) if self.local_windows else 0


def build_local_window_context(
    rank_plan: RankWindowPlan,
    *,
    text_len: int,
    global_windows: int,
    device: torch.device | str,
) -> LocalWindowContext:
    """Build the device-local attention metadata for one rank and one layout."""
    device = torch.device(device)
    window_shapes = rank_plan.window_shapes.to(device=device, dtype=torch.int64, non_blocking=True)
    video_cu_seqlens = rank_plan.video_cu_seqlens.to(device=device, non_blocking=True)
    num_windows = int(window_shapes.shape[0])
    num_video = int(rank_plan.global_token_ids.numel())

    if num_windows:
        lengths = (video_cu_seqlens[1:] - video_cu_seqlens[:-1]).to(torch.int64)
        # The joint offsets come from the shared helper; the packing and unpacking
        # indices below are derived from exactly the same values.
        joint_cu = joint_cu_seqlens(video_cu_seqlens, text_len)
        window_joint_lengths = lengths + int(text_len)
        joint_offsets = torch.cumsum(window_joint_lengths, dim=0)
        base = joint_offsets - window_joint_lengths
        joint_pieces = []
        vid_pieces = []
        txt_pieces = []
        text_range = torch.arange(int(text_len), dtype=torch.int64, device=device)
        for index in range(num_windows):
            start = int(video_cu_seqlens[index])
            count = int(lengths[index])
            window_base = int(base[index])
            text_base = window_base + count
            joint_pieces.append(torch.arange(start, start + count, dtype=torch.int64, device=device))
            joint_pieces.append(torch.arange(num_video, num_video + int(text_len), dtype=torch.int64, device=device))
            vid_pieces.append(torch.arange(window_base, window_base + count, dtype=torch.int64, device=device))
            txt_pieces.append(text_range + text_base)
        joint_order = torch.cat(joint_pieces)
        vid_src = torch.cat(vid_pieces) if vid_pieces else torch.empty(0, dtype=torch.int64, device=device)
        txt_src = torch.cat(txt_pieces) if txt_pieces else torch.empty(0, dtype=torch.int64, device=device)
    else:
        # Empty rank: the same helper must produce a device-local int32 ``[0]``.
        joint_cu = joint_cu_seqlens(video_cu_seqlens, text_len)
        joint_order = torch.empty(0, dtype=torch.int64, device=device)
        vid_src = torch.empty(0, dtype=torch.int64, device=device)
        txt_src = torch.empty(0, dtype=torch.int64, device=device)

    return LocalWindowContext(
        layout_key=rank_plan.layout_key,
        window_shapes=window_shapes,
        video_cu_seqlens=video_cu_seqlens,
        joint_cu_seqlens=joint_cu,
        joint_order=joint_order,
        vid_src=vid_src,
        txt_src=txt_src,
        text_len=int(text_len),
        local_windows=num_windows,
        global_windows=int(global_windows),
    )


def pack_joint_windows(video: torch.Tensor, text: torch.Tensor, ctx: LocalWindowContext) -> torch.Tensor:
    """Interleave ``[video window, replicated text]`` pairs for varlen attention."""
    if ctx.local_windows == 0:
        return torch.empty((0,) + video.shape[1:], dtype=video.dtype, device=video.device)
    joined = torch.cat([video, text], dim=0)
    return joined.index_select(0, ctx.joint_order)


def unpack_joint_windows(joint_out: torch.Tensor, ctx: LocalWindowContext) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the attention output into packed video rows and per-window text rows."""
    if ctx.local_windows == 0:
        empty_video = torch.empty((0,) + joint_out.shape[1:], dtype=joint_out.dtype, device=joint_out.device)
        empty_text = torch.empty((0,) + joint_out.shape[1:], dtype=joint_out.dtype, device=joint_out.device)
        return empty_video, empty_text
    video_out = joint_out.index_select(0, ctx.vid_src)
    text_out = joint_out.index_select(0, ctx.txt_src)
    return video_out, text_out.view(ctx.local_windows, ctx.text_len, *joint_out.shape[1:])


class SeedVR2WindowRuntime:
    """Owns the window-SP plan for one request and drives the layer schedule."""

    def __init__(
        self,
        token_grid: tuple[int, int, int],
        *,
        text_len: int,
        group: dist.ProcessGroup | None = None,
        world_size: int = 1,
        rank: int = 0,
        window: tuple[int, int, int] = DEFAULT_WINDOW,
        methods: tuple[str, ...] = DEFAULT_WINDOW_METHODS,
        num_layers: int = 32,
        planner_version: int = PLANNER_VERSION,
    ) -> None:
        self.text_len = int(text_len)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.group = group
        self.num_layers = int(num_layers)
        self.manager = WindowLayoutManager(
            token_grid,
            group=group,
            world_size=self.world_size,
            rank=self.rank,
            window=window,
            methods=methods,
            num_layers=num_layers,
            planner_version=planner_version,
        )
        self._contexts: dict[WindowLayoutKey, LocalWindowContext] = {}
        self.stats: dict[str, int] = {
            "layout_transitions": 0,
            "network_transitions": 0,
            "local_reorders": 0,
            "remote_video_rows": 0,
            "text_all_reduces": 0,
            "fused_text_mean_calls": 0,
        }

    # -- layouts and contexts ---------------------------------------------
    def layout_for_layer(self, layer_index: int) -> WindowLayout:
        return self.manager.layer_layout(layer_index)

    def context(self, layout: WindowLayout, device: torch.device | str) -> LocalWindowContext:
        ctx = self._contexts.get(layout.key)
        if ctx is None or ctx.joint_order.device != torch.device(device):
            rank_plan = self.manager.rank_plan(layout)
            ctx = build_local_window_context(
                rank_plan,
                text_len=self.text_len,
                global_windows=layout.num_windows,
                device=device,
            )
            self._contexts[layout.key] = ctx
        return ctx

    # -- token placement ---------------------------------------------------
    def local_rows_for(self, canonical_rows: torch.Tensor, layout: WindowLayout) -> torch.Tensor:
        """Select this rank's rows of a canonical ``[N, C]`` tensor."""
        rank_plan = self.manager.rank_plan(layout)
        ids = rank_plan.global_token_ids.to(canonical_rows.device)
        return canonical_rows.index_select(0, ids)

    def to_canonical_rows(self, local_rows: torch.Tensor, layout: WindowLayout) -> torch.Tensor:
        """Gather every rank's rows back into canonical token order (all ranks)."""
        rank_plan = self.manager.rank_plan(layout)
        local_ids = rank_plan.global_token_ids
        num_tokens = layout.num_tokens
        if self.world_size == 1 or self.group is None:
            out = torch.empty((num_tokens,) + local_rows.shape[1:], dtype=local_rows.dtype, device=local_rows.device)
            return out.index_copy(0, local_ids.to(local_rows.device), local_rows)

        counts = torch.tensor([int(local_ids.numel())], dtype=torch.int64, device=local_rows.device)
        all_counts = [torch.zeros_like(counts) for _ in range(self.world_size)]
        dist.all_gather(all_counts, counts, group=self.group)
        sizes = [int(c) for c in all_counts]
        max_rows = max(sizes) if sizes else 0
        padded = torch.zeros((max_rows,) + local_rows.shape[1:], dtype=local_rows.dtype, device=local_rows.device)
        if local_rows.shape[0]:
            padded[: local_rows.shape[0]] = local_rows
        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded, group=self.group)

        out = torch.empty((num_tokens,) + local_rows.shape[1:], dtype=local_rows.dtype, device=local_rows.device)
        for source, rows in enumerate(gathered):
            if sizes[source] == 0:
                continue
            ids = self.manager.rank_plan_for(layout, source).global_token_ids.to(local_rows.device)
            out.index_copy_(0, ids, rows[: sizes[source]])
        return out

    # -- transitions -------------------------------------------------------
    def ensure_layout(self, hidden: torch.Tensor, current: WindowLayoutKey, required: WindowLayoutKey) -> torch.Tensor:
        if current == required:
            return hidden
        self.manager.set_device(hidden.device)
        src_layout = self.manager.layout_for_key(current)
        dst_layout = self.manager.layout_for_key(required)
        plan = self.manager.device_plan(src_layout, dst_layout)
        self.stats["layout_transitions"] += 1
        if plan.network_exchange_required:
            self.stats["network_transitions"] += 1
            self.stats["remote_video_rows"] += sum(
                count for index, count in enumerate(plan.input_split_sizes) if index != self.rank
            )
        else:
            self.stats["local_reorders"] += 1
        return self.manager.ensure_layout(hidden, current, required)

    def remote_video_bytes(self, hidden_channels: int, element_size: int) -> int:
        """Logical source->destination payload of the transitions seen so far."""
        return self.stats["remote_video_rows"] * int(hidden_channels) * int(element_size)

    @property
    def transitions(self) -> int:
        return self.manager.transitions

    # -- text reduction ----------------------------------------------------
    def collective_group(self) -> dist.ProcessGroup | None:
        """Group for the text reduction, or ``None`` for a local mean.

        ``world_size == 1`` is the local case.  A multi-rank runtime without a
        usable group is rejected rather than silently degrading to a rank-local
        mean, which would produce a different text state.
        """
        if self.world_size == 1:
            return None
        if self.group is None:
            raise RuntimeError("window-SP with world_size > 1 requires a process group for the text reduction")
        if dist.get_world_size(self.group) != self.world_size:
            raise RuntimeError(
                f"window-SP group size {dist.get_world_size(self.group)} does not match world_size {self.world_size}"
            )
        return self.group

    def reduce_text(self, local_window_sum: torch.Tensor, global_windows: int) -> torch.Tensor:
        """Global window mean of the per-window text attention outputs.

        Shape adaptation and the runtime statistics live here; the arithmetic and
        the collective live in :func:`global_window_mean`, which the tests and the
        single-rank attention path use as well.
        """
        if local_window_sum.numel() == 0:
            local_window_sum = torch.zeros(
                (self.text_len,) + local_window_sum.shape[1:],
                dtype=local_window_sum.dtype,
                device=local_window_sum.device,
            )
        group = self.collective_group()
        reduced = global_window_mean(local_window_sum, global_windows, group=group)
        if group is None:
            self.stats["fused_text_mean_calls"] += 1
        else:
            self.stats["text_all_reduces"] += 1
        return reduced

    def current_device(self) -> torch.device | None:
        return self.manager._device  # noqa: SLF001 - single internal owner


# Ported from the Apache-2.0 SeedVR2 reference implementation:
#   https://github.com/ByteDance-Seed/SeedVR  (models/dit_v2)
#   https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler (src/models/dit_3b)


logger = init_logger(__name__)

# Reference checkpoint hyper-parameters for the released 3B model.
SEEDVR2_3B_CONFIG: dict = {
    "vid_in_channels": 33,
    "vid_out_channels": 16,
    "vid_dim": 2560,
    "txt_in_dim": 5120,
    "heads": 20,
    "head_dim": 128,
    "expand_ratio": 4,
    "norm_eps": 1e-5,
    "patch_size": (1, 2, 2),
    "num_layers": 32,
    "mm_layers": 10,
    "window": DEFAULT_WINDOW,
    "window_method": DEFAULT_WINDOW_METHODS,
    "rope_dim": 128,
    "vid_out_norm": True,
}


# =============================================================================
# Reference-faithful building blocks
# =============================================================================


class MMArg:
    """Pair of per-stream values (video / text) used by :class:`MMModule`."""

    __slots__ = ("vid", "txt")

    def __init__(self, vid, txt) -> None:
        self.vid = vid
        self.txt = txt


class MMModule(nn.Module):
    """Apply a module to the video and text streams, optionally sharing weights.

    ``shared_weights=True`` builds a single submodule used for both streams
    (the reference's ``all`` prefix in the checkpoint); ``vid_only=True`` drops
    the text branch entirely.
    """

    def __init__(self, factory, dims: MMArg, *, shared_weights: bool = False, vid_only: bool = False) -> None:
        super().__init__()
        self.shared_weights = shared_weights
        self.vid_only = vid_only
        if shared_weights:
            self.all = factory(dims.vid)
        else:
            self.vid = factory(dims.vid)
            self.txt = None if vid_only else factory(dims.txt)

    def forward(self, vid: torch.Tensor, txt: torch.Tensor | None, *args, **kwargs):
        vid_module = self.all if self.shared_weights else self.vid
        vid = vid_module(vid, *args, **kwargs)
        if not self.vid_only and txt is not None:
            txt_module = self.all if self.shared_weights else self.txt
            txt = txt_module(txt, *args, **kwargs)
        return vid, txt


class RMSNorm(nn.Module):
    """Reference ``CustomRMSNorm``: normalise in the input dtype, optional affine."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight.to(x.dtype)
        return x


class AdaSingle(nn.Module):
    """Timestep modulation with per-layer shift / scale / gate parameters."""

    def __init__(self, dim: int, emb_dim: int, layers: Sequence[str], modes: Sequence[str] = ("in", "out")) -> None:
        super().__init__()
        if emb_dim != 6 * dim:
            raise ValueError(f"AdaSingle requires emb_dim == 6 * dim, got {emb_dim} != {6 * dim}")
        self.dim = int(dim)
        self.emb_dim = int(emb_dim)
        self.layers = list(layers)
        modes = set(modes)
        for layer in self.layers:
            if "in" in modes:
                self.register_parameter(f"{layer}_shift", nn.Parameter(torch.randn(dim) / dim**0.5))
                self.register_parameter(f"{layer}_scale", nn.Parameter(torch.randn(dim) / dim**0.5 + 1))
            if "out" in modes:
                self.register_parameter(f"{layer}_gate", nn.Parameter(torch.randn(dim) / dim**0.5))

    def slice(self, emb: torch.Tensor, layer: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(shift, scale, gate)`` embedding slices for ``layer``, each ``[1, dim]``."""
        view = emb.view(emb.shape[0], self.dim, len(self.layers), 3)[..., self.layers.index(layer), :]
        shift, scale, gate = view.unbind(-1)
        return shift, scale, gate

    def forward(self, hid: torch.Tensor, emb: torch.Tensor, layer: str, mode: str) -> torch.Tensor:
        shift_a, scale_a, gate_a = self.slice(emb, layer)
        shift_a = shift_a.to(hid.dtype)
        scale_a = scale_a.to(hid.dtype)
        gate_a = gate_a.to(hid.dtype)
        if mode == "in":
            return hid * (scale_a + getattr(self, f"{layer}_scale")) + (shift_a + getattr(self, f"{layer}_shift"))
        if mode == "out":
            gate_b = getattr(self, f"{layer}_gate", None)
            return hid * (gate_a + gate_b if gate_b is not None else gate_a)
        raise NotImplementedError(mode)


class OutAda(nn.Module):
    """The released ``vid_out_ada`` parameters (see the module docstring)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.out_shift = nn.Parameter(torch.zeros(dim))
        self.out_scale = nn.Parameter(torch.ones(dim))


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer SiLU MLP."""

    def __init__(self, sinusoidal_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.sinusoidal_dim = int(sinusoidal_dim)
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    @staticmethod
    def _sinusoidal(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
        emb = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, timestep, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        timestep = timestep.to(device=device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        emb = self._sinusoidal(timestep, self.sinusoidal_dim).to(dtype)
        emb = self.proj_in(emb)
        emb = self.act(emb)
        emb = self.proj_hid(emb)
        emb = self.act(emb)
        return self.proj_out(emb)


class SwiGLUMLP(nn.Module):
    """Reference SwiGLU MLP (``multiple_of=256``, no biases)."""

    def __init__(self, dim: int, expand_ratio: int, multiple_of: int = 256) -> None:
        super().__init__()
        hidden = int(2 * dim * expand_ratio / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.proj_in_gate = nn.Linear(dim, hidden, bias=False)
        self.proj_in = nn.Linear(dim, hidden, bias=False)
        self.proj_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))


# =============================================================================
# Window attention
# =============================================================================


def grouped_window_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: LocalWindowContext,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Exact window-local attention without a varlen kernel.

    Windows are packed back-to-back in the joint sequence, so this groups the
    equal-length windows and runs one batched SDPA per group (ragged window
    sizes keep the group count small).  It is the portable path used when the
    selected backend has no packed-varlen entry point.
    """
    if ctx.local_windows == 0:
        return torch.empty_like(q)
    lengths = ctx.joint_lengths
    offsets = ctx.joint_cu_seqlens.to(torch.int64)
    out = torch.empty_like(q)
    for length in torch.unique(lengths).tolist():
        window_ids = torch.nonzero(lengths == length, as_tuple=False).flatten().tolist()
        rows = torch.cat(
            [torch.arange(int(offsets[i]), int(offsets[i]) + int(length), device=q.device) for i in window_ids]
        )
        num = len(window_ids)
        qq = q.index_select(0, rows).view(num, int(length), q.shape[1], q.shape[2]).transpose(1, 2)
        kk = k.index_select(0, rows).view(num, int(length), k.shape[1], k.shape[2]).transpose(1, 2)
        vv = v.index_select(0, rows).view(num, int(length), v.shape[1], v.shape[2]).transpose(1, 2)
        attended = F.scaled_dot_product_attention(qq, kk, vv, scale=softmax_scale)
        out.index_copy_(0, rows, attended.transpose(1, 2).reshape(-1, q.shape[1], q.shape[2]))
    return out


class NaSwinAttention(nn.Module):
    """Joint video+text window attention with a globally averaged text output."""

    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm_eps: float,
        rope_dim: int,
        shared_weights: bool,
        use_varlen_kernel: bool = True,
    ) -> None:
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.inner_dim = int(inner_dim)
        self.softmax_scale = 1.0 / math.sqrt(head_dim)
        dims = MMArg(vid_dim, txt_dim)
        self.proj_qkv = MMModule(
            lambda d: nn.Linear(int(d), 3 * inner_dim, bias=qk_bias), dims, shared_weights=shared_weights
        )
        self.proj_out = MMModule(lambda d: nn.Linear(inner_dim, int(d)), dims, shared_weights=shared_weights)
        self.norm_q = MMModule(
            lambda d: RMSNorm(int(d), eps=qk_norm_eps, elementwise_affine=True),
            MMArg(head_dim, head_dim),
            shared_weights=shared_weights,
        )
        self.norm_k = MMModule(
            lambda d: RMSNorm(int(d), eps=qk_norm_eps, elementwise_affine=True),
            MMArg(head_dim, head_dim),
            shared_weights=shared_weights,
        )
        self.rope = NaMMRotaryEmbedding3d(rotary_dim=rope_dim, num_axes=3)
        # Window-local attention is communication free by construction, so the
        # shared layer must never re-shard it through a parallel strategy.
        self.attention = Attention(
            num_heads=heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=self.softmax_scale,
            role="self",
            skip_sequence_parallel=True,
        )
        # ``use_varlen_kernel`` is a request, not a capability proof: a backend
        # that ignores ``cu_seqlens`` would attend across every local window
        # (and the replicated text stream), which is silently wrong and differs
        # by SP degree.  Resolve the request against the selected backend once.
        requested_varlen = bool(use_varlen_kernel)
        backend = self.attention.attn_backend
        supports_varlen = bool(backend is not None and backend.supports_multi_doc_packed_varlen())
        self.use_varlen_kernel = requested_varlen and supports_varlen
        self.attention_backend_name = backend.get_name() if backend is not None else None
        self.attention_path = "packed_varlen" if self.use_varlen_kernel else "grouped_sdpa"
        self.varlen_fallback_reason: str | None = None
        if requested_varlen and not supports_varlen:
            backend_name = self.attention_backend_name or "custom_attention"
            # Stable message without a layer id: 32 layers must warn once.
            self.varlen_fallback_reason = f"backend {backend_name} does not support multi-document packed varlen"
            logger.warning_once(
                "SeedVR2: attention backend %s does not support multi-document packed varlen; "
                "using grouped window SDPA.",
                backend_name,
            )
        self.attention_stats = {"packed_varlen_calls": 0, "grouped_sdpa_calls": 0, "no_local_windows_calls": 0}

    def _split_heads(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return qkv.view(qkv.shape[0], 3, self.heads, self.head_dim).unbind(1)

    def _apply_rope(self, vid_q, vid_k, txt_q, txt_k, ctx: LocalWindowContext):
        device = vid_q.device
        vid_freqs = self.rope.window_freqs_batch(ctx.window_shapes, ctx.text_len, device=device, dtype=torch.float32)
        txt_freqs = self.rope.text_freqs(ctx.text_len, device=device, dtype=torch.float32)
        if vid_freqs.shape[0] != vid_q.shape[0]:
            raise ValueError(f"window RoPE covers {vid_freqs.shape[0]} rows but the video shard has {vid_q.shape[0]}")
        return self.rope(vid_q, vid_k, vid_freqs, txt_q, txt_k, txt_freqs)

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        ctx: LocalWindowContext,
        runtime: SeedVR2WindowRuntime | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        vid_q, vid_k, vid_v = self._split_heads(vid_qkv)
        txt_q, txt_k, txt_v = self._split_heads(txt_qkv)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)
        vid_q, vid_k, txt_q, txt_k = self._apply_rope(vid_q, vid_k, txt_q, txt_k, ctx)

        joint_q = pack_joint_windows(vid_q, txt_q, ctx)
        joint_k = pack_joint_windows(vid_k, txt_k, ctx)
        joint_v = pack_joint_windows(vid_v, txt_v, ctx)

        if not ctx.local_windows:
            self.attention_stats["no_local_windows_calls"] += 1
        elif self.use_varlen_kernel:
            self.attention_stats["packed_varlen_calls"] += 1
        else:
            self.attention_stats["grouped_sdpa_calls"] += 1

        if self.use_varlen_kernel and ctx.local_windows:
            metadata = AttentionMetadata(
                extra={
                    "cu_seqlens_q": ctx.joint_cu_seqlens,
                    "cu_seqlens_k": ctx.joint_cu_seqlens,
                    "max_seqlen_q": ctx.max_joint_len,
                    "max_seqlen_k": ctx.max_joint_len,
                }
            )
            joint_out = self.attention(
                joint_q.unsqueeze(0), joint_k.unsqueeze(0), joint_v.unsqueeze(0), metadata
            ).squeeze(0)
        else:
            joint_out = grouped_window_sdpa(joint_q, joint_k, joint_v, ctx, softmax_scale=self.softmax_scale)

        vid_out, txt_windows = unpack_joint_windows(joint_out, ctx)
        vid_out = vid_out.reshape(-1, self.inner_dim)
        if ctx.local_windows:
            local_text_sum = txt_windows.reshape(ctx.local_windows, ctx.text_len, self.inner_dim).sum(0)
        else:
            # Ranks without windows still join the text reduction; the dtype must
            # match the other ranks' contribution exactly (a mismatch changes the
            # collective's dtype and deadlocks NCCL).
            local_text_sum = torch.zeros((ctx.text_len, self.inner_dim), device=vid.device, dtype=vid.dtype)
        if runtime is not None:
            txt_out = runtime.reduce_text(local_text_sum, ctx.global_windows).to(vid.dtype)
        else:
            # Same reduction as the runtime path, without a collective.
            txt_out = global_window_mean(local_text_sum, ctx.global_windows, group=None, dtype=vid.dtype)

        return self.proj_out(vid_out, txt_out)


# =============================================================================
# Transformer block
# =============================================================================


class NaMMSRTransformerBlock(nn.Module):
    """Reference ``NaMMSRTransformerBlock``: modulated attention + SwiGLU MLP."""

    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm_eps: float,
        qk_bias: bool,
        mlp_type: str,
        shared_weights: bool,
        rope_dim: int,
        is_last_layer: bool,
        use_varlen_kernel: bool = True,
    ) -> None:
        super().__init__()
        if mlp_type != "swiglu":
            raise NotImplementedError(f"unsupported mlp_type {mlp_type!r} for SeedVR2")
        dims = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(
            lambda d: RMSNorm(int(d), eps=norm_eps, elementwise_affine=False), dims, shared_weights=shared_weights
        )
        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm_eps=norm_eps,
            rope_dim=rope_dim,
            shared_weights=shared_weights,
            use_varlen_kernel=use_varlen_kernel,
        )
        self.mlp_norm = MMModule(
            lambda d: RMSNorm(int(d), eps=norm_eps, elementwise_affine=False),
            dims,
            shared_weights=shared_weights,
            vid_only=is_last_layer,
        )
        self.mlp = MMModule(
            lambda d: SwiGLUMLP(int(d), expand_ratio), dims, shared_weights=shared_weights, vid_only=is_last_layer
        )
        self.ada = MMModule(
            lambda d: AdaSingle(int(d), emb_dim, layers=["attn", "mlp"]),
            dims,
            shared_weights=shared_weights,
            vid_only=is_last_layer,
        )
        self.is_last_layer = bool(is_last_layer)

    def _modulate(
        self, vid: torch.Tensor, txt: torch.Tensor | None, emb: torch.Tensor, layer: str, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        vid_ada = self.ada.all if self.ada.shared_weights else self.ada.vid
        vid = vid_ada(vid, emb, layer, mode)
        if txt is not None and not self.ada.vid_only:
            txt_ada = self.ada.all if self.ada.shared_weights else self.ada.txt
            txt = txt_ada(txt, emb, layer, mode)
        return vid, txt

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        emb: torch.Tensor,
        ctx: LocalWindowContext,
        runtime: SeedVR2WindowRuntime | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vid_norm, txt_norm = self.attn_norm(vid, txt)
        vid_norm, txt_norm = self._modulate(vid_norm, txt_norm, emb, "attn", "in")
        vid_attn, txt_attn = self.attn(vid_norm, txt_norm, ctx, runtime)
        vid_attn, txt_attn = self._modulate(vid_attn, txt_attn, emb, "attn", "out")
        vid_attn = vid_attn + vid
        txt_attn = txt_attn + txt

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self._modulate(vid_mlp, txt_mlp, emb, "mlp", "in")
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self._modulate(vid_mlp, txt_mlp, emb, "mlp", "out")
        return vid_mlp + vid_attn, txt_mlp + txt_attn


# =============================================================================
# NaDiT
# =============================================================================


class NaDiTOutput:
    """Container matching the reference ``NaDiTOutput``."""

    def __init__(self, vid_sample: torch.Tensor) -> None:
        self.vid_sample = vid_sample


class SeedVR2NaDiT(nn.Module):
    """SeedVR2 3B ``NaDiT`` transformer with window-aligned SP support."""

    def __init__(
        self,
        *,
        vid_in_channels: int = 33,
        vid_out_channels: int = 16,
        vid_dim: int = 2560,
        txt_in_dim: int = 5120,
        emb_dim: int | None = None,
        heads: int = 20,
        head_dim: int = 128,
        expand_ratio: int = 4,
        norm_eps: float = 1e-5,
        patch_size: Sequence[int] = (1, 2, 2),
        num_layers: int = 32,
        mm_layers: int = 10,
        mlp_type: str = "swiglu",
        window: Sequence[int] = DEFAULT_WINDOW,
        window_method: Sequence[str] = DEFAULT_WINDOW_METHODS,
        rope_type: str = "mmrope3d",
        rope_dim: int = 128,
        vid_out_norm: bool = True,
        use_varlen_kernel: bool = True,
    ) -> None:
        super().__init__()
        if rope_type != "mmrope3d":
            raise NotImplementedError(f"unsupported rope_type {rope_type!r} for SeedVR2")
        txt_dim = vid_dim
        emb_dim = emb_dim or 6 * vid_dim
        self.patch_size = tuple(int(v) for v in patch_size)
        if self.patch_size[0] != 1:
            raise NotImplementedError("only the released temporal patch size 1 is supported")
        self.vid_dim = int(vid_dim)
        self.txt_dim = int(txt_dim)
        self.num_layers = int(num_layers)
        self.window = tuple(int(v) for v in window)
        self.window_method = tuple(window_method)

        self.vid_in = _NaPatchIn(in_channels=vid_in_channels, patch_size=self.patch_size, dim=vid_dim)
        self.txt_in = nn.Linear(txt_in_dim, txt_dim) if txt_in_dim != txt_dim else nn.Identity()
        self.emb_in = TimeEmbedding(sinusoidal_dim=256, hidden_dim=max(vid_dim, txt_dim), output_dim=emb_dim)

        self.blocks = nn.ModuleList(
            [
                NaMMSRTransformerBlock(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm_eps=norm_eps,
                    qk_bias=False,
                    mlp_type=mlp_type,
                    shared_weights=not (index < mm_layers),
                    rope_dim=rope_dim,
                    is_last_layer=(index == num_layers - 1),
                    use_varlen_kernel=use_varlen_kernel,
                )
                for index in range(num_layers)
            ]
        )

        self.vid_out_norm = RMSNorm(vid_dim, eps=norm_eps, elementwise_affine=True) if vid_out_norm else None
        self.vid_out_ada = OutAda(vid_dim)
        self.vid_out = _NaPatchOut(out_channels=vid_out_channels, patch_size=self.patch_size, dim=vid_dim)

    # -- runtime -----------------------------------------------------------
    def build_runtime(
        self,
        token_grid: tuple[int, int, int],
        *,
        text_len: int,
        group=None,
        world_size: int = 1,
        rank: int = 0,
        parallel_config: DiffusionParallelConfig | None = None,
    ) -> SeedVR2WindowRuntime:
        """Create the window-SP driver for one request (SP=1 included)."""
        if parallel_config is not None:
            validate_seedvr2_parallel_config(parallel_config)
            sp = get_sp_group()
            if sp.world_size != parallel_config.sequence_parallel_size:
                raise ValueError("SeedVR2 SP group size does not match its parallel configuration")
            group, world_size, rank = sp.device_group, sp.world_size, sp.rank_in_group
        return SeedVR2WindowRuntime(
            token_grid,
            text_len=text_len,
            group=group,
            world_size=world_size,
            rank=rank,
            window=self.window,
            methods=self.window_method,
            num_layers=self.num_layers,
        )

    def token_grid_for(self, vid_shape: torch.Tensor) -> tuple[int, int, int]:
        frames, height, width = (int(v) for v in vid_shape[0].tolist())
        t, h, w = self.patch_size
        if t > 1 and frames % t != 1:
            raise ValueError(f"frame count {frames} must satisfy frames % {t} == 1")
        return frames // t, height // h, width // w

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        vid_shape: torch.Tensor,
        txt_shape: torch.Tensor,
        timestep,
        runtime: SeedVR2WindowRuntime | None = None,
    ) -> NaDiTOutput:
        """Run the transformer.

        ``vid`` is the flattened pre-patchify latent ``[T*H*W, C]`` with
        ``vid_shape`` holding the raw latent ``(T, H, W)``; ``txt`` is
        ``[L, txt_in_dim]``.  With ``runtime`` set, only this rank's windows are
        ever materialised; without it the model runs the SP=1 path through the
        same planner, so a regular -> shifted transition still permutes rows.
        """
        weight = next(self.vid_in.parameters())
        frames, height, width = (int(v) for v in vid_shape[0].tolist())
        token_grid = self.token_grid_for(vid_shape)

        txt_tokens = self.txt_in(txt.to(weight.dtype))
        text_len = int(txt_tokens.shape[0])

        canonical_rows, _ = patchify(vid, (frames, height, width), self.patch_size)
        canonical_rows = canonical_rows.to(weight.dtype)

        if runtime is None:
            runtime = self.build_runtime(token_grid, text_len=text_len, world_size=1, rank=0)

        first_layout = runtime.layout_for_layer(0)
        local_rows = runtime.local_rows_for(canonical_rows, first_layout)
        vid_hidden = self.vid_in.proj(local_rows)
        emb = self.emb_in(timestep, device=vid_hidden.device, dtype=vid_hidden.dtype)

        current = first_layout.key
        for index, block in enumerate(self.blocks):
            layout = runtime.layout_for_layer(index)
            vid_hidden = runtime.ensure_layout(vid_hidden, current, layout.key)
            current = layout.key
            ctx = runtime.context(layout, vid_hidden.device)
            vid_hidden, txt_tokens = block(vid_hidden, txt_tokens, emb, ctx, runtime)

        vid_hidden = self._output_projection(vid_hidden, emb)
        vid_hidden = runtime.to_canonical_rows(vid_hidden, runtime.layout_for_layer(self.num_layers - 1))
        return NaDiTOutput(vid_sample=unpatchify(vid_hidden, token_grid, self.patch_size))

    # -- attention path reporting -----------------------------------------
    def attention_path_summary(self) -> dict:
        """Requested vs resolved attention path, backends and per-path call counts."""
        layers_per_path: dict[str, int] = {}
        backends: set[str] = set()
        fallbacks: set[str] = set()
        stats = {"packed_varlen_calls": 0, "grouped_sdpa_calls": 0, "no_local_windows_calls": 0}
        for block in self.blocks:
            attn = block.attn
            layers_per_path[attn.attention_path] = layers_per_path.get(attn.attention_path, 0) + 1
            if attn.attention_backend_name:
                backends.add(attn.attention_backend_name)
            if attn.varlen_fallback_reason:
                fallbacks.add(attn.varlen_fallback_reason)
            for key, value in attn.attention_stats.items():
                stats[key] += value
        return {
            "layers_per_path": layers_per_path,
            "backend_names": sorted(backends),
            "varlen_fallback_reasons": sorted(fallbacks),
            **stats,
        }

    def reset_attention_stats(self) -> None:
        for block in self.blocks:
            for key in block.attn.attention_stats:
                block.attn.attention_stats[key] = 0

    # -- internals ---------------------------------------------------------
    def _output_projection(self, vid_hidden: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.vid_out_norm is not None:
            vid_hidden = self.vid_out_norm(vid_hidden)
            block_ada = self.blocks[0].ada
            ada = block_ada.all if block_ada.shared_weights else block_ada.vid
            shift_a, scale_a, _ = ada.slice(emb, "attn")
            vid_hidden = vid_hidden * (scale_a.to(vid_hidden.dtype) + self.vid_out_ada.out_scale) + (
                shift_a.to(vid_hidden.dtype) + self.vid_out_ada.out_shift
            )
        return self.vid_out.proj(vid_hidden)


# =============================================================================
# Patch in / out
# =============================================================================


class _NaPatchIn(nn.Module):
    """Reference ``NaPatchIn``: ``(T t)(H h)(W w) c -> T H W (t h w c)`` + Linear."""

    def __init__(self, in_channels: int, patch_size: Sequence[int], dim: int) -> None:
        super().__init__()
        t, h, w = (int(v) for v in patch_size)
        self.patch_size = (t, h, w)
        self.proj = nn.Linear(in_channels * t * h * w, dim)


class _NaPatchOut(nn.Module):
    """Reference ``NaPatchOut``: Linear then ``T H W (t h w c) -> (T t)(H h)(W w) c``."""

    def __init__(self, out_channels: int, patch_size: Sequence[int], dim: int) -> None:
        super().__init__()
        t, h, w = (int(v) for v in patch_size)
        self.patch_size = (t, h, w)
        self.proj = nn.Linear(dim, out_channels * t * h * w)


def patchify(vid: torch.Tensor, shape: tuple[int, int, int], patch_size: Sequence[int]):
    """Patch embedding layout of the reference (token-major rows)."""
    frames, height, width = (int(v) for v in shape)
    t, h, w = (int(v) for v in patch_size)
    if t != 1:
        raise NotImplementedError("only temporal patch size 1 is supported")
    channels = vid.shape[-1]
    grid = vid.reshape(frames // t, t, height // h, h, width // w, w, channels)
    rows = grid.permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, t * h * w * channels)
    return rows, (frames // t, height // h, width // w)


def unpatchify(rows: torch.Tensor, token_grid: tuple[int, int, int], patch_size: Sequence[int]) -> torch.Tensor:
    """Inverse of :func:`patchify`: ``[N, t*h*w*c] -> [T*H*W, c]`` (reference order)."""
    t, h, w = (int(v) for v in patch_size)
    frames, height, width = (int(v) for v in token_grid)
    channels = rows.shape[-1] // (t * h * w)
    grid = rows.reshape(frames, height, width, t, h, w, channels)
    return grid.permute(0, 3, 1, 4, 2, 5, 6).reshape(frames * t * height * h * width * w, channels)
