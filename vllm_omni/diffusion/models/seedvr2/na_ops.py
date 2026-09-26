# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Window-SP runtime glue for the SeedVR2 DiT.

This module turns the generic planner in
:mod:`vllm_omni.diffusion.models.seedvr2.window_sp` into the concrete objects the
transformer needs:

* :class:`LocalWindowContext` -- the per-layout, per-rank metadata (window
  shapes, video/joint ``cu_seqlens``, joint packing order, unpack positions),
* :class:`SeedVR2WindowRuntime` -- entry distribution, layer-to-layer
  ``ensure_layout`` transitions, the global window-mean text reduction and the
  final reconstruction back to canonical token order.

The framework provides the regular SP group (``get_sp_group()``), configured
with ``ulysses_degree``. The base runtime owns whole-window routing; the
SeedVR2 Ulysses runtime exchanges QKV into head shards around attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_omni.diffusion.models.seedvr2.window_geometry import (
    DEFAULT_WINDOW,
    DEFAULT_WINDOW_METHODS,
)
from vllm_omni.diffusion.models.seedvr2.window_sp import (
    PLANNER_VERSION,
    RankWindowPlan,
    WindowLayout,
    WindowLayoutKey,
    WindowLayoutManager,
    global_window_mean,
    joint_cu_seqlens,
)


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
    sdpa_groups: tuple[tuple[int, torch.Tensor], ...] = ()

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
    window_shapes = rank_plan.window_shapes
    video_cu_seqlens = rank_plan.video_cu_seqlens
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
        text_range = torch.arange(int(text_len), dtype=torch.int64, device="cpu")
        for index in range(num_windows):
            start = int(video_cu_seqlens[index])
            count = int(lengths[index])
            window_base = int(base[index])
            text_base = window_base + count
            joint_pieces.append(torch.arange(start, start + count, dtype=torch.int64, device="cpu"))
            joint_pieces.append(torch.arange(num_video, num_video + int(text_len), dtype=torch.int64, device="cpu"))
            vid_pieces.append(torch.arange(window_base, window_base + count, dtype=torch.int64, device="cpu"))
            txt_pieces.append(text_range + text_base)
        joint_order = torch.cat(joint_pieces)
        vid_src = torch.cat(vid_pieces) if vid_pieces else torch.empty(0, dtype=torch.int64, device="cpu")
        txt_src = torch.cat(txt_pieces) if txt_pieces else torch.empty(0, dtype=torch.int64, device="cpu")
    else:
        # Empty rank: the same helper must produce a device-local int32 ``[0]``.
        joint_cu = joint_cu_seqlens(video_cu_seqlens, text_len)
        joint_order = torch.empty(0, dtype=torch.int64, device="cpu")
        vid_src = torch.empty(0, dtype=torch.int64, device="cpu")
        txt_src = torch.empty(0, dtype=torch.int64, device="cpu")

    # The planner owns CPU metadata. Build each length group once per layout,
    # then reuse device row indices across layers without scalar GPU reads.
    grouped_rows: dict[int, list[torch.Tensor]] = {}
    offsets = joint_cu.tolist()
    for start, stop in zip(offsets, offsets[1:]):
        grouped_rows.setdefault(stop - start, []).append(torch.arange(start, stop, device="cpu"))
    sdpa_groups = tuple((length, torch.cat(grouped_rows[length]).to(device)) for length in sorted(grouped_rows))
    return LocalWindowContext(
        layout_key=rank_plan.layout_key,
        window_shapes=window_shapes.to(device),
        video_cu_seqlens=video_cu_seqlens.to(device),
        joint_cu_seqlens=joint_cu.to(device),
        joint_order=joint_order.to(device),
        vid_src=vid_src.to(device),
        txt_src=txt_src.to(device),
        text_len=int(text_len),
        local_windows=num_windows,
        global_windows=int(global_windows),
        sdpa_groups=sdpa_groups,
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
