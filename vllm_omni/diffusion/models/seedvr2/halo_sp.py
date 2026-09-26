# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed height slabs with boundary-window transfers for SeedVR2."""

import torch
import torch.distributed as dist

from .na_ops import LocalWindowContext, SeedVR2WindowRuntime, build_local_window_context
from .window_geometry import DEFAULT_WINDOW, DEFAULT_WINDOW_METHODS, get_window_op
from .window_sp import (
    DeviceWindowRedistributionPlan,
    WindowAssignment,
    WindowLayout,
    WindowLayoutKey,
    build_rank_window_plan,
    build_redistribution_plans,
    materialize_redistribution_plan,
)


def _balanced_window_owners(local_rows: list[list[int]], world_size: int) -> tuple[list[int], list[int]]:
    """Rebalance majority ownership using the cheapest beneficial boundary moves."""
    sizes = [sum(rows) for rows in local_rows]
    owners = [max(range(world_size), key=rows.__getitem__) for rows in local_rows]
    counts = [sum(size for size, owner in zip(sizes, owners) if owner == rank) for rank in range(world_size)]
    while True:
        source = max(range(world_size), key=counts.__getitem__)
        target = min(range(world_size), key=counts.__getitem__)
        gap = counts[source] - counts[target]
        moves = [
            (rows[source] - rows[target], -size * (gap - size), index)
            for index, (rows, size, owner) in enumerate(zip(local_rows, sizes, owners))
            if owner == source and rows[target] > 0 and size < gap
        ]
        if not moves:
            return owners, counts
        _, _, index = min(moves)
        owners[index] = target
        counts[source] -= sizes[index]
        counts[target] += sizes[index]


class SeedVR2HaloRuntime(SeedVR2WindowRuntime):
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
    ) -> None:
        super().__init__(
            token_grid,
            text_len=text_len,
            group=group,
            world_size=world_size,
            rank=rank,
            window=window,
            methods=methods,
            num_layers=num_layers,
        )
        t, h, w = token_grid
        regular = self.manager.layout(self.manager.methods[0])
        slices = get_window_op(self.manager.methods[0])(token_grid, self.manager.window)
        edges = sorted({0, h, *(sh.stop for _, sh, _ in slices)})
        cuts = [edges[(len(edges) - 1) * rank // self.world_size] for rank in range(self.world_size + 1)]
        canonical = torch.arange(t * h * w).reshape(t, h, w)
        pieces = [canonical[:, cuts[r] : cuts[r + 1], :].flatten() for r in range(self.world_size)]
        offsets = [0]
        for piece in pieces:
            offsets.append(offsets[-1] + piece.numel())
        key = WindowLayoutKey(
            token_grid,
            "height_slabs",
            self.manager.window,
            regular.key.geometry_version,
            regular.key.geometry_fingerprint,
        )
        self.home = WindowLayout(
            key,
            t * h * w,
            torch.tensor(offsets),
            torch.cat(pieces),
            torch.tensor([(t, cuts[r + 1] - cuts[r], w) for r in range(self.world_size)]),
        )
        self.home_assignment = WindowAssignment(
            key,
            self.world_size,
            1,
            torch.arange(self.world_size),
            tuple(torch.tensor([r]) for r in range(self.world_size)),
            tuple(p.numel() for p in pieces),
            (1,) * self.world_size,
        )
        self.owners = torch.empty(t * h * w, dtype=torch.int64)
        for rank, piece in enumerate(pieces):
            self.owners[piece] = rank
        self.assignments: dict[WindowLayoutKey, WindowAssignment] = {}
        self.routes: dict[tuple[WindowLayoutKey, bool, torch.device], DeviceWindowRedistributionPlan] = {}
        self.local_routes: dict[tuple[WindowLayoutKey, bool, torch.device], torch.Tensor] = {}

    def assignment(self, layout: WindowLayout) -> WindowAssignment:
        if layout.key not in self.assignments:
            local_rows = []
            for index in range(layout.num_windows):
                ids = layout.window_token_ids[int(layout.window_offsets[index]) : int(layout.window_offsets[index + 1])]
                local_rows.append(torch.bincount(self.owners[ids], minlength=self.world_size).tolist())
            owners, counts = _balanced_window_owners(local_rows, self.world_size)
            buckets = [[] for _ in range(self.world_size)]
            for index, owner in enumerate(owners):
                buckets[owner].append(index)
            self.assignments[layout.key] = WindowAssignment(
                layout.key,
                self.world_size,
                1,
                torch.tensor(owners),
                tuple(torch.tensor(b, dtype=torch.int64) for b in buckets),
                tuple(counts),
                tuple(len(b) for b in buckets),
            )
        return self.assignments[layout.key]

    def context(self, layout: WindowLayout, device: torch.device | str) -> LocalWindowContext:
        if layout.key not in self._contexts:
            plan = build_rank_window_plan(layout, self.assignment(layout), self.rank)
            self._contexts[layout.key] = build_local_window_context(
                plan, text_len=self.text_len, global_windows=layout.num_windows, device=device
            )
        return self._contexts[layout.key]

    def local_rows_for(self, canonical_rows: torch.Tensor, layout: WindowLayout) -> torch.Tensor:
        ids = build_rank_window_plan(self.home, self.home_assignment, self.rank).global_token_ids.to(
            canonical_rows.device
        )
        return canonical_rows.index_select(0, ids)

    def ensure_layout(self, hidden: torch.Tensor, current: WindowLayoutKey, required: WindowLayoutKey) -> torch.Tensor:
        return hidden

    def transfer(self, hidden: torch.Tensor, layout: WindowLayout, *, returning: bool = False) -> torch.Tensor:
        key = (layout.key, returning, hidden.device)
        if key not in self.routes:
            src, sa, dst, da = self.home, self.home_assignment, layout, self.assignment(layout)
            if returning:
                src, sa, dst, da = dst, da, src, sa
            plan = build_redistribution_plans(src, sa, dst, da)[self.rank]
            self.routes[key] = materialize_redistribution_plan(plan, device=hidden.device)
            if not plan.network_exchange_required:
                self.local_routes[key] = plan.send_indices.index_select(0, plan.recv_to_dst_indices).to(hidden.device)
        plan = self.routes[key]
        if not plan.network_exchange_required:
            return hidden.index_select(0, self.local_routes[key])
        send = hidden.index_select(0, plan.send_indices)
        receive = hidden.new_empty((plan.num_recv_rows, hidden.shape[1]))
        # Keep interior rows local; only boundary rows enter the collective.
        send_sizes = list(plan.input_split_sizes)
        recv_sizes = list(plan.output_split_sizes)
        si, ri = sum(send_sizes[: self.rank]), sum(recv_sizes[: self.rank])
        local = send_sizes[self.rank]
        receive[ri : ri + local] = send[si : si + local]
        remote_send = torch.cat((send[:si], send[si + local :]))
        remote_receive = hidden.new_empty((plan.num_recv_rows - local, hidden.shape[1]))
        send_sizes[self.rank] = recv_sizes[self.rank] = 0
        if self.world_size > 1:
            dist.all_to_all_single(
                remote_receive,
                remote_send,
                output_split_sizes=recv_sizes,
                input_split_sizes=send_sizes,
                group=self.group,
            )
        receive[:ri] = remote_receive[:ri]
        receive[ri + local :] = remote_receive[ri:]
        self.stats["remote_video_rows"] += remote_send.shape[0]
        return receive.index_select(0, plan.recv_to_dst_indices)

    def to_canonical_rows(self, local_rows: torch.Tensor, layout: WindowLayout) -> torch.Tensor:
        output = local_rows.new_zeros((self.home.num_tokens, local_rows.shape[1]))
        ids = build_rank_window_plan(self.home, self.home_assignment, self.rank).global_token_ids.to(local_rows.device)
        output.index_copy_(0, ids, local_rows)
        if self.world_size > 1:
            dist.all_reduce(output, group=self.group)
        return output
