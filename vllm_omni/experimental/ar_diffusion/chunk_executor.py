# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunk pipeline slot loop: microbatch, chain P2P, noisy KV exchange."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

from vllm_omni.experimental.ar_diffusion.chunk_schedule import (
    ChunkPlan,
    ChunkStep,
    Inflight,
    can_admit,
    rank_work,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import NoisyKVState

#: Wall time of each *active* slot (rank has work) on rank 0.
#: Experimental telemetry only; a harness clears it per request.
SLOT_TIMES: list[float] = []

#: End-to-end wall of one ``run_chunk_pipeline`` call on rank 0 (seconds).
PIPELINE_TIMES: list[float] = []


def _slot_times_file() -> str | None:
    return os.environ.get("AR_DIFFUSION_SLOT_TIMES_FILE") or None


def _pipeline_times_file() -> str | None:
    return os.environ.get("AR_DIFFUSION_PIPELINE_TIMES_FILE") or None


def _cuda_sync() -> None:
    try:
        import torch

        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
    except Exception:
        return


def _record_slot(delta: float) -> None:
    SLOT_TIMES.append(delta)
    path = _slot_times_file()
    if path:
        with open(path, "a") as fh:
            fh.write(f"{delta}\n")


def _record_pipeline(delta: float) -> None:
    PIPELINE_TIMES.append(delta)
    path = _pipeline_times_file()
    if path:
        with open(path, "a") as fh:
            fh.write(f"{delta}\n")


logger = init_logger(__name__)


def resolve_pp_rank_and_group() -> tuple[int, Any | None]:
    """Read the existing PP group; never creates a new process group."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_pipeline_parallel_rank,
            get_pp_group,
        )

        return int(get_pipeline_parallel_rank()), get_pp_group()
    except Exception:
        return 0, None


@dataclass(frozen=True)
class ChunkTopology:
    stages: int
    layer_groups: int

    @property
    def world(self) -> int:
        return self.stages * self.layer_groups

    def split_rank(self, rank: int) -> tuple[int, int]:
        if self.stages == 1:
            return 0, rank
        return rank // self.layer_groups, rank % self.layer_groups

    def is_stage_first(self, rank: int) -> bool:
        _, g = self.split_rank(rank)
        return g == 0

    def is_stage_last(self, rank: int) -> bool:
        _, g = self.split_rank(rank)
        return g == self.layer_groups - 1


@dataclass
class ChunkRunSpec:
    topology: ChunkTopology
    max_batch_size: int = 1
    rank: int = 0
    pp_group: Any | None = None


@dataclass(frozen=True)
class PendingAdmit:
    req: str
    plan: ChunkPlan
    chunk_tokens: int


@dataclass
class ARDiffusionChunkContext:
    spec: ChunkRunSpec
    kv: NoisyKVState
    inflight: list[Inflight] = field(default_factory=list)
    pending: list[PendingAdmit] = field(default_factory=list)

    def enqueue(self, req: str, plan: ChunkPlan, *, chunk_tokens: int) -> None:
        self.pending.append(PendingAdmit(req=req, plan=plan, chunk_tokens=chunk_tokens))

    def admit(self, req: str, plan: ChunkPlan, *, chunk_tokens: int, slot: int) -> bool:
        pending = tuple(self.inflight)
        if not can_admit(pending, slot, plan, max_batch_size=self.spec.max_batch_size):
            return False
        inflight_tokens = {self.kv._chunk_tokens[item.req] for item in self.inflight}
        if inflight_tokens and chunk_tokens not in inflight_tokens:
            return False
        self.kv.begin_request(req, plan, chunk_tokens=chunk_tokens, t0=slot)
        self.inflight.append(Inflight(req=req, t0=slot, plan=plan))
        self.kv.set_inflight(tuple(self.inflight))
        return True

    def drop(self, req: str) -> None:
        self.inflight = [item for item in self.inflight if item.req != req]
        self.pending = [item for item in self.pending if item.req != req]
        self.kv.end_request(req)
        self.kv.set_inflight(tuple(self.inflight))

    def admit_pending(self, slot: int) -> None:
        still: list[PendingAdmit] = []
        for item in self.pending:
            if still:
                still.append(item)
                continue
            if self.admit(item.req, item.plan, chunk_tokens=item.chunk_tokens, slot=slot):
                continue
            still.append(item)
        self.pending = still


class ChunkAdapter:
    """Model-side chunk step. ``forward`` receives the microbatch on this rank."""

    def forward(
        self,
        tasks: tuple[tuple[str, ChunkStep], ...],
        kv_contexts: list[list[Any]],
        *,
        hidden: Any | None,
    ) -> Any:
        raise NotImplementedError

    def pack_activation(self, output: Any) -> dict:
        raise NotImplementedError

    def unpack_activation(self, payload: dict) -> Any:
        raise NotImplementedError


def _wait(handles: list[Any]) -> None:
    for handle in handles:
        if handle is not None:
            handle.wait()


def _assert_i8(tasks: tuple[tuple[str, ChunkStep], ...]) -> None:
    reqs = [req for req, _task in tasks]
    if len(reqs) != len(set(reqs)):
        raise RuntimeError(f"I8 violated: same request appears twice in one microbatch: {tasks}")


def run_chunk_pipeline(
    *,
    ctx: ARDiffusionChunkContext,
    adapter: ChunkAdapter,
    max_slots: int | None = None,
) -> None:
    """Drive inflight plans to completion on this rank.

    Each slot waits on two disjoint sets only: the handles this rank posted
    (outbound not left dangling) and the inbound versions the *next* slot
    consumes. Everything else keeps overlapping with the next forward.
    """
    spec = ctx.spec
    kv = ctx.kv
    kv.bind_rank(spec.rank, spec.pp_group)
    pp = spec.pp_group
    logger.info(
        "[chunk_pipeline] start: stages=%d layer_groups=%d rank=%d inflight=%d pending=%d",
        spec.topology.stages,
        spec.topology.layer_groups,
        spec.rank,
        len(ctx.inflight),
        len(ctx.pending),
    )
    slot = 0
    limit = max_slots if max_slots is not None else 10**9
    pending_hidden: Any | None = None
    pipeline_started = time.perf_counter()
    try:
        while (ctx.inflight or ctx.pending) and slot < limit:
            slot_started = time.perf_counter()
            ctx.admit_pending(slot)
            kv.set_inflight(tuple(ctx.inflight))
            tasks = rank_work(tuple(ctx.inflight), slot, spec.rank)
            _assert_i8(tasks)
            contexts = kv.prepare(tasks) if tasks else []
            output = None
            if tasks:
                output = adapter.forward(tasks, contexts, hidden=pending_hidden)
                kv.publish(tasks)
            comm_handles: list[Any] = []
            world = spec.topology.world
            if pp is not None and world > 1:
                last_rank = world - 1
                # S=1: last→0 is latent feedback for the next denoise step.
                # S=T+1: the next step lives on the next ranks; rank 0 starts new chunks.
                if spec.topology.stages == 1 and rank_work(tuple(ctx.inflight), slot, last_rank):
                    if output is not None and spec.rank == last_rank:
                        comm_handles.extend(pp.isend_tensor_dict(adapter.pack_activation(output), dst=0))
                    elif spec.rank == 0:
                        recv, recv_handles, _ = pp.irecv_tensor_dict(src=last_rank)
                        comm_handles.extend(recv_handles)
                        pending_hidden = adapter.unpack_activation(recv)
                for src in range(last_rank):
                    dst = src + 1
                    if not rank_work(tuple(ctx.inflight), slot, src):
                        continue
                    if spec.rank == src and output is not None:
                        comm_handles.extend(pp.isend_tensor_dict(adapter.pack_activation(output), dst=dst))
                    elif spec.rank == dst:
                        recv, recv_handles, _ = pp.irecv_tensor_dict(src=src)
                        comm_handles.extend(recv_handles)
                        pending_hidden = adapter.unpack_activation(recv)
            comm_handles.extend(kv.exchange(slot))
            # ① Outbound not left dangling: only this rank's own posted handles.
            _wait(comm_handles)
            still = []
            for item in ctx.inflight:
                local = slot - item.t0
                if local + 1 < item.plan.num_slots:
                    still.append(item)
                else:
                    kv.end_request(item.req)
            ctx.inflight = still
            kv.set_inflight(tuple(ctx.inflight))
            # ② Next slot's sources: wait only the inbound versions slot+1 reads.
            kv.evict(slot)
            kv.await_ready(slot)
            if spec.rank == 0 and tasks:
                # Active slots only — idle bubbles would dominate SERIAL medians.
                _cuda_sync()
                _record_slot(time.perf_counter() - slot_started)
            slot += 1
        # Teardown: land anything the narrowed waits skipped.
        kv.drain()
        if spec.rank == 0:
            _cuda_sync()
            _record_pipeline(time.perf_counter() - pipeline_started)
    except Exception:
        # Fail-closed: release VersionPool slots so later requests are not stranded.
        for req in {item.req for item in ctx.inflight} | {item.req for item in ctx.pending}:
            ctx.drop(req)
        raise


@contextmanager
def bind_ar_diffusion_chunk_context(owner: Any, ctx: ARDiffusionChunkContext) -> Iterator[ARDiffusionChunkContext]:
    prev = getattr(owner, "_ar_diffusion_chunk_context", None)
    owner._ar_diffusion_chunk_context = ctx
    try:
        yield ctx
    finally:
        owner._ar_diffusion_chunk_context = prev
