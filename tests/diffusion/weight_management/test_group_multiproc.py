# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Multiprocess Gloo tests for RFC 01-4 replica confirmation and AllGather groups."""

from __future__ import annotations

import os
import socket
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.helpers.distributed import collective_context, configure_cpu_collective_worker
from vllm_omni.diffusion.weight_management import group
from vllm_omni.diffusion.weight_management.errors import (
    ConfirmationTimeoutError,
    GroupConfirmationError,
    PeerFailureError,
    PlanningError,
    WeightStackError,
    wrap_step_error,
)
from vllm_omni.diffusion.weight_management.interfaces import (
    ParallelCoordinate,
    RankCoordinates,
    RankFingerprints,
    StartupContext,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _coords(rank: int, world_size: int) -> RankCoordinates:
    one = ParallelCoordinate(1, 0)
    return RankCoordinates(
        tp=ParallelCoordinate(2, rank % 2) if world_size == 4 else one,
        sp=one,
        ulysses=one,
        ring=one,
        cfg=one,
        dp=ParallelCoordinate(world_size // 2, rank // 2) if world_size == 4 else ParallelCoordinate(world_size, rank),
        pp=one,
        ep=None,
        fs=None,
    )


def _ctx(rank: int, world_size: int) -> StartupContext:
    return StartupContext(
        od_config=SimpleNamespace(stage_id=0),  # type: ignore[arg-type]
        vllm_config=object(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        load_format="default",
        custom_pipeline_name=None,
        memory_pools=object(),  # type: ignore[arg-type]
        policy=None,
        compile_enabled=False,
        faults=None,
    )


def _round(ch: group.GroupConfirmation, point: str, payload=None, pending=None):
    msgs = ch.exchange(point, group.RoundMessage.of(point, ch.rank, pending, payload))
    group.raise_if_any_failed(msgs, pending)
    return tuple(m.payload for m in msgs)


def _worker(rank: int, world_size: int, port: int, scenario: str, timeout: float, results) -> None:
    configure_cpu_collective_worker()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30))
    cpu_group = dist.new_group(list(range(world_size)), backend="gloo")
    ch = group.GroupConfirmation(world_size=world_size, rank=rank, cpu_group=cpu_group, timeout_seconds=timeout)
    try:
        if scenario in ("hang-last", "hang-zero"):
            missing = 0 if scenario == "hang-zero" else world_size - 1
            if rank == missing:
                time.sleep(timeout + 2.0)
                results.put((rank, {"hung": True}))
                return
            started = time.monotonic()
            try:
                _round(ch, "point0_topology")
            except ConfirmationTimeoutError as exc:
                results.put(
                    (rank, {"error": type(exc).__name__, "message": str(exc), "elapsed": time.monotonic() - started})
                )
                return
            raise AssertionError("expected ConfirmationTimeoutError")

        if scenario in ("fail-preflight", "fail-confirm"):
            if scenario == "fail-confirm":
                fps = tuple(RankFingerprints(r, {"dit": "same"}, "local", {"dit": "sig"}) for r in range(world_size))
                payloads = tuple(group.Point1Payload(fp, frozenset(), frozenset()) for fp in fps)
                group.create_allgather_groups(fps, payloads, backend="gloo")
            point = "point1_fingerprints" if scenario == "fail-preflight" else "point1_elections"
            phase = "preflight" if scenario == "fail-preflight" else "confirm"
            pending = wrap_step_error(phase, RuntimeError(f"boom {phase}")) if rank == world_size - 1 else None
            if pending is not None:
                pending.rank = rank
            try:
                _round(ch, point, pending=pending)
            except WeightStackError as exc:
                results.put((rank, {"error": type(exc).__name__, "own": exc is pending, "message": str(exc)}))
                return
            raise AssertionError("expected WeightStackError")

        # Full point0 + point1 flow (used for success, reuse, mismatch scenarios).
        dep = ("a" * 32 if rank == 0 else "b" * 32, False)
        if scenario == "deployment-mismatch" and rank == world_size - 1:
            dep = ("d" * 32, True)
        p0_gathered = _round(ch, "point0_topology", group.Point0Payload(f"host-{rank}", _coords(rank, world_size), dep))
        topo = group.assemble_topology(p0_gathered, _ctx(rank, world_size))

        # TP=2 on 4 ranks -> two groups (0, 2) and (1, 3); vae only on ranks (2, 3).
        tp_id = f"tp{rank % 2}" if world_size == 4 else "shared"
        identities = {"dit": tp_id}
        sigs = {"dit": tp_id}
        if world_size == 4 and rank >= 2:
            identities["vae"], sigs["vae"] = "vae", "vae-sig"
        if scenario == "digest-mismatch" and rank == world_size - 1:
            sigs["dit"] = "diverged"
        fp = RankFingerprints(rank, identities, f"local-{rank}", sigs)
        unmatched = frozenset({"vae"}) if world_size == 4 and rank < 2 else frozenset()
        if scenario == "unmatched":
            unmatched |= {"typo"}

        p1_gathered = _round(ch, "point1_fingerprints", group.Point1Payload(fp, group.group_table_keys(), unmatched))
        all_fps = tuple(p.fingerprints for p in p1_gathered)
        group.check_unmatched_policy_keys(p1_gathered)
        group.check_collective_parts(all_fps)
        groups = group.create_allgather_groups(all_fps, p1_gathered, backend="gloo")

        # Exercise a real collective on each group this rank belongs to.
        collected = {}
        for comp, comp_groups in groups.items():
            for entry in comp_groups:
                if entry.group is not None:
                    out = [None] * len(entry.ranks)
                    dist.all_gather_object(out, rank, group=entry.group)
                    collected[(comp, entry.ranks)] = tuple(out)

        # Second round (reinit) must reuse cached groups without calling dist.new_group.
        new_group_calls = 0
        orig_new_group = dist.new_group

        def _counting_new_group(*args, **kwargs):
            nonlocal new_group_calls
            new_group_calls += 1
            return orig_new_group(*args, **kwargs)

        dist.new_group = _counting_new_group
        try:
            p1_again = _round(ch, "point1_fingerprints", group.Point1Payload(fp, group.group_table_keys(), unmatched))
            group.create_allgather_groups(tuple(p.fingerprints for p in p1_again), p1_again, backend="gloo")
        finally:
            dist.new_group = orig_new_group

        results.put(
            (
                rank,
                {
                    "deployment_id": topo.deployment_id,
                    "groups": {c: [g.ranks for g in gs] for c, gs in groups.items()},
                    "collected": collected,
                    "reinit_new_group_calls": new_group_calls,
                },
            )
        )
    except Exception as exc:
        results.put((rank, {"error": type(exc).__name__, "message": str(exc)}))
    finally:
        group.destroy_allgather_groups()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(world_size: int, scenario: str, timeout: float = 10.0) -> dict[int, dict]:
    ctx = collective_context("cpu")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, port, scenario, timeout, queue))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"
    return dict(queue.get(timeout=5) for _ in range(world_size))


def test_four_rank_topology_groups_and_reinit_reuse():
    res = _run(4, "success")
    for rank in range(4):
        assert res[rank]["deployment_id"] == "a" * 32
        assert res[rank]["groups"] == {"dit": [(0, 2), (1, 3)], "vae": [(2, 3)]}
        assert res[rank]["reinit_new_group_calls"] == 0
    assert res[0]["collected"] == {("dit", (0, 2)): (0, 2)}
    assert res[3]["collected"] == {("dit", (1, 3)): (1, 3), ("vae", (2, 3)): (2, 3)}


@pytest.mark.parametrize("scenario", ["fail-preflight", "fail-confirm"])
def test_single_rank_failure_fails_all_ranks(scenario):
    res = _run(2, scenario)
    assert res[1]["own"] is True
    assert res[0]["error"] == PeerFailureError.__name__
    assert "rank 1" in res[0]["message"]


@pytest.mark.parametrize("scenario", ["hang-last", "hang-zero"])
def test_hanging_rank_times_out_within_bound(scenario):
    res = _run(2, scenario, timeout=1.5)
    active = 1 if scenario == "hang-zero" else 0
    assert res[active]["error"] == ConfirmationTimeoutError.__name__
    assert res[active]["elapsed"] < 5.0
    if active == 0:
        assert "1" in res[0]["message"]


@pytest.mark.parametrize(
    ("scenario", "expected_error"),
    [
        ("deployment-mismatch", GroupConfirmationError.__name__),
        ("digest-mismatch", GroupConfirmationError.__name__),
        ("unmatched", PlanningError.__name__),
    ],
)
def test_point0_and_point1_validation_fails_all_ranks(scenario, expected_error):
    res = _run(2, scenario)
    for rank in range(2):
        assert res[rank]["error"] == expected_error
