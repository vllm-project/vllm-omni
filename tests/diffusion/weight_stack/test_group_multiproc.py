# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Multiprocess tests for the replica-wide group confirmation (RFC 9536 01-4).

The workers initialize a real Gloo process group on CPU, so the confirmation
protocol, the AllGather group creation/reuse and the process-level table are
exercised with real collectives.  Two and four ranks are used; a three-rank case
covers the timeout message.
"""

from __future__ import annotations

import queue
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from tests.helpers.distributed import collective_context
from vllm_omni.diffusion.weight_stack.errors import (
    ConfirmationTimeoutError,
    GroupConfirmationError,
    PeerFailureError,
    PlanningError,
    WeightStackError,
    wrap_step_error,
)
from vllm_omni.diffusion.weight_stack.group import (
    GroupConfirmation,
    Point0Payload,
    Point1Payload,
    RoundMessage,
    assemble_topology,
    check_collective_parts,
    check_unmatched_policy_keys,
    create_allgather_groups,
    destroy_allgather_groups,
    group_table_keys,
    raise_if_any_failed,
    read_rank_coordinates,
)
from vllm_omni.diffusion.weight_stack.interfaces import (
    Elections,
    ParallelCoordinate,
    RankCoordinates,
    RankFingerprints,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu, pytest.mark.core_model]

CONFIRMATION_TIMEOUT_SECONDS = 30.0

# Hang scenarios: a 5 s confirmation timeout must return well before the hang ends.
# The gap leaves room for CI scheduling jitter while still proving the wait is bounded
# by the timeout rather than by the hung rank.
HANG_CONFIRMATION_TIMEOUT_SECONDS = 5.0
HANG_SECONDS = 12.0
HANG_WAIT_UPPER_BOUND_SECONDS = 10.0
HANG_SETTLE_LOWER_BOUND_SECONDS = 11.0

# Local startup step -> the confirmation point where its failure is reported (RFC 01 section 6.2).
STEP_TO_POINT = {
    "construct_modules": "point0_topology",
    "build_structure": "point0_topology",
    "plan": "point1_fingerprints",
    "preflight": "point1_fingerprints",
    "confirm": "point1_elections",
    "admit": "point2_ready",
    "allocate": "point3_commit",
    "connect_writer": "point3_commit",
    "write": "point3_commit",
    "finalize": "point3_commit",
    "commit": "point3_commit",
}

# RFC 01 section 3, pinned here instead of reading the implementation's table.
STEP_EXPECTED_ERRORS = {
    "construct_modules": "StackUnavailableError",
    "build_structure": "ConstructionError",
    "plan": "PlanningError",
    "preflight": "PreflightError",
    "confirm": "GroupConfirmationError",
    "admit": "GroupConfirmationError",
    "allocate": "AllocationError",
    "connect_writer": "AllocationError",
    "write": "WriteError",
    "finalize": "FinalizeError",
    "commit": "CommitError",
}

WORLD_ONE = ParallelCoordinate(size=1, rank=0)


# ---------------------------------------------------------------------------
# Worker-side helpers
# ---------------------------------------------------------------------------


def _coordinates(rank: int, world_size: int) -> RankCoordinates:
    return RankCoordinates(
        tp=ParallelCoordinate(size=world_size, rank=rank),
        sp=WORLD_ONE,
        ulysses=WORLD_ONE,
        ring=WORLD_ONE,
        cfg=WORLD_ONE,
        dp=WORLD_ONE,
        pp=WORLD_ONE,
        ep=None,
        fs=None,
    )


def _fingerprints(
    rank: int,
    content_identities: dict[str, str],
    collective: dict[str, str],
) -> RankFingerprints:
    return RankFingerprints(
        rank=rank,
        content_identities=content_identities,
        plan_fingerprint=f"plan-{rank}",
        collective=collective,
    )


def _point1(
    fingerprints: RankFingerprints,
    group_keys: frozenset[tuple[tuple[int, ...], str]],
    unmatched_policy_keys: frozenset[str] = frozenset(),
) -> Point1Payload:
    return Point1Payload(
        fingerprints=fingerprints,
        group_keys=group_keys,
        unmatched_policy_keys=unmatched_policy_keys,
    )


def _ctx(world_size: int) -> SimpleNamespace:
    return SimpleNamespace(world_size=world_size, od_config=SimpleNamespace(stage_id=0))


def _group_view(groups) -> dict[str, list[dict[str, object]]]:
    return {
        component: [{"ranks": list(group.ranks), "member": group.group is not None} for group in component_groups]
        for component, component_groups in groups.items()
    }


def _payloads(gathered) -> tuple:
    return tuple(message.payload for message in gathered)


def _round(channel: GroupConfirmation, rank: int, point: str, payload=None):
    message = RoundMessage.of(point, rank, None, payload)
    return channel.exchange(point, message)


def _install_new_group_counter() -> list[tuple[tuple[int, ...], str]]:
    created: list[tuple[tuple[int, ...], str]] = []
    real_new_group = torch.distributed.new_group

    def counting_new_group(ranks, backend=None):
        created.append((tuple(ranks), backend))
        return real_new_group(ranks, backend=backend)

    torch.distributed.new_group = counting_new_group
    return created


def _scenario_happy(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    payload = Point0Payload(
        record={"rank": rank},
        coordinates=_coordinates(rank, world_size),
        deployment=("d" * 32, True),
    )
    gathered0 = _round(channel, rank, "point0_topology", payload)
    topology = assemble_topology(_payloads(gathered0), _ctx(world_size))

    fingerprints = _fingerprints(rank, {"dit": "id-0"}, {"dit": "sig"})
    gathered1 = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys()))
    all_fps = tuple(message.payload.fingerprints for message in gathered1)
    check_unmatched_policy_keys(_payloads(gathered1))
    check_collective_parts(all_fps)
    groups = create_allgather_groups(all_fps, _payloads(gathered1), backend="gloo")

    local_election = {"rank": rank, "write_process": rank == 0}
    gathered2 = _round(channel, rank, "point1_elections", local_election)
    elections = Elections(local=local_election, records=_payloads(gathered2))

    _round(channel, rank, "point2_ready")
    _round(channel, rank, "point3_commit")
    return {
        "topology": topology,
        "groups": _group_view(groups),
        "elections": {"local": elections.local, "records": list(elections.records)},
        "created": list(created),
    }


def _scenario_peer_failure(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    outcomes: dict[str, dict | None] = {}
    for index, (step, point) in enumerate(STEP_TO_POINT.items()):
        failing_rank = index % world_size
        pending = None
        if rank == failing_rank:
            pending = wrap_step_error(step, RuntimeError(f"injected failure in {step}"))
            pending.rank = rank
            pending.stage_id = 0
        message = RoundMessage.of(point, rank, pending, None)
        gathered = channel.exchange(point, message)
        try:
            raise_if_any_failed(gathered, pending)
        except WeightStackError as exc:
            outcomes[step] = {"type": type(exc).__name__, "message": str(exc)}
        else:
            outcomes[step] = None
    return {"outcomes": outcomes}


def _scenario_confirm_failure(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    payload = Point0Payload(
        record={"rank": rank},
        coordinates=_coordinates(rank, world_size),
        deployment=("d" * 32, True),
    )
    _round(channel, rank, "point0_topology", payload)

    fingerprints = _fingerprints(rank, {"dit": "id-0"}, {"dit": "sig"})
    gathered1 = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys()))
    all_fps = tuple(message.payload.fingerprints for message in gathered1)
    check_collective_parts(all_fps)
    create_allgather_groups(all_fps, _payloads(gathered1), backend="gloo")
    created_after_groups = list(created)

    pending = None
    if rank == 1:
        pending = GroupConfirmationError("injected failure in confirm", phase="confirm")
        pending.rank = rank
    message = RoundMessage.of("point1_elections", rank, pending, None)
    gathered2 = channel.exchange("point1_elections", message)
    error_type = None
    try:
        raise_if_any_failed(gathered2, pending)
    except WeightStackError as exc:
        error_type = type(exc).__name__
    return {"error": error_type, "created_after_groups": created_after_groups, "total_created": len(created)}


def _scenario_hang(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    hang_rank = world_size - 1
    timeout_channel = GroupConfirmation(
        world_size=world_size,
        cpu_group=torch.distributed.group.WORLD,
        timeout_seconds=HANG_CONFIRMATION_TIMEOUT_SECONDS,
        rank=rank,
    )
    if rank == hang_rank:
        time.sleep(HANG_SECONDS)
    try:
        timeout_channel.exchange("point0_topology", RoundMessage.of("point0_topology", rank, None, {"rank": rank}))
    except Exception as exc:  # noqa: BLE001 - the test inspects the failure type
        return {
            "error": type(exc).__name__,
            "message": str(exc),
            "is_timeout": isinstance(exc, ConfirmationTimeoutError),
        }
    return {"error": None, "message": None, "is_timeout": False}


def _scenario_tp2(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    tp_rank = rank % 2
    content = {"dit": f"tp-{tp_rank}"}
    collective = {"dit": f"sig-{tp_rank}"}
    if rank in (0, 2):
        content["text_encoder"] = "te"
        collective["text_encoder"] = "te-sig"
    fingerprints = _fingerprints(rank, content, collective)
    gathered = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys()))
    all_fps = tuple(message.payload.fingerprints for message in gathered)
    check_collective_parts(all_fps)
    groups = create_allgather_groups(all_fps, _payloads(gathered), backend="gloo")
    return {"created": list(created), "groups": _group_view(groups)}


def _scenario_reuse(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    fingerprints = _fingerprints(rank, {"dit": "id-0"}, {"dit": "sig"})
    gathered1 = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys()))
    all_fps = tuple(message.payload.fingerprints for message in gathered1)
    create_allgather_groups(all_fps, _payloads(gathered1), backend="gloo")
    created_after_first = list(created)

    gathered2 = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys()))
    groups2 = create_allgather_groups(all_fps, _payloads(gathered2), backend="gloo")
    return {
        "created_after_first": created_after_first,
        "total_created": len(created),
        "groups": _group_view(groups2),
    }


def _scenario_missing_table(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    fingerprints = _fingerprints(rank, {"dit": "id-0"}, {"dit": "sig"})
    gathered1 = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys()))
    all_fps = tuple(message.payload.fingerprints for message in gathered1)
    create_allgather_groups(all_fps, _payloads(gathered1), backend="gloo")
    if rank == world_size - 1:
        # This rank forgets the group, the other ranks keep it.
        destroy_allgather_groups()
    gathered2 = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys()))
    groups2 = create_allgather_groups(all_fps, _payloads(gathered2), backend="gloo")
    return {"created": list(created), "groups": _group_view(groups2)}


def _scenario_collective_mismatch(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    signature = "sig-a" if rank == 0 else "sig-b"
    fingerprints = _fingerprints(rank, {"dit": "id-0"}, {"dit": signature})
    gathered = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys()))
    all_fps = tuple(message.payload.fingerprints for message in gathered)
    try:
        check_collective_parts(all_fps)
    except GroupConfirmationError as exc:
        return {"error": type(exc).__name__, "message": str(exc), "created": list(created)}
    return {"error": None, "message": None, "created": list(created)}


def _scenario_unmatched(rank: int, world_size: int, channel: GroupConfirmation, created: list, mode: str) -> dict:
    if mode == "disjoint":
        unmatched = frozenset({"key-a"} if rank == 0 else {"key-b"})
    else:
        unmatched = frozenset({"shared-key"})
    fingerprints = _fingerprints(rank, {"dit": "id-0"}, {"dit": "sig"})
    gathered = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys(), unmatched))
    payloads = _payloads(gathered)
    all_fps = tuple(message.payload.fingerprints for message in gathered)
    try:
        check_unmatched_policy_keys(payloads)
    except PlanningError as exc:
        return {"error": type(exc).__name__, "message": str(exc), "created": list(created)}
    groups = create_allgather_groups(all_fps, payloads, backend="gloo")
    return {"error": None, "message": None, "created": list(created), "groups": _group_view(groups)}


def _scenario_deployment_mismatch(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    deployment = (("a" if rank == 0 else "b") * 32, True)
    payload = Point0Payload(
        record={"rank": rank},
        coordinates=_coordinates(rank, world_size),
        deployment=deployment,
    )
    gathered = _round(channel, rank, "point0_topology", payload)
    try:
        assemble_topology(_payloads(gathered), _ctx(world_size))
    except GroupConfirmationError as exc:
        return {"error": type(exc).__name__, "message": str(exc)}
    return {"error": None, "message": None}


def _scenario_default_backend(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    """Exercise the production backend path (``current_omni_platform.dist_backend``)."""
    fingerprints = _fingerprints(rank, {"dit": "id-0"}, {"dit": "sig"})
    gathered1 = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys()))
    all_fps = tuple(message.payload.fingerprints for message in gathered1)
    check_collective_parts(all_fps)
    create_allgather_groups(all_fps, _payloads(gathered1))  # no explicit backend
    created_after_first = list(created)

    gathered2 = _round(channel, rank, "point1_fingerprints", _point1(fingerprints, group_table_keys()))
    groups2 = create_allgather_groups(all_fps, _payloads(gathered2))
    return {
        "created_after_first": created_after_first,
        "total_created": len(created),
        "groups": _group_view(groups2),
    }


def _scenario_hang_rank0(rank: int, world_size: int, channel: GroupConfirmation, created: list) -> dict:
    """Hang rank 0 itself: only the local watchdog can bound the other ranks."""
    timeout_channel = GroupConfirmation(
        world_size=world_size,
        cpu_group=torch.distributed.group.WORLD,
        timeout_seconds=HANG_CONFIRMATION_TIMEOUT_SECONDS,
        rank=rank,
    )
    if rank == 0:
        time.sleep(HANG_SECONDS)
    try:
        timeout_channel.exchange("point0_topology", RoundMessage.of("point0_topology", rank, None, {"rank": rank}))
    except Exception as exc:  # noqa: BLE001 - the test inspects the failure type
        return {
            "error": type(exc).__name__,
            "message": str(exc),
            "is_timeout": isinstance(exc, ConfirmationTimeoutError),
        }
    return {"error": None, "message": None, "is_timeout": False}


SCENARIOS = {
    "happy": _scenario_happy,
    "peer_failure": _scenario_peer_failure,
    "confirm_failure": _scenario_confirm_failure,
    "hang": _scenario_hang,
    "hang_rank0": _scenario_hang_rank0,
    "tp2": _scenario_tp2,
    "reuse": _scenario_reuse,
    "missing_table": _scenario_missing_table,
    "collective_mismatch": _scenario_collective_mismatch,
    "deployment_mismatch": _scenario_deployment_mismatch,
    "default_backend": _scenario_default_backend,
    "unmatched_disjoint": lambda rank, world_size, channel, created: _scenario_unmatched(
        rank, world_size, channel, created, "disjoint"
    ),
    "unmatched_intersection": lambda rank, world_size, channel, created: _scenario_unmatched(
        rank, world_size, channel, created, "intersection"
    ),
}


def _group_worker(rank: int, world_size: int, init_method: str, scenario: str, result_queue) -> None:
    from tests.helpers.distributed import configure_cpu_collective_worker

    result: dict = {"rank": rank}
    try:
        # Install the CPU Gloo platform so the production backend lookup
        # (``current_omni_platform.dist_backend``) resolves to "gloo".
        configure_cpu_collective_worker()
        torch.distributed.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=120),
        )
        created = _install_new_group_counter()
        channel = GroupConfirmation(
            world_size=world_size,
            cpu_group=torch.distributed.group.WORLD,
            timeout_seconds=CONFIRMATION_TIMEOUT_SECONDS,
            rank=rank,
        )
        started = time.monotonic()
        result["payload"] = SCENARIOS[scenario](rank, world_size, channel, created)
        result["elapsed"] = time.monotonic() - started
        result["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001 - the worker reports any failure to the parent
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            destroy_allgather_groups()
        except Exception:  # noqa: BLE001 - best-effort worker teardown
            pass
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:  # noqa: BLE001 - best-effort worker teardown
            pass
    result_queue.put((rank, result))


def _run_workers(worker, nprocs: int, build_args) -> dict[int, dict]:
    """Run workers without joining first, so a deadlocked worker fails the test instead of hanging CI."""
    context = collective_context("cpu")
    manager = context.Manager()
    result_queue = manager.Queue()
    process_context = None
    results: dict[int, dict] = {}
    try:
        args = build_args(result_queue)
        if context.get_start_method() == "spawn":
            process_context = torch.multiprocessing.spawn(worker, args=args, nprocs=nprocs, join=False)
        else:
            process_context = torch.multiprocessing.start_processes(
                worker,
                args=args,
                nprocs=nprocs,
                start_method=context.get_start_method(),
                join=False,
            )
        try:
            missing = set(range(nprocs))
            for _ in range(nprocs):
                rank, result = result_queue.get(timeout=2 * CONFIRMATION_TIMEOUT_SECONDS)
                results[rank] = result
                missing.discard(rank)
        except queue.Empty as exc:
            raise AssertionError(
                f"workers did not report within {2 * CONFIRMATION_TIMEOUT_SECONDS:.0f}s; "
                f"missing ranks: {sorted(missing)}; reported: {sorted(results)}"
            ) from exc
        finally:
            for process in process_context.processes:
                process.join(timeout=10)
            for process in process_context.processes:
                if process.is_alive():
                    process.terminate()
            for process in process_context.processes:
                process.join(timeout=10)
        return results
    finally:
        manager.shutdown()


def _run_scenario(scenario: str, nprocs: int) -> dict[int, dict]:
    from vllm.utils.network_utils import get_file_store_init_method

    return _run_workers(
        _group_worker,
        nprocs,
        lambda result_queue: (nprocs, get_file_store_init_method(), scenario, result_queue),
    )


def _assert_all_ok(results: dict[int, dict]) -> None:
    failures = {rank: result.get("error") for rank, result in results.items() if result["status"] != "ok"}
    assert not failures, f"worker failures: {failures}"


def _assert_all_rank_values_equal(results: dict[int, dict], key: str) -> None:
    values = [result["payload"][key] for result in results.values()]
    for value in values[1:]:
        assert value == values[0]


def _real_coordinates_worker(rank: int, world_size: int, init_method: str, result_queue) -> None:
    """Initialize the real Omni parallel state on CPU and read the rank coordinates from it."""
    from tests.helpers.distributed import configure_cpu_collective_worker

    result: dict = {"rank": rank}
    parallel_state = None
    started = time.monotonic()
    try:
        configure_cpu_collective_worker()
        from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state

        parallel_state = omni_parallel_state
        omni_parallel_state.init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=init_method,
            local_rank=0,
            backend="gloo",
        )
        omni_parallel_state.initialize_model_parallel(
            sequence_parallel_size=world_size,
            ulysses_degree=world_size,
            backend="gloo",
        )
        coordinates = read_rank_coordinates()
        result["payload"] = {
            name: None if coordinate is None else [coordinate.size, coordinate.rank]
            for name, coordinate in (
                ("tp", coordinates.tp),
                ("sp", coordinates.sp),
                ("ulysses", coordinates.ulysses),
                ("ring", coordinates.ring),
                ("cfg", coordinates.cfg),
                ("dp", coordinates.dp),
                ("pp", coordinates.pp),
                ("ep", coordinates.ep),
                ("fs", coordinates.fs),
            )
        }
        result["elapsed"] = time.monotonic() - started
        result["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001 - the worker reports any failure to the parent
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if parallel_state is not None:
            try:
                parallel_state.destroy_distributed_env()
            except Exception:  # noqa: BLE001 - best-effort worker teardown
                pass
    result_queue.put((rank, result))


def _run_real_coordinates(nprocs: int) -> dict[int, dict]:
    from vllm.utils.network_utils import get_file_store_init_method

    return _run_workers(
        _real_coordinates_worker,
        nprocs,
        lambda result_queue: (nprocs, get_file_store_init_method(), result_queue),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_all_ranks_agree():
    results = _run_scenario("happy", 2)
    _assert_all_ok(results)
    _assert_all_rank_values_equal(results, "topology")
    _assert_all_rank_values_equal(results, "groups")
    _assert_all_rank_values_equal(results, "created")

    topology = results[0]["payload"]["topology"]
    assert topology.world_size == 2
    assert topology.stage_id == 0
    assert topology.records == ({"rank": 0}, {"rank": 1})  # ordered by rank
    assert len(topology.coordinates) == 2
    assert topology.deployment_id == "d" * 32
    assert results[0]["payload"]["created"] == [((0, 1), "gloo")]
    assert results[0]["payload"]["groups"] == {"dit": [{"ranks": [0, 1], "member": True}]}
    assert all(result["elapsed"] < CONFIRMATION_TIMEOUT_SECONDS for result in results.values())

    election_records = [results[rank]["payload"]["elections"]["records"] for rank in results]
    assert election_records[0] == election_records[1]
    assert sorted(record["rank"] for record in election_records[0]) == [0, 1]


def test_peer_failure_at_every_local_step_fails_the_whole_replica():
    results = _run_scenario("peer_failure", 2)
    _assert_all_ok(results)

    for index, (step, point) in enumerate(STEP_TO_POINT.items()):
        failing_rank = index % 2
        failing = results[failing_rank]["payload"]["outcomes"][step]
        assert failing is not None
        assert failing["type"] == STEP_EXPECTED_ERRORS[step]
        assert step in failing["message"]

        peer = results[1 - failing_rank]["payload"]["outcomes"][step]
        assert peer is not None
        assert peer["type"] == PeerFailureError.__name__
        assert f"rank {failing_rank}:" in peer["message"]
        assert point in peer["message"]

    # Every round returned within the confirmation timeout instead of hanging.
    assert all(result["elapsed"] < CONFIRMATION_TIMEOUT_SECONDS for result in results.values())


def test_confirm_failure_after_group_creation_does_not_strand_ranks():
    results = _run_scenario("confirm_failure", 2)
    _assert_all_ok(results)

    created = [results[rank]["payload"]["created_after_groups"] for rank in results]
    assert created[0] == created[1] == [((0, 1), "gloo")]
    assert results[0]["payload"]["total_created"] == results[1]["payload"]["total_created"] == 1

    assert results[1]["payload"]["error"] == GroupConfirmationError.__name__
    assert results[0]["payload"]["error"] == PeerFailureError.__name__
    assert all(result["elapsed"] < CONFIRMATION_TIMEOUT_SECONDS for result in results.values())


def test_hang_at_confirmation_point_times_out_and_lists_missing_rank():
    results = _run_scenario("hang", 3)
    _assert_all_ok(results)

    for rank in (0, 1):
        payload = results[rank]["payload"]
        assert payload["is_timeout"] is True
        assert payload["error"] == ConfirmationTimeoutError.__name__
        # The barrier timeout is 5s; the waiting ranks must return well before the
        # 12s hang ends, even with CI scheduling jitter.
        assert 4.0 <= results[rank]["elapsed"] <= HANG_WAIT_UPPER_BOUND_SECONDS
    # Rank 0 reports the missing rank; the other ranks only see the bounded timeout.
    assert "rank(s) [2]" in results[0]["payload"]["message"]
    # The hanging rank only returns once its own sleep is over.
    assert results[2]["elapsed"] >= HANG_SETTLE_LOWER_BOUND_SECONDS


def test_hang_of_rank_zero_is_bounded_for_the_other_ranks():
    """PyTorch only bounds rank 0, so the local watchdog must bound the other ranks."""
    results = _run_scenario("hang_rank0", 3)
    _assert_all_ok(results)

    for rank in (1, 2):
        payload = results[rank]["payload"]
        assert payload["is_timeout"] is True
        assert payload["error"] == ConfirmationTimeoutError.__name__
        # 5s barrier timeout + 1s watchdog grace; the rank must not wait for rank 0.
        assert 4.5 <= results[rank]["elapsed"] <= HANG_WAIT_UPPER_BOUND_SECONDS
    # Rank 0 was the hung one, so it only returns after its own sleep.
    assert results[0]["elapsed"] >= HANG_SETTLE_LOWER_BOUND_SECONDS


def test_default_backend_path_uses_the_platform_shim_and_reuses_groups():
    results = _run_scenario("default_backend", 2)
    _assert_all_ok(results)
    _assert_all_rank_values_equal(results, "created_after_first")
    _assert_all_rank_values_equal(results, "groups")

    for rank in results:
        assert results[rank]["payload"]["created_after_first"] == [((0, 1), "gloo")]
        assert results[rank]["payload"]["total_created"] == 1  # second startup reused
    assert results[0]["payload"]["groups"] == {"dit": [{"ranks": [0, 1], "member": True}]}


def test_tp2_allgather_groups_are_created_in_the_same_order_on_every_rank():
    results = _run_scenario("tp2", 4)
    _assert_all_ok(results)
    _assert_all_rank_values_equal(results, "created")

    assert results[0]["payload"]["created"] == [((0, 2), "gloo"), ((1, 3), "gloo"), ((0, 2), "gloo")]

    # The group objects are per-process, so compare the group structure and then
    # each rank's own membership.
    structure = {
        rank: {
            component: [group["ranks"] for group in groups]
            for component, groups in results[rank]["payload"]["groups"].items()
        }
        for rank in results
    }
    assert structure[0] == structure[1] == structure[2] == structure[3]
    assert structure[0] == {"dit": [[0, 2], [1, 3]], "text_encoder": [[0, 2]]}

    membership = {
        rank: {
            component: [group["member"] for group in groups]
            for component, groups in results[rank]["payload"]["groups"].items()
        }
        for rank in results
    }
    assert membership[0] == {"dit": [True, False], "text_encoder": [True]}
    assert membership[1] == {"dit": [False, True], "text_encoder": [False]}
    assert membership[2] == {"dit": [True, False], "text_encoder": [True]}
    assert membership[3] == {"dit": [False, True], "text_encoder": [False]}


def test_second_startup_reuses_groups_without_calling_new_group():
    results = _run_scenario("reuse", 2)
    _assert_all_ok(results)
    _assert_all_rank_values_equal(results, "created_after_first")
    _assert_all_rank_values_equal(results, "groups")

    for rank in results:
        assert results[rank]["payload"]["created_after_first"] == [((0, 1), "gloo")]
        assert results[rank]["payload"]["total_created"] == 1


def test_one_rank_missing_the_group_table_makes_every_rank_recreate():
    results = _run_scenario("missing_table", 2)
    _assert_all_ok(results)
    _assert_all_rank_values_equal(results, "created")
    _assert_all_rank_values_equal(results, "groups")

    for rank in results:
        assert results[rank]["payload"]["created"] == [((0, 1), "gloo"), ((0, 1), "gloo")]
    assert results[0]["payload"]["groups"] == {"dit": [{"ranks": [0, 1], "member": True}]}


def test_inconsistent_collective_fingerprints_fail_all_ranks_before_group_creation():
    results = _run_scenario("collective_mismatch", 2)
    _assert_all_ok(results)

    for rank in results:
        payload = results[rank]["payload"]
        assert payload["error"] == GroupConfirmationError.__name__
        assert payload["created"] == []
        assert "'dit'" in payload["message"]


def test_inconsistent_deployment_identity_fails_all_ranks():
    results = _run_scenario("deployment_mismatch", 2)
    _assert_all_ok(results)

    for rank in results:
        payload = results[rank]["payload"]
        assert payload["error"] == GroupConfirmationError.__name__
        assert "rank 1" in payload["message"]
        assert "scoped to its own stage" in payload["message"]


def test_disjoint_unmatched_policy_keys_continue_the_startup():
    results = _run_scenario("unmatched_disjoint", 2)
    _assert_all_ok(results)
    _assert_all_rank_values_equal(results, "created")

    for rank in results:
        assert results[rank]["payload"]["error"] is None
        assert results[rank]["payload"]["created"] == [((0, 1), "gloo")]


def test_unmatched_policy_key_intersection_fails_all_ranks_before_group_creation():
    results = _run_scenario("unmatched_intersection", 2)
    _assert_all_ok(results)

    messages = []
    for rank in results:
        payload = results[rank]["payload"]
        assert payload["error"] == PlanningError.__name__
        assert payload["created"] == []
        assert "offload policy key(s) 'shared-key'" in payload["message"]
        messages.append(payload["message"])
    assert messages[0] == messages[1]


def test_read_rank_coordinates_matches_the_real_parallel_state():
    results = _run_real_coordinates(4)
    _assert_all_ok(results)

    for rank, result in results.items():
        payload = result["payload"]
        # sequence_parallel_size=ulysses_degree=4: the SP and Ulysses ranks follow the
        # global rank; the other dimensions are 1 and EP/HSDP are not enabled.
        assert payload["sp"] == [4, rank]
        assert payload["ulysses"] == [4, rank]
        assert payload["tp"] == [1, 0]
        assert payload["ring"] == [1, 0]
        assert payload["cfg"] == [1, 0]
        assert payload["dp"] == [1, 0]
        assert payload["pp"] == [1, 0]
        assert payload["ep"] is None
        assert payload["fs"] is None
