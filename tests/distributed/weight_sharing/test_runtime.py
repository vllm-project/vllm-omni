# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sharing lifecycle tests with the real backend and a CPU codec, not CUDA evidence."""

import asyncio
import importlib
import runpy
from dataclasses import replace
from pathlib import Path

import msgspec
import pytest

from tests.distributed.weight_sharing.helpers import FakeCodec, make_weights

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Host:
    def __init__(self, package):
        self.api = importlib.import_module(f"{package.__name__}.contracts")
        self.runtime = importlib.import_module(f"{package.__name__}.runtime")
        self.worker = importlib.import_module(f"{package.__name__}.worker_session")
        self.package = package
        self.sessions = {}
        self.events = []
        self.fail_at = None
        self.response = lambda stage, op, result: result
        self.external = []
        self.detach_error = False
        self.terminate_error = False
        self.codec = FakeCodec(self.api)

    def plan(self, tp=2, **kwargs):
        return self.runtime.SharingPlan(
            0,
            1,
            tuple(f"GPU-{rank}" for rank in range(tp)),
            (self.runtime.DeviceBudget(1024, 128, 128, 64),) * tp,
            model_id="model",
            checkpoint_revision="revision",
            layout="layout",
            timeout=0.1,
            poll_interval=0.001,
            **kwargs,
        )

    async def launch(self, stage, bootstrap):
        self.events.append((stage, "launch"))
        # Exercise exactly the JSON-safe startup representation used by the host.
        bootstrap = msgspec.convert(msgspec.to_builtins(bootstrap), type=self.runtime.Bootstrap)
        self.sessions[stage] = []
        for group in bootstrap.groups:
            context = self.worker.build_session_context(group, group.rank)
            bindings, spec = make_weights(self.api, two_storages=True)
            backend = self.package.TorchCudaIpcBackend(codec=self.codec)
            if group.role == "provider":
                handle = backend.prepare_export(bindings, spec, context)
            else:
                handle = backend.open_import(group.payload, spec, context)
            self.sessions[stage].append(
                self.worker.WorkerSharingSession(
                    backend,
                    context,
                    handle,
                    memory_usage=lambda: (64, 64, None),
                    detach=self.detach,
                )
            )
            if self.fail_at == (stage, "launch"):
                raise RuntimeError("partial startup failure")

    def detach(self):
        if self.detach_error:
            raise RuntimeError("model detach failed")

    async def rpc(self, stage, operation, messages):
        self.events.append((stage, operation))
        results = [session.control(operation, messages) for session in self.sessions[stage]]
        if self.fail_at == (stage, operation):
            raise TimeoutError("response lost after execution")
        return self.response(stage, operation, results)

    async def quiesce(self, stage):
        self.events.append((stage, "quiesce"))

    def check_health(self, stage):
        self.events.append((stage, "health"))
        if self.fail_at == (stage, "health"):
            raise RuntimeError("peer process exited")

    async def terminate(self, stage):
        self.events.append((stage, "terminate"))
        if stage == 1 and self.terminate_error:
            raise RuntimeError("consumer still alive")


@pytest.fixture
def host(weight_sharing):
    return Host(weight_sharing)


def test_tp_start_and_close_use_receipts_before_provider_release(host):
    async def run():
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        await runtime.start()
        assert runtime.state == "READY"
        assert len({i.session_id for i in runtime.identities}) == 2
        assert host.events == [(0, "launch"), (0, "claim"), (1, "launch"), (1, "status"), (0, "confirm_import")]
        await runtime.close()
        assert runtime.state == "CLOSED" and runtime.termination_complete
        assert host.events[-7:] == [
            (0, "quiesce"),
            (1, "quiesce"),
            (1, "close"),
            (0, "confirm_release"),
            (0, "release"),
            (1, "terminate"),
            (0, "terminate"),
        ]
        events = list(host.events)
        await runtime.close()
        assert host.events == events
        with pytest.raises(RuntimeError, match="twice"):
            await runtime.start()

    asyncio.run(run())


@pytest.mark.parametrize("failure", [(0, "claim"), (1, "launch"), (1, "status"), (0, "confirm_import")])
def test_uncertain_start_never_replays_or_releases(host, failure):
    async def run():
        host.fail_at = failure
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        with pytest.raises((RuntimeError, TimeoutError)):
            await runtime.start()
        assert runtime.state == "FAILED"
        assert host.events.count((0, "claim")) == 1
        assert host.events.count((1, "launch")) <= 1
        assert (0, "release") not in host.events
        if (1, "launch") in host.events:
            assert host.events[-2:] == [(1, "terminate"), (0, "terminate")]

    asyncio.run(run())


@pytest.mark.parametrize("change", ["missing", "duplicate", "generation", "budget", "poison"])
def test_no_ready_with_invalid_rank_response(host, change):
    def response(stage, operation, results):
        if stage == 1:
            if change == "missing":
                return results[:-1]
            if change == "duplicate":
                return [results[0], results[0]]
            if change == "generation":
                results[0]["identity"]["generation"] += 1
            if change == "budget":
                results[0]["private_bytes"] = 1024
            if change == "poison":
                results[0]["state"] = "POISONED"
        return results

    async def run():
        host.response = response
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        with pytest.raises(RuntimeError):
            await runtime.start()
        assert runtime.state == "FAILED" and (0, "confirm_import") not in host.events

    asyncio.run(run())


def test_held_consumer_retries_close_only_until_external_reference_dropped(host):
    async def run():
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        await runtime.start()
        # One rank remains HELD while the other has already closed.
        host.external = [host.sessions[1][0].handle["first"]]

        async def drop():
            while (1, "close") not in host.events:
                await asyncio.sleep(0)
            assert (0, "release") not in host.events
            host.external.clear()

        await asyncio.gather(runtime.close(), drop())
        assert host.events.count((1, "close")) >= 2
        assert host.events.count((0, "claim")) == 1
        assert runtime.state == "CLOSED"

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["held", "detach", "lost_close", "lost_release_receipt"])
def test_close_failure_terminates_without_premature_release(host, failure):
    async def run():
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        await runtime.start()
        if failure == "held":
            host.external = [host.sessions[1][0].handle["first"]]
        elif failure == "detach":
            host.detach_error = True
        else:
            host.fail_at = (1, "close") if failure == "lost_close" else (0, "confirm_release")
        with pytest.raises((RuntimeError, TimeoutError)):
            await runtime.close()
        assert runtime.state == "FAILED"
        assert (0, "release") not in host.events
        assert host.events[-2:] == [(1, "terminate"), (0, "terminate")]
        if failure == "detach":
            session = host.sessions[1][0]
            assert session.backend.session_status(session.handle).state == host.api.SessionState.POISONED

    asyncio.run(run())


def test_unconfirmed_consumer_exit_retains_provider(host):
    async def run():
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        await runtime.start()
        host.terminate_error = True
        runtime.fail("peer lost")
        with pytest.raises(RuntimeError, match="still alive"):
            await runtime.close()
        assert not runtime.termination_complete
        assert (0, "terminate") not in host.events
        assert (0, "release") not in host.events
        assert runtime.identities[0].group_id in runtime._retained
        host.terminate_error = False
        await runtime.close()
        assert runtime.identities[0].group_id not in runtime._retained

    asyncio.run(run())


def test_health_failure_freezes_group(host):
    async def run():
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        await runtime.start()
        host.fail_at = (1, "health")
        with pytest.raises(RuntimeError, match="process exited"):
            await runtime.monitor()
        assert runtime.state == "FAILED"
        await runtime.close()
        assert (0, "release") not in host.events

    asyncio.run(run())


def test_worker_control_rejects_non_owner_thread(host):
    async def run():
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        await runtime.start()
        with pytest.raises(host.api.ThreadOwnershipError):
            await asyncio.to_thread(host.sessions[0][0].control, "release")
        assert runtime.state == "READY"
        await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_uuids": ("GPU-0", "GPU-0")},
        {"device_uuids": ("0", "1")},
        {"consumer_stage": 0},
        {"timeout": 0},
        {"poll_interval": float("nan")},
        {"generation": -1},
        {"budgets": ()},
    ],
)
def test_invalid_plan_rejected_before_launch(host, kwargs):
    with pytest.raises(ValueError):
        replace(host.plan(), **kwargs)
    assert not host.events


@pytest.mark.parametrize("role", ["provider", "consumer"])
def test_bootstrap_routes_the_loader_config_by_rank(host, role):
    plan = host.plan()
    runtime = host.runtime.SharingRuntime(plan, host)
    bundles = (b"bundle-0", b"bundle-1") if role == "consumer" else ()
    data = {host.runtime.CONFIG_KEY: msgspec.to_builtins(runtime.bootstrap(role, bundles))}
    from types import SimpleNamespace

    config = SimpleNamespace(additional_config=data)
    host.worker.configure_worker(config, role=role, rank=1)
    context = host.worker.build_session_context(config.weight_sharing, 0)
    assert context.identity == runtime.identities[1] and context.device_index == 0
    assert config.weight_sharing.payload == (b"bundle-1" if role == "consumer" else None)
    with pytest.raises(ValueError, match="rank"):
        host.worker.configure_worker(config, role=role, rank=2)


def test_overcommitted_budget_rejected(host):
    plan = host.plan()
    budget = msgspec.structs.replace(plan.budgets[0], capacity_bytes=1)
    with pytest.raises(ValueError, match="capacity"):
        replace(plan, budgets=(budget, budget))


@pytest.mark.parametrize("stage", [0, 1])
def test_insufficient_free_memory_fails_before_admission(host, stage):
    def response(current, operation, results):
        if current == stage:
            results[0]["free_bytes"] = 0
        return results

    async def run():
        host.response = response
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        with pytest.raises(RuntimeError, match="insufficient memory"):
            await runtime.start()
        assert runtime.state == "FAILED"
        if stage == 0:
            assert (1, "launch") not in host.events
        assert (0, "confirm_import") not in host.events

    asyncio.run(run())


def test_dit_kv_is_included_in_capacity_and_private_budget(host):
    plan = host.plan()
    budget = msgspec.structs.replace(plan.budgets[0], consumer_kv_cache_bytes=1024)
    with pytest.raises(ValueError, match="capacity"):
        replace(plan, budgets=(budget, budget))


@pytest.mark.parametrize(
    "change",
    ["fits", "group_limit", "physical_free", "simultaneous_peaks", "different_storage_sizes"],
)
def test_measured_group_counts_shared_once_and_reserves_both_private_peaks(host, change):
    # A 1000-byte card at 90%: 500 shared + 200 AR + 150 DiT = 850, not 1350.
    # Private measurements already include the 128/64-byte KV caches.
    budget = host.runtime.DeviceBudget(900, 100, 128, 64)
    plan = replace(host.plan(), budgets=(budget, budget))

    def response(stage, operation, results):
        for result in results:
            result.update(
                shared_bytes=500,
                private_bytes=200 if stage == 0 else 150,
                resident_bytes=200 if stage == 0 else 150,
                free_bytes=400 if stage == 0 else 150,
            )
            if stage == 1:
                if change == "group_limit":
                    result["private_bytes"] = 250  # Each stage fits alone; combined usage does not.
                elif change == "physical_free":
                    result["free_bytes"] = 99  # Another process consumes the safety margin.
                elif change == "simultaneous_peaks":
                    result["resident_bytes"] = 90  # Needs 60 + 100 bytes free, only 150 available.
                elif change == "different_storage_sizes":
                    result["shared_bytes"] += 1
        return results

    async def run():
        host.response = response
        runtime = host.runtime.SharingRuntime(plan, host)
        if change == "fits":
            await runtime.start()
            assert runtime.state == "READY"
            host.response = lambda stage, operation, results: results
            await runtime.close()
        else:
            with pytest.raises(RuntimeError):
                await runtime.start()
            assert host.events[-2:] == [(1, "terminate"), (0, "terminate")]

    asyncio.run(run())


def test_provider_measurements_can_reject_budget_before_consumer_launch(host):
    def response(stage, operation, results):
        for result in results:
            result["shared_bytes"] = 1000
        return results

    async def run():
        host.response = response
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            await runtime.start()
        assert (1, "launch") not in host.events

    asyncio.run(run())


@pytest.fixture
def parse_policy():
    # The public policy has no serving dependencies; exercise it on CPU hosts too.
    path = Path(__file__).resolve().parents[3] / "vllm_omni/config/weight_sharing.py"
    return runpy.run_path(str(path))["parse_weight_sharing"]


def test_ratio_policy_defaults_and_disabled_path(parse_policy):
    assert parse_policy(None) is None
    policy = parse_policy(dict(provider_stage=0, consumer_stage=1))
    assert policy.gpu_memory_utilization == 0.9 and policy.timeout == 120


@pytest.mark.parametrize("ratio", [0.5, 0.95, 1])
def test_ratio_policy_accepts_valid_fraction(parse_policy, ratio):
    policy = parse_policy(dict(provider_stage=0, consumer_stage=1, gpu_memory_utilization=ratio))
    assert policy.gpu_memory_utilization == ratio


@pytest.mark.parametrize(
    "extra",
    [{"gpu_memory_utilization": value} for value in (0, -0.1, 1.1, float("nan"), float("inf"), True, "0.9")]
    + [
        {"consumer_stage": 0},
        {"provider_stage": True},
        {"timeout": float("nan")},
        {"timeout": True},
    ]
    + [
        {key: 1}
        for key in ("shared_bytes", "provider_bytes", "consumer_bytes", "reserve_bytes", "capacity_bytes", "unknown")
    ],
)
def test_invalid_ratio_policy_rejected(parse_policy, extra):
    with pytest.raises((TypeError, ValueError)):
        parse_policy(dict(provider_stage=0, consumer_stage=1) | extra)


def test_cancelled_import_confirmation_poisons_without_replay(host):
    async def run():
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        rpc = host.rpc

        async def cancelled(stage, operation, messages):
            if operation == "confirm_import":
                raise asyncio.CancelledError()
            return await rpc(stage, operation, messages)

        host.rpc = cancelled
        with pytest.raises(asyncio.CancelledError):
            await runtime.start()
        assert runtime.state == "FAILED" and runtime.termination_complete
        assert (0, "release") not in host.events
        for sessions in host.sessions.values():
            for session in sessions:
                assert session.backend.session_status(session.handle).state == host.api.SessionState.POISONED

    asyncio.run(run())


def test_malformed_bootstrap_cannot_fall_back_to_full_loading(host):
    from types import SimpleNamespace

    config = SimpleNamespace(additional_config={})
    host.worker.configure_worker(config, role="provider", rank=0)
    assert not hasattr(config, "weight_sharing")
    config.additional_config[host.runtime.CONFIG_KEY] = {}
    with pytest.raises(msgspec.ValidationError):
        host.worker.configure_worker(config, role="provider", rank=0)


def test_hung_rpc_has_a_deadline_and_never_reimports(host):
    async def run():
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        rpc = host.rpc

        async def hung(stage, operation, messages):
            if (stage, operation) == (1, "status"):
                await asyncio.Event().wait()
            return await rpc(stage, operation, messages)

        host.rpc = hung
        with pytest.raises(TimeoutError):
            await runtime.start()
        assert runtime.state == "FAILED" and runtime.termination_complete
        assert host.events.count((1, "launch")) == 1
        assert (0, "release") not in host.events

    asyncio.run(run())


def test_health_monitor_never_queues_rpc_behind_inference(host):
    async def run():
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        await runtime.start()
        host.events.clear()
        monitor = asyncio.create_task(runtime.monitor())
        while (1, "health") not in host.events:
            await asyncio.sleep(0)
        monitor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitor
        assert all(operation == "health" for _, operation in host.events)
        await runtime.close()

    asyncio.run(run())


def test_health_monitor_stops_with_orchestrator_before_group_close(host):
    async def run():
        runtime = host.runtime.SharingRuntime(host.plan(), host)
        await runtime.start()
        stop = asyncio.Event()
        monitor = asyncio.create_task(runtime.monitor(stop))
        await asyncio.sleep(0)
        stop.set()
        await asyncio.wait_for(monitor, timeout=1)
        assert runtime.state == "READY"
        await runtime.close()
        assert runtime.termination_complete

    asyncio.run(run())
