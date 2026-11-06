# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import importlib
import multiprocessing
from types import SimpleNamespace
from typing import cast

import msgspec
import pytest

from tests.distributed.weight_sharing.helpers import load_class_methods, load_weight_sharing

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _register_child(data, start, result, hold):
    package = load_weight_sharing()
    cls = importlib.import_module(f"{package.__name__}.worker_registry").ConsumerWorkerRegistry
    registry = cls.__new__(cls)
    registry._data = data
    assert start.wait(15)
    registered = registry.register(0)
    result.send(registered)
    if registered:
        hold.recv()


@pytest.mark.parametrize("late", [False, True])
def test_partial_startup_tracks_existing_peers_and_blocks_late_imports(weight_sharing, late):
    cls = importlib.import_module(f"{weight_sharing.__name__}.worker_registry").ConsumerWorkerRegistry
    ctx = multiprocessing.get_context("spawn")
    registry = cls(ctx, 1)
    start = ctx.Event()
    hold, release = ctx.Pipe(duplex=False)
    reader, writer = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_register_child, args=(registry._data, start, writer, hold))
    child.start()
    writer.close()
    try:
        if late:
            registry.terminate(5)
            start.set()
            assert reader.poll(15) and reader.recv() is False
        else:
            start.set()
            assert reader.poll(15) and reader.recv() is True
            registry.terminate(5)
        child.join(10)
        assert not child.is_alive()
        registry.terminate(5)  # Cleanup may be retried after the manager dies.
    finally:
        if child.is_alive():
            child.kill()
        child.join(10)
        reader.close()
        hold.close()
        release.close()


@pytest.fixture
def host(weight_sharing, mocker):
    """Use real host startup/cleanup methods without importing serving/model packages."""
    runtime = importlib.import_module(f"{weight_sharing.__name__}.runtime")
    registry = importlib.import_module(f"{weight_sharing.__name__}.worker_registry")
    namespace = dict(
        asyncio=asyncio,
        cast=cast,
        msgspec=msgspec,
        Bootstrap=runtime.Bootstrap,
        CONFIG_KEY=runtime.CONFIG_KEY,
        ConsumerStartupError=registry.ConsumerStartupError,
        ConsumerWorkerRegistry=registry.ConsumerWorkerRegistry,
        get_mp_context=lambda: multiprocessing.get_context("spawn"),
        StageDiffusionClient=object,
    )
    cls = load_class_methods(
        "vllm_omni/engine/weight_sharing_runtime.py", "StageSharingHost", ["launch", "terminate"], namespace
    )
    host = cls()
    host.plan = runtime.SharingPlan(
        0, 1, ("GPU-0",), (runtime.DeviceBudget(1024, 128, 128, 64),), "model", "revision", "layout"
    )
    host.runtime = mocker.Mock(_stage_init_timeout=5)
    host.runtime._init_cancelled.is_set.return_value = False
    config = SimpleNamespace(additional_config={})
    host.stages = {1: mocker.Mock(replicas=[mocker.Mock(stage_cfg=mocker.Mock(diffusion_config=config))])}
    host.clients = {}
    host._consumer_started = False
    host._consumer_registry = None
    host.runtime_module = runtime
    host.registry_module = registry
    return host


@pytest.mark.parametrize("confirmed", [False, True])
def test_consumer_startup_ack_controls_provider_retention(host, mocker, confirmed):
    registry = mocker.Mock(spec=host.registry_module.ConsumerWorkerRegistry)
    if not confirmed:
        registry.terminate.side_effect = RuntimeError("consumer exit unconfirmed")
    mocker.patch.object(
        host.runtime,
        "_initialize_replica",
        side_effect=host.registry_module.ConsumerStartupError("startup failed", registry=registry),
    )
    group = host.runtime_module.SharingRuntime(host.plan, host)

    async def run():
        with pytest.raises(host.registry_module.ConsumerStartupError):
            await host.launch(1, group.bootstrap("consumer", (b"a",)))
        if confirmed:
            await host.terminate(1)
        else:
            with pytest.raises(RuntimeError, match="unconfirmed"):
                await host.terminate(1)

    asyncio.run(run())


def test_failure_before_consumer_process_creation_can_confirm_termination(host, mocker):
    group = host.runtime_module.SharingRuntime(host.plan, host)
    mocker.patch.object(host.runtime, "_initialize_replica", side_effect=RuntimeError("startup failed"))

    async def run():
        with pytest.raises(RuntimeError, match="startup failed"):
            await host.launch(1, group.bootstrap("consumer", (b"a",)))
        registry = host._consumer_registry
        assert registry is host.stages[1].replicas[0].stage_cfg._sharing_worker_registry
        provider = mocker.Mock()
        host.clients[0] = provider
        host.rpc = mocker.AsyncMock()
        group._started = [0, 1]
        group.fail("consumer configuration failed")
        await group.terminate()
        assert group.termination_complete
        provider.shutdown.assert_called_once()
        assert not registry.register(0)  # Late startup cannot import after cleanup.

    asyncio.run(run())


@pytest.mark.parametrize("registered", [False, True])
def test_consumer_registry_must_confirm_exit_before_provider_can_stop(host, mocker, registered):
    client = mocker.Mock()
    registry = mocker.Mock(spec=host.registry_module.ConsumerWorkerRegistry) if registered else None
    client._proc_manager = mocker.Mock(sharing_workers=registry)
    host.clients[1] = client
    host._consumer_started = True
    if registered:
        registry.terminate.side_effect = RuntimeError("consumer worker exit unconfirmed")
    with pytest.raises(RuntimeError, match="unconfirmed"):
        asyncio.run(host.terminate(1))
    client.shutdown.assert_called_once()
    assert host.clients[1] is client
    if registered:
        registry.terminate.side_effect = None
        asyncio.run(host.terminate(1))
        assert 1 not in host.clients
