# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Deferred multi-API handshakes own their socket on the readiness thread."""

import contextlib
import itertools
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
import zmq
from vllm.utils import network_utils
from vllm.v1.engine.utils import EngineZmqAddresses

from vllm_omni.engine import stage_engine_core_proc_manager, stage_engine_startup

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def fake_startup(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(
        events=[],
        managers=[],
        launches=[],
        config=SimpleNamespace(parallel_config=object(), cache_config=object()),
    )
    paths = itertools.count()
    monkeypatch.setattr(network_utils, "get_open_zmq_ipc_path", lambda: f"ipc://unused-{next(paths)}")
    monkeypatch.setattr(
        stage_engine_startup,
        "get_engine_zmq_addresses",
        lambda _config: EngineZmqAddresses(inputs=["ipc://single-in"], outputs=["ipc://single-out"]),
    )
    monkeypatch.setattr(
        stage_engine_startup, "scoped_spawn_device_env", lambda *_args, **_kwargs: contextlib.nullcontext()
    )

    def manager(**kwargs: object) -> SimpleNamespace:
        state.events.append(("spawn", threading.get_ident()))
        result = SimpleNamespace(kwargs=kwargs)
        state.managers.append(result)
        return result

    @contextlib.contextmanager
    def socket_ctx(path: str, socket_type: int, *, bind: bool) -> Iterator[SimpleNamespace]:
        assert path.startswith("ipc://unused-")
        assert socket_type == stage_engine_startup.zmq.ROUTER
        assert bind is True
        owner = threading.get_ident()
        state.events.append(("socket-create", owner))
        socket = SimpleNamespace(owner=owner, closed=False)
        try:
            yield socket
        finally:
            assert threading.get_ident() == owner, "Socket destroyed on a different thread"
            socket.closed = True
            state.events.append(("socket-close", threading.get_ident()))

    def wait_for_startup(socket, engines, parallel_config, coordinated_dp, cache_config, launch) -> None:
        assert socket.owner == threading.get_ident(), "Socket used on a different thread"
        assert not socket.closed
        assert len(engines) == 1 and engines[0].local
        assert parallel_config is state.config.parallel_config
        assert coordinated_dp is False
        assert cache_config is state.config.cache_config
        assert launch.engine_manager is state.managers[0]
        state.events.append(("socket-use", threading.get_ident()))
        state.launches.append(launch)

    monkeypatch.setattr(stage_engine_core_proc_manager, "StageEngineCoreProcManager", manager)
    monkeypatch.setattr(stage_engine_startup, "zmq_socket_ctx", socket_ctx)
    monkeypatch.setattr(stage_engine_startup, "wait_for_engine_startup", wait_for_startup)
    return state


def test_multi_api_socket_lifetime_stays_on_readiness_worker(fake_startup: SimpleNamespace) -> None:
    main_thread = threading.get_ident()
    frontends: list[Any] = []
    context = stage_engine_startup.launch_stage_replica(
        vllm_config=fake_startup.config,
        executor_class=object,
        log_stats=False,
        stage_id=0,
        num_api_servers=4,
        watched_frontend_processes=frontends,
    )
    resources = context.__enter__()
    assert resources.manager is fake_startup.managers[0]
    assert len(resources.addresses.inputs) == len(resources.addresses.outputs) == 4
    assert fake_startup.events == [("spawn", main_thread)]
    # As in StageRuntime, API processes become known only after all cores
    # have spawned and before the worker resumes the readiness handshake.
    frontend = object()
    frontends.append(frontend)
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(context.__exit__, None, None, None).result(timeout=5) is False

    worker_thread = fake_startup.events[1][1]
    assert worker_thread != main_thread
    assert fake_startup.events == [
        ("spawn", main_thread),
        ("socket-create", worker_thread),
        ("socket-use", worker_thread),
        ("socket-close", worker_thread),
    ]
    assert fake_startup.launches[0].addresses is resources.addresses
    assert fake_startup.launches[0].watched_frontend_processes is frontends
    assert fake_startup.launches[0].watched_frontend_processes == [frontend]


def test_multi_api_body_failure_does_not_create_socket_or_handshake(fake_startup: SimpleNamespace) -> None:
    with pytest.raises(RuntimeError, match="API process startup failed"):
        with stage_engine_startup.launch_stage_replica(
            vllm_config=fake_startup.config,
            executor_class=object,
            log_stats=False,
            stage_id=0,
            num_api_servers=4,
        ):
            assert fake_startup.events == [("spawn", threading.get_ident())]
            raise RuntimeError("API process startup failed")

    assert fake_startup.events == [("spawn", threading.get_ident())]
    assert fake_startup.launches == []


def test_multi_api_dealer_queues_hello_before_router_binds(
    fake_startup: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    hello_sent = threading.Event()
    reply_received = threading.Event()
    ipc_dir = tmp_path_factory.mktemp("ipc")
    paths = itertools.count()
    monkeypatch.setattr(network_utils, "get_open_zmq_ipc_path", lambda: f"ipc://{ipc_dir}/{next(paths)}")

    @contextlib.contextmanager
    def real_socket_ctx(*args, **kwargs):
        assert hello_sent.is_set(), "The ROUTER must bind after the DEALER queues HELLO"
        with network_utils.zmq_socket_ctx(*args, **kwargs) as socket:
            yield socket

    def wait_for_startup(socket, *_args) -> None:
        assert socket.poll(timeout=3000), "Queued HELLO did not reach the late-bound ROUTER"
        identity, hello = socket.recv_multipart()
        assert hello == b"HELLO"
        socket.send_multipart([identity, b"INIT"])
        # The real startup handshake waits for READY. Keep this ROUTER alive
        # until the reply arrives too, so linger=0 cannot discard queued INIT.
        assert reply_received.wait(timeout=3), "The DEALER did not receive INIT"

    monkeypatch.setattr(stage_engine_startup, "zmq_socket_ctx", real_socket_ctx)
    monkeypatch.setattr(stage_engine_startup, "wait_for_engine_startup", wait_for_startup)
    context = stage_engine_startup.launch_stage_replica(
        vllm_config=fake_startup.config,
        executor_class=object,
        log_stats=False,
        stage_id=0,
        num_api_servers=2,
    )
    resources = context.__enter__()

    def dealer_client() -> None:
        with zmq.Context() as zmq_context, zmq_context.socket(zmq.DEALER) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(resources.manager.kwargs["handshake_address"])
            socket.send(b"HELLO")
            hello_sent.set()
            assert socket.poll(timeout=3000), "The late-bound ROUTER did not reply"
            assert socket.recv() == b"INIT"
            reply_received.set()

    # Each socket remains on one worker. Bounded polls make executor shutdown
    # terminate even when either side fails before sending its message.
    with ThreadPoolExecutor(max_workers=2) as executor:
        client = executor.submit(dealer_client)
        assert hello_sent.wait(timeout=3), "The DEALER did not queue HELLO"
        handshake = executor.submit(context.__exit__, None, None, None)
        assert handshake.result(timeout=5) is False
        client.result(timeout=5)


@pytest.mark.parametrize("body_fails", [False, True])
def test_single_api_retains_socket_around_context_body(fake_startup: SimpleNamespace, body_fails: bool) -> None:
    owner = threading.get_ident()
    expected_error = pytest.raises(RuntimeError, match="body failed") if body_fails else contextlib.nullcontext()
    with expected_error:
        with stage_engine_startup.launch_stage_replica(
            vllm_config=fake_startup.config,
            executor_class=object,
            log_stats=False,
            stage_id=0,
        ) as resources:
            assert resources.addresses.inputs == ["ipc://single-in"]
            assert fake_startup.events == [("spawn", owner), ("socket-create", owner)]
            assert fake_startup.launches == []
            if body_fails:
                raise RuntimeError("body failed")

    expected = [("spawn", owner), ("socket-create", owner)]
    if not body_fails:
        expected.append(("socket-use", owner))
    expected.append(("socket-close", owner))
    assert fake_startup.events == expected
    assert len(fake_startup.launches) == (0 if body_fails else 1)
