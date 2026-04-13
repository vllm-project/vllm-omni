"""Unit tests for AsyncOmniEngine single-stage mode and OmniMasterServer.

These tests cover:
- OmniMasterServer address pre-allocation & ZMQ registration handshake
- AsyncOmniEngine single_stage_mode detection / _single_stage_id_filter setup
- _initialize_stages stage routing (local launch vs. remote-wait) in
  single_stage_mode
- _create_remote_llm_stage delegation to connect_remote_engine_cores
- _launch_llm_stage delegation to launch_omni_core_engines in
  single_stage_mode

All tests run without real hardware by mocking ZMQ, vllm_config, and the
heavy initialization helpers.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_mock import MockerFixture
from vllm.v1.engine.utils import EngineZmqAddresses

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_engine_startup import (
    OmniMasterServer,
    StageAllocation,
    StageCoordinatorAddresses,
    connect_remote_engine_cores,
    launch_omni_core_engines,
)
from vllm_omni.engine.stage_init_utils import StartedLlmStage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stage_cfg(stage_id: int, stage_type: str = "llm"):
    """Return a lightweight stage config mock."""
    return SimpleNamespace(
        stage_id=stage_id,
        stage_type=stage_type,
        engine_args=SimpleNamespace(
            async_chunk=False,
            model_stage=None,
            engine_output_type=None,
        ),
    )


def _make_started_llm_stage(stage_id: int) -> StartedLlmStage:
    """Return a minimal StartedLlmStage for mocking."""
    addresses = SimpleNamespace(
        inputs=["tcp://127.0.0.1:5000"],
        outputs=["tcp://127.0.0.1:5001"],
        frontend_stats_publish_address=None,
    )
    return StartedLlmStage(
        stage_id=stage_id,
        metadata=SimpleNamespace(stage_id=stage_id),
        vllm_config=SimpleNamespace(),
        executor_class=SimpleNamespace(),
        engine_manager=SimpleNamespace(),
        coordinator=SimpleNamespace(),
        addresses=addresses,
    )


# ---------------------------------------------------------------------------
# OmniMasterServer – address pre-allocation
# ---------------------------------------------------------------------------


class TestOmniMasterServerAllocation:
    """Test address pre-allocation in OmniMasterServer.__init__."""

    def test_public_address_and_port_properties_expose_registration_endpoint(self):
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15000,
            stage_ids=[0],
        )
        assert server.address == "127.0.0.1"
        assert server.port == 15000

    def test_allocations_created_for_each_stage_id(self):
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15000,
            stage_ids=[0, 1, 2],
        )
        assert set(server._allocations.keys()) == {0, 1, 2}

    def test_each_allocation_is_stage_allocation(self):
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15000,
            stage_ids=[0, 1],
        )
        for sid in (0, 1):
            alloc = server._allocations[sid]
            assert isinstance(alloc, StageAllocation)

    def test_allocation_addresses_reference_master_address(self):
        server = OmniMasterServer(
            master_address="192.168.1.10",
            master_port=20000,
            stage_ids=[0],
        )
        alloc = server._allocations[0]
        for addr in (
            alloc.handshake_bind_address,
            alloc.handshake_connect_address,
            alloc.input_bind_address,
            alloc.input_connect_address,
            alloc.output_bind_address,
            alloc.output_connect_address,
        ):
            assert "192.168.1.10" in addr, f"Expected master address in {addr}"

    def test_port_uniqueness_within_single_allocation(self):
        """Each allocation uses three distinct ports."""
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15001,
            stage_ids=[0],
        )
        alloc = server._allocations[0]
        hs_port = int(alloc.handshake_bind_address.split(":")[-1])
        inp_port = int(alloc.input_bind_address.split(":")[-1])
        out_port = int(alloc.output_bind_address.split(":")[-1])
        assert len({hs_port, inp_port, out_port}) == 3, "Expected three distinct ports per stage allocation"

    def test_get_zmq_addresses_returns_bind_addresses(self):
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15002,
            stage_ids=[0],
        )
        alloc = server._allocations[0]
        zmq_addrs = server.get_zmq_addresses(0)
        assert zmq_addrs.inputs == [alloc.input_bind_address]
        assert zmq_addrs.outputs == [alloc.output_bind_address]

    def test_get_engine_zmq_addresses_returns_connect_addresses(self):
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15003,
            stage_ids=[0],
        )
        alloc = server._allocations[0]
        engine_addrs = server.get_engine_zmq_addresses(0)
        assert engine_addrs.inputs == [alloc.input_connect_address]
        assert engine_addrs.outputs == [alloc.output_connect_address]

    def test_get_allocation_returns_correct_object(self):
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15004,
            stage_ids=[3],
        )
        assert server.get_allocation(3) is server._allocations[3]


# ---------------------------------------------------------------------------
# OmniMasterServer – ZMQ registration flow
# ---------------------------------------------------------------------------


class TestOmniMasterServerRegistration:
    """Test that the server correctly handles a stage registration."""

    def test_registration_reply_contains_handshake_address(self):
        """A DEALER client that sends a registration msg gets the handshake
        address back from the ROUTER registration socket."""
        import msgspec
        import zmq
        from vllm.utils.network_utils import get_open_port

        master_port = get_open_port()
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=master_port,
            stage_ids=[0],
        )
        server.start()
        expected_hs = server._allocations[0].handshake_connect_address

        ctx = zmq.Context()
        try:
            sock = ctx.socket(zmq.DEALER)
            sock.connect(f"tcp://127.0.0.1:{master_port}")
            sock.send(msgspec.msgpack.encode({"stage_id": 0}))
            if not sock.poll(timeout=5_000):
                pytest.fail("No reply received from OmniMasterServer within 5 s")
            reply = msgspec.msgpack.decode(sock.recv())
            assert reply["handshake_address"] == expected_hs
        finally:
            sock.close(linger=0)
            ctx.term()
            server.stop()

    def test_server_handles_unknown_stage_id_gracefully(self):
        """A registration for an unrecognised stage_id must not crash the server."""
        import msgspec
        import zmq
        from vllm.utils.network_utils import get_open_port

        master_port = get_open_port()
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=master_port,
            stage_ids=[0],
        )
        server.start()

        ctx = zmq.Context()
        try:
            bad_sock = ctx.socket(zmq.DEALER)
            bad_sock.connect(f"tcp://127.0.0.1:{master_port}")
            # Send unknown stage_id=99
            bad_sock.send(msgspec.msgpack.encode({"stage_id": 99}))
            # Server should NOT reply for an unknown id; wait briefly
            has_reply = bad_sock.poll(timeout=500)
            assert not has_reply, "Server should not reply to unknown stage_id"
            # Then register the valid stage so the server thread can exit
            good_sock = ctx.socket(zmq.DEALER)
            good_sock.connect(f"tcp://127.0.0.1:{master_port}")
            good_sock.send(msgspec.msgpack.encode({"stage_id": 0}))
            good_sock.poll(timeout=2_000)
        finally:
            for s in (bad_sock, good_sock):
                try:
                    s.close(linger=0)
                except Exception:
                    pass
            ctx.term()
            server.stop()

    def test_registration_stores_stage_config(self):
        """Stage registration should persist the sender's stage config."""
        import msgspec
        import zmq
        from vllm.utils.network_utils import get_open_port

        master_port = get_open_port()
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=master_port,
            stage_ids=[0],
        )
        server.start()

        payload = {
            "stage_id": 0,
            "stage_config": {
                "stage_id": 0,
                "stage_type": "llm",
                "engine_args": {"model": "fake-model"},
            },
        }

        ctx = zmq.Context()
        try:
            sock = ctx.socket(zmq.DEALER)
            sock.connect(f"tcp://127.0.0.1:{master_port}")
            sock.send(msgspec.msgpack.encode(payload))
            assert sock.poll(timeout=5_000)
            sock.recv()

            stored = server.get_stage_config(0, timeout_s=0.1)
            assert stored == payload["stage_config"]
        finally:
            sock.close(linger=0)
            ctx.term()
            server.stop()

    def test_registration_stores_coordinator_addresses(self):
        """Stage registration should persist optional coordinator addresses."""
        import msgspec
        import zmq
        from vllm.utils.network_utils import get_open_port

        master_port = get_open_port()
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=master_port,
            stage_ids=[0],
        )
        server.start()

        payload = {
            "stage_id": 0,
            "stage_config": {"stage_id": 0},
            "coordinator_input": "tcp://127.0.0.1:31001",
            "coordinator_output": "tcp://127.0.0.1:31002",
            "frontend_stats_publish_address": "tcp://127.0.0.1:31003",
        }

        ctx = zmq.Context()
        try:
            sock = ctx.socket(zmq.DEALER)
            sock.connect(f"tcp://127.0.0.1:{master_port}")
            sock.send(msgspec.msgpack.encode(payload))
            assert sock.poll(timeout=5_000)
            sock.recv()

            stored = server.get_stage_coordinator_addresses(0, timeout_s=0.1)
            assert stored == StageCoordinatorAddresses(
                coordinator_input=payload["coordinator_input"],
                coordinator_output=payload["coordinator_output"],
                frontend_stats_publish_address=payload["frontend_stats_publish_address"],
            )
        finally:
            sock.close(linger=0)
            ctx.term()
            server.stop()

    def test_stop_joins_server_thread(self):
        from vllm.utils.network_utils import get_open_port

        master_port = get_open_port()
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=master_port,
            stage_ids=[],  # no stages → thread exits immediately
        )
        server.start()
        assert server._thread is not None
        server.stop()
        # Thread should have exited (joined with timeout=10 inside stop())
        assert not server._thread.is_alive()


# ---------------------------------------------------------------------------
# AsyncOmniEngine – single_stage_mode detection in __init__
# ---------------------------------------------------------------------------


class TestSingleStageModeDetection:
    """Test __init__ single_stage_mode / _single_stage_id_filter setup.

    We bypass the real __init__ by patching _resolve_stage_configs and
    the orchestrator thread, so no actual engines are started.
    """

    def _make_engine_no_thread(self, mocker: MockerFixture, **kwargs: Any) -> AsyncOmniEngine:
        """Create an AsyncOmniEngine without starting the orchestrator thread."""
        stage_cfg = _make_stage_cfg(0)
        mock_stage_configs = [stage_cfg]

        mocker.patch.object(
            AsyncOmniEngine,
            "_resolve_stage_configs",
            return_value=("/fake/path", mock_stage_configs),
        )
        mocker.patch.object(
            AsyncOmniEngine,
            "_bootstrap_orchestrator",
        )
        mock_thread_cls = mocker.patch("threading.Thread")
        mock_future_cls = mocker.patch("concurrent.futures.Future")

        mock_future = mocker.Mock()
        mock_future.result.return_value = mocker.Mock()  # simulates a loop
        mock_future_cls.return_value = mock_future

        mock_thread = mocker.Mock()
        mock_thread.is_alive.return_value = False
        mock_thread_cls.return_value = mock_thread

        engine = AsyncOmniEngine(model="fake-model", **kwargs)
        return engine

    def test_explicit_single_stage_mode_true(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
            single_stage_mode=True,
            omni_master_address="127.0.0.1",
            omni_master_port=20000,
        )
        assert engine.single_stage_mode is True

    def test_stage_id_kwarg_promotes_to_single_stage_mode(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
            stage_id=0,
            omni_master_address="127.0.0.1",
            omni_master_port=20001,
        )
        assert engine.single_stage_mode is True

    def test_stage_id_kwarg_sets_filter(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
            stage_id=1,
            omni_master_address="127.0.0.1",
            omni_master_port=20002,
        )
        assert engine._single_stage_id_filter == 1

    def test_no_stage_id_no_single_stage_mode(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
        )
        assert engine.single_stage_mode is False
        assert engine._single_stage_id_filter is None

    def test_single_stage_mode_without_stage_id_has_no_filter(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
            single_stage_mode=True,
            omni_master_address="127.0.0.1",
            omni_master_port=20003,
        )
        assert engine._single_stage_id_filter is None

    def test_master_address_and_port_stored(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
            stage_id=0,
            omni_master_address="10.0.0.1",
            omni_master_port=12345,
        )
        assert engine._omni_master_address == "10.0.0.1"
        assert engine._omni_master_port == 12345

    def test_omni_master_server_starts_as_none(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
        )
        assert engine._omni_master_server is None


# ---------------------------------------------------------------------------
# AsyncOmniEngine – _initialize_stages stage routing
# ---------------------------------------------------------------------------


class TestInitializeStagesRouting:
    """Verify that _initialize_stages routes each stage to the correct launch
    function depending on single_stage_mode and _single_stage_id_filter."""

    _COMMON_PATCHES = [
        "vllm_omni.engine.async_omni_engine.prepare_engine_environment",
        "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
        "vllm_omni.engine.async_omni_engine.get_stage_connector_spec",
        "vllm_omni.engine.async_omni_engine.resolve_omni_kv_config_for_stage",
        "vllm_omni.engine.async_omni_engine.extract_stage_metadata",
        "vllm_omni.engine.async_omni_engine.finalize_initialized_stages",
    ]

    def _build_engine_skeleton(
        self,
        stage_cfgs: list[Any],
        single_stage_mode: bool,
        stage_id_filter: int | None,
        omni_master_address: str = "127.0.0.1",
        omni_master_port: int = 25000,
    ) -> AsyncOmniEngine:
        """Build a bare AsyncOmniEngine without launching any threads."""
        engine = object.__new__(AsyncOmniEngine)
        engine.model = "fake-model"
        engine.config_path = "/fake"
        engine.stage_configs = stage_cfgs
        engine.num_stages = len(stage_cfgs)
        engine.async_chunk = False
        engine.single_stage_mode = single_stage_mode
        engine._single_stage_id_filter = stage_id_filter
        engine._omni_master_address = omni_master_address
        engine._omni_master_port = omni_master_port
        engine._omni_master_server = None
        engine._llm_stage_launch_lock = __import__("threading").Lock()
        engine.diffusion_batch_size = 1
        engine.stage_clients = []
        engine.stage_vllm_configs = []
        engine.output_processors = []
        engine.input_processor = None
        engine.supported_tasks = ("generate",)
        engine.default_sampling_params_list = []
        engine.stage_metadata = []
        engine.prompt_expand_func = None
        return engine

    def _fake_metadata(self, mocker: MockerFixture, stage_id: int, stage_type: str = "llm") -> Any:
        meta = mocker.Mock()
        meta.stage_id = stage_id
        meta.stage_type = stage_type
        meta.runtime_cfg = {}
        meta.prompt_expand_func = None
        meta.engine_output_type = None
        meta.is_comprehension = False
        meta.final_output = True if stage_id == 0 else False
        meta.final_output_type = None
        return meta

    def _run_initialize_stages_mocked(
        self,
        mocker: MockerFixture,
        engine: AsyncOmniEngine,
        stage_cfgs: list[Any],
        *,
        launch_side_effect: Any = None,
        remote_side_effect: Any = None,
        attach_result: Any = None,
    ) -> tuple[Any, Any]:
        """Execute _initialize_stages with all heavy helpers mocked.

        Returns (mock_launch_llm_stage, mock_create_remote_llm_stage).
        """
        started_by_stage: dict[int, StartedLlmStage] = {
            cfg.stage_id: _make_started_llm_stage(cfg.stage_id)
            for cfg in stage_cfgs
            if getattr(cfg, "stage_type", "llm") != "diffusion"
        }

        default_attach = (mocker.Mock(), mocker.Mock(), mocker.Mock(), mocker.Mock())

        mock_launch = mocker.Mock(
            side_effect=launch_side_effect
            or (lambda cfg, meta, spec, timeout, llm_stage_launch_lock, kv: started_by_stage[meta.stage_id])
        )
        mock_remote = mocker.Mock(
            side_effect=remote_side_effect or (lambda cfg, meta, spec, timeout, srv: started_by_stage[meta.stage_id])
        )
        mock_attach = mocker.Mock(return_value=attach_result or default_attach)

        mock_oms = mocker.Mock(spec=OmniMasterServer)
        mock_oms.get_zmq_addresses.side_effect = lambda sid: mocker.Mock()

        finalized = (
            [mocker.Mock() for _ in stage_cfgs],
            [mocker.Mock() for _ in stage_cfgs],
            [{"final_output": True, "final_output_type": None, "stage_type": "llm"} for _ in stage_cfgs],
        )

        mocker.patch.object(engine, "_launch_llm_stage", mock_launch)
        mocker.patch.object(engine, "_create_remote_llm_stage", mock_remote)
        mocker.patch.object(engine, "_attach_llm_stage", mock_attach)
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.OmniMasterServer",
            return_value=mock_oms,
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.prepare_engine_environment",
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
            return_value=None,
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.get_stage_connector_spec",
            return_value={},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.resolve_omni_kv_config_for_stage",
            return_value=(None, None, None),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.extract_stage_metadata",
            side_effect=lambda cfg: self._fake_metadata(
                mocker,
                cfg.stage_id,
                getattr(cfg, "stage_type", "llm"),
            ),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.finalize_initialized_stages",
            return_value=finalized,
        )

        engine._initialize_stages(stage_init_timeout=60)

        return mock_launch, mock_remote

    # -- single-stage mode: stage matches filter → local launch ---------------

    def test_matching_stage_uses_launch_llm_stage(self, mocker: MockerFixture):
        """stage_id == _single_stage_id_filter → _launch_llm_stage is called."""
        stage_cfgs = [_make_stage_cfg(0), _make_stage_cfg(1)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=0)
        mock_launch, mock_remote = self._run_initialize_stages_mocked(mocker, engine, stage_cfgs)

        launched_ids = [c.args[1].stage_id for c in mock_launch.call_args_list]
        assert 0 in launched_ids, "_launch_llm_stage should be called for stage 0"

    def test_non_matching_stage_uses_create_remote_llm_stage(self, mocker: MockerFixture):
        """stage_id != _single_stage_id_filter → _create_remote_llm_stage is called."""
        stage_cfgs = [_make_stage_cfg(0), _make_stage_cfg(1)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=0)
        mock_launch, mock_remote = self._run_initialize_stages_mocked(mocker, engine, stage_cfgs)

        remote_ids = [c.args[1].stage_id for c in mock_remote.call_args_list]
        assert 1 in remote_ids, "_create_remote_llm_stage should be called for stage 1"

    def test_filter_1_routes_correctly(self, mocker: MockerFixture):
        """With filter=1, stage 0 is remote and stage 1 is local."""
        stage_cfgs = [_make_stage_cfg(0), _make_stage_cfg(1)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=1)
        mock_launch, mock_remote = self._run_initialize_stages_mocked(mocker, engine, stage_cfgs)

        launched_ids = [c.args[1].stage_id for c in mock_launch.call_args_list]
        remote_ids = [c.args[1].stage_id for c in mock_remote.call_args_list]
        assert 1 in launched_ids, "stage 1 should be launched locally with filter=1"
        assert 0 in remote_ids, "stage 0 should use remote path with filter=1"

    def test_no_filter_all_stages_use_launch_path(self, mocker: MockerFixture):
        """single_stage_mode=True but no filter → all stages use _launch_llm_stage."""
        stage_cfgs = [_make_stage_cfg(0), _make_stage_cfg(1)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=None)
        mock_launch, mock_remote = self._run_initialize_stages_mocked(mocker, engine, stage_cfgs)

        assert mock_remote.call_count == 0, "No remote launches without a filter"
        launched_ids = [c.args[1].stage_id for c in mock_launch.call_args_list]
        assert set(launched_ids) == {0, 1}

    def test_non_single_stage_mode_never_calls_create_remote(self, mocker: MockerFixture):
        """Outside single_stage_mode, _create_remote_llm_stage must not be called."""
        stage_cfgs = [_make_stage_cfg(0), _make_stage_cfg(1)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=False, stage_id_filter=None)
        mock_launch, mock_remote = self._run_initialize_stages_mocked(mocker, engine, stage_cfgs)

        assert mock_remote.call_count == 0

    def test_omni_master_server_started_in_single_stage_mode(self, mocker: MockerFixture):
        """OmniMasterServer.start() must be called when single_stage_mode=True."""
        stage_cfgs = [_make_stage_cfg(0)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=0)
        mock_oms = mocker.Mock(spec=OmniMasterServer)
        mock_oms.get_zmq_addresses.return_value = mocker.Mock()
        finalized = (
            [mocker.Mock()],
            [mocker.Mock()],
            [{"final_output": True, "final_output_type": None, "stage_type": "llm"}],
        )

        mocker.patch.object(engine, "_launch_llm_stage", return_value=_make_started_llm_stage(0))
        mocker.patch.object(engine, "_create_remote_llm_stage", return_value=_make_started_llm_stage(0))
        mocker.patch.object(
            engine,
            "_attach_llm_stage",
            return_value=(mocker.Mock(), mocker.Mock(), mocker.Mock(), mocker.Mock()),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.OmniMasterServer",
            return_value=mock_oms,
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.prepare_engine_environment")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
            return_value=None,
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.get_stage_connector_spec",
            return_value={},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.resolve_omni_kv_config_for_stage",
            return_value=(None, None, None),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.extract_stage_metadata",
            side_effect=lambda cfg: self._fake_metadata(mocker, cfg.stage_id),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.finalize_initialized_stages",
            return_value=finalized,
        )

        engine._initialize_stages(stage_init_timeout=60)

        mock_oms.start.assert_called_once()

    def test_omni_master_server_uses_configured_stage_ids(self, mocker: MockerFixture):
        """Configured stage IDs, not list indexes, should drive pre-allocation."""
        stage_cfgs = [_make_stage_cfg(7), _make_stage_cfg(11)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=7)
        mock_oms = mocker.Mock(spec=OmniMasterServer)
        mock_oms.get_zmq_addresses.return_value = mocker.Mock()
        finalized = (
            [mocker.Mock(), mocker.Mock()],
            [mocker.Mock(), mocker.Mock()],
            [{"final_output": False, "final_output_type": None, "stage_type": "llm"} for _ in stage_cfgs],
        )

        mocker.patch.object(
            engine,
            "_launch_llm_stage",
            side_effect=[_make_started_llm_stage(7), _make_started_llm_stage(11)],
        )
        mocker.patch.object(
            engine,
            "_create_remote_llm_stage",
            return_value=_make_started_llm_stage(11),
        )
        mocker.patch.object(
            engine,
            "_attach_llm_stage",
            return_value=(mocker.Mock(), mocker.Mock(), mocker.Mock(), mocker.Mock()),
        )
        mock_oms_cls = mocker.patch(
            "vllm_omni.engine.async_omni_engine.OmniMasterServer",
            return_value=mock_oms,
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.prepare_engine_environment")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
            return_value=None,
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.get_stage_connector_spec",
            return_value={},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.resolve_omni_kv_config_for_stage",
            return_value=(None, None, None),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.extract_stage_metadata",
            side_effect=lambda cfg: self._fake_metadata(mocker, cfg.stage_id),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.finalize_initialized_stages",
            return_value=finalized,
        )

        engine._initialize_stages(stage_init_timeout=60)

        mock_oms_cls.assert_called_once_with(
            master_address=engine._omni_master_address,
            master_port=engine._omni_master_port,
            stage_ids=[7, 11],
        )

    def test_single_stage_filter_uses_configured_stage_ids(self, mocker: MockerFixture):
        """Local/remote dispatch should compare against configured stage IDs."""
        stage_cfgs = [_make_stage_cfg(7), _make_stage_cfg(11)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=7)
        mock_oms = mocker.Mock(spec=OmniMasterServer)
        finalized = (
            [mocker.Mock(), mocker.Mock()],
            [mocker.Mock(), mocker.Mock()],
            [{"final_output": False, "final_output_type": None, "stage_type": "llm"} for _ in stage_cfgs],
        )

        mock_launch = mocker.patch.object(
            engine,
            "_launch_llm_stage",
            side_effect=[_make_started_llm_stage(7)],
        )
        mock_remote = mocker.patch.object(
            engine,
            "_create_remote_llm_stage",
            return_value=_make_started_llm_stage(11),
        )
        mocker.patch.object(
            engine,
            "_attach_llm_stage",
            return_value=(mocker.Mock(), mocker.Mock(), mocker.Mock(), mocker.Mock()),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.OmniMasterServer",
            return_value=mock_oms,
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.prepare_engine_environment")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
            return_value=None,
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.get_stage_connector_spec",
            return_value={},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.resolve_omni_kv_config_for_stage",
            return_value=(None, None, None),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.extract_stage_metadata",
            side_effect=lambda cfg: self._fake_metadata(mocker, cfg.stage_id),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.finalize_initialized_stages",
            return_value=finalized,
        )

        engine._initialize_stages(stage_init_timeout=60)

        assert [call.args[1].stage_id for call in mock_launch.call_args_list] == [7]
        assert [call.args[1].stage_id for call in mock_remote.call_args_list] == [11]

    def test_omni_master_server_preallocates_diffusion_stage_ids(self, mocker: MockerFixture):
        """Diffusion stages should also receive OmniMasterServer allocations."""
        stage_cfgs = [_make_stage_cfg(7), _make_stage_cfg(11, stage_type="diffusion")]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=7)
        mock_oms = mocker.Mock(spec=OmniMasterServer)
        finalized = (
            [mocker.Mock(), mocker.Mock()],
            [mocker.Mock(), mocker.Mock()],
            [
                {"final_output": False, "final_output_type": None, "stage_type": "llm"},
                {"final_output": False, "final_output_type": None, "stage_type": "diffusion"},
            ],
        )

        mocker.patch.object(engine, "_launch_llm_stage", return_value=_make_started_llm_stage(7))
        mocker.patch.object(engine, "_create_remote_llm_stage", return_value=_make_started_llm_stage(7))
        mocker.patch.object(engine, "_launch_diffusion_stage", return_value=mocker.Mock())
        mocker.patch.object(
            engine,
            "_create_remote_diffusion_stage",
            return_value=mocker.Mock(),
        )
        mocker.patch.object(
            engine,
            "_attach_llm_stage",
            return_value=(mocker.Mock(), mocker.Mock(), mocker.Mock(), mocker.Mock()),
        )
        mock_oms_cls = mocker.patch(
            "vllm_omni.engine.async_omni_engine.OmniMasterServer",
            return_value=mock_oms,
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.prepare_engine_environment")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
            return_value=None,
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.get_stage_connector_spec",
            return_value={},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.resolve_omni_kv_config_for_stage",
            return_value=(None, None, None),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.extract_stage_metadata",
            side_effect=lambda cfg: self._fake_metadata(
                mocker,
                cfg.stage_id,
                getattr(cfg, "stage_type", "llm"),
            ),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.finalize_initialized_stages",
            return_value=finalized,
        )

        engine._initialize_stages(stage_init_timeout=60)

        mock_oms_cls.assert_called_once_with(
            master_address=engine._omni_master_address,
            master_port=engine._omni_master_port,
            stage_ids=[7, 11],
        )

    def test_duplicate_llm_stage_ids_raise(self, mocker: MockerFixture):
        """Duplicate configured LLM stage IDs should fail fast."""
        stage_cfgs = [_make_stage_cfg(3), _make_stage_cfg(3)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=3)

        mocker.patch("vllm_omni.engine.async_omni_engine.prepare_engine_environment")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
            return_value=None,
        )
        with pytest.raises(ValueError, match="Duplicate stage_id"):
            engine._initialize_stages(stage_init_timeout=60)

    def test_omni_master_server_not_started_in_normal_mode(self, mocker: MockerFixture):
        """OmniMasterServer must NOT be instantiated outside single_stage_mode."""
        stage_cfgs = [_make_stage_cfg(0)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=False, stage_id_filter=None)
        finalized = (
            [mocker.Mock()],
            [mocker.Mock()],
            [{"final_output": True, "final_output_type": None, "stage_type": "llm"}],
        )

        mocker.patch.object(engine, "_launch_llm_stage", return_value=_make_started_llm_stage(0))
        mocker.patch.object(
            engine,
            "_attach_llm_stage",
            return_value=(mocker.Mock(), mocker.Mock(), mocker.Mock(), mocker.Mock()),
        )
        mock_oms_cls = mocker.patch("vllm_omni.engine.async_omni_engine.OmniMasterServer")
        mocker.patch("vllm_omni.engine.async_omni_engine.prepare_engine_environment")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
            return_value=None,
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.get_stage_connector_spec",
            return_value={},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.resolve_omni_kv_config_for_stage",
            return_value=(None, None, None),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.extract_stage_metadata",
            side_effect=lambda cfg: self._fake_metadata(mocker, cfg.stage_id),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.finalize_initialized_stages",
            return_value=finalized,
        )

        engine._initialize_stages(stage_init_timeout=60)

        mock_oms_cls.assert_not_called()

    def test_single_stage_mode_missing_master_address_raises(self, mocker: MockerFixture):
        """single_stage_mode without master address/port raises ValueError."""
        stage_cfgs = [_make_stage_cfg(0)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=0)
        engine._omni_master_address = None  # missing
        engine._omni_master_port = None

        mocker.patch("vllm_omni.engine.async_omni_engine.prepare_engine_environment")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
            return_value=None,
        )
        with pytest.raises(ValueError, match="omni_master_address"):
            engine._initialize_stages(stage_init_timeout=60)

    def test_matching_diffusion_stage_uses_local_registered_launch(self, mocker: MockerFixture):
        """A local diffusion stage should use the registered single-stage launch path."""
        stage_cfgs = [_make_stage_cfg(0, stage_type="diffusion"), _make_stage_cfg(1)]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=0)
        mock_oms = mocker.Mock(spec=OmniMasterServer)
        diffusion_client = mocker.Mock(stage_type="diffusion")
        finalized = (
            [diffusion_client, mocker.Mock()],
            [mocker.Mock(), mocker.Mock()],
            [
                {"final_output": False, "final_output_type": None, "stage_type": "diffusion"},
                {"final_output": False, "final_output_type": None, "stage_type": "llm"},
            ],
        )

        mock_local_diff = mocker.patch.object(
            engine,
            "_launch_diffusion_stage",
            return_value=diffusion_client,
        )
        mock_remote_diff = mocker.patch.object(engine, "_create_remote_diffusion_stage")
        mocker.patch.object(engine, "_launch_llm_stage", return_value=_make_started_llm_stage(1))
        mocker.patch.object(engine, "_create_remote_llm_stage", return_value=_make_started_llm_stage(1))
        mocker.patch.object(
            engine,
            "_attach_llm_stage",
            return_value=(mocker.Mock(), mocker.Mock(), mocker.Mock(), mocker.Mock()),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.OmniMasterServer",
            return_value=mock_oms,
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.prepare_engine_environment")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
            return_value=None,
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.get_stage_connector_spec",
            return_value={},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.resolve_omni_kv_config_for_stage",
            return_value=(None, None, None),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.extract_stage_metadata",
            side_effect=lambda cfg: self._fake_metadata(
                mocker,
                cfg.stage_id,
                getattr(cfg, "stage_type", "llm"),
            ),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.finalize_initialized_stages",
            return_value=finalized,
        )

        engine._initialize_stages(stage_init_timeout=60)

        assert mock_local_diff.call_count == 1
        assert mock_local_diff.call_args.args[1].stage_id == 0
        mock_remote_diff.assert_not_called()

    def test_non_matching_diffusion_stage_uses_remote_diffusion_client(self, mocker: MockerFixture):
        """A non-local diffusion stage should attach via the remote diffusion path."""
        stage_cfgs = [_make_stage_cfg(0), _make_stage_cfg(1, stage_type="diffusion")]
        engine = self._build_engine_skeleton(stage_cfgs, single_stage_mode=True, stage_id_filter=0)
        mock_oms = mocker.Mock(spec=OmniMasterServer)
        remote_diffusion_client = mocker.Mock(stage_type="diffusion")
        finalized = (
            [mocker.Mock(), remote_diffusion_client],
            [mocker.Mock(), mocker.Mock()],
            [
                {"final_output": False, "final_output_type": None, "stage_type": "llm"},
                {"final_output": False, "final_output_type": None, "stage_type": "diffusion"},
            ],
        )

        mock_local_diff = mocker.patch.object(engine, "_launch_diffusion_stage")
        mock_remote_diff = mocker.patch.object(
            engine,
            "_create_remote_diffusion_stage",
            return_value=remote_diffusion_client,
        )
        mocker.patch.object(engine, "_launch_llm_stage", return_value=_make_started_llm_stage(0))
        mocker.patch.object(engine, "_create_remote_llm_stage", return_value=_make_started_llm_stage(0))
        mocker.patch.object(
            engine,
            "_attach_llm_stage",
            return_value=(mocker.Mock(), mocker.Mock(), mocker.Mock(), mocker.Mock()),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.OmniMasterServer",
            return_value=mock_oms,
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.prepare_engine_environment")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.load_omni_transfer_config_for_model",
            return_value=None,
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.get_stage_connector_spec",
            return_value={},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.resolve_omni_kv_config_for_stage",
            return_value=(None, None, None),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.extract_stage_metadata",
            side_effect=lambda cfg: self._fake_metadata(
                mocker,
                cfg.stage_id,
                getattr(cfg, "stage_type", "llm"),
            ),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.finalize_initialized_stages",
            return_value=finalized,
        )

        engine._initialize_stages(stage_init_timeout=60)

        mock_local_diff.assert_not_called()
        assert mock_remote_diff.call_count == 1
        assert mock_remote_diff.call_args.args[0].stage_id == 1


# ---------------------------------------------------------------------------
# AsyncOmniEngine – _launch_diffusion_stage
# ---------------------------------------------------------------------------


class TestLaunchDiffusionStage:
    """Test local diffusion stage launch wiring."""

    def test_registers_stage_with_public_master_properties(self, mocker: MockerFixture):
        engine = object.__new__(AsyncOmniEngine)
        engine.model = "fake-model"
        engine.diffusion_batch_size = 4

        stage_cfg = _make_stage_cfg(5, stage_type="diffusion")
        metadata = mocker.Mock(stage_id=5)
        omni_master_server = mocker.Mock(spec=OmniMasterServer)
        omni_master_server.address = "127.0.0.1"
        omni_master_server.port = 25000

        proc = mocker.Mock()
        diffusion_client = mocker.Mock()

        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_diffusion_config",
            return_value="diffusion-config",
        )
        mock_register = mocker.patch(
            "vllm_omni.engine.async_omni_engine.register_stage_with_omni_master",
            return_value=(
                "tcp://127.0.0.1:25001",
                "tcp://127.0.0.1:25002",
                "tcp://127.0.0.1:25003",
            ),
        )
        mock_spawn = mocker.patch(
            "vllm_omni.engine.async_omni_engine.spawn_diffusion_proc",
            return_value=(proc, None, None, None),
        )
        mock_handshake = mocker.patch("vllm_omni.engine.async_omni_engine.complete_diffusion_handshake")
        mock_from_addresses = mocker.patch(
            "vllm_omni.engine.async_omni_engine.StageDiffusionClient.from_addresses",
            return_value=diffusion_client,
        )

        result = engine._launch_diffusion_stage(
            stage_cfg=stage_cfg,
            metadata=metadata,
            omni_master_server=omni_master_server,
        )

        mock_register.assert_called_once_with(
            omni_master_address="127.0.0.1",
            omni_master_port=25000,
            omni_stage_id=5,
            omni_stage_config=stage_cfg,
            return_addresses=True,
        )
        mock_spawn.assert_called_once_with(
            "fake-model",
            "diffusion-config",
            handshake_address="tcp://127.0.0.1:25001",
            request_address="tcp://127.0.0.1:25002",
            response_address="tcp://127.0.0.1:25003",
        )
        mock_handshake.assert_called_once_with(proc, "tcp://127.0.0.1:25001")
        mock_from_addresses.assert_called_once_with(
            metadata,
            request_address="tcp://127.0.0.1:25002",
            response_address="tcp://127.0.0.1:25003",
            proc=proc,
            batch_size=4,
        )
        assert result is diffusion_client


# ---------------------------------------------------------------------------
# AsyncOmniEngine – _create_remote_llm_stage
# ---------------------------------------------------------------------------


class TestCreateRemoteLlmStage:
    """Test _create_remote_llm_stage delegates correctly."""

    def _engine(self, mocker: MockerFixture) -> AsyncOmniEngine:
        engine = object.__new__(AsyncOmniEngine)
        engine.model = "fake-model"
        engine.single_stage_mode = True
        engine._single_stage_id_filter = 0
        engine._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        engine._omni_master_server.get_zmq_addresses.return_value = mocker.Mock()
        engine._omni_master_server.get_allocation.return_value = mocker.Mock()
        engine._omni_master_server.get_stage_config.return_value = {
            "stage_id": 0,
            "stage_type": "llm",
            "engine_args": {},
        }
        return engine

    def _mock_build_and_connect(self, mocker: MockerFixture, stage_id: int):
        fake_vllm_config = mocker.Mock()
        fake_executor_cls = mocker.Mock()
        fake_addresses = mocker.Mock()
        fake_addresses.inputs = ["tcp://127.0.0.1:5000"]
        fake_addresses.outputs = ["tcp://127.0.0.1:5001"]
        fake_addresses.frontend_stats_publish_address = None

        eng_mgr = mocker.Mock()
        coordinator = mocker.Mock()

        @contextmanager
        def fake_connect_cm(*args, **kwargs):
            yield eng_mgr, coordinator, fake_addresses

        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_engine_args_dict",
            return_value={"model": "fake", "stage_id": stage_id},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_vllm_config",
            return_value=(fake_vllm_config, fake_executor_cls),
        )
        mock_connect = mocker.patch(
            "vllm_omni.engine.async_omni_engine.connect_remote_engine_cores",
            return_value=fake_connect_cm(),
        )

        return mock_connect, fake_vllm_config, fake_executor_cls, fake_addresses

    def test_returns_started_llm_stage_with_correct_stage_id(self, mocker: MockerFixture):
        engine = self._engine(mocker)
        stage_cfg = _make_stage_cfg(1)
        metadata = mocker.Mock(stage_id=1)
        omni_ms = engine._omni_master_server
        omni_ms.get_stage_config.return_value = {
            "stage_id": 1,
            "stage_type": "llm",
            "engine_args": {},
        }

        self._mock_build_and_connect(mocker, 1)
        result = engine._create_remote_llm_stage(
            stage_cfg=stage_cfg,
            metadata=metadata,
            stage_connector_spec={},
            stage_init_timeout=60,
            omni_master_server=omni_ms,
        )
        assert isinstance(result, StartedLlmStage)
        assert result.stage_id == 1

    def test_connect_remote_engine_cores_called_with_stage_id(self, mocker: MockerFixture):
        engine = self._engine(mocker)
        stage_cfg = _make_stage_cfg(2)
        metadata = mocker.Mock(stage_id=2)
        omni_ms = engine._omni_master_server
        omni_ms.get_zmq_addresses.return_value = mocker.Mock(inputs=["x"], outputs=["y"])
        omni_ms.get_stage_config.return_value = {
            "stage_id": 2,
            "stage_type": "llm",
            "engine_args": {},
        }

        fake_vllm_config = mocker.Mock()
        fake_executor_cls = mocker.Mock()
        fake_addresses = mocker.Mock()
        fake_addresses.inputs = ["tcp://127.0.0.1:5000"]
        fake_addresses.outputs = ["tcp://127.0.0.1:5001"]
        fake_addresses.frontend_stats_publish_address = None

        @contextmanager
        def fake_connect_cm(*args, **kwargs):
            yield mocker.Mock(), mocker.Mock(), fake_addresses

        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_engine_args_dict",
            return_value={"model": "fake", "stage_id": 2},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_vllm_config",
            return_value=(fake_vllm_config, fake_executor_cls),
        )
        mock_connect = mocker.patch(
            "vllm_omni.engine.async_omni_engine.connect_remote_engine_cores",
            return_value=fake_connect_cm(),
        )

        engine._create_remote_llm_stage(
            stage_cfg=stage_cfg,
            metadata=metadata,
            stage_connector_spec={},
            stage_init_timeout=60,
            omni_master_server=omni_ms,
        )

        mock_connect.assert_called_once()
        _, kwargs = mock_connect.call_args
        assert kwargs.get("stage_id") == 2 or mock_connect.call_args.args[-1] == 2
        omni_ms.get_stage_config.assert_called_once_with(2, timeout_s=60)

    def test_missing_registered_stage_config_raises_value_error(self, mocker: MockerFixture):
        engine = self._engine(mocker)
        stage_cfg = _make_stage_cfg(3)
        metadata = mocker.Mock(stage_id=3)
        omni_ms = engine._omni_master_server
        omni_ms.get_stage_config.return_value = None

        mock_build_args = mocker.patch("vllm_omni.engine.async_omni_engine.build_engine_args_dict")
        with pytest.raises(
            ValueError,
            match="Remote stage 3 registered without stage config",
        ):
            engine._create_remote_llm_stage(
                stage_cfg=stage_cfg,
                metadata=metadata,
                stage_connector_spec={},
                stage_init_timeout=60,
                omni_master_server=omni_ms,
            )

        mock_build_args.assert_not_called()

    def test_exception_during_connect_closes_started_stage(self, mocker: MockerFixture):
        """If an error occurs after StartedLlmStage creation, close_started_llm_stage is called."""
        engine = self._engine(mocker)
        stage_cfg = _make_stage_cfg(1)
        metadata = mocker.Mock(stage_id=1)
        omni_ms = engine._omni_master_server
        omni_ms.get_stage_config.return_value = {
            "stage_id": 1,
            "stage_type": "llm",
            "engine_args": {},
        }

        @contextmanager
        def boom(*args, **kwargs):
            yield mocker.Mock(), mocker.Mock(), mocker.Mock()
            raise RuntimeError("handshake failed")

        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_engine_args_dict",
            return_value={"model": "fake", "stage_id": 1},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_vllm_config",
            return_value=(mocker.Mock(), mocker.Mock()),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.connect_remote_engine_cores",
            return_value=boom(),
        )
        mock_close = mocker.patch("vllm_omni.engine.async_omni_engine.close_started_llm_stage")
        with pytest.raises(RuntimeError, match="handshake failed"):
            engine._create_remote_llm_stage(
                stage_cfg=stage_cfg,
                metadata=metadata,
                stage_connector_spec={},
                stage_init_timeout=60,
                omni_master_server=omni_ms,
            )
        mock_close.assert_called_once()


class TestConnectRemoteEngineCoresCoordinator:
    """Test coordinator launch parity with launch_core_engines."""

    @staticmethod
    def _build_vllm_config(
        mocker: MockerFixture, *, dp_rank: int = 0, offline_mode: bool = False, needs_dp_coordinator: bool = True
    ) -> Any:
        parallel_config = mocker.Mock()
        parallel_config.data_parallel_size_local = 1
        parallel_config.data_parallel_size = 2
        parallel_config.data_parallel_rank = dp_rank
        parallel_config.data_parallel_rank_local = 0 if offline_mode else None

        vllm_config = mocker.Mock()
        vllm_config.parallel_config = parallel_config
        vllm_config.needs_dp_coordinator = needs_dp_coordinator
        vllm_config.model_config = mocker.Mock(is_moe=False)
        return vllm_config

    def test_uses_registered_coordinator_addresses(self, mocker: MockerFixture):
        vllm_config = self._build_vllm_config(mocker, dp_rank=0, offline_mode=False, needs_dp_coordinator=True)

        omni_master_server = mocker.Mock(spec=OmniMasterServer)
        omni_master_server.get_zmq_addresses.return_value = EngineZmqAddresses(
            inputs=["tcp://client-in"], outputs=["tcp://client-out"]
        )
        omni_master_server.get_allocation.return_value = mocker.Mock(handshake_bind_address="tcp://127.0.0.1:26001")
        omni_master_server.get_stage_coordinator_addresses.return_value = StageCoordinatorAddresses(
            coordinator_input="tcp://coord-in",
            coordinator_output="tcp://coord-out",
            frontend_stats_publish_address="tcp://stats",
        )

        @contextmanager
        def fake_socket_ctx(*args, **kwargs):
            yield mocker.Mock()

        mocker.patch(
            "vllm_omni.engine.stage_engine_startup.zmq_socket_ctx",
            return_value=fake_socket_ctx(),
        )
        mock_wait = mocker.patch("vllm_omni.engine.stage_engine_startup._wait_for_omni_engine_startup")
        with connect_remote_engine_cores(
            vllm_config=vllm_config,
            omni_master_server=omni_master_server,
            stage_id=7,
        ) as (_, yielded_coordinator, yielded_addresses):
            assert yielded_coordinator is None
            assert yielded_addresses.coordinator_input == "tcp://coord-in"
            assert yielded_addresses.coordinator_output == "tcp://coord-out"
            assert yielded_addresses.frontend_stats_publish_address == "tcp://stats"

        omni_master_server.get_stage_coordinator_addresses.assert_called_once_with(7)
        mock_wait.assert_called_once()

    def test_defaults_to_no_coordinator_addresses_when_none_registered(self, mocker: MockerFixture):
        vllm_config = self._build_vllm_config(
            mocker,
            dp_rank=0,
            offline_mode=False,
            needs_dp_coordinator=True,
        )

        omni_master_server = mocker.Mock(spec=OmniMasterServer)
        omni_master_server.get_zmq_addresses.return_value = EngineZmqAddresses(
            inputs=["tcp://client-in"], outputs=["tcp://client-out"]
        )
        omni_master_server.get_allocation.return_value = mocker.Mock(handshake_bind_address="tcp://127.0.0.1:26001")
        omni_master_server.get_stage_coordinator_addresses.return_value = StageCoordinatorAddresses()

        @contextmanager
        def fake_socket_ctx(*args, **kwargs):
            yield mocker.Mock()

        mocker.patch(
            "vllm_omni.engine.stage_engine_startup.zmq_socket_ctx",
            return_value=fake_socket_ctx(),
        )
        mocker.patch("vllm_omni.engine.stage_engine_startup._wait_for_omni_engine_startup")
        with connect_remote_engine_cores(
            vllm_config=vllm_config,
            omni_master_server=omni_master_server,
            stage_id=7,
        ) as (_, yielded_coordinator, yielded_addresses):
            assert yielded_coordinator is None
            assert yielded_addresses.coordinator_input is None
            assert yielded_addresses.coordinator_output is None
            assert yielded_addresses.frontend_stats_publish_address is None


class TestLaunchOmniCoreEngines:
    """Tests for local omni engine launch wiring."""

    def test_registers_stage_once_and_reuses_handshake_for_all_local_engines(self, mocker: MockerFixture):
        parallel_config = mocker.Mock(
            data_parallel_size_local=2,
            data_parallel_size=4,
            data_parallel_rank=3,
        )
        vllm_config = mocker.Mock(parallel_config=parallel_config)

        omni_master_server = mocker.Mock(spec=OmniMasterServer)
        omni_master_server.address = "127.0.0.1"
        omni_master_server.port = 26000
        omni_master_server.get_allocation.return_value = mocker.Mock(handshake_bind_address="tcp://127.0.0.1:26001")

        stage_config = {"stage_id": 7, "stage_type": "llm"}
        local_engine_manager = mocker.Mock()

        @contextmanager
        def fake_socket_ctx(*args, **kwargs):
            yield mocker.Mock()

        mock_register = mocker.patch(
            "vllm_omni.engine.stage_engine_startup.register_stage_with_omni_master",
            return_value="tcp://127.0.0.1:26001",
        )
        mocker.patch(
            "vllm_omni.engine.stage_engine_startup.zmq_socket_ctx",
            return_value=fake_socket_ctx(),
        )
        mock_manager_cls = mocker.patch(
            "vllm_omni.engine.stage_engine_startup.CoreEngineProcManager",
            return_value=local_engine_manager,
        )
        mocker.patch("vllm_omni.engine.stage_engine_startup.wait_for_engine_startup")
        with launch_omni_core_engines(
            vllm_config=vllm_config,
            executor_class=mocker.Mock(),
            log_stats=False,
            omni_master_server=omni_master_server,
            stage_id=7,
            stage_config=stage_config,
        ) as (yielded_manager, yielded_coordinator, yielded_addresses):
            assert yielded_manager is local_engine_manager
            assert yielded_coordinator is None

        mock_register.assert_called_once_with(
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
            omni_stage_id=7,
            omni_stage_config=stage_config,
            coordinator=None,
        )
        mock_manager_cls.assert_called_once()
        manager_kwargs = mock_manager_cls.call_args.kwargs
        assert manager_kwargs["local_engine_count"] == 2
        assert manager_kwargs["start_index"] == 3
        assert manager_kwargs["local_start_index"] == 0
        assert manager_kwargs["vllm_config"] is vllm_config
        assert manager_kwargs["local_client"] is True
        assert manager_kwargs["handshake_address"] == "tcp://127.0.0.1:26001"
        assert manager_kwargs["executor_class"] is not None

    def test_registers_stage_with_coordinator_when_started(self, mocker: MockerFixture):
        parallel_config = mocker.Mock(
            data_parallel_size_local=1,
            data_parallel_size=2,
            data_parallel_rank=0,
        )
        vllm_config = mocker.Mock(parallel_config=parallel_config)
        vllm_config.needs_dp_coordinator = True
        vllm_config.model_config = mocker.Mock(is_moe=False)

        omni_master_server = mocker.Mock(spec=OmniMasterServer)
        omni_master_server.address = "127.0.0.1"
        omni_master_server.port = 26000
        omni_master_server.get_zmq_addresses.return_value = EngineZmqAddresses(
            inputs=["tcp://client-in"], outputs=["tcp://client-out"]
        )
        omni_master_server.get_allocation.return_value = mocker.Mock(handshake_bind_address="tcp://127.0.0.1:26001")

        coordinator = mocker.Mock()
        coordinator.proc.pid = 1234
        coordinator.get_engine_socket_addresses.return_value = ("tcp://coord-in", "tcp://coord-out")
        coordinator.get_stats_publish_address.return_value = "tcp://stats"

        @contextmanager
        def fake_socket_ctx(*args, **kwargs):
            yield mocker.Mock()

        mocker.patch("vllm_omni.engine.stage_engine_startup.DPCoordinator", return_value=coordinator)
        mock_register = mocker.patch(
            "vllm_omni.engine.stage_engine_startup.register_stage_with_omni_master",
            return_value="tcp://127.0.0.1:26001",
        )
        mocker.patch(
            "vllm_omni.engine.stage_engine_startup.zmq_socket_ctx",
            return_value=fake_socket_ctx(),
        )
        mock_manager_cls = mocker.patch(
            "vllm_omni.engine.stage_engine_startup.CoreEngineProcManager",
            return_value=mocker.Mock(),
        )
        mock_wait = mocker.patch("vllm_omni.engine.stage_engine_startup.wait_for_engine_startup")
        with launch_omni_core_engines(
            vllm_config=vllm_config,
            executor_class=mocker.Mock(),
            log_stats=False,
            omni_master_server=omni_master_server,
            stage_id=7,
            stage_config={"stage_id": 7},
        ):
            pass

        mock_register.assert_called_once_with(
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
            omni_stage_id=7,
            omni_stage_config={"stage_id": 7},
            coordinator=coordinator,
        )
        manager_kwargs = mock_manager_cls.call_args.kwargs
        assert manager_kwargs["log_stats"] is False
        mock_wait.assert_called_once()


# ---------------------------------------------------------------------------
# AsyncOmniEngine – _launch_llm_stage single_stage_mode codepath
# ---------------------------------------------------------------------------


class TestLaunchLlmStageSingleStageMode:
    """Test that _launch_llm_stage selects launch_omni_core_engines when
    single_stage_mode=True and _omni_master_server is set."""

    def _build_engine_with_oms(self, mocker: MockerFixture) -> AsyncOmniEngine:
        engine = object.__new__(AsyncOmniEngine)
        engine.model = "fake-model"
        engine.single_stage_mode = True
        engine._single_stage_id_filter = 0
        engine._llm_stage_launch_lock = threading.Lock()
        mock_oms = mocker.Mock(spec=OmniMasterServer)
        mock_oms.address = "127.0.0.1"
        mock_oms.port = 25000
        alloc = mocker.Mock()
        alloc.handshake_bind_address = "tcp://127.0.0.1:25001"
        mock_oms.get_allocation.return_value = alloc
        fake_addresses = mocker.Mock()
        fake_addresses.inputs = ["tcp://127.0.0.1:5000"]
        fake_addresses.outputs = ["tcp://127.0.0.1:5001"]
        fake_addresses.frontend_stats_publish_address = None
        mock_oms.get_zmq_addresses.return_value = fake_addresses
        engine._omni_master_server = mock_oms
        return engine

    def _mock_launch_omni(self, mocker: MockerFixture, stage_id: int):
        fake_vllm_config = mocker.Mock()
        fake_executor_cls = mocker.Mock()
        fake_addresses = mocker.Mock()
        fake_addresses.inputs = ["tcp://127.0.0.1:5000"]
        fake_addresses.outputs = ["tcp://127.0.0.1:5001"]
        fake_addresses.frontend_stats_publish_address = None

        eng_mgr = mocker.Mock()

        @contextmanager
        def fake_launch_omni(*args, **kwargs):
            yield eng_mgr, None, fake_addresses

        mocker.patch("vllm_omni.engine.async_omni_engine.setup_stage_devices")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_engine_args_dict",
            return_value={"model": "fake", "stage_id": stage_id},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_vllm_config",
            return_value=(fake_vllm_config, fake_executor_cls),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.acquire_device_locks",
            return_value=[],
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.release_device_locks")
        return mocker.patch(
            "vllm_omni.engine.async_omni_engine.launch_omni_core_engines",
            return_value=fake_launch_omni(),
        )

    def test_launch_omni_core_engines_used_in_single_stage_mode(self, mocker: MockerFixture):
        """single_stage_mode + _omni_master_server → launch_omni_core_engines."""
        engine = self._build_engine_with_oms(mocker)
        metadata = mocker.Mock(stage_id=0, runtime_cfg={})
        stage_cfg = _make_stage_cfg(0)

        mock_launch_omni = self._mock_launch_omni(mocker, 0)
        result = engine._launch_llm_stage(
            stage_cfg=stage_cfg,
            metadata=metadata,
            stage_connector_spec={},
            stage_init_timeout=60,
            llm_stage_launch_lock=threading.Lock(),
        )

        mock_launch_omni.assert_called_once()
        assert mock_launch_omni.call_args.kwargs["stage_config"] is stage_cfg
        assert isinstance(result, StartedLlmStage)
        assert result.stage_id == 0

    def test_spawn_stage_core_used_in_normal_mode(self, mocker: MockerFixture):
        """~single_stage_mode → spawn_stage_core + complete_stage_handshake."""
        engine = object.__new__(AsyncOmniEngine)
        engine.model = "fake-model"
        engine.single_stage_mode = False
        engine._omni_master_server = None
        engine._llm_stage_launch_lock = threading.Lock()

        fake_vllm_config = mocker.Mock()
        fake_executor_cls = mocker.Mock()
        fake_addresses = mocker.Mock()
        fake_addresses.inputs = ["tcp://127.0.0.1:5000"]
        fake_addresses.outputs = ["tcp://127.0.0.1:5001"]
        fake_addresses.frontend_stats_publish_address = None

        fake_proc = mocker.Mock()
        fake_handshake_address = "ipc:///tmp/fake-handshake"
        stage_init_timeout = 60

        mocker.patch("vllm_omni.engine.async_omni_engine.setup_stage_devices")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_engine_args_dict",
            return_value={"model": "fake", "stage_id": 0},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_vllm_config",
            return_value=(fake_vllm_config, fake_executor_cls),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.acquire_device_locks",
            return_value=[],
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.release_device_locks")
        mock_spawn = mocker.patch(
            "vllm_omni.engine.async_omni_engine.spawn_stage_core",
            return_value=(fake_addresses, fake_proc, fake_handshake_address),
        )
        mock_handshake = mocker.patch("vllm_omni.engine.async_omni_engine.complete_stage_handshake")
        mock_omni = mocker.patch("vllm_omni.engine.async_omni_engine.launch_omni_core_engines")
        metadata = mocker.Mock(stage_id=0, runtime_cfg={})
        result = engine._launch_llm_stage(
            stage_cfg=_make_stage_cfg(0),
            metadata=metadata,
            stage_connector_spec={},
            stage_init_timeout=stage_init_timeout,
            llm_stage_launch_lock=threading.Lock(),
        )

        mock_spawn.assert_called_once_with(
            vllm_config=fake_vllm_config,
            executor_class=fake_executor_cls,
            log_stats=False,
        )
        mock_handshake.assert_called_once_with(
            fake_proc,
            fake_handshake_address,
            fake_addresses,
            fake_vllm_config,
            stage_init_timeout,
        )
        mock_omni.assert_not_called()
        assert isinstance(result, StartedLlmStage)
        assert result.proc is fake_proc

    def test_launch_omni_passes_stage_id_and_master_server(self, mocker: MockerFixture):
        """launch_omni_core_engines receives the correct stage_id and omni_master_server."""
        engine = self._build_engine_with_oms(mocker)
        metadata = mocker.Mock(stage_id=0, runtime_cfg={})

        captured_kwargs: dict[str, Any] = {}

        @contextmanager
        def capturing_launch(*args, **kwargs):
            captured_kwargs.update(kwargs)
            fake_addresses = mocker.Mock()
            fake_addresses.inputs = ["tcp://127.0.0.1:5000"]
            fake_addresses.outputs = ["tcp://127.0.0.1:5001"]
            fake_addresses.frontend_stats_publish_address = None
            yield mocker.Mock(), None, fake_addresses

        mocker.patch("vllm_omni.engine.async_omni_engine.setup_stage_devices")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_engine_args_dict",
            return_value={"model": "fake", "stage_id": 0},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_vllm_config",
            return_value=(mocker.Mock(), mocker.Mock()),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.acquire_device_locks",
            return_value=[],
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.release_device_locks")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.launch_omni_core_engines",
            side_effect=capturing_launch,
        )

        engine._launch_llm_stage(
            stage_cfg=_make_stage_cfg(0),
            metadata=metadata,
            stage_connector_spec={},
            stage_init_timeout=60,
            llm_stage_launch_lock=threading.Lock(),
        )

        assert captured_kwargs.get("stage_id") == 0
        assert captured_kwargs.get("omni_master_server") is engine._omni_master_server

    def test_launch_omni_context_exits_before_stage_cleanup_on_error(self, mocker: MockerFixture):
        """Errors after entering the omni launch context still unwind it first."""
        engine = self._build_engine_with_oms(mocker)
        metadata = mocker.Mock(stage_id=0, runtime_cfg={})

        fake_addresses = mocker.Mock()
        fake_addresses.inputs = ["tcp://127.0.0.1:5000"]
        fake_addresses.outputs = ["tcp://127.0.0.1:5001"]
        fake_addresses.frontend_stats_publish_address = None

        events: list[str] = []

        @contextmanager
        def fake_launch_omni(*args, **kwargs):
            try:
                yield mocker.Mock(), None, fake_addresses
            finally:
                events.append("launch_exit")

        mocker.patch("vllm_omni.engine.async_omni_engine.setup_stage_devices")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_engine_args_dict",
            return_value={"model": "fake", "stage_id": 0},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_vllm_config",
            return_value=(mocker.Mock(), mocker.Mock()),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.acquire_device_locks",
            return_value=[],
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.release_device_locks")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.launch_omni_core_engines",
            return_value=fake_launch_omni(),
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.logger.info", side_effect=RuntimeError("boom"))
        mock_close_stage = mocker.patch(
            "vllm_omni.engine.async_omni_engine.close_started_llm_stage",
            side_effect=lambda _started: events.append("stage_close"),
        )
        with pytest.raises(RuntimeError, match="boom"):
            engine._launch_llm_stage(
                stage_cfg=_make_stage_cfg(0),
                metadata=metadata,
                stage_connector_spec={},
                stage_init_timeout=60,
                llm_stage_launch_lock=threading.Lock(),
            )

        mock_close_stage.assert_called_once()
        assert events == ["launch_exit", "stage_close"]

    def test_base_exception_propagates_without_started_stage_cleanup(self, mocker: MockerFixture):
        """BaseException subclasses should bypass the Exception cleanup path."""
        engine = self._build_engine_with_oms(mocker)
        metadata = mocker.Mock(stage_id=0, runtime_cfg={})

        fake_addresses = mocker.Mock()
        fake_addresses.inputs = ["tcp://127.0.0.1:5000"]
        fake_addresses.outputs = ["tcp://127.0.0.1:5001"]
        fake_addresses.frontend_stats_publish_address = None

        events: list[str] = []

        class FatalLaunchInterrupt(BaseException):
            pass

        @contextmanager
        def fake_launch_omni(*args, **kwargs):
            try:
                yield mocker.Mock(), None, fake_addresses
            finally:
                events.append("launch_exit")

        mocker.patch("vllm_omni.engine.async_omni_engine.setup_stage_devices")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_engine_args_dict",
            return_value={"model": "fake", "stage_id": 0},
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.build_vllm_config",
            return_value=(mocker.Mock(), mocker.Mock()),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.acquire_device_locks",
            return_value=[],
        )
        mocker.patch("vllm_omni.engine.async_omni_engine.release_device_locks")
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.launch_omni_core_engines",
            return_value=fake_launch_omni(),
        )
        mocker.patch(
            "vllm_omni.engine.async_omni_engine.logger.info",
            side_effect=FatalLaunchInterrupt("stop"),
        )
        mock_close_stage = mocker.patch("vllm_omni.engine.async_omni_engine.close_started_llm_stage")
        with pytest.raises(FatalLaunchInterrupt, match="stop"):
            engine._launch_llm_stage(
                stage_cfg=_make_stage_cfg(0),
                metadata=metadata,
                stage_connector_spec={},
                stage_init_timeout=60,
                llm_stage_launch_lock=threading.Lock(),
            )

        mock_close_stage.assert_not_called()
        assert events == ["launch_exit"]
