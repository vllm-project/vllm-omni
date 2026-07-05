"""Unit tests for AsyncOmniEngine single-stage mode and OmniMasterServer."""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_mock import MockerFixture
from vllm.v1.engine.utils import EngineZmqAddresses

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase
from vllm_omni.engine.stage_engine_startup import (
    OmniMasterServer,
    StageAllocation,
    StageCoordinatorAddresses,
    StageRegistrationResponse,
    StageReplicaResources,
    _launch_omni_core_engines,
    connect_remote_diffusion_proc,
    connect_remote_engine_cores,
)
from vllm_omni.engine.stage_init_utils import LogicalStageInitPlan, ReplicaInitPlan
from vllm_omni.engine.stage_runtime import DistStageRuntime, StageRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stage_cfg(stage_id: int, stage_type: str = "llm"):
    """Return a lightweight stage config mock."""
    return SimpleNamespace(
        stage_id=stage_id,
        stage_type=stage_type,
        runtime=SimpleNamespace(devices="0"),
        engine_args=SimpleNamespace(
            async_chunk=False,
            model_stage=None,
            engine_output_type=None,
        ),
    )


def _make_llm_plan(
    stage_idx: int,
    *,
    stage_id: int,
    launch_mode: str,
    vllm_config: Any | None = None,
) -> LogicalStageInitPlan:
    stage_cfg = _make_stage_cfg(stage_id)
    metadata = SimpleNamespace(
        stage_id=stage_id,
        stage_type="llm",
        runtime_cfg={"devices": "0"},
        prompt_expand_func=None,
        final_output=False,
        final_output_type=None,
        default_sampling_params=SimpleNamespace(),
        custom_process_input_func=None,
        engine_input_source=[],
        engine_output_type="token_ids",
        replica_id=0,
    )
    return LogicalStageInitPlan(
        stage_idx=stage_idx,
        stage_id=stage_id,
        replicas=[
            ReplicaInitPlan(
                replica_id=0,
                num_replicas=1,
                launch_mode=launch_mode,
                stage_cfg=stage_cfg,
                metadata=metadata,
                stage_connector_spec={},
                omni_kv_connector=(None, None, None),
                stage_vllm_config=vllm_config
                or SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size_local=1)),
                executor_class=object,
                engine_args_dict={},
            )
        ],
    )


def _make_diffusion_plan(
    stage_idx: int,
    *,
    stage_id: int,
    launch_mode: str,
) -> LogicalStageInitPlan:
    stage_cfg = _make_stage_cfg(stage_id, stage_type="diffusion")
    metadata = SimpleNamespace(
        stage_id=stage_id,
        stage_type="diffusion",
        runtime_cfg={"devices": "0"},
        prompt_expand_func=None,
        final_output=True,
        final_output_type="image",
        default_sampling_params=SimpleNamespace(),
        custom_process_input_func=None,
        engine_input_source=[],
        cfg_kv_collect_func=None,
        replica_id=0,
    )
    return LogicalStageInitPlan(
        stage_idx=stage_idx,
        stage_id=stage_id,
        replicas=[
            ReplicaInitPlan(
                replica_id=0,
                num_replicas=1,
                launch_mode=launch_mode,
                stage_cfg=stage_cfg,
                metadata=metadata,
                stage_connector_spec={},
                omni_kv_connector=(None, None, None),
            )
        ],
    )


# ---------------------------------------------------------------------------
# OmniMasterServer address pre-allocation
# ---------------------------------------------------------------------------


class TestOmniMasterServerAllocation:
    def test_public_address_and_port_properties_expose_registration_endpoint(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15000, stage_ids=[0])
        assert server.address == "127.0.0.1"
        assert server.port == 15000

    def test_allocations_created_for_each_stage_id(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15000, stage_ids=[0, 1, 2])
        assert set(server._stage_routes.keys()) == {(0, 0), (1, 0), (2, 0)}

    def test_each_allocation_is_stage_allocation(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15000, stage_ids=[0, 1])
        for sid in (0, 1):
            assert isinstance(server.get_allocation(sid), StageAllocation)

    def test_replica_allocations_are_distinct(self):
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15000,
            stage_ids=[0],
            stage_replica_counts={0: 3},
        )

        allocations = [server.get_allocation(0, replica_id) for replica_id in range(3)]
        assert len({alloc.handshake_bind_address for alloc in allocations}) == 3
        assert server.get_allocation(0) is allocations[0]

    def test_replica_stage_configs_are_isolated(self):
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15000,
            stage_ids=[0],
            stage_replica_counts={0: 2},
        )

        server.register_stage_config(0, {"replica": 0}, replica_id=0)
        server.register_stage_config(0, {"replica": 1}, replica_id=1)

        assert server.get_stage_config(0, replica_id=0) == {"replica": 0}
        assert server.get_stage_config(0, replica_id=1) == {"replica": 1}

    def test_allocation_addresses_reference_master_address(self):
        server = OmniMasterServer(master_address="192.168.1.10", master_port=20000, stage_ids=[0])
        alloc = server.get_allocation(0)
        for address in (
            alloc.handshake_bind_address,
            alloc.handshake_connect_address,
            alloc.input_bind_address,
            alloc.input_connect_address,
            alloc.output_bind_address,
            alloc.output_connect_address,
        ):
            assert "192.168.1.10" in address

    def test_port_uniqueness_within_single_allocation(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15001, stage_ids=[0])
        alloc = server.get_allocation(0)
        handshake_port = int(alloc.handshake_bind_address.split(":")[-1])
        input_port = int(alloc.input_bind_address.split(":")[-1])
        output_port = int(alloc.output_bind_address.split(":")[-1])
        assert len({handshake_port, input_port, output_port}) == 3

    def test_get_zmq_addresses_returns_bind_addresses(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15002, stage_ids=[0])
        alloc = server.get_allocation(0)
        zmq_addrs = server.get_zmq_addresses(0)
        assert zmq_addrs.inputs == [alloc.input_bind_address]
        assert zmq_addrs.outputs == [alloc.output_bind_address]

    def test_get_engine_zmq_addresses_returns_connect_addresses(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15003, stage_ids=[0])
        alloc = server.get_allocation(0)
        zmq_addrs = server.get_engine_zmq_addresses(0)
        assert zmq_addrs.inputs == [alloc.input_connect_address]
        assert zmq_addrs.outputs == [alloc.output_connect_address]

    def test_get_allocation_returns_correct_object(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15004, stage_ids=[3])
        assert server.get_allocation(3) is server._stage_routes[(3, 0)]

    def test_next_free_replica_id_skips_head_local_slot_until_filled(self):
        # Head pre-allocates slot (0, 0) for its own register_stage_with_omni_master
        # call. A same-host headless that registers with auto-assign BEFORE the
        # head's own registration must NOT be handed slot 0 — it should land on
        # slot 1 instead. Without the head_local_replicas reservation,
        # _next_free_replica_id would see (0, 0) absent from _stage_configs and
        # return 0, colliding with the head's own bound sockets.
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15010,
            stage_ids=[0],
            head_local_replicas={0: [0]},
        )
        assert server._next_free_replica_id(0) == 1

    def test_next_free_replica_id_uses_remote_slot_when_unowned(self):
        # When the head pre-allocates a remote-only slot (the head's _initialize_*
        # path waits on get_stage_config), auto-assign SHOULD fill it so the
        # head's wait unblocks. This is the original behavior, preserved.
        server = OmniMasterServer(
            master_address="127.0.0.1",
            master_port=15011,
            stage_ids=[0],
        )
        assert server._next_free_replica_id(0) == 0


# ---------------------------------------------------------------------------
# OmniMasterServer registration flow
# ---------------------------------------------------------------------------


class TestOmniMasterServerRegistration:
    def test_registration_reply_contains_handshake_address(self):
        import msgspec
        import zmq
        from vllm.utils.network_utils import get_open_port

        master_port = get_open_port()
        server = OmniMasterServer(master_address="127.0.0.1", master_port=master_port, stage_ids=[0])
        server.start()
        expected_hs = server.get_allocation(0).handshake_connect_address

        ctx = zmq.Context()
        try:
            sock = ctx.socket(zmq.DEALER)
            sock.connect(f"tcp://127.0.0.1:{master_port}")
            sock.send(msgspec.msgpack.encode({"stage_id": 0}))
            assert sock.poll(timeout=5_000)
            reply = msgspec.msgpack.decode(sock.recv())
            assert reply["handshake_address"] == expected_hs
        finally:
            sock.close(linger=0)
            ctx.term()
            server.stop()

    def test_registration_stores_stage_config(self):
        import msgspec
        import zmq
        from vllm.utils.network_utils import get_open_port

        master_port = get_open_port()
        server = OmniMasterServer(master_address="127.0.0.1", master_port=master_port, stage_ids=[0])
        server.start()

        stage_config = {"stage_id": 0, "stage_type": "llm"}
        ctx = zmq.Context()
        try:
            sock = ctx.socket(zmq.DEALER)
            sock.connect(f"tcp://127.0.0.1:{master_port}")
            sock.send(msgspec.msgpack.encode({"stage_id": 0, "stage_config": stage_config}))
            assert sock.poll(timeout=5_000)
            sock.recv()
            assert server.get_stage_config(0) == stage_config
        finally:
            sock.close(linger=0)
            ctx.term()
            server.stop()

    def test_registration_stores_coordinator_addresses(self):
        import msgspec
        import zmq
        from vllm.utils.network_utils import get_open_port

        master_port = get_open_port()
        server = OmniMasterServer(master_address="127.0.0.1", master_port=master_port, stage_ids=[0])
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
            assert server.get_stage_coordinator_addresses(0) == StageCoordinatorAddresses(
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
        server = OmniMasterServer(master_address="127.0.0.1", master_port=master_port, stage_ids=[])
        server.start()

        assert server._thread is not None
        server.stop()
        assert not server._thread.is_alive()


# ---------------------------------------------------------------------------
# AsyncOmniEngine single_stage_mode detection in __init__
# ---------------------------------------------------------------------------


class TestSingleStageModeDetection:
    def _make_engine_no_thread(
        self,
        mocker: MockerFixture,
        *,
        stage_cfgs: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncOmniEngine:
        mock_stage_configs = stage_cfgs or [_make_stage_cfg(0)]

        mocker.patch.object(
            AsyncOmniEngine,
            "_resolve_stage_configs",
            return_value=("/fake/path", mock_stage_configs),
        )
        mocker.patch.object(AsyncOmniEngine, "_bootstrap_orchestrator")
        mock_thread_cls = mocker.patch("threading.Thread")
        mock_future_cls = mocker.patch("concurrent.futures.Future")

        mock_future = mocker.Mock()
        mock_future.result.return_value = mocker.Mock()
        mock_future_cls.return_value = mock_future

        mock_thread = mocker.Mock()
        mock_thread.is_alive.return_value = False
        mock_thread_cls.return_value = mock_thread

        return AsyncOmniEngine(model="fake-model", **kwargs)

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
        engine = self._make_engine_no_thread(mocker)
        assert engine.single_stage_mode is False
        assert engine._single_stage_id_filter is None

    def test_single_stage_mode_without_stage_id_has_no_filter(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
            single_stage_mode=True,
            omni_master_address="127.0.0.1",
            omni_master_port=20003,
        )
        assert engine.single_stage_mode is True
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
        engine = self._make_engine_no_thread(mocker)
        assert not hasattr(engine, "_omni_master_server")


# ---------------------------------------------------------------------------
# AsyncOmniEngine single-stage initialization paths
# ---------------------------------------------------------------------------


class TestSingleStageInitialization:
    def _build_runtime(self, stage_cfgs: list[Any], *, stage_id_filter: int | None) -> DistStageRuntime:
        return DistStageRuntime(
            stage_configs=stage_cfgs,
            model="fake-model",
            config_path="/fake/stages.yaml",
            stage_init_timeout=60,
            diffusion_batch_size=2,
            async_chunk=False,
            single_stage_id_filter=stage_id_filter,
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
        )

    def test_build_logical_stage_init_plans_marks_non_matching_stage_remote(self, mocker: MockerFixture):
        import vllm_omni.engine.stage_runtime as runtime_mod

        stage_cfgs = [_make_stage_cfg(0), _make_stage_cfg(1)]
        runtime = self._build_runtime(stage_cfgs, stage_id_filter=0)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            runtime_mod,
            "extract_stage_metadata",
            lambda cfg: SimpleNamespace(
                stage_id=cfg.stage_id,
                stage_type=getattr(cfg, "stage_type", "llm"),
                prompt_expand_func=None,
                runtime_cfg={},
            ),
        )
        monkeypatch.setattr(runtime_mod, "get_stage_connector_spec", lambda **_: {})
        monkeypatch.setattr(runtime_mod, "resolve_omni_kv_config_for_stage", lambda *_: (None, None, None))
        monkeypatch.setattr(runtime_mod, "build_engine_args_dict", lambda *_, **__: {})
        monkeypatch.setattr(runtime_mod, "build_vllm_config", lambda *_, **__: (SimpleNamespace(), object))
        try:
            stage_plans = runtime._build_logical_stage_init_plans(None, [1, 1], {})
        finally:
            monkeypatch.undo()

        assert [plan.replicas[0].launch_mode for plan in stage_plans] == ["local", "remote"]

    def test_build_logical_stage_init_plans_rejects_non_contiguous_stage_ids(self, mocker: MockerFixture):
        import vllm_omni.engine.stage_runtime as runtime_mod

        runtime = self._build_runtime([_make_stage_cfg(7), _make_stage_cfg(11)], stage_id_filter=7)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            runtime_mod,
            "extract_stage_metadata",
            lambda cfg: SimpleNamespace(
                stage_id=cfg.stage_id,
                stage_type=getattr(cfg, "stage_type", "llm"),
                prompt_expand_func=None,
                runtime_cfg={},
            ),
        )
        try:
            with pytest.raises(ValueError, match="stage_id must match its position"):
                runtime._build_logical_stage_init_plans(None, [1, 1], {})
        finally:
            monkeypatch.undo()

    def test_start_omni_master_server_uses_stage_ids(self, mocker: MockerFixture):
        import vllm_omni.engine.stage_runtime as runtime_mod
        from vllm_omni.distributed import omni_coordinator as omni_coord_mod

        runtime = self._build_runtime([], stage_id_filter=0)
        mock_oms = mocker.Mock(spec=OmniMasterServer)
        mocker.patch.object(runtime_mod, "OmniMasterServer", return_value=mock_oms)
        mocker.patch.object(
            omni_coord_mod,
            "OmniCoordinatorRuntime",
            return_value=mocker.Mock(router_address="tcp://127.0.0.1:9999"),
        )

        stage_plans = [
            _make_llm_plan(0, stage_id=0, launch_mode="local"),
            _make_diffusion_plan(1, stage_id=1, launch_mode="remote"),
        ]

        runtime._start_omni_master_server(stage_plans)

        call_kwargs = runtime_mod.OmniMasterServer.call_args.kwargs
        assert call_kwargs["master_address"] == "127.0.0.1"
        assert call_kwargs["master_port"] == 26000
        assert call_kwargs["stage_ids"] == [0, 1]
        assert call_kwargs["stage_replica_counts"] == {0: 1, 1: 1}
        # head_local_replicas reserves slots that the head will register
        # itself (launch_mode == "local"). Stage 1 is remote, so it must
        # NOT appear in the head-owned set — that slot is for the headless
        # to fill via auto-assign.
        assert call_kwargs["head_local_replicas"] == {0: [0]}
        mock_oms.start.assert_called_once()

    def test_start_omni_master_server_duplicate_stage_ids_raise(self):
        runtime = self._build_runtime([], stage_id_filter=0)
        stage_plans = [
            _make_llm_plan(0, stage_id=0, launch_mode="local"),
            _make_llm_plan(1, stage_id=0, launch_mode="remote"),
        ]

        with pytest.raises(ValueError, match="Duplicate stage_id"):
            runtime._start_omni_master_server(stage_plans)

    def test_start_omni_master_server_missing_address_raises(self):
        runtime = self._build_runtime([], stage_id_filter=0)
        runtime._omni_master_address = None

        with pytest.raises(ValueError, match="requires both"):
            runtime._start_omni_master_server([_make_llm_plan(0, stage_id=0, launch_mode="local")])

    def test_build_logical_stage_init_plans_preserves_runtime_cfg_for_local_llm_in_single_stage_mode(
        self, mocker: MockerFixture
    ):
        import vllm_omni.engine.stage_runtime as runtime_mod

        runtime = self._build_runtime([_make_stage_cfg(0)], stage_id_filter=0)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            runtime_mod,
            "extract_stage_metadata",
            lambda cfg: SimpleNamespace(
                stage_id=cfg.stage_id,
                stage_type="llm",
                prompt_expand_func=None,
                runtime_cfg={"devices": "0"},
            ),
        )
        monkeypatch.setattr(runtime_mod, "get_stage_connector_spec", lambda **_: {})
        monkeypatch.setattr(runtime_mod, "resolve_omni_kv_config_for_stage", lambda *_: (None, None, None))
        monkeypatch.setattr(runtime_mod, "build_engine_args_dict", lambda *_, **__: {})
        monkeypatch.setattr(
            runtime_mod,
            "build_vllm_config",
            lambda *_, **__: (SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size_local=1)), object),
        )
        try:
            stage_plans = runtime._build_logical_stage_init_plans(None, [1], {})
        finally:
            monkeypatch.undo()

        assert stage_plans[0].replicas[0].metadata.runtime_cfg == {"devices": "0"}

    def test_validate_single_stage_mode_allows_diffusion_replicas(self):
        stage_cfg = _make_stage_cfg(0, stage_type="diffusion")
        stage_cfg.runtime.num_replicas = 2
        runtime = self._build_runtime([stage_cfg], stage_id_filter=0)

        runtime._validate_single_stage_mode_replica_constraints()

    def test_validate_single_stage_mode_does_not_swallow_num_replica_validation_errors(self):
        class RuntimeConfig:
            @property
            def num_replicas(self):
                return 1

            @num_replicas.setter
            def num_replicas(self, value):
                raise ValueError("invalid num_replicas")

        stage_cfg = _make_stage_cfg(0, stage_type="llm")
        stage_cfg.runtime = RuntimeConfig()
        runtime = self._build_runtime([stage_cfg], stage_id_filter=0)

        with pytest.raises(ValueError, match="invalid num_replicas"):
            runtime._validate_single_stage_mode_replica_constraints()

    def test_build_logical_stage_init_plans_preserves_diffusion_runtime_cfg_in_single_stage_mode(
        self, mocker: MockerFixture
    ):
        import vllm_omni.engine.stage_runtime as runtime_mod

        stage_cfg = _make_stage_cfg(0, stage_type="diffusion")
        stage_cfg.runtime.devices = "0,1"
        runtime = self._build_runtime([stage_cfg], stage_id_filter=0)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            runtime_mod,
            "extract_stage_metadata",
            lambda cfg: SimpleNamespace(
                stage_id=cfg.stage_id,
                stage_type="diffusion",
                prompt_expand_func=None,
                runtime_cfg={"devices": cfg.runtime.devices},
                final_output=True,
                final_output_type="image",
                default_sampling_params=SimpleNamespace(),
                custom_process_input_func=None,
                engine_input_source=[],
                cfg_kv_collect_func=None,
                replica_id=0,
            ),
        )
        monkeypatch.setattr(runtime_mod, "get_stage_connector_spec", lambda **_: {})
        monkeypatch.setattr(runtime_mod, "resolve_omni_kv_config_for_stage", lambda *_: (None, None, None))
        try:
            stage_plans = runtime._build_logical_stage_init_plans(None, [2], {0: ["0", "1"]})
        finally:
            monkeypatch.undo()

        replicas = stage_plans[0].replicas
        assert [replica.replica_id for replica in replicas] == [0, 1]
        assert [replica.stage_cfg.runtime.devices for replica in replicas] == ["0", "1"]
        assert [replica.metadata.runtime_cfg for replica in replicas] == [{"devices": "0"}, {"devices": "1"}]

    def test_initialize_stages_calls_master_server_only_in_single_stage_mode(self, mocker: MockerFixture):
        import vllm_omni.engine.stage_runtime as runtime_mod

        stage_cfgs = [_make_stage_cfg(0)]
        runtime = self._build_runtime(stage_cfgs, stage_id_filter=0)
        stage_plan = _make_llm_plan(0, stage_id=0, launch_mode="local")
        client = SimpleNamespace(
            stage_type="llm",
            is_comprehension=False,
            final_output=True,
            final_output_type=None,
            default_sampling_params=SimpleNamespace(),
        )

        mocker.patch.object(runtime_mod, "prepare_engine_environment")
        mocker.patch.object(runtime_mod, "load_omni_transfer_config_for_model", return_value=None)
        mocker.patch.object(runtime_mod, "compute_replica_layout", return_value=([1], {}))
        mocker.patch.object(runtime, "_prepare_stage_plans", return_value=[stage_plan])
        mock_start = mocker.patch.object(runtime, "_start_omni_master_server")
        mocker.patch.object(runtime, "_initialize_stage_replicas", return_value={0: [client]})
        mocker.patch.object(runtime_mod, "build_llm_stage_output_processor", return_value=object())

        runtime.initialize()
        mock_start.assert_called_once()

        plain_runtime = StageRuntime(
            stage_configs=stage_cfgs,
            model="fake-model",
            config_path="/fake/stages.yaml",
            stage_init_timeout=60,
            diffusion_batch_size=2,
            async_chunk=False,
        )
        mocker.patch.object(runtime_mod, "prepare_engine_environment")
        mocker.patch.object(runtime_mod, "load_omni_transfer_config_for_model", return_value=None)
        mocker.patch.object(runtime_mod, "compute_replica_layout", return_value=([1], {}))
        mocker.patch.object(plain_runtime, "_prepare_stage_plans", return_value=[stage_plan])
        mocker.patch.object(plain_runtime, "_initialize_stage_replicas", return_value={0: [client]})
        mocker.patch.object(runtime_mod, "build_llm_stage_output_processor", return_value=object())

        plain_runtime.initialize()

    def test_initialize_stages_stops_master_server_and_shuts_down_initialized_clients_on_failure(
        self,
        mocker: MockerFixture,
    ):
        import vllm_omni.engine.stage_runtime as runtime_mod

        stage_cfgs = [_make_stage_cfg(0)]
        runtime = self._build_runtime(stage_cfgs, stage_id_filter=0)
        stage_plan = _make_llm_plan(0, stage_id=0, launch_mode="local")
        initialized_client = mocker.Mock()
        mock_master = mocker.Mock(spec=OmniMasterServer)

        mocker.patch.object(runtime_mod, "prepare_engine_environment")
        mocker.patch.object(runtime_mod, "load_omni_transfer_config_for_model", return_value=None)
        mocker.patch.object(runtime_mod, "compute_replica_layout", return_value=([1], {}))
        mocker.patch.object(runtime, "_prepare_stage_plans", return_value=[stage_plan])

        def _start_master(_plans):
            runtime._omni_master_server = mock_master

        mocker.patch.object(runtime, "_start_omni_master_server", side_effect=_start_master)
        mocker.patch.object(runtime, "_initialize_stage_replicas", return_value={0: [initialized_client]})
        mocker.patch.object(runtime, "_finalize_initialized_stages", side_effect=RuntimeError("assemble failed"))
        mock_shutdown = mocker.patch.object(runtime, "_shutdown_initialized_clients")

        with pytest.raises(RuntimeError, match="assemble failed"):
            runtime.initialize()

        mock_shutdown.assert_called_once_with([initialized_client])
        mock_master.stop.assert_called_once()


class TestSingleStageReplicaInitialization:
    def test_initialize_llm_replica_remote_uses_connect_remote_engine_cores(self, mocker: MockerFixture):
        import vllm_omni.engine.stage_runtime as runtime_mod

        runtime = DistStageRuntime(
            stage_configs=[],
            model="fake-model",
            config_path="/fake/stages.yaml",
            stage_init_timeout=60,
            diffusion_batch_size=2,
            async_chunk=False,
            single_stage_id_filter=None,
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
        )
        runtime._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        runtime._omni_master_server.get_stage_config.return_value = {"stage_id": 1, "stage_type": "llm"}

        fake_vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size_local=1))
        fake_addresses = SimpleNamespace(
            inputs=["tcp://in"], outputs=["tcp://out"], frontend_stats_publish_address=None
        )
        fake_manager = mocker.Mock()
        fake_coordinator = mocker.Mock()
        events: list[str] = []

        @contextmanager
        def _fake_connect(**kwargs):
            events.append("enter")
            try:
                yield StageReplicaResources(
                    manager=fake_manager,
                    coordinator=fake_coordinator,
                    addresses=fake_addresses,
                )
            finally:
                events.append("exit")

        plan = _make_llm_plan(
            0,
            stage_id=1,
            launch_mode="remote",
            vllm_config=fake_vllm_config,
        ).replicas[0]
        sentinel_client = SimpleNamespace()

        mock_connect = mocker.patch.object(runtime_mod, "connect_remote_engine_cores", side_effect=_fake_connect)
        mocker.patch.object(
            StageEngineCoreClientBase,
            "make_async_mp_client",
            side_effect=lambda **_: (events.append("attach"), sentinel_client)[1],
        )

        result = runtime._initialize_remote_replica(plan, stage_init_timeout=60)

        assert result is sentinel_client
        runtime._omni_master_server.get_stage_config.assert_called_once_with(1, timeout_s=60, replica_id=0)
        assert mock_connect.call_args.kwargs["vllm_config"].parallel_config.data_parallel_size_local == 0
        assert mock_connect.call_args.kwargs["stage_id"] == 1
        assert mock_connect.call_args.kwargs["replica_id"] == 0
        assert events == ["enter", "exit", "attach"]

    def test_initialize_llm_replica_remote_missing_registered_stage_config_raises(self, mocker: MockerFixture):
        runtime = DistStageRuntime(
            stage_configs=[],
            model="fake-model",
            config_path="/fake/stages.yaml",
            stage_init_timeout=60,
            diffusion_batch_size=2,
            async_chunk=False,
            single_stage_id_filter=None,
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
        )
        runtime._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        runtime._omni_master_server.get_stage_config.return_value = None

        plan = _make_llm_plan(0, stage_id=1, launch_mode="remote").replicas[0]

        with pytest.raises(ValueError, match="registered without stage config"):
            runtime._initialize_remote_replica(plan, stage_init_timeout=60)

    def test_initialize_llm_replica_remote_attach_failure_cleans_up_started_resources(self, mocker: MockerFixture):
        import vllm_omni.engine.stage_runtime as runtime_mod

        runtime = DistStageRuntime(
            stage_configs=[],
            model="fake-model",
            config_path="/fake/stages.yaml",
            stage_init_timeout=60,
            diffusion_batch_size=2,
            async_chunk=False,
            single_stage_id_filter=None,
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
        )
        runtime._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        runtime._omni_master_server.get_stage_config.return_value = {"stage_id": 1, "stage_type": "llm"}

        fake_vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size_local=1))
        fake_addresses = SimpleNamespace(
            inputs=["tcp://in"], outputs=["tcp://out"], frontend_stats_publish_address=None
        )
        fake_manager = mocker.Mock()
        fake_coordinator = mocker.Mock()

        @contextmanager
        def _fake_connect(**kwargs):
            yield StageReplicaResources(
                manager=fake_manager,
                coordinator=fake_coordinator,
                addresses=fake_addresses,
            )

        plan = _make_llm_plan(
            0,
            stage_id=1,
            launch_mode="remote",
            vllm_config=fake_vllm_config,
        ).replicas[0]
        mocker.patch.object(runtime_mod, "connect_remote_engine_cores", side_effect=_fake_connect)
        mocker.patch.object(
            StageEngineCoreClientBase,
            "make_async_mp_client",
            side_effect=RuntimeError("attach failed"),
        )

        with pytest.raises(RuntimeError, match="attach failed"):
            runtime._initialize_remote_replica(plan, stage_init_timeout=60)

        fake_manager.shutdown.assert_called_once()
        fake_coordinator.shutdown.assert_called_once()

    def test_initialize_local_llm_replica_uses_shared_launcher(self, mocker: MockerFixture):
        import vllm_omni.engine.stage_runtime as runtime_mod
        from vllm_omni.platforms import current_omni_platform

        runtime = DistStageRuntime(
            stage_configs=[],
            model="fake-model",
            config_path="/fake/stages.yaml",
            stage_init_timeout=60,
            diffusion_batch_size=2,
            async_chunk=False,
            single_stage_id_filter=None,
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
        )
        runtime._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        runtime._coordinator_runtime = None
        runtime._stage_configs = []

        fake_vllm_config = SimpleNamespace(parallel_config=SimpleNamespace())
        fake_addresses = SimpleNamespace(
            inputs=["tcp://in"], outputs=["tcp://out"], frontend_stats_publish_address=None
        )

        @contextmanager
        def _fake_launch(**kwargs):
            yield StageReplicaResources(
                manager=mocker.Mock(),
                addresses=fake_addresses,
            )

        plan = _make_llm_plan(
            0,
            stage_id=0,
            launch_mode="local",
            vllm_config=fake_vllm_config,
        ).replicas[0]
        sentinel_client = SimpleNamespace()

        device_env_var = current_omni_platform.device_control_env_var
        prev_device_env = os.environ.get(device_env_var)
        os.environ[device_env_var] = "0"
        runtime._init_visible_devices_baseline = "0"

        mocker.patch.object(runtime_mod, "build_engine_args_dict", return_value={})
        mocker.patch.object(runtime_mod, "acquire_device_locks", return_value=[])
        mocker.patch.object(runtime_mod, "release_device_locks")
        mock_launch = mocker.patch.object(runtime_mod, "launch_stage_replica", side_effect=_fake_launch)
        mocker.patch.object(
            StageEngineCoreClientBase,
            "make_async_mp_client",
            side_effect=lambda **_: sentinel_client,
        )
        try:
            result = runtime._initialize_local_llm_replica(plan, stage_init_timeout=60)
        finally:
            if prev_device_env is None:
                os.environ.pop(device_env_var, None)
            else:
                os.environ[device_env_var] = prev_device_env

        assert result is sentinel_client
        assert mock_launch.call_args.kwargs["stage_id"] == 0
        assert mock_launch.call_args.kwargs["stage_config"] is plan.stage_cfg
        assert mock_launch.call_args.kwargs["replica_id"] == 0
        assert mock_launch.call_args.kwargs["omni_master_server"] is runtime._omni_master_server

    def test_initialize_diffusion_replica_remote_uses_from_addresses(self, mocker: MockerFixture):
        import vllm_omni.engine.stage_runtime as runtime_mod

        runtime = DistStageRuntime(
            stage_configs=[],
            model="fake-model",
            config_path="/fake/stages.yaml",
            stage_init_timeout=60,
            diffusion_batch_size=4,
            async_chunk=False,
            single_stage_id_filter=None,
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
        )
        runtime._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        runtime._omni_master_server.get_stage_config.return_value = {"stage_id": 1, "stage_type": "diffusion"}

        def _fake_connect(**kwargs):
            @contextmanager
            def _ctx():
                yield StageReplicaResources(
                    addresses=SimpleNamespace(
                        inputs=["tcp://in"],
                        outputs=["tcp://out"],
                    )
                )

            return _ctx()

        remote_metadata = _make_diffusion_plan(1, stage_id=1, launch_mode="remote").replicas[0].metadata
        plan = _make_diffusion_plan(1, stage_id=1, launch_mode="remote").replicas[0]
        sentinel_client = SimpleNamespace()

        mocker.patch.object(runtime_mod, "extract_stage_metadata", return_value=remote_metadata)
        mock_connect = mocker.patch.object(runtime_mod, "connect_remote_diffusion_proc", side_effect=_fake_connect)
        mock_from_addresses = mocker.patch(
            "vllm_omni.diffusion.stage_diffusion_client.StageDiffusionClient.from_addresses",
            return_value=sentinel_client,
        )

        result = runtime._initialize_remote_replica(plan, stage_init_timeout=60)

        assert result is sentinel_client
        runtime._omni_master_server.get_stage_config.assert_called_once_with(1, timeout_s=60, replica_id=0)
        mock_connect.assert_called_once_with(
            omni_master_server=runtime._omni_master_server,
            stage_id=1,
            replica_id=0,
        )
        mock_from_addresses.assert_called_once()

    def test_initialize_local_diffusion_replica_registers_with_master(self, mocker: MockerFixture):
        import vllm_omni.engine.stage_runtime as runtime_mod
        from vllm_omni.platforms import current_omni_platform

        runtime = DistStageRuntime(
            stage_configs=[],
            model="fake-model",
            config_path="/fake/stages.yaml",
            stage_init_timeout=60,
            diffusion_batch_size=4,
            async_chunk=False,
            single_stage_id_filter=None,
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
        )
        runtime._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        runtime._omni_master_server.address = "127.0.0.1"
        runtime._omni_master_server.port = 25000
        runtime._coordinator_runtime = None

        plan = _make_diffusion_plan(0, stage_id=0, launch_mode="local").replicas[0]
        sentinel_client = SimpleNamespace()
        proc = mocker.Mock()

        device_env_var = current_omni_platform.device_control_env_var
        prev_device_env = os.environ.get(device_env_var)
        os.environ[device_env_var] = "0"
        runtime._init_visible_devices_baseline = "0"

        mocker.patch.object(runtime_mod, "inject_kv_stage_info")
        mocker.patch("vllm_omni.engine.stage_engine_startup.build_diffusion_config", return_value="diffusion-config")
        mock_register = mocker.patch(
            "vllm_omni.engine.stage_engine_startup.register_stage_with_omni_master",
            return_value=StageRegistrationResponse(
                handshake_address="tcp://127.0.0.1:26001",
                input_address="tcp://127.0.0.1:26002",
                output_address="tcp://127.0.0.1:26003",
                replica_id=0,
                coordinator_router_address=None,
            ),
        )
        fake_manager = SimpleNamespace(
            proc=proc,
            addresses=SimpleNamespace(
                inputs=["tcp://127.0.0.1:26002"],
                outputs=["tcp://127.0.0.1:26003"],
            ),
        )
        mock_manager = mocker.patch(
            "vllm_omni.diffusion.stage_diffusion_proc.StageDiffusionProcManager",
            return_value=fake_manager,
        )
        mock_from_addresses = mocker.patch(
            "vllm_omni.diffusion.stage_diffusion_client.StageDiffusionClient.from_addresses",
            return_value=sentinel_client,
        )

        try:
            result = runtime._initialize_local_diffusion_replica(plan, stage_init_timeout=60)
        finally:
            if prev_device_env is None:
                os.environ.pop(device_env_var, None)
            else:
                os.environ[device_env_var] = prev_device_env

        assert result is sentinel_client
        mock_register.assert_called_once_with(
            omni_master_address="127.0.0.1",
            omni_master_port=25000,
            omni_stage_id=0,
            omni_stage_config=plan.stage_cfg,
            replica_id=0,
        )
        mock_manager.assert_called_once_with(
            model="fake-model",
            od_config="diffusion-config",
            stage_init_timeout=60,
            handshake_address="tcp://127.0.0.1:26001",
            addresses=mocker.ANY,
            omni_coordinator_address=None,
            omni_stage_id=0,
            omni_replica_id=0,
        )
        assert mock_manager.call_args.kwargs["addresses"].inputs == ["tcp://127.0.0.1:26002"]
        assert mock_manager.call_args.kwargs["addresses"].outputs == ["tcp://127.0.0.1:26003"]
        mock_from_addresses.assert_called_once_with(
            plan.metadata,
            request_address="tcp://127.0.0.1:26002",
            response_address="tcp://127.0.0.1:26003",
            proc_manager=mocker.ANY,
            batch_size=4,
        )
        assert mock_from_addresses.call_args.kwargs["proc_manager"].proc is proc

    def test_initialize_local_diffusion_replica_failure_terminates_proc(self, mocker: MockerFixture):
        import vllm_omni.engine.stage_runtime as runtime_mod
        from vllm_omni.platforms import current_omni_platform

        runtime = DistStageRuntime(
            stage_configs=[],
            model="fake-model",
            config_path="/fake/stages.yaml",
            stage_init_timeout=60,
            diffusion_batch_size=4,
            async_chunk=False,
            single_stage_id_filter=None,
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
        )
        runtime._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        runtime._omni_master_server.address = "127.0.0.1"
        runtime._omni_master_server.port = 25000
        runtime._coordinator_runtime = None

        plan = _make_diffusion_plan(0, stage_id=0, launch_mode="local").replicas[0]

        device_env_var = current_omni_platform.device_control_env_var
        prev_device_env = os.environ.get(device_env_var)
        os.environ[device_env_var] = "0"
        runtime._init_visible_devices_baseline = "0"

        mocker.patch.object(runtime_mod, "inject_kv_stage_info")
        mocker.patch("vllm_omni.engine.stage_engine_startup.build_diffusion_config", return_value="diffusion-config")
        mocker.patch(
            "vllm_omni.engine.stage_engine_startup.register_stage_with_omni_master",
            return_value=StageRegistrationResponse(
                handshake_address="tcp://127.0.0.1:26001",
                input_address="tcp://127.0.0.1:26002",
                output_address="tcp://127.0.0.1:26003",
                replica_id=0,
                coordinator_router_address=None,
            ),
        )
        mocker.patch(
            "vllm_omni.diffusion.stage_diffusion_proc.StageDiffusionProcManager",
            side_effect=RuntimeError("handshake failed"),
        )

        try:
            with pytest.raises(RuntimeError, match="handshake failed"):
                runtime._initialize_local_diffusion_replica(plan, stage_init_timeout=60)
        finally:
            if prev_device_env is None:
                os.environ.pop(device_env_var, None)
            else:
                os.environ[device_env_var] = prev_device_env


# ---------------------------------------------------------------------------
# Stage engine startup helpers
# ---------------------------------------------------------------------------


class TestConnectRemoteEngineCoresCoordinator:
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
            inputs=["tcp://client-in"],
            outputs=["tcp://client-out"],
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

        mocker.patch("vllm_omni.engine.stage_engine_startup.zmq_socket_ctx", return_value=fake_socket_ctx())
        mock_wait = mocker.patch("vllm_omni.engine.stage_engine_startup.wait_for_engine_startup")
        with connect_remote_engine_cores(
            vllm_config=vllm_config,
            omni_master_server=omni_master_server,
            stage_id=7,
            replica_id=2,
        ) as resources:
            assert resources.coordinator is None
            yielded_addresses = resources.addresses
            assert yielded_addresses is not None
            assert yielded_addresses.coordinator_input == "tcp://coord-in"
            assert yielded_addresses.coordinator_output == "tcp://coord-out"
            assert yielded_addresses.frontend_stats_publish_address == "tcp://stats"

        omni_master_server.get_zmq_addresses.assert_called_once_with(7, replica_id=2)
        omni_master_server.get_stage_coordinator_addresses.assert_called_once_with(7, replica_id=2)
        omni_master_server.get_allocation.assert_called_once_with(7, replica_id=2)
        mock_wait.assert_called_once()

    def test_defaults_to_no_coordinator_addresses_when_none_registered(self, mocker: MockerFixture):
        vllm_config = self._build_vllm_config(mocker, dp_rank=0, offline_mode=False, needs_dp_coordinator=True)

        omni_master_server = mocker.Mock(spec=OmniMasterServer)
        omni_master_server.get_zmq_addresses.return_value = EngineZmqAddresses(
            inputs=["tcp://client-in"],
            outputs=["tcp://client-out"],
        )
        omni_master_server.get_allocation.return_value = mocker.Mock(handshake_bind_address="tcp://127.0.0.1:26001")
        omni_master_server.get_stage_coordinator_addresses.return_value = StageCoordinatorAddresses()

        @contextmanager
        def fake_socket_ctx(*args, **kwargs):
            yield mocker.Mock()

        mocker.patch("vllm_omni.engine.stage_engine_startup.zmq_socket_ctx", return_value=fake_socket_ctx())
        mocker.patch("vllm_omni.engine.stage_engine_startup.wait_for_engine_startup")
        with connect_remote_engine_cores(
            vllm_config=vllm_config,
            omni_master_server=omni_master_server,
            stage_id=7,
        ) as resources:
            assert resources.coordinator is None
            yielded_addresses = resources.addresses
            assert yielded_addresses is not None
            assert yielded_addresses.coordinator_input is None
            assert yielded_addresses.coordinator_output is None
            assert yielded_addresses.frontend_stats_publish_address is None

    def test_connect_remote_diffusion_proc_waits_for_headless_remote(self, mocker: MockerFixture):
        omni_master_server = mocker.Mock(spec=OmniMasterServer)
        omni_master_server.get_zmq_addresses.return_value = EngineZmqAddresses(
            inputs=["tcp://client-in"],
            outputs=["tcp://client-out"],
        )
        omni_master_server.get_allocation.return_value = mocker.Mock(handshake_bind_address="tcp://127.0.0.1:26001")

        @contextmanager
        def fake_socket_ctx(*args, **kwargs):
            yield mocker.Mock()

        mocker.patch("vllm_omni.engine.stage_engine_startup.zmq_socket_ctx", return_value=fake_socket_ctx())
        mock_wait = mocker.patch("vllm_omni.engine.stage_engine_startup.wait_for_engine_startup")

        with connect_remote_diffusion_proc(
            omni_master_server=omni_master_server,
            stage_id=7,
            replica_id=2,
        ) as resources:
            assert resources.addresses is omni_master_server.get_zmq_addresses.return_value

        omni_master_server.get_zmq_addresses.assert_called_once_with(7, replica_id=2)
        omni_master_server.get_allocation.assert_called_once_with(7, replica_id=2)
        mock_wait.assert_called_once()
        _, _, core_engines, parallel_config, *_ = mock_wait.call_args.args
        assert core_engines[0].local is False
        assert parallel_config.data_parallel_size_local == 0


class TestLaunchOmniCoreEngines:
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
            return_value=StageRegistrationResponse(
                handshake_address="tcp://127.0.0.1:26001",
                input_address="tcp://client-in",
                output_address="tcp://client-out",
                replica_id=2,
                coordinator_router_address=None,
            ),
        )
        mocker.patch("vllm_omni.engine.stage_engine_startup.zmq_socket_ctx", return_value=fake_socket_ctx())
        mock_manager_cls = mocker.patch(
            "vllm_omni.engine.stage_engine_startup.CoreEngineProcManager",
            return_value=local_engine_manager,
        )
        mocker.patch("vllm_omni.engine.stage_engine_startup.wait_for_engine_startup")
        with _launch_omni_core_engines(
            vllm_config=vllm_config,
            executor_class=mocker.Mock(),
            log_stats=False,
            omni_master_server=omni_master_server,
            stage_id=7,
            stage_config=stage_config,
            replica_id=2,
        ) as (yielded_manager, yielded_coordinator, yielded_addresses):
            assert yielded_manager is local_engine_manager
            assert yielded_coordinator is None
            assert yielded_addresses is not None

        mock_register.assert_called_once_with(
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
            omni_stage_id=7,
            omni_stage_config=stage_config,
            coordinator=None,
            replica_id=2,
        )
        omni_master_server.get_zmq_addresses.assert_called_once_with(7, replica_id=2)
        omni_master_server.get_allocation.assert_called_once_with(7, replica_id=2)
        manager_kwargs = mock_manager_cls.call_args.kwargs
        assert manager_kwargs["local_engine_count"] == 2
        assert manager_kwargs["start_index"] == 3
        assert manager_kwargs["local_start_index"] == 0
        assert manager_kwargs["handshake_address"] == "tcp://127.0.0.1:26001"

    def test_registers_stage_with_coordinator_when_started(self, mocker: MockerFixture):
        parallel_config = mocker.Mock(
            data_parallel_size_local=1,
            data_parallel_size=2,
            data_parallel_rank=0,
        )
        vllm_config = mocker.Mock(
            parallel_config=parallel_config,
            needs_dp_coordinator=True,
            model_config=mocker.Mock(is_moe=False),
            cache_config=mocker.Mock(),
        )

        omni_master_server = mocker.Mock(spec=OmniMasterServer)
        omni_master_server.address = "127.0.0.1"
        omni_master_server.port = 26000
        omni_master_server.get_zmq_addresses.return_value = EngineZmqAddresses(
            inputs=["tcp://client-in"],
            outputs=["tcp://client-out"],
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
            return_value=StageRegistrationResponse(
                handshake_address="tcp://127.0.0.1:26001",
                input_address="tcp://client-in",
                output_address="tcp://client-out",
                replica_id=3,
                coordinator_router_address=None,
            ),
        )
        mocker.patch("vllm_omni.engine.stage_engine_startup.zmq_socket_ctx", return_value=fake_socket_ctx())
        mocker.patch(
            "vllm_omni.engine.stage_engine_startup.CoreEngineProcManager",
            return_value=mocker.Mock(),
        )
        mock_wait = mocker.patch("vllm_omni.engine.stage_engine_startup.wait_for_engine_startup")
        with _launch_omni_core_engines(
            vllm_config=vllm_config,
            executor_class=mocker.Mock(),
            log_stats=False,
            omni_master_server=omni_master_server,
            stage_id=7,
            stage_config={"stage_id": 7},
            replica_id=3,
        ) as (_, yielded_coordinator, yielded_addresses):
            assert yielded_coordinator is coordinator
            assert yielded_addresses.coordinator_input == "tcp://coord-in"
            assert yielded_addresses.coordinator_output == "tcp://coord-out"
            assert yielded_addresses.frontend_stats_publish_address == "tcp://stats"

        mock_register.assert_called_once_with(
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
            omni_stage_id=7,
            omni_stage_config={"stage_id": 7},
            coordinator=coordinator,
            replica_id=3,
        )
        mock_wait.assert_called_once()
