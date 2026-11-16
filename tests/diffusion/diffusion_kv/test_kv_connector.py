# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.outputs import KVConnectorOutput

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv import kv_connector as connector_module
from vllm_omni.diffusion.diffusion_kv.kv_connector import (
    KVTransferRegistrationError,
    commit_kv_load,
    parse_kv_transfer_config,
    shutdown_kv_connector,
    wait_for_kv_load,
)
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

KV_TRANSFER_CONFIG = {
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer",
    "engine_id": "dit-engine-1",
    "kv_connector_extra_config": {"mooncake_protocol": "tcp"},
}


@pytest.fixture(autouse=True)
def _fixed_master_port(monkeypatch) -> None:
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda _self: 29500)


def test_config_roundtrip_to_kv_vllm_config() -> None:
    od_config = OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=1,
        max_model_len=64,
        kv_transfer_config=dict(KV_TRANSFER_CONFIG),
    )

    assert isinstance(od_config.kv_transfer_config, KVTransferConfig)
    assert od_config.kv_transfer_config.engine_id == "dit-engine-1"
    vllm_config = create_diffusion_vllm_config(torch.device("cpu"), od_config)
    assert vllm_config.kv_transfer_config is od_config.kv_transfer_config


def test_parse_requires_explicit_engine_id() -> None:
    payload = dict(KV_TRANSFER_CONFIG)
    payload.pop("engine_id")
    with pytest.raises(ValueError, match="non-empty engine_id"):
        parse_kv_transfer_config(payload)


def test_diffusion_projection_rejects_missing_engine_id_before_materialize() -> None:
    """Structured resolve must not let upstream auto-mint engine_id first."""
    from vllm_omni.config.omni_config import _DiffusionConfigProjection

    payload = dict(KV_TRANSFER_CONFIG)
    payload.pop("engine_id")
    with pytest.raises(ValueError, match="non-empty engine_id"):
        _DiffusionConfigProjection.from_kwargs(kv_transfer_config=payload)


def test_diffusion_projection_preserves_explicit_engine_id() -> None:
    from vllm_omni.config.omni_config import _DiffusionConfigProjection

    projection = _DiffusionConfigProjection.from_kwargs(kv_transfer_config=dict(KV_TRANSFER_CONFIG))
    assert isinstance(projection.kv_transfer_config, KVTransferConfig)
    assert projection.kv_transfer_config.engine_id == "dit-engine-1"

    od_config = OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=1,
        kv_transfer_config=projection.kv_transfer_config,
    )
    assert od_config.kv_transfer_config is not None
    assert od_config.kv_transfer_config.engine_id == "dit-engine-1"


class _ConcreteScheduler(BaseScheduler):
    def update_from_output(self, sched_output, output) -> set[str]:
        del sched_output, output
        return set()


def test_scheduler_assembles_kv_stub_and_shuts_it_down() -> None:
    scheduler = _ConcreteScheduler()
    fake_connector = mock.Mock()
    od_config = OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=1,
        max_model_len=64,
        kv_transfer_config=dict(KV_TRANSFER_CONFIG),
    )
    kv_vllm_config = mock.Mock()
    kv_vllm_config.model_config.max_model_len = 64
    kv_vllm_config.max_in_flight_tokens = 64

    with (
        mock.patch(
            "vllm_omni.diffusion.diffusion_kv.kv_connector.KVConnectorFactory.create_connector",
            return_value=fake_connector,
        ),
        mock.patch("vllm_omni.diffusion.sched.base_scheduler.DiffusionKVCacheManager"),
    ):
        scheduler.initialize(
            od_config,
            kv_cache_config=mock.sentinel.kv_cache_config,
            scheduler_block_size=16,
            hash_block_size=16,
            kv_vllm_config=kv_vllm_config,
        )

    assert scheduler.kv_connector is fake_connector
    with mock.patch("vllm_omni.diffusion.diffusion_kv.kv_connector.shutdown_kv_connector") as shutdown:
        scheduler.close()
    shutdown.assert_called_once_with(scheduler_connector=fake_connector)


def test_native_config_requires_paged_scheduler() -> None:
    with pytest.raises(ValueError, match="requires diffusion_kv_mode='paged_scheduler'"):
        OmniDiffusionConfig.from_kwargs(kv_transfer_config=dict(KV_TRANSFER_CONFIG))


def test_kv_and_legacy_transfer_configs_are_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one KV transfer path"):
        OmniDiffusionConfig.from_kwargs(
            kv_transfer_config=dict(KV_TRANSFER_CONFIG),
            omni_kv_config={"need_recv_cache": True},
        )


def test_shutdown_kv_connector_is_idempotent() -> None:
    scheduler_connector = mock.Mock()
    with mock.patch("vllm_omni.diffusion.diffusion_kv.kv_connector.ensure_kv_transfer_shutdown") as ensure_shutdown:
        shutdown_kv_connector(scheduler_connector=scheduler_connector)
        shutdown_kv_connector()

    assert ensure_shutdown.call_count == 2
    scheduler_connector.shutdown.assert_called_once()


@pytest.mark.parametrize("matched_tokens", [[7, 5], [0, 0], [7, 0]])
def test_commit_aligns_cfg_transport_pages_without_extending_computed_prefix(matched_tokens) -> None:
    requests = tuple(
        SimpleNamespace(
            request_id=f"req/{sequence_id}",
            kv_transfer_params={"num_transfer_tokens": 9},
            num_tokens=16,
            num_computed_tokens=0,
        )
        for sequence_id in range(2)
    )
    manager = SimpleNamespace(
        get_blocks=lambda _request_id: KVCacheBlocks(([0, 1, 2, 3],)),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4))]),
    )
    connector = mock.Mock()

    expected = commit_kv_load(connector, manager, requests, matched_tokens)

    assert expected == {"req/0", "req/1"}
    calls = connector.update_state_after_alloc.call_args_list
    assert [len(call.args[1].blocks[0]) for call in calls] == [3, 3]
    assert [call.args[2] for call in calls] == [9, 9]
    assert [request.num_computed_tokens for request in requests] == matched_tokens


def test_wait_without_pending_load_uses_vllm_post_forward_once() -> None:
    connector = mock.Mock()
    output = mock.Mock(invalid_block_ids=set())
    active_connector = mock.Mock(kv_connector=connector)
    active_connector.post_forward.return_value = output
    scheduler_output = SimpleNamespace(
        kv_transfer_request_ids=set(),
        finished_req_ids={"finished"},
        kv_finished_request_ids={"finished/diffusion-kv/0", "finished/diffusion-kv/1"},
    )

    result = wait_for_kv_load(active_connector, scheduler_output, timeout=1.0)

    assert result is output
    active_connector.pre_forward.assert_called_once_with(scheduler_output)
    active_connector.post_forward.assert_called_once_with({"finished/diffusion-kv/0", "finished/diffusion-kv/1"})
    connector.get_finished.assert_not_called()


def test_timeout_returns_partial_completion_without_releasing_pages():
    active = mock.Mock()
    active.kv_connector.get_finished.return_value = (set(), {"cfg0"})
    active.kv_connector.get_block_ids_with_load_errors.return_value = set()
    active.post_forward.return_value = KVConnectorOutput()
    scheduled = SimpleNamespace(kv_transfer_request_ids={"cfg0", "cfg1"}, kv_finished_request_ids=set())
    output = wait_for_kv_load(active, scheduled, timeout=0)
    assert output.finished_recving == {"cfg0"}
    active.kv_connector.request_finished.assert_not_called()
    active.post_forward.assert_called_once_with(set())


@pytest.fixture
def cfg_registration():
    requests = tuple(
        SimpleNamespace(
            request_id=f"cfg{i}",
            status=None,
            num_tokens=8,
            kv_transfer_params={"num_transfer_tokens": 4, "transfer_id": "ticket", "do_remote_prefill": True},
        )
        for i in range(2)
    )
    manager = SimpleNamespace(
        get_blocks=lambda _: KVCacheBlocks(([0, 1],)),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4))]),
    )
    return requests, manager


def test_cfg_validation_is_atomic_before_connector_registration(cfg_registration):
    requests, manager = cfg_registration
    requests[1].kv_transfer_params["num_transfer_tokens"] = 9
    connector = mock.Mock()
    with pytest.raises(KVTransferRegistrationError, match="boundary"):
        commit_kv_load(connector, manager, requests, [4, 4])
    connector.update_state_after_alloc.assert_not_called()


@pytest.mark.parametrize("fail_init", [False, True])
def test_sp_connector_init_snapshots_topology_and_restores_model_tp(monkeypatch, fail_init):
    import vllm.distributed.parallel_state as parallel_state

    import vllm_omni.diffusion.diffusion_kv.kv_connector as module
    import vllm_omni.diffusion.distributed.parallel_state as omni_parallel

    tp = SimpleNamespace(world_size=1, rank_in_group=0)
    sp = SimpleNamespace(world_size=2, rank_in_group=1)
    monkeypatch.setattr(parallel_state, "_TP", tp)
    monkeypatch.setattr(omni_parallel, "get_sp_group", lambda: sp)
    snapshot: dict[str, int] = {}

    def initialize(*args):
        snapshot.update(
            rank=parallel_state.get_tensor_model_parallel_rank(),
            size=parallel_state.get_tensor_model_parallel_world_size(),
        )
        if fail_init:
            raise RuntimeError("initialization failed")

    monkeypatch.setattr(module, "ensure_kv_transfer_initialized", initialize)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnector", engine_id="dit"),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
    )
    with pytest.raises(RuntimeError, match="initialization failed") if fail_init else nullcontext():
        module.init_worker_kv_connector(config, object())
    assert parallel_state._TP is tp
    assert snapshot == {"rank": 1, "size": 2}


def test_partial_mooncake_registration_is_replaced_by_empty_notifications(cfg_registration):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeConnector,
        MooncakeConnectorScheduler,
    )

    scheduler = object.__new__(MooncakeConnectorScheduler)
    scheduler.is_kv_producer = False
    scheduler.is_kv_consumer = True
    scheduler._reqs_need_recv = {}
    connector = object.__new__(MooncakeConnector)
    connector.connector_scheduler = scheduler
    requests, manager = cfg_registration
    for request in requests:
        request.kv_transfer_params.update(remote_engine_id="ar", remote_bootstrap_addr="http://localhost:8998")

    def register(request, blocks, count):
        scheduler._reqs_need_recv[request.request_id] = (request, blocks)
        request.kv_transfer_params["do_remote_prefill"] = False
        if request.request_id == "cfg1":
            raise RuntimeError("partial registration")

    connector.update_state_after_alloc = register
    with pytest.raises(KVTransferRegistrationError, match="partial registration"):
        commit_kv_load(connector, manager, requests, [4, 4])
    assert set(scheduler._reqs_need_recv) == {"cfg0", "cfg1"}
    assert all(blocks == [] for _, blocks in scheduler._reqs_need_recv.values())
    metadata = scheduler.build_connector_meta(None)
    # Even partial registration failure sends every row in one per-rank batch,
    # preserving the producer's complete fan-out count for the shared ticket.
    assert set(metadata.reqs_to_recv["ar"]) == {"cfg0", "cfg1"}
    assert all(
        meta.transfer_id == "ticket" and meta.local_block_ids == [] for meta in metadata.reqs_to_recv["ar"].values()
    )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), "60"])
def test_native_transfer_timeout_must_be_positive_finite(timeout):
    with pytest.raises(ValueError, match="transfer_timeout"):
        parse_kv_transfer_config(
            {
                **KV_TRANSFER_CONFIG,
                "kv_connector_extra_config": {"transfer_timeout": timeout},
            }
        )


def test_receive_completion_requires_all_ranks_across_polls():
    from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
    from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput

    executor = SimpleNamespace(od_config=SimpleNamespace(num_gpus=2))
    executor.collective_rpc = mock.Mock(
        side_effect=[
            [KVConnectorOutput(finished_recving={"cfg0", "cfg1"}), KVConnectorOutput(finished_recving={"cfg0"})],
            [KVConnectorOutput(), KVConnectorOutput(finished_recving={"cfg1"})],
        ]
    )
    scheduled = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=0,
        num_waiting_reqs=0,
        kv_connector_metadata=object(),
        kv_transfer_request_ids={"cfg0", "cfg1"},
    )
    result = DiffusionExecutor.prepare_kv_for_forward(executor, scheduled)
    assert result.finished_recving == {"cfg0"}
    scheduled.kv_transfer_request_ids = set()
    result = DiffusionExecutor.prepare_kv_for_forward(executor, scheduled)
    assert result.finished_recving == {"cfg1"}
    assert not executor._kv_receive_completed_ranks


def test_native_transfer_rejects_sleep_without_changing_legacy_sleep():
    with pytest.raises(ValueError, match="registered pages must remain mapped"):
        OmniDiffusionConfig.from_kwargs(
            diffusion_kv_mode="paged_scheduler",
            diffusion_kv_max_rows_per_request=2,
            kv_transfer_config=dict(KV_TRANSFER_CONFIG),
            enable_sleep_mode=True,
        )
    assert OmniDiffusionConfig.from_kwargs(enable_sleep_mode=True).enable_sleep_mode


@pytest.mark.parametrize("reusable_tokens", [0, 5])
def test_native_mooncake_receives_pages_without_extending_reusable_prefix(reusable_tokens):
    native = pytest.importorskip("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector")
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

    scheduler = object.__new__(native.MooncakeConnectorScheduler)
    scheduler._reqs_in_batch = set()
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {}
    request = SimpleNamespace(
        request_id="cfg0",
        num_tokens=16,
        num_computed_tokens=0,
        kv_transfer_params={
            "num_transfer_tokens": 9,
            "do_remote_prefill": True,
            "remote_block_ids": ([10, 11, 12],),
            "remote_engine_id": "ar",
            "remote_request_id": "source",
            "remote_host": "localhost",
            "remote_port": 10000,
            "remote_block_size": 4,
        },
    )
    blocks = KVCacheBlocks(([SimpleNamespace(block_id=i, block_hash=None, is_null=False) for i in range(1, 5)],))
    manager = SimpleNamespace(
        get_blocks=lambda _: blocks,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4))]),
    )
    commit_kv_load(scheduler, manager, (request,), [reusable_tokens])
    metadata = scheduler.build_connector_meta(None).requests["cfg0"]
    worker = object.__new__(native.MooncakeConnectorWorker)
    worker.block_size = 4
    worker.block_size_scale = [[1]]
    local, remote = worker._get_kernel_block_ids(
        [0], metadata, 0, {"kv_cache_spec_type": "FullAttentionSpec", "kv_cache_group_id": 0}
    )
    assert local == [1, 2, 3]
    assert remote == [10, 11, 12]
    assert request.num_computed_tokens == reusable_tokens


def test_worker_connector_factory_keeps_upstream_gpu_initialization(monkeypatch):
    import builtins

    import vllm.v1.worker.gpu.kv_connector as active_module

    import vllm_omni.platforms as platforms

    constructor = mock.Mock()
    monkeypatch.setattr(active_module, "ActiveKVConnector", constructor)
    monkeypatch.setattr(platforms, "current_omni_platform", SimpleNamespace(is_npu=lambda: False))
    original_import = builtins.__import__

    def without_ascend(name, *args, **kwargs):
        if name.startswith("vllm_ascend"):
            raise AssertionError("GPU connector must not import Ascend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_ascend)
    config, caches = object(), {"layer": object()}
    assert connector_module.create_worker_kv_connector(config, caches) is constructor.return_value
    constructor.assert_called_once_with(config, caches)


@pytest.fixture
def ascend_connector(monkeypatch):
    import vllm.distributed.kv_transfer as kv_transfer
    import vllm.v1.worker.gpu.kv_connector as active_module

    import vllm_omni.platforms as platforms

    pytest.importorskip("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector")
    native = mock.Mock(connector_worker=None)
    native.get_finished.return_value = (set(), set())
    native.get_block_ids_with_load_errors.return_value = set()
    monkeypatch.setattr(kv_transfer, "get_kv_transfer_group", lambda: native)
    monkeypatch.setattr(platforms, "current_omni_platform", SimpleNamespace(is_npu=lambda: True))
    monkeypatch.setattr(active_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(active_module, "get_forward_context", lambda: None)
    caches = {"layer": object()}
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnectorV1"))
    active = connector_module.create_worker_kv_connector(config, caches)
    native.register_kv_caches.assert_called_once_with(caches)
    native.set_host_xfer_buffer_ops.assert_not_called()
    return active, native


@pytest.mark.parametrize("invalid_pages", [False, True])
def test_load_completion_and_errors_use_shared_lifecycle(ascend_connector, invalid_pages):
    active, native = ascend_connector
    native.get_finished.side_effect = [(set(), {"req/kv/0"}), (set(), set())]
    native.get_block_ids_with_load_errors.return_value = {3} if invalid_pages else set()
    scheduler = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=0,
        num_waiting_reqs=0,
        kv_connector_metadata=object(),
        kv_transfer_request_ids={"req/kv/0"},
        kv_finished_request_ids={"previous/kv/0"},
    )
    if invalid_pages:
        with pytest.raises(RuntimeError, match="invalid remote pages"):
            wait_for_kv_load(active, scheduler, timeout=1)
    else:
        assert wait_for_kv_load(active, scheduler, timeout=1).finished_recving == {"req/kv/0"}
        native.wait_for_save.assert_called_once()
        native.clear_connector_metadata.assert_called_once()
    native.bind_connector_metadata.assert_called_once_with(scheduler.kv_connector_metadata)
    native.start_load_kv.assert_called_once_with(None)


def _source_meta(request_id="source", *, virtual=False):
    return SimpleNamespace(
        do_virtual=virtual,
        remote_engine_id="ar-engine",
        remote_request_id=request_id,
        remote_pcp_size=1,
        remote_dcp_size=1,
        remote_ptp_size=4,
        remote_port=10000,
        remote_host="ar-host",
        remote_multi_nodes_meta_mapping={},
    )


def _fanout_worker(receiver, ranks):
    worker = SimpleNamespace()
    worker.kv_recv_thread = receiver
    worker._is_hma_required = False
    worker.pcp_size = worker.dcp_size = 1
    worker._prefill_tp_size = 4
    worker._get_remote_ranks_for_req = mock.Mock(return_value=ranks)
    worker._get_remote_host_info_by_port = mock.Mock(return_value=("ar-host", "ar-engine"))
    return worker


@pytest.mark.parametrize(
    "ranks,expected",
    [
        ([[0], [1], [2], [3]], {10000: 2, 10001: 2, 10002: 2, 10003: 2}),
        ([[0], [0], [1], [1]], {10000: 4, 10001: 4}),
        ([[0, 1], [2, 3]], {10000: 2, 10001: 2, 10002: 2, 10003: 2}),
        ([[0]], None),
    ],
)
def test_cfg_ack_counts_include_all_consumers_and_tp_ranks(ranks, expected):
    receiver = mock.Mock()
    worker = _fanout_worker(receiver, ranks)
    metadata = SimpleNamespace(requests={"cfg0": _source_meta(), "cfg1": _source_meta()})
    if expected is None:
        metadata.requests = {"a": _source_meta("a"), "virtual": _source_meta("a", virtual=True), "b": _source_meta("b")}
    adapter = connector_module._MooncakeReceiveFanout(worker, metadata, receiver)
    for request_id in metadata.requests:
        adapter.add_request(request_id=request_id, remote_port_send_num=None, local_block_ids=([5],))
    for call in receiver.add_request.call_args_list:
        counts = call.kwargs["remote_port_send_num"]
        if expected is None:
            assert counts is None
        else:
            assert {port: value["num"] for port, value in counts.items()} == expected
            assert all(value["host"] == "ar-host" for value in counts.values())
        assert call.kwargs["local_block_ids"] == ([5],)
    assert adapter.task_tracker is receiver.task_tracker


@pytest.mark.parametrize("fail", [False, True])
def test_fanout_submission_restores_native_receiver(ascend_connector, fail):
    active, native = ascend_connector
    receiver = mock.Mock()
    native_module = pytest.importorskip("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector")
    worker = object.__new__(native_module.MooncakeConnectorWorker)
    worker.__dict__.update(vars(_fanout_worker(receiver, [[0]])))
    native.connector_worker = worker
    metadata = SimpleNamespace(requests={"cfg0": _source_meta(), "cfg1": _source_meta()})

    def submit(_):
        worker.kv_recv_thread.add_request(request_id="cfg0", remote_port_send_num=None)
        if fail:
            raise RuntimeError("submission failed")
        worker.kv_recv_thread.add_request(request_id="cfg1", remote_port_send_num=None)

    native.start_load_kv.side_effect = submit
    scheduler = SimpleNamespace(kv_connector_metadata=metadata, has_sync_kv_loads=True)
    if fail:
        with pytest.raises(RuntimeError, match="submission failed"):
            active.pre_forward(scheduler)
    else:
        active.pre_forward(scheduler)
    assert worker.kv_recv_thread is receiver
    assert receiver.add_request.call_args_list[0].kwargs["remote_port_send_num"][10000]["num"] == 2
