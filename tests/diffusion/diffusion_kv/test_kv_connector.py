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
from vllm_omni.diffusion.diffusion_kv.kv_connector import (
    KVTransferRegistrationError,
    commit_kv_load,
    parse_kv_transfer_config,
    shutdown_kv_connector,
    wait_for_kv_load,
)
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
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


def test_commit_aligns_cfg_transport_pages_without_extending_computed_prefix() -> None:
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

    expected = commit_kv_load(connector, manager, requests, [7, 5])

    assert expected == {"req/0", "req/1"}
    calls = connector.update_state_after_alloc.call_args_list
    assert [len(call.args[1].blocks[0]) for call in calls] == [3, 3]
    assert [call.args[2] for call in calls] == [7, 5]
    assert [request.num_computed_tokens for request in requests] == [7, 5]


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
