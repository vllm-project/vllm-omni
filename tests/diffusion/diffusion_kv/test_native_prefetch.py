# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from vllm.config import KVTransferConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.outputs import KVConnectorOutput

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.kv_connector import KVReceiveProgress, native_prefetch_enabled
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import DiffusionRequestStatus, RequestScheduler
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _output(*, submit=(), required=(), retire=(), metadata=None):
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=0,
        num_waiting_reqs=0,
        kv_connector_metadata=metadata,
        kv_transfer_request_ids=set(submit),
        kv_required_request_ids=set(required),
        kv_finished_request_ids=set(retire),
        kv_poll_only=metadata is None,
    )


def _completed(*ids, invalid=()):
    return KVConnectorOutput(finished_recving=set(ids), invalid_block_ids=set(invalid))


def test_prefetch_survives_post_forward_and_is_not_submitted_twice():
    progress = KVReceiveProgress()
    active = Mock()
    # B finishes before A, in a different get_finished/post_forward call.
    active.post_forward.side_effect = [_completed("B0"), _completed("A0"), _completed(), _completed()]
    first = _output(submit=["A0", "B0"], required=["A0"], metadata=object())
    assert progress.prepare(active, first, 1).finished_recving == {"A0", "B0"}
    second = _output(required=["B0"], retire=["A0"])
    assert progress.prepare(active, second, 1).finished_recving == {"B0"}
    active.pre_forward.assert_called_once_with(first)
    progress.prepare(active, _output(retire=["B0"]), 1)
    assert not progress.submitted
    assert not progress.received
    assert not progress.deadlines


def test_submit_b_does_not_wait_for_b():
    progress = KVReceiveProgress()
    active = Mock()
    active.post_forward.return_value = _completed("A0")
    result = progress.prepare(active, _output(submit=["A0", "B0"], required=["A0"], metadata=object()), 10)
    assert result.finished_recving == {"A0"}
    assert progress.submitted == {"A0", "B0"}
    active.post_forward.assert_called_once()
    active.post_forward.side_effect = [_completed(), _completed("B0")]
    progress.prepare(active, _output(required=["B0"]), 10)
    assert progress.received == {"A0", "B0"}
    assert active.pre_forward.call_count == 1


def test_duplicate_submission_and_unknown_required_fail_before_transfer():
    progress = KVReceiveProgress(submitted={"B0"})
    active = Mock()
    with pytest.raises(RuntimeError, match="Duplicate"):
        progress.prepare(active, _output(submit=["B0"], metadata=object()), 1)
    with pytest.raises(RuntimeError, match="never submitted"):
        progress.prepare(active, _output(required=["C0"]), 1)
    active.pre_forward.assert_not_called()


def test_poll_checks_timeout_without_resetting_it(monkeypatch):
    now = [1.0]
    monkeypatch.setattr("vllm_omni.diffusion.diffusion_kv.kv_connector.time.monotonic", lambda: now[0])
    progress = KVReceiveProgress()
    active = Mock()
    active.post_forward.side_effect = lambda *_: _completed()
    progress.prepare(active, _output(submit=["B0"], metadata=object()), 5)
    now[0] = 7
    with pytest.raises(TimeoutError, match="B0"):
        progress.prepare(active, _output(), 5)
    assert progress.submitted == {"B0"}


def test_poll_checks_load_errors_and_refuses_early_retirement():
    progress = KVReceiveProgress(submitted={"B0"})
    active = Mock()
    active.post_forward.return_value = _completed(invalid=[42])
    with pytest.raises(RuntimeError, match="invalid remote pages"):
        progress.prepare(active, _output(), 1)
    active.post_forward.return_value = _completed()
    with pytest.raises(RuntimeError, match="unfinished"):
        progress.prepare(active, _output(retire=["B0"]), 1)
    assert progress.submitted == {"B0"}


def test_late_unsubmitted_cancellation_ack_is_not_retained():
    progress = KVReceiveProgress()
    active = Mock()
    active.post_forward.return_value = _completed("cancelled-before-admission")
    result = progress.prepare(active, _output(), 1)
    assert not result.finished_recving
    assert not progress.received


def test_executor_intersects_rank_completions_and_waits_only_required():
    executor = SimpleNamespace(od_config=SimpleNamespace(num_gpus=2), collective_rpc=Mock())
    output = _output(submit=["A0", "B0"], required=["A0"], metadata=object())
    executor.collective_rpc.return_value = [_completed("A0", "B0"), _completed("A0")]
    result = DiffusionExecutor.prepare_kv_for_forward(executor, output)
    assert result.finished_recving == {"A0"}
    assert executor.collective_rpc.call_args.kwargs["exec_all_ranks"] is True
    executor.collective_rpc.return_value = [_completed("A0", "B0"), _completed("A0", "B0")]
    assert DiffusionExecutor.prepare_kv_for_forward(executor, _output()).finished_recving == {"A0", "B0"}
    executor.collective_rpc.return_value = [_completed("B0"), _completed()]
    # Background polling can return early, but compute B requires every rank.
    assert not DiffusionExecutor.prepare_kv_for_forward(executor, _output()).finished_recving
    with pytest.raises(RuntimeError, match="every rank"):
        DiffusionExecutor.prepare_kv_for_forward(executor, _output(required=["B0"]))


def test_executor_submits_prefetch_after_current_rank_barrier():
    calls = []

    def rpc(method, **kwargs):
        output = kwargs["args"][0]
        calls.append(output)
        if len(calls) == 1:
            assert output.kv_transfer_request_ids == output.kv_required_request_ids == {"A0"}
            assert output.has_sync_kv_loads
        else:
            assert output.kv_transfer_request_ids == {"B0"}
            assert not output.kv_required_request_ids
            assert not output.has_sync_kv_loads
            assert output.kv_connector_metadata == "B metadata"
        # B is still in flight on both ranks. This must not block compute A.
        return [_completed("A0"), _completed("A0")]

    executor = SimpleNamespace(od_config=SimpleNamespace(num_gpus=2), collective_rpc=rpc)
    executor.prepare_kv_for_forward = lambda out: DiffusionExecutor.prepare_kv_for_forward(executor, out)
    output = _output(submit=["A0", "B0"], required=["A0"], metadata="A metadata")
    output.kv_prefetch_connector_metadata = "B metadata"
    output.kv_prefetch_request_ids = {"B0"}
    assert executor.prepare_kv_for_forward(output).finished_recving == {"A0"}
    assert len(calls) == 2


def _config():
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="MooncakeConnector",
            kv_role="kv_consumer",
            engine_id="prefetch-test",
            kv_connector_extra_config={"mooncake_protocol": "tcp", "enable_kv_async_prefetch": True},
        ),
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        diffusion_kv_max_rows_per_request=2,
        max_num_seqs=1,
        model_class_name="TestDiffusionPipeline",
    )


def test_mock_config_does_not_enable_native_prefetch():
    assert not native_prefetch_enabled(Mock())


@pytest.mark.parametrize("enabled", ["true", "false", 0, 1, None])
def test_native_prefetch_rejects_non_boolean_flag(enabled):
    config = _config()
    config.kv_transfer_config.kv_connector_extra_config["enable_kv_async_prefetch"] = enabled
    with pytest.raises(ValueError, match="must be a boolean"):
        native_prefetch_enabled(config)


def test_opt_in_and_platform_validation(monkeypatch):
    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform.is_cuda", lambda: True)
    config = _config()
    assert native_prefetch_enabled(config)
    config.model_class_name = "AnotherDiffusionPipeline"
    assert native_prefetch_enabled(config)
    config.max_num_seqs = 2
    with pytest.raises(ValueError, match="max_num_seqs=1"):
        native_prefetch_enabled(config)
    config.max_num_seqs = 1
    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform.is_cuda", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        native_prefetch_enabled(config)
    config.kv_transfer_config.kv_connector_extra_config["enable_kv_async_prefetch"] = False
    assert not native_prefetch_enabled(config)
    assert not native_prefetch_enabled(SimpleNamespace())


class _Connector:
    def __init__(self):
        self.commits = []
        self.metas = []

    def get_num_new_matched_tokens(self, request, computed):
        return 4, True

    def update_state_after_alloc(self, request, blocks, tokens):
        self.commits.append(request.request_id)
        self.metas.append(request.request_id)
        # Upstream Mooncake mutates this flag after staging the receive.
        request.kv_transfer_params["do_remote_prefill"] = False

    def build_connector_meta(self, output):
        meta, self.metas = tuple(self.metas), []
        return meta

    def update_connector_output(self, output):
        pass

    def request_finished(self, request, blocks):
        return False, None


def _scheduler(monkeypatch, num_blocks=24):
    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform.is_cuda", lambda: True)
    connector = _Connector()
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_kv.kv_connector.create_scheduler_kv_connector", lambda *args: connector
    )
    native_kv_managers.register_all_kvcache_specs(None)
    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * num_blocks,
                layers=["layer0"],
                layer_stride=spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=spec)],
    )
    scheduler = RequestScheduler()
    scheduler.initialize(
        _config(),
        kv_cache_config=config,
        scheduler_block_size=4,
        hash_block_size=4,
        kv_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64),
            max_in_flight_tokens=64,
            cache_config=SimpleNamespace(enable_prefix_caching=False),
        ),
    )
    return scheduler, connector


def _request(rid):
    return OmniDiffusionRequest(
        prompt=rid,
        request_id=rid,
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        diffusion_kv_requests=tuple(
            DiffusionKVRequest(
                f"{rid}/{row}", sequence_id=row, prefix_len=4, target_len=4, seq_len=8, prompt_token_ids=[1] * 4
            )
            for row in range(2)
        ),
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": f"xfer-{rid}",
            "remote_engine_id": "ar",
            "remote_bootstrap_addr": "localhost:8998",
            "num_transfer_tokens": 4,
        },
    )


def test_scheduler_prefetches_one_waiting_request_and_reuses_allocation(monkeypatch):
    scheduler, connector = _scheduler(monkeypatch)
    for rid in ("A", "B", "C"):
        scheduler.add_request(_request(rid))
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["A"]
    assert output.kv_transfer_request_ids == {"A/0", "A/1", "B/0", "B/1"}
    assert output.kv_required_request_ids == {"A/0", "A/1"}
    assert output.kv_connector_metadata == ("A/0", "A/1")
    assert output.kv_prefetch_connector_metadata == ("B/0", "B/1")
    assert scheduler.get_request_state("B").status == DiffusionRequestStatus.WAITING
    manager = scheduler._diffusion_kv_manager
    b_meta = manager.get_metadata("B")
    assert not manager.has_request("C")
    a_blocks = {b for row in manager.get_metadata("A").sequences for group in row.block_ids for b in group}
    b_blocks = {b for row in b_meta.sequences for group in row.block_ids for b in group}
    assert a_blocks.isdisjoint(b_blocks)
    scheduler.update_kv_connector_output(_completed("A/0", "A/1", "B/0", "B/1"))
    scheduler.finish_requests("A", DiffusionRequestStatus.FINISHED_COMPLETED)
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["B"]
    assert manager.get_metadata("B") is b_meta
    assert output.kv_transfer_request_ids == {"C/0", "C/1"}
    assert not output.kv_required_request_ids
    assert output.kv_finished_request_ids == {"A/0", "A/1"}
    assert connector.commits.count("B/0") == connector.commits.count("B/1") == 1


def test_prefetch_capacity_failure_rolls_back_all_cfg_rows(monkeypatch):
    # One sentinel + A's four pages + only one of B's two CFG rows.
    scheduler, connector = _scheduler(monkeypatch, num_blocks=7)
    for rid in ("A", "B"):
        scheduler.add_request(_request(rid))
    output = scheduler.schedule()
    assert output.kv_transfer_request_ids == {"A/0", "A/1"}
    assert not scheduler._diffusion_kv_manager.has_request("B")
    assert scheduler._native_prefetch_request_id is None
    assert connector.commits == ["A/0", "A/1"]


def test_abort_drains_b_before_releasing_pages(monkeypatch):
    scheduler, _ = _scheduler(monkeypatch)
    for rid in ("A", "B"):
        scheduler.add_request(_request(rid))
    scheduler.schedule()
    scheduler.update_kv_connector_output(_completed("A/0", "A/1"))
    scheduler.finish_requests("B", DiffusionRequestStatus.FINISHED_ABORTED)
    assert "B" in scheduler._kv_draining_requests
    assert scheduler._diffusion_kv_manager.has_request("B")
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = scheduler
    engine._remove_diffusion_kv_requests = Mock()
    engine.abort_queue = SimpleNamespace(empty=lambda: True)

    def drain(output):
        assert output.kv_required_request_ids == {"B/0", "B/1"}
        assert scheduler._diffusion_kv_manager.has_request("B")
        engine._remove_diffusion_kv_requests.assert_not_called()
        return _completed("A/0", "A/1", "B/0", "B/1")

    engine.executor = SimpleNamespace(prepare_kv_for_forward=drain)
    engine._abort_requests(["B"])
    assert not scheduler._diffusion_kv_manager.has_request("B")
    assert scheduler.get_request_state("B").status == DiffusionRequestStatus.FINISHED_ABORTED
    assert scheduler._native_prefetch_request_id is None


def test_pending_prefetch_reuses_transfer_when_admitted(monkeypatch):
    scheduler, connector = _scheduler(monkeypatch)
    for rid in ("A", "B"):
        scheduler.add_request(_request(rid))
    scheduler.schedule()
    scheduler.update_kv_connector_output(_completed("A/0", "A/1", "B/0"))
    assert "B" in scheduler._kv_loading_request_ids
    scheduler.finish_requests("A", DiffusionRequestStatus.FINISHED_COMPLETED)
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["B"]
    assert output.kv_required_request_ids == {"B/0", "B/1"}
    assert not output.kv_transfer_request_ids
    assert len(connector.commits) == 4


def test_missing_source_skips_prefetch_without_allocating(monkeypatch):
    scheduler, connector = _scheduler(monkeypatch)
    scheduler.add_request(_request("A"))
    b = _request("B")
    del b.kv_transfer_params["remote_bootstrap_addr"]
    scheduler.add_request(b)
    scheduler.schedule()
    assert not scheduler._diffusion_kv_manager.has_request("B")
    assert connector.commits == ["A/0", "A/1"]


def test_prefetch_reservation_error_does_not_fail_current_request(monkeypatch):
    scheduler, connector = _scheduler(monkeypatch)
    for rid in ("A", "B"):
        scheduler.add_request(_request(rid))
    manager = scheduler._diffusion_kv_manager
    reserve = manager.reserve_request

    def fail_b(request_id, requests):
        if request_id == "B":
            raise ValueError("invalid B layout")
        return reserve(request_id, requests)

    monkeypatch.setattr(manager, "reserve_request", fail_b)
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["A"]
    assert connector.commits == ["A/0", "A/1"]
    assert not manager.has_request("B")
    scheduler.update_kv_connector_output(_completed("A/0", "A/1"))
    scheduler.finish_requests("A", DiffusionRequestStatus.FINISHED_COMPLETED)
    output = scheduler.schedule()
    assert scheduler.get_request_state("B").status == DiffusionRequestStatus.FINISHED_ERROR
    assert "B" in output.finished_req_ids


def test_invalid_prefetch_boundary_is_rejected_before_reservation(monkeypatch):
    scheduler, connector = _scheduler(monkeypatch)
    for rid in ("A", "B"):
        scheduler.add_request(_request(rid))
    b_rows = scheduler.get_request_state("B").diffusion_kv_requests
    b_rows[1].kv_transfer_params["num_transfer_tokens"] = b_rows[1].num_tokens + 1
    manager = scheduler._diffusion_kv_manager
    reserve = manager.reserve_request

    def fail_if_b_reserved(request_id, requests):
        if request_id == "B":
            raise AssertionError("invalid B must be rejected before reservation")
        return reserve(request_id, requests)

    monkeypatch.setattr(manager, "reserve_request", fail_if_b_reserved)
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["A"]
    assert scheduler.get_request_state("B").status == DiffusionRequestStatus.WAITING
    assert not manager.has_request("B")


def test_abort_timeout_keeps_target_pages(monkeypatch):
    scheduler, _ = _scheduler(monkeypatch)
    for rid in ("A", "B"):
        scheduler.add_request(_request(rid))
    scheduler.schedule()
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = scheduler
    engine.executor = SimpleNamespace(prepare_kv_for_forward=Mock(side_effect=TimeoutError("receive")))
    engine._fail_engine = Mock()
    engine._remove_diffusion_kv_requests = Mock()
    with pytest.raises(TimeoutError):
        engine._abort_requests(["B"])
    engine._fail_engine.assert_called_once()
    engine._remove_diffusion_kv_requests.assert_not_called()
    assert scheduler._diffusion_kv_manager.has_request("B")


def test_sleep_rejects_live_prefetch_before_touching_allocator():
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    worker = SimpleNamespace(model_runner=SimpleNamespace(_kv_receive_progress=KVReceiveProgress(submitted={"B/0"})))
    with pytest.raises(RuntimeError, match="Cannot sleep"):
        DiffusionWorker.sleep(worker)


def test_registration_failure_retires_b_and_flushes_cancellation(monkeypatch):
    from vllm_omni.diffusion.diffusion_kv import kv_connector

    scheduler, connector = _scheduler(monkeypatch)
    for rid in ("A", "B"):
        scheduler.add_request(_request(rid))
    commit = kv_connector.commit_kv_load

    def fail_b(conn, manager, requests, matched):
        if requests[0].request_id.startswith("B/"):
            conn.metas.append("B cancellation")
            raise kv_connector.KVTransferRegistrationError("registration rollback")
        return commit(conn, manager, requests, matched)

    monkeypatch.setattr("vllm_omni.diffusion.sched.base_scheduler.commit_kv_load", fail_b)
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["A"]
    assert output.finished_req_ids == {"B"}
    assert output.kv_finished_request_ids == {"B/0", "B/1"}
    assert output.kv_prefetch_connector_metadata == ("B cancellation",)
    assert not output.kv_prefetch_request_ids
    assert output.num_waiting_reqs == 0
    assert scheduler.get_request_state("B").status == DiffusionRequestStatus.FINISHED_ERROR
    assert not scheduler._diffusion_kv_manager.has_request("B")
    assert scheduler._diffusion_kv_manager.has_request("A")


def test_partial_cfg_registration_rolls_back_every_row(monkeypatch):
    from vllm_omni.diffusion.diffusion_kv import kv_connector

    class FakeMooncakeConnector:
        def __init__(self):
            self.staged = []
            self.cancelled = []

        def update_state_after_alloc(self, request, blocks, tokens):
            self.staged.append(request.request_id)
            if request.request_id.endswith("/1"):
                raise RuntimeError("second CFG row failed")

        def request_finished(self, request, blocks):
            self.cancelled.append(request.request_id)
            return False, None

    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.MooncakeConnector",
        FakeMooncakeConnector,
    )
    scheduler, _ = _scheduler(monkeypatch)
    scheduler.add_request(_request("B"))
    state = scheduler.get_request_state("B")
    rows = state.diffusion_kv_requests
    for row in rows:
        row.kv_transfer_params = dict(state.req.kv_transfer_params)
    allocation = scheduler._diffusion_kv_manager.reserve_request("B", rows)
    connector = FakeMooncakeConnector()

    with pytest.raises(kv_connector.KVTransferRegistrationError, match="Could not register"):
        kv_connector.commit_kv_load(
            connector,
            scheduler._diffusion_kv_manager.native_manager,
            rows,
            [4, 4],
        )
    assert allocation is not None
    assert connector.staged == ["B/0", "B/1"]
    assert connector.cancelled == ["B/0", "B/1"]


def test_drain_after_terminal_request_state_was_popped(monkeypatch):
    scheduler, _ = _scheduler(monkeypatch)
    for rid in ("A", "B"):
        scheduler.add_request(_request(rid))
    scheduler.schedule()
    scheduler.finish_requests("B", DiffusionRequestStatus.FINISHED_ABORTED)
    scheduler.pop_request_state("B")
    output = scheduler.native_kv_poll_output(drain_request_ids=["B"])
    assert output.kv_required_request_ids == {"B/0", "B/1"}
    assert scheduler._diffusion_kv_manager.has_request("B")


@pytest.mark.parametrize("prefetch", [False, True])
def test_real_active_connector_starts_required_load_before_polling(prefetch, monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    from vllm.v1.worker.gpu import kv_connector as native_kv_connector

    from vllm_omni.diffusion.diffusion_kv.kv_connector import wait_for_kv_load

    connector = Mock(spec=KVConnectorBase_V1)
    connector.get_block_ids_with_load_errors.return_value = set()
    # Exercise upstream's compatibility adapter instead of returning an
    # unconstrained Mock from the v0.30 get_transfer_results API.
    connector.get_transfer_results.side_effect = lambda ids: KVConnectorBase_V1.get_transfer_results(connector, ids)
    monkeypatch.setattr(native_kv_connector, "get_kv_transfer_group", lambda: connector)
    context = object()
    monkeypatch.setattr(native_kv_connector, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(native_kv_connector, "get_forward_context", lambda: context)
    active = native_kv_connector.ActiveKVConnector(Mock(), {})

    completions: Iterator[tuple[set[str], set[str]]] = iter([(set(), {"A0"}), (set(), set())])

    def finished(_):
        connector.start_load_kv.assert_called_once_with(context)
        return next(completions)

    connector.get_finished.side_effect = finished
    output = _output(submit=["A0"], required=["A0"], metadata=object())
    if prefetch:
        result = KVReceiveProgress().prepare(active, output, 1)
    else:
        output.kv_required_request_ids = None
        result = wait_for_kv_load(active, output, 1)
    assert result.finished_recving == {"A0"}
    assert result.finished_sending == set()
    connector.get_transfer_results.assert_called_once_with(set())
    connector.start_load_kv.assert_called_once_with(context)
