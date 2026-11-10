# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Joint KV/HS protocol checks with deterministic Store and native KV seams."""

import hashlib
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field, replace

import pytest
import torch
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BlobBlockHashes,
    LoadSpec,
    MooncakeLookupResult,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
    TailKeyBoundary,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    KVCacheStoreSendingThread,
    MooncakeStoreWorker,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import CrossAttentionSpec, FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_omni.distributed.kv_transfer.mooncake.config import MooncakeConfig
from vllm_omni.distributed.kv_transfer.mooncake.connector import (
    Attempt,
    JointMetadata,
    JointStoreWorker,
    OmniMooncakeStoreConnector,
)
from vllm_omni.distributed.kv_transfer.mooncake.hs import RequestHS, encode_hs, hs_key

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class MemoryStore:
    """Pinned Store's immutable duplicate-put behavior without native buffers."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_put: set[str] = set()

    def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def put(self, key: str, payload: bytes) -> int:
        if key in self.fail_put:
            return -1
        self.objects.setdefault(key, payload)
        return 0

    def is_exist(self, key: str) -> int:
        return int(key in self.objects)

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [self.is_exist(key) for key in keys]

    def remove(self, key: str, force: bool = False) -> int:
        return 0 if self.objects.pop(key, None) is not None else -1


@dataclass
class TokenDatabase:
    prefix: str = "kv:"

    def key_for(self, block_hash: bytes) -> str:
        return self.prefix + block_hash.hex()


@dataclass
class WorkerSeam:
    store: MemoryStore
    token_dbs: list[TokenDatabase] = field(default_factory=lambda: [TokenDatabase()])


@dataclass
class SchedulerSeam:
    load_specs: dict[str, LoadSpec] = field(default_factory=dict)


@dataclass
class PlatformSeam:
    cuda: bool

    def is_cuda(self) -> bool:
        return self.cuda


@dataclass
class HFConfigSeam:
    model_type: str = "qwen3_omni_moe"


@dataclass
class ModelConfigSeam:
    model_stage: str = "thinker"
    hf_config: HFConfigSeam = field(default_factory=HFConfigSeam)
    session_mode: str = "turn"
    engine_output_type: str = "latent"


@dataclass
class ParallelConfigSeam:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    prefill_context_parallel_size: int = 1


@dataclass
class CacheConfigSeam:
    enable_prefix_caching: bool = False


@dataclass
class ServingConfigSeam:
    kv_transfer_config: KVTransferConfig
    use_v2_model_runner: bool = True
    model_config: ModelConfigSeam = field(default_factory=ModelConfigSeam)
    parallel_config: ParallelConfigSeam = field(default_factory=ParallelConfigSeam)
    cache_config: CacheConfigSeam = field(default_factory=CacheConfigSeam)
    speculative_config: object | None = None
    lora_config: object | None = None


class ManualExecutor:
    """A submitted copy completes only when the test explicitly advances it."""

    def __init__(self):
        self.jobs = []

    def submit(self, function, *args):
        future: Future[object] = Future()
        self.jobs.append((future, function, args))
        return future

    def run(self, index=0):
        future, function, args = self.jobs[index]
        try:
            result = function(*args)
        except (ValueError, RuntimeError, OSError, TypeError) as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)
        return future


def captures(start, end):
    rows = torch.arange(start * 3, end * 3, dtype=torch.float32).reshape(end - start, 3)
    return {"embedding": rows, "hidden": rows + 100}


@pytest.fixture
def connector(monkeypatch):
    connector = OmniMooncakeStoreConnector.__new__(OmniMooncakeStoreConnector)
    connector._closed = True  # No real native resources to close in __del__.
    connector._lock = threading.RLock()
    connector._executor = ManualExecutor()
    connector._attempts = {}
    connector._reads = {}
    connector._saves = {}
    connector._reserved_bytes = 0
    connector._scheduler_attempts = {}
    connector._bypass = set()
    connector._retired = set()
    connector._cancelled = set()
    connector._delivered_until = {}
    connector.settings = MooncakeConfig(
        config_file="unused-in-unit-test.json",
        namespace="test-model-revision",
        model_revision="pinned-model-revision",
        max_hs_bytes=1024 * 1024,
        max_pending_transfers=4,
        load_timeout_seconds=30,
    )
    connector.namespace = "test-model-revision"
    connector.block_size = 4
    connector.hidden_layer = 24
    connector.hidden_size = 3
    connector.hs_dtype = torch.float32
    connector._row_bytes = 24
    connector.connector_worker = WorkerSeam(MemoryStore())
    connector.connector_scheduler = SchedulerSeam()
    connector.native_result = KVConnectorTransferResults()
    connector.native_retired = set()
    connector.native_loads = 0

    def poll(self, retired):
        self.native_retired.update(retired)
        result, self.native_result = self.native_result, KVConnectorTransferResults()
        return result

    def start(self, _context, **_kwargs):
        self.native_loads += 1

    monkeypatch.setattr(MooncakeStoreConnector, "get_transfer_results", poll)
    monkeypatch.setattr(MooncakeStoreConnector, "get_block_ids_with_load_errors", lambda _self: set())
    monkeypatch.setattr(MooncakeStoreConnector, "start_load_kv", start)
    monkeypatch.setattr(MooncakeStoreConnector, "update_connector_output", lambda *_: None)
    monkeypatch.setattr(
        MooncakeStoreConnector, "bind_connector_metadata", lambda self, meta: setattr(self, "meta", meta)
    )
    monkeypatch.setattr(MooncakeStoreConnector, "_get_connector_metadata", lambda self: self.meta)
    return connector


def install_attempt(connector, *, req_id="request", token="attempt", prompt_len=9):
    attempt = Attempt(req_id, token, prompt_len, [b"first", b"second", b"third"])
    connector._attempts[req_id] = attempt
    return attempt


def save_prefix(connector, attempt, blocks):
    store = connector.connector_worker.store
    for index, block_hash in enumerate(attempt.hashes[:blocks]):
        payload = encode_hs(connector._spec(block_hash, index), captures(index * 4, (index + 1) * 4))
        key = hs_key(connector.namespace, block_hash)
        store.objects[key] = payload
        store.objects[key + ":ready"] = hashlib.sha256(payload).digest()
        store.objects["kv:" + block_hash.hex()] = b"kv"


def begin_read(connector, attempt, tokens=8):
    connector.meta = MooncakeStoreConnectorMetadata({attempt.token}, set())
    connector.meta.requests.append(
        ReqMeta(
            req_id=attempt.token,
            token_len_chunk=tokens,
            block_hashes=attempt.hashes,
            num_prompt_tokens=attempt.prompt_len,
            load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=tokens, can_load=True),
            block_ids=([11, 12, 13],),
        )
    )
    connector.start_load_kv(None)
    return connector._reads[attempt.token]


def joint_metadata(*, omni_retired, omni_cancelled, omni_finished, omni_attempts, requests):
    native = MooncakeStoreConnectorMetadata(set(omni_attempts), omni_retired - omni_finished)
    native.requests = requests
    return JointMetadata(native, omni_attempts, omni_finished, omni_cancelled)


def capture_meta(attempt):
    return ReqMeta(
        req_id=attempt.req_id,
        token_len_chunk=0,
        block_ids=([11, 12, 13],),
        block_hashes=attempt.hashes,
        num_prompt_tokens=attempt.prompt_len,
    )


def test_lookup_stops_at_missing_hs_pair_and_clamps_native_boundary(monkeypatch):
    worker = JointStoreWorker.__new__(JointStoreWorker)
    worker.hs_namespace, worker.block_size, worker.max_hs_tokens = "ns", 4, 20
    worker.store = MemoryStore()
    hashes = [b"a", b"b", b"c"]
    for block_hash in hashes:
        worker.store.objects[hs_key("ns", block_hash)] = b"hs"
        worker.store.objects[hs_key("ns", block_hash) + ":ready"] = b"ready"
    del worker.store.objects[hs_key("ns", b"b") + ":ready"]
    monkeypatch.setattr(MooncakeStoreWorker, "lookup", lambda *_: MooncakeLookupResult(12, (TailKeyBoundary(0, 12),)))
    result = worker.lookup(12, hashes)
    assert result.hit_length == 4
    assert result.tail_key_boundaries == (TailKeyBoundary(0, 4),)


def test_lookup_caps_read_memory_at_complete_block(monkeypatch):
    worker = JointStoreWorker.__new__(JointStoreWorker)
    worker.hs_namespace, worker.block_size, worker.max_hs_tokens = "ns", 4, 7
    worker.store = MemoryStore()
    for block_hash in (b"a", b"b"):
        worker.store.objects[hs_key("ns", block_hash)] = b"hs"
        worker.store.objects[hs_key("ns", block_hash) + ":ready"] = b"ready"
    monkeypatch.setattr(MooncakeStoreWorker, "lookup", lambda *_: MooncakeLookupResult(8, (TailKeyBoundary(0, 8),)))
    assert worker.lookup(8, [b"a", b"b"]).hit_length == 4


def test_lookup_rpc_hash_views_match_scheduler_byte_keys(monkeypatch):
    worker = JointStoreWorker.__new__(JointStoreWorker)
    worker.hs_namespace, worker.block_size, worker.max_hs_tokens = "ns", 4, 8
    worker.store = MemoryStore()
    for block_hash in (b"aaaa", b"bbbb"):
        worker.store.objects[hs_key("ns", block_hash)] = b"hs"
        worker.store.objects[hs_key("ns", block_hash) + ":ready"] = b"ready"
    monkeypatch.setattr(MooncakeStoreWorker, "lookup", lambda *_: MooncakeLookupResult(8, (TailKeyBoundary(0, 8),)))
    hashes = BlobBlockHashes(memoryview(b"aaaabbbb"), 4)
    assert worker.lookup(8, hashes).hit_length == 8


def test_kv_completion_waits_for_hs_before_readmission(connector):
    attempt = install_attempt(connector)
    save_prefix(connector, attempt, 2)
    pending = begin_read(connector, attempt)
    connector.native_result = KVConnectorTransferResults(finished_recving={attempt.token})
    result = connector.get_transfer_results(set())
    assert result.finished_recving == set()
    assert attempt.rows is None
    assert pending.kv_done
    connector._executor.run()
    result = connector.get_transfer_results(set())
    assert result.finished_recving == {attempt.req_id}
    assert result.failed_recving == set()
    assert attempt.rows is not None
    assert connector._reserved_bytes == 0


@pytest.mark.parametrize("damage", ["missing", "truncated", "wrong_commit"])
def test_invalid_hs_reports_failure_and_waits_for_native_copy(connector, damage):
    attempt = install_attempt(connector)
    save_prefix(connector, attempt, 2)
    key = hs_key(connector.namespace, attempt.hashes[0])
    store = connector.connector_worker.store
    if damage == "missing":
        del store.objects[key]
    elif damage == "truncated":
        store.objects[key] = store.objects[key][:-1]
        store.objects[key + ":ready"] = hashlib.sha256(store.objects[key]).digest()
    else:
        store.objects[key + ":ready"] = b"other-writer"
    begin_read(connector, attempt)
    connector._executor.run()
    first = connector.get_transfer_results(set())
    assert first.failed_recving == {attempt.req_id}
    assert first.finished_recving == set()
    assert attempt.rows is None
    assert connector._reads
    connector.native_result = KVConnectorTransferResults(finished_recving={attempt.token})
    second = connector.get_transfer_results(set())
    assert second.finished_recving == {attempt.req_id}
    assert second.failed_recving == set()
    assert connector._reserved_bytes == 0
    assert not connector._reads


def test_timeout_drains_gpu_before_retry_and_keeps_live_cpu_reservation(connector):
    attempt = install_attempt(connector)
    save_prefix(connector, attempt, 2)
    pending = begin_read(connector, attempt)
    pending.deadline = time.monotonic() - 1
    first = connector.get_transfer_results(set())
    assert first.failed_recving == {attempt.req_id}
    assert first.finished_recving == set()
    assert connector._reserved_bytes == pending.reserved_bytes
    connector.native_result = KVConnectorTransferResults(finished_recving={attempt.token})
    second = connector.get_transfer_results(set())
    assert second.finished_recving == {attempt.req_id}
    assert connector._reserved_bytes == pending.reserved_bytes
    assert attempt.rows is None
    connector._executor.run()
    third = connector.get_transfer_results(set())
    assert third.finished_recving == set()
    assert connector._reserved_bytes == 0
    assert not connector._reads


def test_failed_load_bypasses_lookup_for_affected_request_only(connector, monkeypatch):
    calls = []

    def lookup(_self, request, _count):
        calls.append(request.request_id)
        return 8, True

    monkeypatch.setattr(
        MooncakeStoreConnector,
        "get_num_new_matched_tokens",
        lookup,
    )
    connector._scheduler_attempts["bad"] = "first-attempt"
    connector.connector_scheduler.load_specs["bad"] = LoadSpec(0, 8, True)
    connector.update_connector_output(KVConnectorOutput(failed_recving={"bad"}))
    assert connector.get_num_new_matched_tokens(
        Request("bad", [1, 2], SamplingParams(max_tokens=1), pooling_params=None), 0
    ) == (0, False)
    assert "bad" not in connector.connector_scheduler.load_specs
    assert connector.get_num_new_matched_tokens(
        Request("other", [1, 2], SamplingParams(max_tokens=1), pooling_params=None), 0
    ) == (8, True)
    assert calls == ["other"]


def test_scheduler_metadata_retains_native_contract_and_attempt_ownership(connector, monkeypatch):
    native = MooncakeStoreConnectorMetadata({"active"}, {"preempted"})
    monkeypatch.setattr(MooncakeStoreConnector, "build_connector_meta", lambda *_: native)
    connector._scheduler_attempts = {"finished": "old-finished", "preempted": "old-preempted"}
    connector._bypass = {"finished", "preempted"}
    connector._cancelled = {"finished"}
    step = SchedulerOutput.make_empty()
    step.finished_req_ids = {"finished"}
    step.preempted_req_ids = {"preempted"}
    step.num_scheduled_tokens = {"active": 1}

    metadata = connector.build_connector_meta(step)

    assert isinstance(metadata, JointMetadata)
    assert isinstance(metadata, MooncakeStoreConnectorMetadata)
    assert metadata.requests is native.requests
    assert metadata.unfinished_request_ids == {"active"}
    assert set(metadata.omni_attempts) == {"active"}
    assert metadata.preempted_req_ids == {"preempted"}
    assert metadata.omni_finished == {"finished"}
    assert metadata.omni_cancelled == {"finished", "preempted"}
    assert not connector._bypass
    assert not connector._cancelled


def test_cancellation_and_request_id_reuse_reject_old_completion(connector):
    old = install_attempt(connector, token="old")
    save_prefix(connector, old, 2)
    begin_read(connector, old)
    connector.bind_connector_metadata(
        joint_metadata(
            omni_retired={old.req_id},
            omni_cancelled={old.req_id},
            omni_finished={old.req_id},
            omni_attempts={old.req_id: "new"},
            requests=[],
        )
    )
    assert old.cancelled
    assert connector.current_attempt(old.req_id) == "new"
    assert connector.capture_step(old.req_id, "old", 0, 9, captures(0, 4)) is None
    connector._executor.run()
    connector.native_result = KVConnectorTransferResults(finished_recving={"old"})
    result = connector.get_transfer_results(set())
    assert result.finished_recving == set()
    assert result.failed_recving == set()
    assert "old" in connector.native_retired
    assert connector._attempts[old.req_id].rows is None
    assert connector._reserved_bytes == 0


def test_capture_first_partial_and_full_hit_have_each_prompt_row_once(connector):
    cold = install_attempt(connector, req_id="cold", token="cold", prompt_len=9)
    output = connector.capture_step("cold", "cold", 0, 9, captures(0, 5))
    assert torch.equal(output["embedding"], captures(0, 5)["embedding"])
    output = connector.capture_step("cold", "cold", 5, 9, captures(5, 10))
    assert torch.equal(output["hidden"], captures(5, 10)["hidden"])
    partial = install_attempt(connector, req_id="partial", token="partial", prompt_len=9)
    partial.rows = RequestHS(9, 4, 4096)
    partial.rows.restore(captures(0, 4))
    output = connector.capture_step("partial", "partial", 4, 9, captures(4, 10))
    assert torch.equal(output["embedding"], captures(0, 10)["embedding"])
    full = install_attempt(connector, req_id="full", token="full", prompt_len=8)
    full.rows = RequestHS(8, 4, 4096)
    full.rows.restore(captures(0, 8))
    output = connector.capture_step("full", "full", 7, 8, captures(7, 9))
    assert torch.equal(output["hidden"], captures(0, 9)["hidden"])
    output = connector.capture_step("full", "full", 9, 8, captures(9, 10))
    assert torch.equal(output["embedding"], captures(9, 10)["embedding"])
    assert cold.rows is not None


def test_full_save_queue_preserves_output_without_extra_task(connector):
    attempt = install_attempt(connector)
    connector.settings = replace(connector.settings, max_pending_transfers=1)
    connector._saves[Future()] = 0
    output = connector.capture_step(attempt.req_id, attempt.token, 0, 9, captures(0, 4))
    assert torch.equal(output["embedding"], captures(0, 4)["embedding"])
    assert not connector._executor.jobs
    assert attempt.rows.buffered_bytes == 0


def test_hs_admission_failure_still_waits_for_native_gpu_copy(connector):
    attempt = install_attempt(connector)
    connector.settings = replace(connector.settings, max_hs_bytes=1)
    pending = begin_read(connector, attempt)
    assert pending.future is None
    assert connector.native_loads == 1
    first = connector.get_transfer_results(set())
    assert first.failed_recving == {attempt.req_id}
    assert first.finished_recving == set()
    connector.native_result = KVConnectorTransferResults(finished_recving={attempt.token})
    second = connector.get_transfer_results(set())
    assert second.finished_recving == {attempt.req_id}
    assert attempt.rows is None


def test_save_publishes_commit_after_kv_and_hs_exist(connector):
    attempt = install_attempt(connector)
    block_hash = attempt.hashes[0]
    store = connector.connector_worker.store
    store.objects["kv:" + block_hash.hex()] = b"kv"
    connector._save_hs(attempt, [(0, 4, captures(0, 4))])
    key = hs_key(connector.namespace, block_hash)
    assert store.objects[key + ":ready"] == hashlib.sha256(store.objects[key]).digest()


def test_failed_save_cannot_publish_completion(connector):
    attempt = install_attempt(connector)
    key = hs_key(connector.namespace, attempt.hashes[0])
    store = connector.connector_worker.store
    store.fail_put.add(key)
    connector._save_hs(attempt, [(0, 4, captures(0, 4))])
    assert key not in store.objects
    assert key + ":ready" not in store.objects


def test_duplicate_save_publishes_actual_immutable_winner(connector):
    attempt = install_attempt(connector)
    save_prefix(connector, attempt, 1)
    key = hs_key(connector.namespace, attempt.hashes[0])
    original = connector.connector_worker.store.objects[key]
    different = {name: tensor + 3 for name, tensor in captures(0, 4).items()}
    connector._save_hs(attempt, [(0, 4, different)])
    assert connector.connector_worker.store.objects[key] == original
    assert connector.connector_worker.store.objects[key + ":ready"] == hashlib.sha256(original).digest()


def test_preemption_rejects_pending_handoff_and_preserves_delivered_watermark(connector):
    old = install_attempt(connector)
    published = connector.capture_step(old.req_id, old.token, 0, 9, captures(0, 5))
    assert published["embedding"].shape[0] == 5
    connector.bind_connector_metadata(
        joint_metadata(
            omni_retired={old.req_id},
            omni_cancelled={old.req_id},
            omni_finished=set(),
            omni_attempts={old.req_id: "resumed"},
            requests=[capture_meta(old)],
        )
    )
    # This old step was copied to CPU but not published when preemption arrived.
    assert connector.capture_step(old.req_id, old.token, 5, 9, captures(5, 8)) is None
    replay = connector.capture_step(old.req_id, "resumed", 0, 9, captures(0, 4))
    assert replay["embedding"].shape[0] == 0
    new = connector.capture_step(old.req_id, "resumed", 4, 9, captures(4, 9))
    assert torch.equal(new["embedding"], captures(5, 9)["embedding"])


def test_terminal_request_id_reuse_clears_delivered_watermark(connector):
    old = install_attempt(connector)
    connector.capture_step(old.req_id, old.token, 0, 9, captures(0, 4))
    connector.bind_connector_metadata(
        joint_metadata(
            omni_retired={old.req_id},
            omni_cancelled={old.req_id},
            omni_finished={old.req_id},
            omni_attempts={old.req_id: "new"},
            requests=[capture_meta(old)],
        )
    )
    new = connector.capture_step(old.req_id, "new", 0, 9, captures(0, 4))
    assert torch.equal(new["embedding"], captures(0, 4)["embedding"])


def test_native_block_failure_invalidates_joint_attempt(connector, monkeypatch):
    attempt = install_attempt(connector)
    save_prefix(connector, attempt, 2)
    begin_read(connector, attempt)
    connector._executor.run()
    monkeypatch.setattr(MooncakeStoreConnector, "get_block_ids_with_load_errors", lambda _self: {11})
    connector.native_result = KVConnectorTransferResults(finished_recving={attempt.token})
    result = connector.get_transfer_results(set())
    assert result.finished_recving == {attempt.req_id}
    assert result.failed_recving == {attempt.req_id}
    assert attempt.rows is None
    assert connector.get_block_ids_with_load_errors() == set()


def test_write_without_kv_never_publishes_commit(connector, monkeypatch):
    attempt = install_attempt(connector)
    ticks = iter([0.0, 31.0])
    monkeypatch.setattr("vllm_omni.distributed.kv_transfer.mooncake.connector.time.monotonic", lambda: next(ticks))
    connector._save_hs(attempt, [(0, 4, captures(0, 4))])
    key = hs_key(connector.namespace, attempt.hashes[0])
    assert key in connector.connector_worker.store.objects
    assert key + ":ready" not in connector.connector_worker.store.objects


def test_normal_finish_allows_already_queued_save_to_drain(connector):
    attempt = install_attempt(connector)
    connector.connector_worker.store.objects["kv:" + attempt.hashes[0].hex()] = b"kv"
    connector.capture_step(attempt.req_id, attempt.token, 0, 9, captures(0, 4))
    connector.bind_connector_metadata(
        joint_metadata(
            omni_retired={attempt.req_id},
            omni_cancelled=set(),
            omni_finished={attempt.req_id},
            omni_attempts={},
            requests=[],
        )
    )
    assert not attempt.cancelled
    assert connector.capture_step(attempt.req_id, attempt.token, 4, 9, captures(4, 8)) is None
    connector._executor.run()
    connector.get_transfer_results(set())
    key = hs_key(connector.namespace, attempt.hashes[0])
    assert key + ":ready" in connector.connector_worker.store.objects
    assert connector._reserved_bytes == 0


@pytest.mark.parametrize("reason", ["finished", "aborted", "preempted", "retry"])
def test_native_save_job_retirement_preserves_only_normal_finish(connector, reason):
    """Exercise upstream's ledger, which distinguishes finish from preemption."""
    attempt = install_attempt(connector)
    send = KVCacheStoreSendingThread.__new__(KVCacheStoreSendingThread)
    send.done_task_lock = threading.Lock()
    send.stored_requests = {attempt.token: {17}}
    send._completed_saves = {}
    send._saved_offset = {attempt.token: 4}
    send._skip_store_requests = set()
    send._retry_token_ids = {}
    worker = MooncakeStoreWorker.__new__(MooncakeStoreWorker)
    worker.kv_send_thread = send
    worker.finished_store_req = set()
    job = replace(capture_meta(attempt), req_id=attempt.token, store_job_id=17)
    connector.bind_connector_metadata(
        joint_metadata(
            omni_retired=set() if reason == "retry" else {attempt.req_id},
            omni_cancelled={attempt.req_id} if reason in ("aborted", "preempted") else set(),
            omni_finished={attempt.req_id} if reason in ("finished", "aborted") else set(),
            omni_attempts={attempt.req_id: "retry"} if reason == "retry" else {},
            requests=[],
        )
    )
    connector.get_transfer_results(set())
    worker._close_ended_store_requests(connector.native_retired, connector.meta)
    assert send.is_live_store_job(job) == (reason == "finished")
    if reason == "finished":
        assert send._saved_offset[attempt.token] == 4
        assert worker.finished_store_req == {attempt.token}
    send.finish_store_job(job)
    worker._close_ended_store_requests(set(), MooncakeStoreConnectorMetadata(set(), set()))
    assert attempt.token not in send.stored_requests
    assert attempt.token not in send._saved_offset
    assert send.take_completed_saves() == {17: 1}


@pytest.fixture
def supported_config(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.distributed.kv_transfer.mooncake.connector.current_platform", PlatformSeam(cuda=True)
    )
    config = ServingConfigSeam(
        kv_transfer_config=KVTransferConfig(
            kv_connector="OmniMooncakeStoreConnector", kv_load_failure_policy="recompute", kv_role="kv_both"
        )
    )
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=3, dtype=torch.float32)
    return config, KVCacheConfig(
        num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)]
    )


def test_supported_turn_configuration_passes_startup_guard(supported_config):
    OmniMooncakeStoreConnector._validate_supported(*supported_config)


@pytest.mark.parametrize(
    ("owner", "field", "value", "message"),
    [
        (None, "use_v2_model_runner", False, "Model Runner V2"),
        ("parallel_config", "tensor_parallel_size", 2, "single TP=1"),
        ("parallel_config", "pipeline_parallel_size", 2, "single TP=1"),
        ("parallel_config", "data_parallel_size", 2, "single TP=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "single TP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "single TP=1"),
        ("model_config", "model_stage", "talker", "Thinker only"),
        ("model_config", "session_mode", "duplex", "duplex"),
        ("model_config", "engine_output_type", "text", "capture outputs"),
        ("cache_config", "enable_prefix_caching", True, "local prefix caching"),
        (None, "speculative_config", object(), "speculation"),
        (None, "lora_config", object(), "LoRA"),
        ("kv_transfer_config", "kv_load_failure_policy", "fail", "recompute"),
        ("kv_transfer_config", "kv_role", "kv_producer", "kv_both"),
    ],
)
def test_unsupported_serving_configuration_fails_at_startup(supported_config, owner, field, value, message):
    config, kv_config = supported_config
    target = config if owner is None else getattr(config, owner)
    setattr(target, field, value)
    with pytest.raises(ValueError, match=message):
        OmniMooncakeStoreConnector._validate_supported(config, kv_config)


def test_non_cuda_platform_fails_at_startup(supported_config, monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.distributed.kv_transfer.mooncake.connector.current_platform", PlatformSeam(cuda=False)
    )
    with pytest.raises(ValueError, match="CUDA"):
        OmniMooncakeStoreConnector._validate_supported(*supported_config)


@pytest.mark.parametrize(
    "groups",
    [
        [],
        [
            KVCacheGroupSpec(
                ["layer"], CrossAttentionSpec(block_size=4, num_kv_heads=1, head_size=3, dtype=torch.float32)
            )
        ],
    ],
)
def test_unsupported_kv_layout_fails_at_startup(supported_config, groups):
    config, _ = supported_config
    with pytest.raises(ValueError, match="full.attention"):
        OmniMooncakeStoreConnector._validate_supported(
            config, KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=groups)
        )
