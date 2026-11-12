# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Joint KV/HS protocol checks with deterministic Store and native KV seams."""

import hashlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import transformers.configuration_utils
from transformers import Qwen3OmniMoeConfig
from vllm import SamplingParams
from vllm.config import CacheConfig, DeviceConfig, KVTransferConfig, SchedulerConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BlobBlockHashes,
    LoadSpec,
    MooncakeLookupResult,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
    TailKeyBoundary,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    KVCacheStoreSendingThread,
    LookupKeyClient,
    MooncakeStoreWorker,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import CrossAttentionSpec, FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.distributed.kv_transfer.mooncake import connector as connector_module
from vllm_omni.distributed.kv_transfer.mooncake.config import MooncakeConfig
from vllm_omni.distributed.kv_transfer.mooncake.connector import (
    Attempt,
    JointMetadata,
    JointStoreWorker,
    OmniMooncakeStoreConnector,
    PendingSave,
    cache_namespace,
)
from vllm_omni.distributed.kv_transfer.mooncake.hs import RequestHS, encode_hs, hs_key

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class MemoryStore:
    """Pinned Store's immutable duplicate-put behavior without native buffers."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_put: set[str] = set()
        self.batch_reads: list[int] = []

    def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def get_batch(self, keys: list[str]) -> list[bytes]:
        # The pinned binding returns empty bytes for each missing key.
        self.batch_reads.append(len(keys))
        return [self.objects.get(key, b"") for key in keys]

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
class LookupClientSeam:
    discarded: list[str] = field(default_factory=list)

    def discard(self, req_id: str) -> None:
        self.discarded.append(req_id)


@dataclass
class SchedulerSeam:
    load_specs: dict[str, LoadSpec] = field(default_factory=dict)
    client: LookupClientSeam = field(default_factory=LookupClientSeam)


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
    hf_config: HFConfigSeam | Qwen3OmniMoeConfig = field(default_factory=HFConfigSeam)
    session_mode: str = "turn"
    engine_output_type: str = "latent"
    hf_text_config: object | None = None
    dtype: torch.dtype = torch.bfloat16
    quantization: str | None = None


@dataclass
class ParallelConfigSeam:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    data_parallel_index: int = 0


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
    quant_config: object | None = None


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
    connector._role = KVConnectorRole.WORKER
    connector._lock = threading.RLock()
    connector._read_executor = ManualExecutor()
    connector._save_executor = ManualExecutor()
    connector._attempts = {}
    connector._reads = {}
    connector._saves = {}
    connector._reserved_bytes = 0
    connector._scheduler_attempts = {}
    connector._lookup_started = {}
    connector._lookup_probe = None
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
    connector._read_executor.run()
    result = connector.get_transfer_results(set())
    assert result.finished_recving == {attempt.req_id}
    assert result.failed_recving == set()
    assert attempt.rows is not None
    assert connector._reserved_bytes == 0


@pytest.mark.parametrize("damage", ["missing", "truncated", "wrong_commit", "failed_batch"])
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
    elif damage == "wrong_commit":
        store.objects[key + ":ready"] = b"other-writer"
    else:
        # The pinned binding's result when its client is unavailable.
        store.get_batch = lambda keys: [b""]
    begin_read(connector, attempt)
    connector._read_executor.run()
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
    connector._read_executor.run()
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
    connector._read_executor.run()
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


def test_full_save_queue_preserves_output_and_stops_later_saves(connector):
    attempt = install_attempt(connector)
    connector.settings = replace(connector.settings, max_pending_transfers=1)
    queued: Future[None] = Future()
    connector._saves[queued] = PendingSave(attempt, [], 0)
    output = connector.capture_step(attempt.req_id, attempt.token, 0, 9, captures(0, 4))
    assert torch.equal(output["embedding"], captures(0, 4)["embedding"])
    assert not connector._save_executor.jobs
    assert attempt.rows.buffered_bytes == 0
    # Lookups stop at the dropped block, so later blocks are not saved even
    # once the queue has room again.
    del connector._saves[queued]
    output = connector.capture_step(attempt.req_id, attempt.token, 4, 9, captures(4, 8))
    assert torch.equal(output["embedding"], captures(4, 8)["embedding"])
    assert not connector._save_executor.jobs


def test_admission_gap_preserves_previously_queued_prefix(connector):
    attempt = install_attempt(connector)
    connector.settings = replace(connector.settings, max_pending_transfers=1)
    store = connector.connector_worker.store
    store.objects["kv:" + attempt.hashes[0].hex()] = b"kv"
    connector.capture_step(attempt.req_id, attempt.token, 0, 9, captures(0, 4))
    connector.capture_step(attempt.req_id, attempt.token, 4, 9, captures(4, 8))
    assert not attempt.rows.save_enabled
    connector._save_executor.run().result()
    assert hs_key(connector.namespace, attempt.hashes[0]) + ":ready" in store.objects
    assert hs_key(connector.namespace, attempt.hashes[1]) not in store.objects


class LateKVStore(MemoryStore):
    """The first block's KV object appears only at the second existence poll."""

    def __init__(self, late_kv_key: str) -> None:
        super().__init__()
        self.late_kv_key = late_kv_key
        self.polls = 0
        self.records: list[str] = []
        self.records_before_late_kv: int | None = None

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        self.polls += 1
        if self.polls == 2:
            self.records_before_late_kv = len(self.records)
            self.objects[self.late_kv_key] = b"kv"
        return super().batch_is_exist(keys)

    def put(self, key: str, payload: bytes) -> int:
        if key.endswith(":ready"):
            self.records.append(key)
        return super().put(key, payload)


def test_save_publishes_completion_records_in_block_order(connector):
    attempt = install_attempt(connector)
    first_kv, second_kv = ("kv:" + block_hash.hex() for block_hash in attempt.hashes[:2])
    store = LateKVStore(first_kv)
    store.objects[second_kv] = b"kv"
    connector.connector_worker = WorkerSeam(store)
    connector._save_hs(PendingSave(attempt, [(0, 4, captures(0, 4)), (4, 8, captures(4, 8))], 0))
    # The second block's KV existed first, but its record waited for the first.
    assert store.records_before_late_kv == 0
    assert store.records == [hs_key(connector.namespace, block_hash) + ":ready" for block_hash in attempt.hashes[:2]]


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
    connector._save_hs(PendingSave(attempt, [(0, 4, captures(0, 4))], 0))
    key = hs_key(connector.namespace, block_hash)
    assert store.objects[key + ":ready"] == hashlib.sha256(store.objects[key]).digest()


def test_failed_save_cannot_publish_completion(connector):
    attempt = install_attempt(connector)
    key = hs_key(connector.namespace, attempt.hashes[0])
    store = connector.connector_worker.store
    store.fail_put.add(key)
    connector._save_hs(PendingSave(attempt, [(0, 4, captures(0, 4))], 0))
    assert key not in store.objects
    assert key + ":ready" not in store.objects


@pytest.mark.parametrize("failure", ["put", "readback", "timeout", "ready", "remove", "exception"])
def test_save_failure_stops_already_queued_and_future_saves(connector, monkeypatch, failure):
    attempt = install_attempt(connector, prompt_len=13)
    store = connector.connector_worker.store
    first, second = (hs_key(connector.namespace, block_hash) for block_hash in attempt.hashes[:2])
    for block_hash in attempt.hashes:
        store.objects["kv:" + block_hash.hex()] = b"kv"
    if failure in ("put", "ready"):
        store.fail_put.add(first if failure == "put" else first + ":ready")
    elif failure == "remove":
        store.objects[first + ":ready"] = b"stale"
        monkeypatch.setattr(store, "remove", lambda *_args, **_kwargs: -1)
    elif failure == "timeout":
        del store.objects["kv:" + attempt.hashes[0].hex()]
        connector.settings = replace(connector.settings, load_timeout_seconds=0.01)
    else:
        original_get = store.get

        def failed_read(key):
            if key == first:
                if failure == "exception":
                    raise OSError("HS read-back failed")
                return b""
            return original_get(key)

        monkeypatch.setattr(store, "get", failed_read)
    for start in (0, 4):
        output = connector.capture_step(attempt.req_id, attempt.token, start, 13, captures(start, start + 4))
        assert torch.equal(output["embedding"], captures(start, start + 4)["embedding"])
    connector._save_executor.run(0)
    connector._save_executor.run(1).result()
    assert not attempt.rows.save_enabled
    assert second not in store.objects
    assert second + ":ready" not in store.objects
    assert store.objects.get(first + ":ready") == (b"stale" if failure == "remove" else None)
    output = connector.capture_step(attempt.req_id, attempt.token, 8, 13, captures(8, 12))
    assert torch.equal(output["hidden"], captures(8, 12)["hidden"])
    assert len(connector._save_executor.jobs) == 2
    connector.get_transfer_results(set())
    assert connector._reserved_bytes == 0


def test_save_failure_preserves_successful_leading_blocks(connector):
    attempt = install_attempt(connector, prompt_len=13)
    store = connector.connector_worker.store
    keys = [hs_key(connector.namespace, block_hash) for block_hash in attempt.hashes]
    for block_hash in attempt.hashes:
        store.objects["kv:" + block_hash.hex()] = b"kv"
    store.fail_put.add(keys[1])
    connector.capture_step(attempt.req_id, attempt.token, 0, 13, captures(0, 8))
    connector.capture_step(attempt.req_id, attempt.token, 8, 13, captures(8, 12))
    connector._save_executor.run(0).result()
    connector._save_executor.run(1).result()
    assert keys[0] + ":ready" in store.objects
    assert keys[1] not in store.objects
    assert keys[2] not in store.objects
    assert connector._reserved_bytes == 0


def test_concurrent_saves_wait_for_the_same_attempts_preceding_save(connector, monkeypatch):
    attempt = install_attempt(connector)
    store = connector.connector_worker.store
    first, second = (hs_key(connector.namespace, block_hash) for block_hash in attempt.hashes[:2])
    store.fail_put.add(first)
    store.objects["kv:" + attempt.hashes[1].hex()] = b"kv"
    first_writing, second_started, release = (threading.Event() for _ in range(3))
    write = connector._write_hs_block
    save_hs = connector._save_hs

    def blocked_write(store, owner, block):
        if block[0] == 0:
            first_writing.set()
            assert release.wait(timeout=5)
        return write(store, owner, block)

    def observe_save(save, *args):
        if save.blocks[0][0] == 4:
            second_started.set()
        return save_hs(save, *args)

    monkeypatch.setattr(connector, "_write_hs_block", blocked_write)
    monkeypatch.setattr(connector, "_save_hs", observe_save)
    with ThreadPoolExecutor(max_workers=2) as executor:
        connector._save_executor = executor
        try:
            connector.capture_step(attempt.req_id, attempt.token, 0, 9, captures(0, 4))
            assert first_writing.wait(timeout=5)
            connector.capture_step(attempt.req_id, attempt.token, 4, 9, captures(4, 8))
            assert second_started.wait(timeout=5)
        finally:
            release.set()
        for future in connector._saves:
            future.result(timeout=5)
    assert second not in store.objects
    assert not attempt.rows.save_enabled
    assert connector._reserved_bytes == 0


def test_duplicate_save_publishes_actual_immutable_winner(connector):
    attempt = install_attempt(connector)
    save_prefix(connector, attempt, 1)
    key = hs_key(connector.namespace, attempt.hashes[0])
    original = connector.connector_worker.store.objects[key]
    different = {name: tensor + 3 for name, tensor in captures(0, 4).items()}
    connector._save_hs(PendingSave(attempt, [(0, 4, different)], 0))
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
    connector._read_executor.run()
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
    connector._save_hs(PendingSave(attempt, [(0, 4, captures(0, 4))], 0))
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
    connector._save_executor.run()
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


@pytest.mark.parametrize("configured,resolved", [(True, False), (False, True), (True, True)])
def test_quantization_rejected_for_an_otherwise_identical_checkpoint(supported_config, configured, resolved):
    from vllm.config.quantization import resolve_quantization_config
    from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig

    config, kv_config = supported_config
    config.model_config.hf_config = Qwen3OmniMoeConfig()
    OmniMooncakeStoreConnector._validate_supported(config, kv_config)
    if configured:
        config.model_config.quantization = "fp8_per_tensor"
    if resolved:
        args = resolve_quantization_config("fp8_per_tensor", None)
        assert args is not None
        config.quant_config = OnlineQuantizationConfig(args)
    # The checkpoint config, activation dtype and KV layout remain unchanged;
    # only runtime quantization differs from the accepted unquantized setup.
    with pytest.raises(ValueError, match="quantized Thinker"):
        OmniMooncakeStoreConnector._validate_supported(config, kv_config)


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


def test_queued_saves_do_not_take_read_admission_slots(connector):
    connector.settings = replace(connector.settings, max_pending_transfers=1)
    writer = install_attempt(connector, req_id="writer", token="writer")
    connector._saves[Future()] = PendingSave(writer, [], 0)
    reader = install_attempt(connector, req_id="reader", token="reader")
    save_prefix(connector, reader, 2)
    pending = begin_read(connector, reader)
    assert not pending.failed
    assert pending.future is not None


class KVPollStore(MemoryStore):
    """Signals once a save has polled every expected, never-written KV key."""

    def __init__(self, kv_keys: list[str]) -> None:
        super().__init__()
        self.kv_keys = set(kv_keys)
        self.polled: set[str] = set()
        self.saves_waiting = threading.Event()

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        self.polled.update(self.kv_keys.intersection(keys))
        if self.polled == self.kv_keys:
            self.saves_waiting.set()
        return super().batch_is_exist(keys)


def test_saves_waiting_for_kv_do_not_delay_a_complete_read(connector):
    writers = [
        Attempt(f"writer-{index}", f"writer-{index}", 9, [f"w{index}-{block}".encode() for block in range(3)])
        for index in range(2)
    ]
    store = KVPollStore(["kv:" + writer.hashes[0].hex() for writer in writers])
    connector.connector_worker = WorkerSeam(store)
    reader = install_attempt(connector, req_id="reader", token="reader")
    save_prefix(connector, reader, 2)
    connector._read_executor = ThreadPoolExecutor(max_workers=2)
    connector._save_executor = ThreadPoolExecutor(max_workers=2)
    try:
        for writer in writers:
            connector._attempts[writer.req_id] = writer
            connector.capture_step(writer.req_id, writer.token, 0, 9, captures(0, 4))
        # Both save workers are now polling for KV objects that never appear.
        assert store.saves_waiting.wait(timeout=5)
        assert connector._reserved_bytes == 0
        begin_read(connector, reader)
        connector.native_result = KVConnectorTransferResults(finished_recving={reader.token})
        deadline = time.monotonic() + 5
        finished: set[str] = set()
        while not finished:
            assert time.monotonic() < deadline, "a complete read waited behind saves"
            result = connector.get_transfer_results(set())
            assert not result.failed_recving
            finished = result.finished_recving
            time.sleep(0.001)
        assert finished == {reader.req_id}
        assert reader.rows is not None
        assert connector._saves and not any(future.done() for future in connector._saves)
        assert connector._reserved_bytes == 0
    finally:
        for writer in writers:
            writer.cancelled = True
        connector._read_executor.shutdown(wait=True)
        connector._save_executor.shutdown(wait=True)


@pytest.mark.parametrize("successor", [None, "retry"])
def test_cancelled_load_frees_gpu_blocks_while_its_hs_read_runs(connector, successor):
    attempt = install_attempt(connector)
    save_prefix(connector, attempt, 2)
    pending = begin_read(connector, attempt)
    connector.bind_connector_metadata(
        joint_metadata(
            omni_retired={attempt.req_id},
            omni_cancelled={attempt.req_id},
            omni_finished=set() if successor else {attempt.req_id},
            omni_attempts={attempt.req_id: successor} if successor else {},
            requests=[],
        )
    )
    connector.native_result = KVConnectorTransferResults(finished_recving={attempt.token})
    first = connector.get_transfer_results(set())
    # The native copy has drained, so an abort can free its destination GPU
    # blocks now. A newer attempt for the same request never receives it.
    assert first.finished_recving == (set() if successor else {attempt.req_id})
    assert first.failed_recving == set()
    assert not pending.future.done()
    assert connector._reserved_bytes == pending.reserved_bytes
    connector._read_executor.run()
    second = connector.get_transfer_results(set())
    assert second.finished_recving == set()
    assert second.failed_recving == set()
    assert connector._reserved_bytes == 0
    assert not connector._reads
    assert attempt.rows is None


@pytest.mark.parametrize(("batch_bytes", "store_calls"), [(16 * 1024 * 1024, [4]), (96, [2, 2])])
def test_hs_read_fetches_objects_and_records_in_batches(connector, monkeypatch, batch_bytes, store_calls):
    monkeypatch.setattr(connector_module, "_READ_BATCH_BYTES", batch_bytes)
    attempt = install_attempt(connector)
    save_prefix(connector, attempt, 2)
    begin_read(connector, attempt)
    connector._read_executor.run()
    assert connector.connector_worker.store.batch_reads == store_calls
    connector.native_result = KVConnectorTransferResults(finished_recving={attempt.token})
    assert connector.get_transfer_results(set()).finished_recving == {attempt.req_id}
    restored = attempt.rows.append(8, captures(8, 9))
    assert torch.equal(restored["embedding"], captures(0, 9)["embedding"])
    assert torch.equal(restored["hidden"], captures(0, 9)["hidden"])


def test_online_connector_reset_is_rejected(connector):
    connector._role = KVConnectorRole.SCHEDULER
    connector.connector_scheduler = Mock(spec=MooncakeStoreScheduler)
    assert connector.reset_cache() is False
    connector.connector_scheduler.reset_store.assert_not_called()
    connector._role = KVConnectorRole.WORKER
    assert connector.reset_cache() is None


def namespace_for(monkeypatch, config, *, version="5.19.0", revision="pinned-model-revision"):
    monkeypatch.setattr(transformers.configuration_utils, "__version__", version)
    settings = MooncakeConfig(config_file="unused.json", namespace="qwen-test", model_revision=revision)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=4, head_size=128, dtype=torch.bfloat16)
    return cache_namespace(settings, config, config.talker_config.accept_hidden_layer, torch.bfloat16, spec)


def loaded_from(path):
    config = Qwen3OmniMoeConfig()
    for owner in (config, config.thinker_config, config.thinker_config.text_config):
        owner._name_or_path = path
    return config


def test_namespace_ignores_load_path_and_transformers_version(monkeypatch):
    local = namespace_for(monkeypatch, loaded_from("/models/qwen3-omni"))
    snapshot = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f7"
    assert namespace_for(monkeypatch, loaded_from(snapshot), version="5.19.1") == local
    assert namespace_for(monkeypatch, loaded_from("/models/qwen3-omni"), revision="other-revision") != local
    nested = loaded_from("/models/qwen3-omni")
    nested.thinker_config.text_config.num_hidden_layers -= 1
    assert namespace_for(monkeypatch, nested) != local
    layer = loaded_from("/models/qwen3-omni")
    layer.talker_config.accept_hidden_layer += 1
    assert namespace_for(monkeypatch, layer) != local


BLOCK16_KV_CONFIG = KVCacheConfig(
    num_blocks=16,
    kv_cache_tensors=[],
    kv_cache_groups=[
        KVCacheGroupSpec(["layer"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16))
    ],
)


class LookupRpcGate:
    """Replaces only the Store RPC of the real lookup client until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.rpcs: list[int] = []

    def __call__(self, num_tokens, block_hashes):
        self.rpcs.append(num_tokens)
        # Bounded so that a failing test cannot leave the client's worker blocked.
        self.release.wait(timeout=10)
        return MooncakeLookupResult(0)


def lookup_scheduler(vllm_config, kv_cache_config):
    """The pinned scheduler's lookup path and real client, without its store coordinator."""
    options = vllm_config.kv_transfer_config.kv_connector_extra_config
    scheduler = MooncakeStoreScheduler.__new__(MooncakeStoreScheduler)
    # Same option defaults as the pinned constructor.
    scheduler.enable_lookup = options.get("enable_lookup", True)
    scheduler.lookup_async = options.get("lookup_async", False)
    scheduler.load_async = options.get("load_async", True)
    scheduler.enable_partial_hash_hits = False
    scheduler._block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
    scheduler._hash_block_size = scheduler._block_size
    scheduler.client = LookupKeyClient(vllm_config)
    scheduler.load_specs = {}
    return scheduler


def scheduler_role_config():
    hf_config = Qwen3OmniMoeConfig()
    settings = {"config_file": "unused.json", "namespace": "qwen-test", "model_revision": "pinned-model-revision"}
    return ServingConfigSeam(
        kv_transfer_config=KVTransferConfig(
            kv_connector="OmniMooncakeStoreConnector",
            kv_load_failure_policy="recompute",
            kv_role="kv_both",
            kv_connector_extra_config={"omni_mooncake": settings},
        ),
        model_config=ModelConfigSeam(hf_config=hf_config, hf_text_config=hf_config.thinker_config.text_config),
    )


@pytest.fixture
def scheduler_role(monkeypatch):
    """Patch the platform and the client's IPC path for scheduler-role construction."""
    monkeypatch.setattr(connector_module, "current_platform", PlatformSeam(cuda=True))
    # The real client connects its ZMQ socket lazily; keep the IPC path short.
    monkeypatch.setenv("VLLM_RPC_BASE_PATH", "/tmp")
    return monkeypatch


@pytest.fixture
def lookup_connector(scheduler_role):
    """A scheduler-role connector from its constructor, using the pinned lookup client."""
    scheduler_role.setattr(connector_module, "MooncakeStoreScheduler", lookup_scheduler)
    connector = OmniMooncakeStoreConnector(scheduler_role_config(), KVConnectorRole.SCHEDULER, BLOCK16_KV_CONFIG)
    client = connector.connector_scheduler.client
    gate = LookupRpcGate()
    # lookup(), discard() and the executor stay those of the pinned client.
    scheduler_role.setattr(client, "_lookup", gate)
    yield connector, gate
    gate.release.set()
    client.close()


def test_scheduler_role_uses_the_pinned_async_lookup_client(lookup_connector):
    connector, _ = lookup_connector
    options = connector._kv_transfer_config.kv_connector_extra_config
    assert options["lookup_async"] is True
    assert options["load_async"] is True
    assert options["cache_prefix"] == connector.namespace
    assert connector.connector_scheduler.lookup_async
    assert type(connector.connector_scheduler.client) is LookupKeyClient


@pytest.mark.parametrize("change", ["two_lookup_workers", "synchronous_lookup"])
def test_constructor_rejects_lookup_clients_that_break_the_stall_guard(scheduler_role, change):
    clients = []

    def changed_scheduler(vllm_config, kv_cache_config):
        scheduler = lookup_scheduler(vllm_config, kv_cache_config)
        clients.append(scheduler.client)
        if change == "two_lookup_workers":
            scheduler.client.executor.shutdown()
            scheduler.client.executor = ThreadPoolExecutor(max_workers=2)
        else:
            scheduler.lookup_async = False
        return scheduler

    scheduler_role.setattr(connector_module, "MooncakeStoreScheduler", changed_scheduler)
    try:
        with pytest.raises(ValueError, match="one at a time"):
            OmniMooncakeStoreConnector(scheduler_role_config(), KVConnectorRole.SCHEDULER, BLOCK16_KV_CONFIG)
    finally:
        for client in clients:
            client.close()


def wait_for_lookup(connector, request, timeout=5.0):
    deadline = time.monotonic() + timeout
    while (result := connector.get_num_new_matched_tokens(request, 0))[0] is None:
        assert time.monotonic() < deadline, "the lookup did not finish"
        time.sleep(0.001)
    return result


def test_lookup_timeout_is_a_miss_until_the_lookup_queue_drains(lookup_connector):
    connector, gate = lookup_connector
    slow = Request("slow", list(range(32)), SamplingParams(max_tokens=1), None)
    queued = Request("queued", list(range(32)), SamplingParams(max_tokens=1), None)
    assert connector.get_num_new_matched_tokens(slow, 0) == (None, False)
    assert connector.get_num_new_matched_tokens(queued, 0) == (None, False)
    # Age the stalled lookup past its deadline instead of sleeping.
    connector._lookup_started[slow.request_id] -= connector.settings.load_timeout_seconds
    assert connector.get_num_new_matched_tokens(slow, 0) == (0, False)
    # A lookup queued behind the stalled reply is cancelled, not waited for.
    assert connector.get_num_new_matched_tokens(queued, 0) == (0, False)
    assert connector.connector_scheduler.client.futures == {}
    assert gate.rpcs == [32]
    gate.release.set()
    connector._lookup_probe.result(timeout=5)
    # Once the stalled reply returns, lookups reach the Store again; the
    # timed-out request keeps computing without the cache.
    assert wait_for_lookup(connector, queued) == (0, False)
    assert connector.get_num_new_matched_tokens(slow, 0) == (0, False)
    assert gate.rpcs == [32, 32]


def ar_scheduler(connector):
    """Production AR scheduler and KV allocator; only lookups reach the connector."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.model_config = SimpleNamespace(
        multimodal_config=None,
        uses_mrope=False,
        is_encoder_decoder=False,
        max_model_len=64,
        runner_type="generate",
        is_diffusion=False,
        supports_multimodal_inputs=False,
        return_sampling_mask=False,
        recompute_preemption="fail",
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=64, max_num_seqs=4, max_num_batched_tokens=128, is_encoder_decoder=False
    )
    config.cache_config = CacheConfig(block_size=16, enable_prefix_caching=False)
    config.cache_config.num_gpu_blocks = 16
    register_all_kvcache_specs(config)
    scheduler = OmniARScheduler(config, BLOCK16_KV_CONFIG, Mock(spec=StructuredOutputManager), 16)
    native = Mock(spec=KVConnectorBase_V1)
    native.requires_kv_delivery = False
    native.get_num_new_matched_tokens.side_effect = connector.get_num_new_matched_tokens
    native.has_pending_block_frees.return_value = False
    native.has_pending_push_work.return_value = False
    native.request_finished.return_value = (False, None)
    native.build_connector_meta.return_value = None
    native.take_events.return_value = ()
    native.get_kv_connector_stats.return_value = None
    scheduler.connector = native
    return scheduler


def sampled(*requests):
    ids = [request.request_id for request in requests]
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(ids)},
        sampled_token_ids=[[1] for _ in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_stalled_lookup_does_not_delay_a_decoding_request(lookup_connector):
    connector, gate = lookup_connector
    scheduler = ar_scheduler(connector)
    decoding = Request("decoding", list(range(8)), SamplingParams(max_tokens=16), None)
    scheduler.add_request(decoding)
    step = scheduler.schedule()
    # Shorter than one block, so this request never queries the Store.
    assert step.num_scheduled_tokens == {"decoding": 8}
    scheduler.update_from_output(step, sampled(decoding))

    stalled = Request("stalled", list(range(32)), SamplingParams(max_tokens=1), None)
    scheduler.add_request(stalled)
    for _ in range(2):
        started = time.monotonic()
        step = scheduler.schedule()
        # A synchronous lookup would hold this step until the Store replied.
        assert time.monotonic() - started < 5
        assert step.num_scheduled_tokens == {"decoding": 1}
        scheduler.update_from_output(step, sampled(decoding))
    assert gate.rpcs == [32]

    # Past its deadline, the stalled request computes its whole prompt. A
    # request arriving during the stall does not queue behind it.
    connector._lookup_started[stalled.request_id] -= connector.settings.load_timeout_seconds
    arriving = Request("arriving", list(range(32)), SamplingParams(max_tokens=1), None)
    scheduler.add_request(arriving)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"decoding": 1, "stalled": 32, "arriving": 32}
    assert gate.rpcs == [32]


def test_rejected_connector_reset_fails_the_scheduler_reset(lookup_connector):
    connector, _ = lookup_connector
    scheduler = ar_scheduler(connector)
    scheduler.connector.reset_cache.side_effect = connector.reset_cache
    # A local reset still succeeds; a connector reset reaches the caller as a
    # failure, which AsyncOmni and the HTTP endpoint report unchanged.
    assert scheduler.reset_prefix_cache() is True
    assert scheduler.reset_prefix_cache(reset_connector=True) is False
