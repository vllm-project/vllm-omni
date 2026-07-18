"""Test that aborting a request during KV transfer cleans tracking sets
and frees KV blocks.

Regression test: when a request is aborted while a KV transfer is
active (or pending), stale entries in the scheduler's tracking sets
(active_kv_transfers, waiting_for_transfer_free, etc.) cause
has_unfinished_requests() to return True forever, spinning the engine
loop indefinitely and blocking all subsequent requests.
"""

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

BLOCK_SIZE = 16


@pytest.fixture(scope="session", autouse=True)
def _init_hashing():
    init_none_hash(sha256)


NUM_BLOCKS = 10000


def _create_vllm_config(need_send_cache: bool = False) -> VllmConfig:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    if need_send_cache:
        model_config.omni_kv_config = {
            "need_send_cache": True,
            "kv_transfer_criteria": {
                "type": "prefill_finished",
                "stop_after_transfer": True,
            },
        }
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    cache_config.num_gpu_blocks = NUM_BLOCKS
    return VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )


def _create_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def _create_scheduler(need_send_cache: bool = False) -> OmniARScheduler:
    vllm_config = _create_vllm_config(need_send_cache=need_send_cache)
    return OmniARScheduler(
        vllm_config=vllm_config,
        kv_cache_config=_create_kv_cache_config(),
        block_size=BLOCK_SIZE,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _create_request(req_id: str, num_tokens: int = 10) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=[0] * num_tokens,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _make_model_runner_output(scheduler_output, sampled_token_ids: dict[str, list[int]]) -> ModelRunnerOutput:
    """Build a ModelRunnerOutput matching the scheduled requests."""
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
    tokens = [sampled_token_ids.get(rid, []) for rid in req_ids]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=tokens,
    )


def _prefill_and_trigger_kv_transfer(
    sched: OmniARScheduler,
    req_ids: list[str],
) -> None:
    """Schedule and process output to trigger KV transfer for all
    requests.  After this call, requests are RUNNING with deferred
    stop pending (in transfer_triggered_requests and
    pending_stop_after_extraction)."""
    sched_out = sched.schedule()
    model_out = _make_model_runner_output(sched_out, {rid: [42] for rid in req_ids})
    sched.update_from_output(sched_out, model_out)


def _assert_tracking_clean(sched: OmniARScheduler, req_id: str):
    assert req_id not in sched.requests
    assert req_id not in sched.requests_needing_kv_transfer
    assert req_id not in sched.active_kv_transfers
    assert req_id not in sched.transfer_triggered_requests
    assert req_id not in sched.waiting_for_transfer_free
    assert req_id not in sched.pending_stop_after_extraction
    assert req_id not in sched._omits_kv_transfer_cache


class TestAbortDuringKVTransfer:
    def test_abort_cleans_tracking_and_frees_blocks(self):
        """Abort a request after KV transfer was triggered.

        With deferred stop (pending_stop_after_extraction), the
        request stays RUNNING until KV extraction completes.  Abort
        before that happens — the override must clean all tracking
        sets and free KV blocks.
        """
        sched = _create_scheduler(need_send_cache=True)
        req = _create_request("req-1")
        sched.add_request(req)
        free_before = sched.kv_cache_manager.block_pool.get_num_free_blocks()

        _prefill_and_trigger_kv_transfer(sched, ["req-1"])

        # Deferred stop: request stays RUNNING, transfer is pending.
        assert req.status == RequestStatus.RUNNING
        assert "req-1" in sched.transfer_triggered_requests
        assert "req-1" in sched.pending_stop_after_extraction
        free_during = sched.kv_cache_manager.block_pool.get_num_free_blocks()
        assert free_during < free_before

        sched.finish_requests("req-1", RequestStatus.FINISHED_ABORTED)

        _assert_tracking_clean(sched, "req-1")
        assert not sched.has_unfinished_requests()
        free_after = sched.kv_cache_manager.block_pool.get_num_free_blocks()
        assert free_after == free_before

    def test_abort_during_active_extraction(self):
        """Abort while KV extraction is in-flight (active_kv_transfers).

        After the trigger, get_finished_requests_needing_kv_transfer()
        moves the request into active_kv_transfers.  Abort before
        the extraction ack arrives.
        """
        sched = _create_scheduler(need_send_cache=True)
        req = _create_request("req-2")
        sched.add_request(req)
        free_before = sched.kv_cache_manager.block_pool.get_num_free_blocks()

        _prefill_and_trigger_kv_transfer(sched, ["req-2"])
        sched.get_finished_requests_needing_kv_transfer()
        assert "req-2" in sched.active_kv_transfers

        sched.finish_requests("req-2", RequestStatus.FINISHED_ABORTED)

        _assert_tracking_clean(sched, "req-2")
        assert not sched.has_unfinished_requests()
        free_after = sched.kv_cache_manager.block_pool.get_num_free_blocks()
        assert free_after == free_before

    def test_abort_does_not_affect_other_requests(self):
        """Aborting one request must not disturb another's tracking."""
        sched = _create_scheduler(need_send_cache=True)
        req_a = _create_request("req-a")
        req_b = _create_request("req-b")
        sched.add_request(req_a)
        sched.add_request(req_b)

        _prefill_and_trigger_kv_transfer(sched, ["req-a", "req-b"])

        # Deferred stop: both requests are RUNNING with pending transfer.
        assert "req-a" in sched.transfer_triggered_requests
        assert "req-b" in sched.transfer_triggered_requests

        sched.finish_requests("req-a", RequestStatus.FINISHED_ABORTED)

        _assert_tracking_clean(sched, "req-a")
        assert "req-b" in sched.transfer_triggered_requests
        assert "req-b" in sched.pending_stop_after_extraction
        assert sched.has_unfinished_requests()

    def test_abort_already_finished_with_held_blocks(self):
        """Abort a request that finished naturally (EOS/max_tokens)
        while extraction was active.

        The request is already FINISHED_STOPPED in
        waiting_for_transfer_free, so the base class skips it.
        The override must still free the held KV blocks.
        """
        sched = _create_scheduler(need_send_cache=True)
        req = _create_request("req-1")
        sched.add_request(req)
        free_before = sched.kv_cache_manager.block_pool.get_num_free_blocks()

        _prefill_and_trigger_kv_transfer(sched, ["req-1"])
        sched.get_finished_requests_needing_kv_transfer()
        assert "req-1" in sched.active_kv_transfers

        # Simulate request finishing naturally (EOS) while extraction
        # is active: set status, call _free_request (which holds blocks
        # in waiting_for_transfer_free), and remove from running queue.
        req.status = RequestStatus.FINISHED_STOPPED
        sched._free_request(req)
        sched.running = [r for r in sched.running if r is not req]
        assert "req-1" in sched.waiting_for_transfer_free
        free_during = sched.kv_cache_manager.block_pool.get_num_free_blocks()
        assert free_during < free_before

        # Abort — base class skips (already finished), override frees.
        sched.finish_requests("req-1", RequestStatus.FINISHED_ABORTED)

        _assert_tracking_clean(sched, "req-1")
        assert not sched.has_unfinished_requests()
        free_after = sched.kv_cache_manager.block_pool.get_num_free_blocks()
        assert free_after == free_before

    def test_abort_request_without_kv_transfer(self):
        """Aborting a request on a scheduler without KV transfer
        config is a no-op for tracking sets."""
        sched = _create_scheduler(need_send_cache=False)
        req = _create_request("req-3")
        sched.add_request(req)
        sched.schedule()

        result = sched.finish_requests("req-3", RequestStatus.FINISHED_ABORTED)

        assert len(result) == 1
        assert not sched.has_unfinished_requests()

    def test_abort_all_requests_cleans_all_tracking(self):
        """Passing request_ids=None aborts all and cleans all tracking."""
        sched = _create_scheduler(need_send_cache=True)
        for i in range(3):
            sched.add_request(_create_request(f"req-{i}"))
        free_before = sched.kv_cache_manager.block_pool.get_num_free_blocks()

        _prefill_and_trigger_kv_transfer(sched, [f"req-{i}" for i in range(3)])

        sched.finish_requests(None, RequestStatus.FINISHED_ABORTED)

        for i in range(3):
            _assert_tracking_clean(sched, f"req-{i}")
        assert not sched.has_unfinished_requests()
        free_after = sched.kv_cache_manager.block_pool.get_num_free_blocks()
        assert free_after == free_before
