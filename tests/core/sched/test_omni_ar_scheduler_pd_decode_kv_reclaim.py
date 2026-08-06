from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _MockBlockPool:
    def __init__(self, sink: list) -> None:
        self._sink = sink

    def free_blocks(self, blocks) -> None:
        self._sink.extend(blocks)


class _MockKVCacheManager:
    def __init__(self, sink: list) -> None:
        self._sink = sink
        self.block_pool = _MockBlockPool(sink)

    def free(self, request) -> None:
        self._sink.extend(request.blocks)

    def pop_blocks_for_free(self, request) -> list:
        return request.blocks


def _mock_request(request_id: str, last_sched_seq: int):
    return SimpleNamespace(
        request_id=request_id,
        blocks=[f"{request_id}-b0", f"{request_id}-b1"],
        last_sched_seq=last_sched_seq,
    )


class _DeferHarness:
    """Exercises the real upstream defer/drain methods around our fix.

    Reusing ``Scheduler._free_request_blocks`` / ``_drain_deferred_frees``
    verbatim keeps this test honest: it validates upstream's actual fencing
    semantics rather than a reimplementation of them.
    """

    _free_request_blocks = VLLMScheduler._free_request_blocks
    _drain_deferred_frees = VLLMScheduler._drain_deferred_frees

    def __init__(self, *, defer_block_free: bool) -> None:
        self.defer_block_free = defer_block_free
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        self.deferred_frees: deque = deque()
        self.returned_to_pool: list = []
        self.kv_cache_manager = _MockKVCacheManager(self.returned_to_pool)

    def run_steps(self, num_steps: int, *, drain: bool) -> None:
        """Simulate ``num_steps`` schedule/finish cycles.

        ``drain=True`` replicates the block this fix adds to
        ``OmniARScheduler.update_from_output``.
        """
        for i in range(num_steps):
            # Upstream advances the fence for every non-empty scheduled step.
            self.sched_step_seq += 1
            self._free_request_blocks(_mock_request(f"r{i}", self.sched_step_seq))
            if drain:
                self.processed_step_seq += 1
                self._drain_deferred_frees()


def test_missing_drain_leaks_every_freed_block():
    """Without the drain, deferred frees accumulate and never reach the pool."""
    harness = _DeferHarness(defer_block_free=True)
    harness.run_steps(50, drain=False)

    assert harness.returned_to_pool == []
    assert len(harness.deferred_frees) == 50


def test_drain_returns_all_blocks_to_pool():
    """With the drain, every freed block is reclaimed and nothing is parked."""
    harness = _DeferHarness(defer_block_free=True)
    harness.run_steps(50, drain=True)

    assert len(harness.returned_to_pool) == 100
    assert not harness.deferred_frees


def test_non_consumer_stage_frees_immediately():
    """defer_block_free=False (prefill / single-node) bypasses the queue."""
    harness = _DeferHarness(defer_block_free=False)
    harness.sched_step_seq = 5
    harness._free_request_blocks(_mock_request("x", 5))

    assert len(harness.returned_to_pool) == 2
    assert not harness.deferred_frees


def _cap_scheduler(
    kv_role: str | None,
    *,
    num_blocks: int = 32076,
    block_size: int = 16,
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
):
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    kv_transfer_config = SimpleNamespace(kv_role=kv_role) if kv_role else None
    scheduler.vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    # kv_cache_groups=None forces the flat block-math fallback, which is what
    # runs when upstream's group-aware helper cannot be used.
    scheduler.kv_cache_config = SimpleNamespace(num_blocks=num_blocks, kv_cache_groups=None)
    scheduler.cache_config = SimpleNamespace(num_gpu_blocks=num_blocks)
    scheduler.block_size = block_size
    scheduler.max_model_len = max_model_len
    scheduler.max_num_running_reqs = max_num_seqs
    scheduler.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
    return scheduler


def test_decode_consumer_capped_to_kv_capacity():
    """513,216 KV tokens / max_model_len 4096 -> 125, down from max_num_seqs 256."""
    scheduler = _cap_scheduler("kv_consumer")
    scheduler._maybe_cap_running_reqs_for_pd_decode()

    assert scheduler.max_num_running_reqs == 125
    # The worker sizes its batch from max_num_seqs; it must stay untouched.
    assert scheduler.scheduler_config.max_num_seqs == 256


@pytest.mark.parametrize("block_size", [16, 64, 128])
def test_cap_is_block_size_invariant(block_size):
    scheduler = _cap_scheduler("kv_consumer", num_blocks=513216 // block_size, block_size=block_size)
    scheduler._maybe_cap_running_reqs_for_pd_decode()

    assert scheduler.max_num_running_reqs == 125


@pytest.mark.parametrize("kv_role", [None, "kv_producer"])
def test_prefill_and_single_node_are_untouched(kv_role):
    """Only the decode (kv_consumer) side may be capped."""
    scheduler = _cap_scheduler(kv_role)
    scheduler._maybe_cap_running_reqs_for_pd_decode()

    assert scheduler.max_num_running_reqs == 256


def test_cap_never_raises_configured_limit():
    scheduler = _cap_scheduler("kv_consumer", max_num_seqs=64)
    scheduler._maybe_cap_running_reqs_for_pd_decode()

    assert scheduler.max_num_running_reqs == 64


def test_env_override_sets_and_disables_cap(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_PD_DECODE_MAX_RUNNING", "4")
    scheduler = _cap_scheduler("kv_consumer")
    scheduler._maybe_cap_running_reqs_for_pd_decode()
    assert scheduler.max_num_running_reqs == 4

    monkeypatch.setenv("VLLM_OMNI_PD_DECODE_MAX_RUNNING", "0")
    scheduler = _cap_scheduler("kv_consumer")
    scheduler._maybe_cap_running_reqs_for_pd_decode()
    assert scheduler.max_num_running_reqs == 256


def test_degenerate_kv_config_leaves_limit_unchanged():
    """Unknown capacity must not cap (and must not raise)."""
    scheduler = _cap_scheduler("kv_consumer", num_blocks=0)
    scheduler.cache_config.num_gpu_blocks = 0
    scheduler._maybe_cap_running_reqs_for_pd_decode()

    assert scheduler.max_num_running_reqs == 256
