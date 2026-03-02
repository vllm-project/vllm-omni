"""Standalone test for the O(n^2) -> O(n) fix in _process_chunk_queue().

Tests the fix without requiring vllm to be installed, by simulating
the exact logic of the original and fixed _process_chunk_queue().
"""
import time
from collections import deque
from types import SimpleNamespace

# Simulate RequestStatus enum
class RequestStatus:
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    WAITING_FOR_CHUNK = "WAITING_FOR_CHUNK"
    FINISHED_STOPPED = "FINISHED_STOPPED"


def _req(req_id, status, external_req_id=None):
    return SimpleNamespace(
        request_id=req_id,
        external_req_id=external_req_id or req_id,
        status=status,
        prompt_token_ids=[],
        num_computed_tokens=0,
        additional_information=None,
        is_finished=lambda: status == RequestStatus.FINISHED_STOPPED,
    )


class DummyWaitingQueue(list):
    """Simulates the scheduler's waiting queue (list subclass)."""
    def prepend_requests(self, requests):
        self[:0] = list(requests)

    def add_request(self, request):
        self.append(request)


class MockAdapter:
    """Minimal mock of OmniChunkTransferAdapter for testing _process_chunk_queue."""
    def __init__(self):
        self.requests_with_ready_chunks = set()
        self.finished_requests = set()
        self.load_async_calls = []

    def load_async(self, request):
        self.load_async_calls.append(request.request_id)


def _process_chunk_queue_original(adapter, queue, waiting_for_chunk_list,
                                   target_status, finished_load_reqs):
    """ORIGINAL code from chunk_transfer_adapter.py:327-353 (before fix)."""
    queue_snapshot = list(queue)
    for request in queue_snapshot:
        if request.status != RequestStatus.WAITING_FOR_CHUNK:
            if request.request_id in adapter.requests_with_ready_chunks:
                continue
            if request.request_id in adapter.finished_requests:
                continue
            adapter.load_async(request)
            request.status = RequestStatus.WAITING_FOR_CHUNK
        else:
            if request.request_id in finished_load_reqs:
                request.status = target_status
                finished_load_reqs.remove(request.request_id)
                adapter.requests_with_ready_chunks.add(request.request_id)
                continue
        queue.remove(request)
        waiting_for_chunk_list.append(request)


def _process_chunk_queue_fixed(adapter, queue, waiting_for_chunk_list,
                                target_status, finished_load_reqs):
    """FIXED code — single-pass O(n) partition."""
    keep = []
    for request in list(queue):
        if request.status != RequestStatus.WAITING_FOR_CHUNK:
            if request.request_id in adapter.requests_with_ready_chunks:
                keep.append(request)
                continue
            if request.request_id in adapter.finished_requests:
                keep.append(request)
                continue
            adapter.load_async(request)
            request.status = RequestStatus.WAITING_FOR_CHUNK
        else:
            if request.request_id in finished_load_reqs:
                request.status = target_status
                finished_load_reqs.remove(request.request_id)
                adapter.requests_with_ready_chunks.add(request.request_id)
                keep.append(request)
                continue
        waiting_for_chunk_list.append(request)
    queue[:] = keep


# =============================================================================
# Tests
# =============================================================================

def test_all_new_requests_moved():
    """All new WAITING requests should be moved to waiting_for_chunk_list."""
    for impl_name, impl in [("original", _process_chunk_queue_original),
                             ("fixed", _process_chunk_queue_fixed)]:
        adapter = MockAdapter()
        reqs = [_req(f"r{i}", RequestStatus.WAITING) for i in range(5)]
        queue = DummyWaitingQueue(list(reqs))
        waiting = deque()

        impl(adapter, queue, waiting, RequestStatus.WAITING, set())

        assert len(queue) == 0, f"[{impl_name}] queue should be empty"
        assert len(waiting) == 5, f"[{impl_name}] all 5 should be moved"
        for r in waiting:
            assert r.status == RequestStatus.WAITING_FOR_CHUNK
        assert len(adapter.load_async_calls) == 5
    print("  PASS: test_all_new_requests_moved")


def test_ready_chunks_stay_in_queue():
    """Requests with ready chunks should stay in the queue (continue path)."""
    for impl_name, impl in [("original", _process_chunk_queue_original),
                             ("fixed", _process_chunk_queue_fixed)]:
        adapter = MockAdapter()
        adapter.requests_with_ready_chunks = {"r0", "r2", "r4"}
        reqs = [_req(f"r{i}", RequestStatus.WAITING) for i in range(5)]
        queue = DummyWaitingQueue(list(reqs))
        waiting = deque()

        impl(adapter, queue, waiting, RequestStatus.WAITING, set())

        kept_ids = sorted(r.request_id for r in queue)
        moved_ids = sorted(r.request_id for r in waiting)
        assert kept_ids == ["r0", "r2", "r4"], f"[{impl_name}] ready requests should stay"
        assert moved_ids == ["r1", "r3"], f"[{impl_name}] non-ready should be moved"
    print("  PASS: test_ready_chunks_stay_in_queue")


def test_finished_requests_stay_in_queue():
    """Finished requests should stay in the queue (continue path)."""
    for impl_name, impl in [("original", _process_chunk_queue_original),
                             ("fixed", _process_chunk_queue_fixed)]:
        adapter = MockAdapter()
        adapter.finished_requests = {"r1", "r3"}
        reqs = [_req(f"r{i}", RequestStatus.WAITING) for i in range(5)]
        queue = DummyWaitingQueue(list(reqs))
        waiting = deque()

        impl(adapter, queue, waiting, RequestStatus.WAITING, set())

        kept_ids = sorted(r.request_id for r in queue)
        moved_ids = sorted(r.request_id for r in waiting)
        assert kept_ids == ["r1", "r3"], f"[{impl_name}] finished requests should stay"
        assert moved_ids == ["r0", "r2", "r4"], f"[{impl_name}] others should be moved"
    print("  PASS: test_finished_requests_stay_in_queue")


def test_waiting_for_chunk_with_finished_load():
    """WAITING_FOR_CHUNK requests whose load finished should stay and get target_status."""
    for impl_name, impl in [("original", _process_chunk_queue_original),
                             ("fixed", _process_chunk_queue_fixed)]:
        adapter = MockAdapter()
        reqs = [
            _req("r0", RequestStatus.WAITING_FOR_CHUNK),
            _req("r1", RequestStatus.WAITING_FOR_CHUNK),
            _req("r2", RequestStatus.WAITING),
        ]
        finished_load = {"r0"}
        queue = DummyWaitingQueue(list(reqs))
        waiting = deque()

        impl(adapter, queue, waiting, RequestStatus.RUNNING, set(finished_load))

        kept_ids = [r.request_id for r in queue]
        assert "r0" in kept_ids, f"[{impl_name}] r0 should stay (load finished)"
        assert queue[kept_ids.index("r0")].status == RequestStatus.RUNNING

        moved_ids = [r.request_id for r in waiting]
        assert "r1" in moved_ids, f"[{impl_name}] r1 should be moved (still waiting)"
        assert "r2" in moved_ids, f"[{impl_name}] r2 should be moved (new request)"
    print("  PASS: test_waiting_for_chunk_with_finished_load")


def test_mixed_scenario():
    """Complex mixed scenario with all code paths exercised."""
    for impl_name, impl in [("original", _process_chunk_queue_original),
                             ("fixed", _process_chunk_queue_fixed)]:
        adapter = MockAdapter()
        adapter.requests_with_ready_chunks = {"r1"}
        adapter.finished_requests = {"r3"}
        reqs = [
            _req("r0", RequestStatus.WAITING),              # new -> move
            _req("r1", RequestStatus.WAITING),              # has ready chunk -> stay
            _req("r2", RequestStatus.WAITING_FOR_CHUNK),    # still waiting -> move
            _req("r3", RequestStatus.WAITING),              # finished -> stay
            _req("r4", RequestStatus.WAITING_FOR_CHUNK),    # load finished -> stay
        ]
        finished_load = {"r4"}
        queue = DummyWaitingQueue(list(reqs))
        waiting = deque()

        impl(adapter, queue, waiting, RequestStatus.WAITING, set(finished_load))

        kept_ids = sorted(r.request_id for r in queue)
        moved_ids = sorted(r.request_id for r in waiting)
        assert kept_ids == ["r1", "r3", "r4"], f"[{impl_name}] kept={kept_ids}"
        assert moved_ids == ["r0", "r2"], f"[{impl_name}] moved={moved_ids}"

        # r4 should have target_status
        r4 = [r for r in queue if r.request_id == "r4"][0]
        assert r4.status == RequestStatus.WAITING
        assert "r4" in adapter.requests_with_ready_chunks
    print("  PASS: test_mixed_scenario")


def test_with_plain_list_queue():
    """running_queue is a plain list, not DummyWaitingQueue."""
    for impl_name, impl in [("original", _process_chunk_queue_original),
                             ("fixed", _process_chunk_queue_fixed)]:
        adapter = MockAdapter()
        reqs = [_req(f"r{i}", RequestStatus.RUNNING) for i in range(3)]
        queue = list(reqs)  # plain list
        waiting = deque()

        impl(adapter, queue, waiting, RequestStatus.RUNNING, set())

        assert len(queue) == 0, f"[{impl_name}] queue should be empty"
        assert len(waiting) == 3, f"[{impl_name}] all should be moved"
    print("  PASS: test_with_plain_list_queue")


def test_empty_queue():
    """Empty queue should be a no-op."""
    for impl_name, impl in [("original", _process_chunk_queue_original),
                             ("fixed", _process_chunk_queue_fixed)]:
        adapter = MockAdapter()
        queue = DummyWaitingQueue()
        waiting = deque()
        impl(adapter, queue, waiting, RequestStatus.WAITING, set())
        assert len(queue) == 0
        assert len(waiting) == 0
    print("  PASS: test_empty_queue")


def test_original_vs_fixed_equivalence():
    """The fixed version produces identical results to the original for all inputs."""
    import random
    random.seed(42)

    for trial in range(100):
        n = random.randint(0, 50)
        adapter_orig = MockAdapter()
        adapter_fixed = MockAdapter()

        # Random state
        ready = set(f"r{i}" for i in range(n) if random.random() < 0.2)
        finished = set(f"r{i}" for i in range(n) if random.random() < 0.1 and f"r{i}" not in ready)
        adapter_orig.requests_with_ready_chunks = set(ready)
        adapter_orig.finished_requests = set(finished)
        adapter_fixed.requests_with_ready_chunks = set(ready)
        adapter_fixed.finished_requests = set(finished)

        statuses = [RequestStatus.WAITING, RequestStatus.WAITING_FOR_CHUNK]
        reqs_orig = [_req(f"r{i}", random.choice(statuses)) for i in range(n)]
        reqs_fixed = [_req(f"r{i}", reqs_orig[i].status) for i in range(n)]

        finished_load = set(f"r{i}" for i in range(n)
                           if reqs_orig[i].status == RequestStatus.WAITING_FOR_CHUNK
                           and random.random() < 0.3)

        q_orig = DummyWaitingQueue(list(reqs_orig))
        w_orig = deque()
        _process_chunk_queue_original(adapter_orig, q_orig, w_orig,
                                       RequestStatus.WAITING, set(finished_load))

        q_fixed = DummyWaitingQueue(list(reqs_fixed))
        w_fixed = deque()
        _process_chunk_queue_fixed(adapter_fixed, q_fixed, w_fixed,
                                    RequestStatus.WAITING, set(finished_load))

        orig_kept = [(r.request_id, r.status) for r in q_orig]
        fixed_kept = [(r.request_id, r.status) for r in q_fixed]
        orig_moved = [(r.request_id, r.status) for r in w_orig]
        fixed_moved = [(r.request_id, r.status) for r in w_fixed]

        assert orig_kept == fixed_kept, f"Trial {trial}: kept mismatch\n  orig={orig_kept}\n  fixed={fixed_kept}"
        assert orig_moved == fixed_moved, f"Trial {trial}: moved mismatch\n  orig={orig_moved}\n  fixed={fixed_moved}"

    print("  PASS: test_original_vs_fixed_equivalence (100 random trials)")


def test_performance_improvement():
    """Verify O(n^2) vs O(n) scaling on worst-case input."""
    def make_worst_case(n):
        """Half requests stay (ready chunks), half get removed."""
        adapter = MockAdapter()
        reqs = []
        for i in range(n // 2):
            r = _req(f"ready-{i}", RequestStatus.WAITING)
            reqs.append(r)
            adapter.requests_with_ready_chunks.add(r.request_id)
        for i in range(n // 2, n):
            r = _req(f"new-{i}", RequestStatus.WAITING)
            reqs.append(r)
        return adapter, reqs

    # Benchmark at N=5000
    n = 5000
    iters = 5

    # Original
    times_orig = []
    for _ in range(iters):
        adapter, reqs = make_worst_case(n)
        q = DummyWaitingQueue(list(reqs))
        w = deque()
        start = time.perf_counter()
        _process_chunk_queue_original(adapter, q, w, RequestStatus.WAITING, set())
        times_orig.append(time.perf_counter() - start)
    avg_orig = sum(times_orig) / len(times_orig) * 1e6

    # Fixed
    times_fixed = []
    for _ in range(iters):
        adapter, reqs = make_worst_case(n)
        q = DummyWaitingQueue(list(reqs))
        w = deque()
        start = time.perf_counter()
        _process_chunk_queue_fixed(adapter, q, w, RequestStatus.WAITING, set())
        times_fixed.append(time.perf_counter() - start)
    avg_fixed = sum(times_fixed) / len(times_fixed) * 1e6

    speedup = avg_orig / avg_fixed if avg_fixed > 0 else float('inf')
    print(f"  PASS: test_performance_improvement")
    print(f"        N={n}: original={avg_orig:.0f}µs, fixed={avg_fixed:.0f}µs, speedup={speedup:.1f}x")
    assert speedup > 5, f"Expected >5x speedup at N={n}, got {speedup:.1f}x"


def test_order_preservation():
    """Verify that the order of kept elements is preserved."""
    for impl_name, impl in [("original", _process_chunk_queue_original),
                             ("fixed", _process_chunk_queue_fixed)]:
        adapter = MockAdapter()
        # r0, r2, r4 stay (ready chunks), r1, r3 move
        adapter.requests_with_ready_chunks = {"r0", "r2", "r4"}
        reqs = [_req(f"r{i}", RequestStatus.WAITING) for i in range(5)]
        queue = DummyWaitingQueue(list(reqs))
        waiting = deque()

        impl(adapter, queue, waiting, RequestStatus.WAITING, set())

        kept_ids = [r.request_id for r in queue]
        moved_ids = [r.request_id for r in waiting]
        assert kept_ids == ["r0", "r2", "r4"], f"[{impl_name}] order preserved in kept"
        assert moved_ids == ["r1", "r3"], f"[{impl_name}] order preserved in moved"
    print("  PASS: test_order_preservation")


if __name__ == "__main__":
    print("Testing _process_chunk_queue O(n^2) -> O(n) fix")
    print("=" * 60)
    test_all_new_requests_moved()
    test_ready_chunks_stay_in_queue()
    test_finished_requests_stay_in_queue()
    test_waiting_for_chunk_with_finished_load()
    test_mixed_scenario()
    test_with_plain_list_queue()
    test_empty_queue()
    test_original_vs_fixed_equivalence()
    test_performance_improvement()
    test_order_preservation()
    print("=" * 60)
    print("ALL TESTS PASSED")
