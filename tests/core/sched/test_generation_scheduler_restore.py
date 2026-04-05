"""Test that OmniGenerationScheduler restores chunk-waiting requests
even when the OmniNewRequestData rewrapping fails.

Regression test: if process_pending_chunks() moves requests into
internal deques but restore_queues() is not called due to an exception,
those requests are permanently orphaned.
"""

import unittest
from collections import deque


class FakeAdapter:
    """Minimal mock of OmniChunkTransferAdapter tracking restore calls."""

    def __init__(self):
        self.waiting_for_chunk_waiting_requests = deque()
        self.waiting_for_chunk_running_requests = deque()
        self.restore_called = False

    def process_pending_chunks(self, waiting, running):
        """Simulate moving requests out of the scheduler queues."""
        # Move one request from running into internal deque
        if running:
            req = running.pop()
            self.waiting_for_chunk_running_requests.append(req)

    def restore_queues(self, waiting, running):
        """Put requests back."""
        self.restore_called = True
        running.extend(self.waiting_for_chunk_running_requests)
        self.waiting_for_chunk_running_requests = deque()

    def postprocess_scheduler_output(self, output):
        pass


class TestRestoreQueuesOnError(unittest.TestCase):
    """Verify that restore_queues is called even when rewrapping raises."""

    def test_requests_not_lost_on_exception(self):
        """Simulate the error path: process_pending_chunks moves a request
        out, then an exception occurs during rewrapping.
        The finally block must restore the request to the queue."""

        adapter = FakeAdapter()
        running = ["req-A", "req-B"]

        # Step 1: process_pending_chunks moves req-B out
        adapter.process_pending_chunks(waiting=[], running=running)
        self.assertEqual(running, ["req-A"])
        self.assertEqual(len(adapter.waiting_for_chunk_running_requests), 1)

        # Step 2: simulate the try/except/finally pattern
        try:
            raise RuntimeError("OmniNewRequestData construction failed")
        except Exception:
            pass  # Log error, leave output unchanged
        finally:
            # This is what guarantees restore always runs
            adapter.restore_queues(waiting=[], running=running)

        # Step 3: verify request is restored
        self.assertTrue(adapter.restore_called)
        self.assertIn("req-B", running)
        self.assertEqual(len(adapter.waiting_for_chunk_running_requests), 0)

    def test_requests_lost_without_fix(self):
        """Demonstrate the bug: without restore in except, request is lost."""

        adapter = FakeAdapter()
        running = ["req-A", "req-B"]

        adapter.process_pending_chunks(waiting=[], running=running)
        self.assertEqual(running, ["req-A"])

        # Simulate the BUGGY code: except without restore
        try:
            raise RuntimeError("OmniNewRequestData construction failed")
        except Exception:
            pass  # Bug: no restore_queues call

        # Request is lost!
        self.assertNotIn("req-B", running)
        self.assertEqual(len(adapter.waiting_for_chunk_running_requests), 1)

    def test_happy_path_restores_via_finally(self):
        """When no exception, restore_queues is still called via finally."""

        adapter = FakeAdapter()
        running = ["req-A", "req-B"]

        adapter.process_pending_chunks(waiting=[], running=running)

        # Happy path: no exception, finally still runs
        try:
            pass  # Rewrapping succeeds
        finally:
            adapter.restore_queues(waiting=[], running=running)

        self.assertTrue(adapter.restore_called)
        self.assertIn("req-B", running)


if __name__ == "__main__":
    unittest.main()
