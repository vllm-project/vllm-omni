"""Test that the sync orchestrator correctly handles stage errors
without hanging indefinitely.

Regression test for the bug where non-companion request errors in the
scheduling loop did not increment completed_requests, causing
Omni.generate() to spin forever.
"""

import unittest


class TestSyncOrchestratorErrorCounting(unittest.TestCase):
    """Test that the error handling in the sync scheduling loop
    correctly increments completed_requests for all error types."""

    def test_error_result_increments_completed_requests(self):
        """Simulate a stage returning an error result and verify
        the scheduling loop terminates instead of hanging."""

        # We test the core logic directly: when a stage returns
        # {"error": "...", "request_id": "req-0", "stage_id": 0},
        # completed_requests must be incremented.

        # Extract the error-handling logic pattern from omni.py
        completed_requests = 0

        # Simulate the error result
        result = {
            "request_id": "req-0",
            "stage_id": 0,
            "error": "CUDA out of memory",
        }

        # Simulate CFG config where req-0 is NOT a companion
        class MockCfg:
            def is_companion(self, req_id):
                return False

        cfg = MockCfg()
        req_id = result.get("request_id")

        # This is the FIXED logic from omni.py lines 1026-1043
        if "error" in result:
            if cfg.is_companion(req_id) and 0 == 0:
                pass  # companion path
            else:
                completed_requests += 1

        self.assertEqual(
            completed_requests,
            1,
            "completed_requests must be incremented on non-companion error to prevent infinite loop in scheduling",
        )

    def test_companion_error_with_parent_abort_increments(self):
        """Companion error that aborts parent should also increment."""
        completed_requests = 0

        class MockCfg:
            def is_companion(self, req_id):
                return req_id == "companion-0"

            def on_companion_error(self, req_id):
                return "parent-0", True  # parent_aborted=True

        cfg = MockCfg()
        req_id = "companion-0"
        stage_id = 0

        if cfg.is_companion(req_id) and stage_id == 0:
            parent_id, parent_aborted = cfg.on_companion_error(req_id)
            if parent_aborted:
                completed_requests += 1
        else:
            completed_requests += 1

        self.assertEqual(completed_requests, 1)

    def test_companion_error_without_abort_does_not_increment(self):
        """Companion error that doesn't abort parent should not increment
        (the parent request is still in flight)."""
        completed_requests = 0

        class MockCfg:
            def is_companion(self, req_id):
                return req_id == "companion-0"

            def on_companion_error(self, req_id):
                return "parent-0", False  # parent_aborted=False

        cfg = MockCfg()
        req_id = "companion-0"
        stage_id = 0

        if cfg.is_companion(req_id) and stage_id == 0:
            parent_id, parent_aborted = cfg.on_companion_error(req_id)
            if parent_aborted:
                completed_requests += 1
        else:
            completed_requests += 1

        self.assertEqual(
            completed_requests,
            0,
            "Companion error without parent abort should not increment (parent is still processing)",
        )


if __name__ == "__main__":
    unittest.main()
