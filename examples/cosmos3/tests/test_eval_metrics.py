"""Check action and time denominators in the standalone evaluator."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from eval_metrics import (  # noqa: E402
    build_report,
    make_request_record,
    summarize_model_throughput,
    summarize_request_timings,
    summarize_throughput,
    write_report,
)


def record(start: float, *, warmup: bool = False, execution_window: int = 0) -> dict:
    return make_request_record(
        request=1,
        endpoint="/predict",
        batch_size=1,
        result={
            "action": [[0.0] * 10 for _ in range(16)],
            "timing_contract": "moh_sample_actions_v1",
            "policy_timing": {"infer_ms": 200},
        },
        http_total_ms=1000,
        started_at_s=start,
        client_preprocess_ms=10,
        client_ttfa_ms=None,
        partial_events=0,
        action_timings=[],
        warmup=warmup,
        execution_window=execution_window,
    )


class EvalMetricsTest(unittest.TestCase):
    def test_pi_client_boundary_excludes_preparation_and_capacity_excludes_unused_suffix(self):
        row = record(2, execution_window=4)
        self.assertEqual(row["client_ms"], 1000)
        self.assertEqual(row["ttfa_ms"], 1000)
        self.assertEqual(row["client_total_ms"], 1010)
        self.assertEqual((row["started_s"], row["completed_s"]), (2, 3))
        self.assertIsNone(row["policy_wall_ms"])
        stats = summarize_throughput([row])
        self.assertEqual(stats["generated_actions"], 16)
        self.assertEqual(stats["active_actions_per_s"], 16)
        self.assertEqual(stats["active_executable_actions_per_s"], 4)

    def test_generated_actions_and_overlapping_request_intervals(self):
        result = summarize_throughput([record(0), record(0.5), record(3)])
        self.assertEqual(result["generated_actions"], 48)
        self.assertEqual(result["http_active_s"], 2.5)
        self.assertEqual(result["active_actions_per_s"], 48 / 2.5)

    def test_request_warmup_does_not_change_closed_loop_actions(self):
        timing = summarize_request_timings([record(0, warmup=True), record(2)])
        self.assertEqual(timing["measured_request_count"], 1)
        self.assertEqual(timing["throughput"]["timed_actions"], 16)
        tasks = [{"task_id": 0, "episode_results": [{"success": True, "steps": 12, "error": None}]}]
        report = build_report([record(0)], tasks, evaluation_wall_s=2, run={"warmup_steps": 10})
        wall = report["throughput"]["evaluation_wall"]
        self.assertEqual(wall["generated_actions_per_s"], 8)
        self.assertEqual(wall["executed_actions_per_s"], 1)

    def test_model_throughput_uses_sampler_time_not_http_or_job_time(self):
        first, second = record(0), record(10)
        first["worker_inference_ms"] = 3000
        second["policy_infer_ms"] = 300
        model = summarize_model_throughput([first, second])
        self.assertTrue(model["complete"])
        self.assertEqual(model["sum_model_time_s"], 0.5)
        self.assertEqual(model["chunks_per_s"], 4)
        self.assertEqual(model["actions_per_s"], 64)

    def test_old_missing_invalid_or_streaming_timing_never_falls_back(self):
        for changes in (
            {"timing_contract": None},
            {"policy_infer_ms": None},
            {"policy_infer_ms": float("nan")},
            {"policy_infer_ms": -1},
            {"streaming": True},
        ):
            with self.subTest(changes=changes):
                row = {**record(0), **changes, "worker_inference_ms": 10}
                model = summarize_model_throughput([record(2), row])
                self.assertFalse(model["complete"])
                self.assertIsNone(model["actions_per_s"])

    def test_zero_or_empty_denominator_is_unavailable(self):
        self.assertIsNone(summarize_model_throughput([])["actions_per_s"])
        self.assertIsNone(summarize_model_throughput([{**record(0), "policy_infer_ms": 0}])["actions_per_s"])

    def test_legacy_roundtrip_value_is_not_recorded_as_model_latency(self):
        row = make_request_record(
            request=1,
            endpoint="/predict",
            batch_size=1,
            result={"action": [[0] * 10], "policy_timing": {"infer_ms": 900}},
            http_total_ms=1000,
            started_at_s=0,
            client_preprocess_ms=0,
            client_ttfa_ms=None,
            partial_events=0,
            action_timings=[],
            warmup=False,
        )
        self.assertIsNone(row["policy_infer_ms"])
        legacy = {**record(0), "policy_infer_ms": 900, "timing_contract": None}
        self.assertIsNone(summarize_request_timings([legacy])["steady_state"]["policy_infer_ms"])

    def test_report_exports_model_and_http_throughput_separately(self):
        report = build_report([record(0)], [], evaluation_wall_s=2, run={})
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["throughput"]["steady_state_inference"]["actions_per_s"], 80)
        self.assertEqual(report["throughput"]["steady_state_http"]["active_actions_per_s"], 16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(root, report)
            self.assertEqual(
                json.loads((root / "metrics.json").read_text())["timing_contract"], "moh_sample_actions_v1"
            )
            self.assertIn("model_actions_per_s: 80.000", (root / "metrics.txt").read_text())
            self.assertIn("steady_state_inference,actions_per_s", (root / "metrics.csv").read_text())


if __name__ == "__main__":
    unittest.main()
