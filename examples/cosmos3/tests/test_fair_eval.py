"""CPU-only checks for the paired evaluation preflight."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("fair_eval", Path(__file__).parents[1] / "fair_eval.py")
assert SPEC and SPEC.loader
fair_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fair_eval)


class FairEvalTest(unittest.TestCase):
    def test_command_pins_timing_and_rollout_settings(self):
        command = fair_eval.eval_command(
            "python", Path("evaluator.py"), "http://localhost:1", Path("output"), streaming=True, render_gpu=7
        )
        self.assertIn("--timing_warmup_requests", command)
        self.assertEqual(command[command.index("--timing_warmup_requests") + 1], "2")
        self.assertIn("--rotate_180", command)
        self.assertIn("--streaming", command)
        smoke = fair_eval.eval_command(
            "python",
            Path("evaluator.py"),
            "http://localhost:1",
            Path("output"),
            streaming=True,
            render_gpu=-1,
            mujoco_gl="osmesa",
            task_ids="0",
            trials=1,
            max_steps=64,
        )
        for flag, expected in (
            ("--mujoco_gl", "osmesa"),
            ("--task_ids", "0"),
            ("--num_trials_per_task", "1"),
            ("--max_steps", "64"),
        ):
            self.assertEqual(smoke[smoke.index(flag) + 1], expected)

    def test_valid_pair_and_video_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats.json"
            stats.write_text("same normalization")
            framework_ckpt = root / "framework"
            vllm_ckpt = root / "vllm"
            framework_ckpt.mkdir()
            vllm_ckpt.mkdir()
            common = {
                **fair_eval.COMMON_SETTINGS,
                "action_time_schedule": "has_euler",
                "action_stats_path": str(stats),
                "streaming": {"supported": True},
            }
            framework = {**common, "checkpoint": str(framework_ckpt)}
            vllm = {**common, "checkpoint": str(vllm_ckpt), "flow_shift": 1.0}
            kwargs = {
                "schedule": "has_euler",
                "framework_checkpoint": framework_ckpt,
                "vllm_checkpoint": vllm_ckpt,
                "stats_sha256": fair_eval.sha256(stats),
            }
            fair_eval.validate(framework, vllm, **kwargs)
            legacy = {k: v for k, v in framework.items() if k != "timing_contract"}
            with self.assertRaisesRegex(ValueError, "timing_contract"):
                fair_eval.validate(legacy, vllm, **kwargs)
            framework["video_decode_enabled"] = True
            with self.assertRaisesRegex(ValueError, "video_decode_enabled"):
                fair_eval.validate(framework, vllm, **kwargs)
            framework["video_decode_enabled"] = False
            vllm["action_stats_path"] = str(root / "missing.json")
            with self.assertRaisesRegex(ValueError, "action_stats_path"):
                fair_eval.validate(framework, vllm, **kwargs)


if __name__ == "__main__":
    unittest.main()
