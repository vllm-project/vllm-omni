"""Preflight and run the same LIBERO evaluator against two Cosmos3 backends.

Both servers must already be running. This tool refuses to start an evaluation
when their inference settings or the action normalization data differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

COMMON_SETTINGS = {
    "num_steps": 10,
    "guidance": 1.0,
    "seed": 0,
    "action_chunk_size": 16,
    "raw_action_dim": 10,
    "fps": 20,
    "faster_alpha": 0.6,
    "faster_u0": 0.9,
    "faster_formula": "paper_v3",
    "return_video": False,
    "video_decode_enabled": False,
    "early_stop_actions": None,
    "timing_contract": "moh_sample_actions_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_info(url: str) -> dict:
    with urllib.request.urlopen(f"{url.rstrip('/')}/info", timeout=10) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid /info response from {url}")
    return data


def validate(
    framework: dict, vllm: dict, *, schedule: str, framework_checkpoint: Path, vllm_checkpoint: Path, stats_sha256: str
) -> None:
    errors = []
    for backend, info, expected_checkpoint in (
        ("framework", framework, framework_checkpoint),
        ("vllm", vllm, vllm_checkpoint),
    ):
        for key, expected in COMMON_SETTINGS.items():
            if key not in info or info[key] != expected:
                errors.append(f"{backend}.{key}: expected {expected!r}, got {info.get(key, '<missing>')!r}")
        if info.get("action_time_schedule") != schedule:
            errors.append(f"{backend}.action_time_schedule: expected {schedule!r}")
        if Path(str(info.get("checkpoint", ""))).resolve() != expected_checkpoint.resolve():
            errors.append(f"{backend}.checkpoint: expected {expected_checkpoint}, got {info.get('checkpoint')}")
        stats = info.get("action_stats_path")
        if not stats or not Path(stats).is_file():
            errors.append(f"{backend}.action_stats_path: inaccessible: {stats!r}")
        elif sha256(Path(stats)) != stats_sha256:
            errors.append(f"{backend}.action_stats_path: SHA256 differs from expected value")
    if vllm.get("flow_shift") != 1.0:
        errors.append(f"vllm.flow_shift: expected 1.0, got {vllm.get('flow_shift')!r}")
    if errors:
        raise ValueError("Fair evaluation preflight failed:\n" + "\n".join(f"- {e}" for e in errors))


def eval_command(
    python: str,
    evaluator: Path,
    url: str,
    output: Path,
    *,
    streaming: bool,
    render_gpu: int,
    mujoco_gl: str = "egl",
    task_ids: str = "",
    trials: int = 50,
    max_steps: int = 0,
) -> list[str]:
    command = [
        python,
        str(evaluator),
        "--server_url",
        url,
        "--task_suite",
        "libero_10",
        "--num_trials_per_task",
        str(trials),
        "--num_envs",
        "1",
        "--action_horizon",
        "16",
        "--camera",
        "agentview,wrist",
        "--image_size",
        "256",
        "--env_image_size",
        "256",
        "--action_space",
        "frame_wise_relative",
        "--rotation_space",
        "6d",
        "--action_dim",
        "10",
        "--gripper_mode",
        "zero_one",
        "--seed",
        "0",
        "--timeout",
        "1800",
        "--rotate_180",
        "--warmup_steps",
        "10",
        "--timing_warmup_requests",
        "2",
        "--mujoco_gl",
        mujoco_gl,
        "--render_gpu_device_id",
        str(render_gpu),
        "--output_dir",
        str(output),
    ]
    if task_ids:
        command.extend(("--task_ids", task_ids))
    if max_steps:
        command.extend(("--max_steps", str(max_steps)))
    if streaming:
        command.append("--streaming")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-url", required=True)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--framework-checkpoint", required=True, type=Path)
    parser.add_argument("--vllm-checkpoint", required=True, type=Path)
    parser.add_argument("--stats-sha256", required=True)
    parser.add_argument("--schedule", choices=("has_euler", "constant_euler"), required=True)
    parser.add_argument("--evaluator", type=Path, default=Path(__file__).with_name("closed_loop_eval.py"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True, help="Python environment that runs the shared evaluator")
    parser.add_argument("--render-gpu", type=int, required=True)
    parser.add_argument("--mujoco-gl", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--task-ids", default="", help="Comma-separated task IDs; empty means all tasks")
    parser.add_argument("--trials", type=int, default=50, help="Trials per selected task")
    parser.add_argument("--max-steps", type=int, default=0, help="Episode limit; 0 uses task default")
    parser.add_argument("--run", action="store_true", help="Run both evaluations after preflight")
    args = parser.parse_args()
    if args.framework_url.rstrip("/") == args.vllm_url.rstrip("/"):
        parser.error("The two service URLs must differ")
    if not args.evaluator.is_file():
        parser.error(f"Evaluator does not exist: {args.evaluator}")
    if not args.framework_checkpoint.exists() or not args.vllm_checkpoint.exists():
        parser.error("Both checkpoints must exist")
    if args.trials < 1 or args.max_steps < 0:
        parser.error("--trials must be positive and --max-steps must be nonnegative")
    if len(args.stats_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.stats_sha256.lower()):
        parser.error("--stats-sha256 must be a SHA256 hex digest")
    framework = get_info(args.framework_url)
    vllm = get_info(args.vllm_url)
    validate(
        framework,
        vllm,
        schedule=args.schedule,
        framework_checkpoint=args.framework_checkpoint,
        vllm_checkpoint=args.vllm_checkpoint,
        stats_sha256=args.stats_sha256.lower(),
    )
    if args.output_root.exists():
        parser.error(f"Output root already exists: {args.output_root}")
    commands = {
        "framework": eval_command(
            args.python,
            args.evaluator,
            args.framework_url,
            args.output_root / "framework",
            streaming=False,
            render_gpu=args.render_gpu,
            mujoco_gl=args.mujoco_gl,
            task_ids=args.task_ids,
            trials=args.trials,
            max_steps=args.max_steps,
        ),
        "vllm": eval_command(
            args.python,
            args.evaluator,
            args.vllm_url,
            args.output_root / "vllm",
            streaming=False,
            render_gpu=args.render_gpu,
            mujoco_gl=args.mujoco_gl,
            task_ids=args.task_ids,
            trials=args.trials,
            max_steps=args.max_steps,
        ),
    }
    lock = {
        "framework_info": framework,
        "vllm_info": vllm,
        "schedule": args.schedule,
        "stats_sha256": args.stats_sha256.lower(),
        "evaluator_sha256": sha256(args.evaluator),
        "commands": commands,
    }
    print(json.dumps(lock, indent=2, ensure_ascii=False))
    if args.run:
        args.output_root.mkdir(parents=True)
        (args.output_root / "run_lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
        env = os.environ.copy()
        env["MUJOCO_GL"] = args.mujoco_gl
        env["PYOPENGL_PLATFORM"] = args.mujoco_gl
        for command in commands.values():
            subprocess.run(command, check=True, env=env)
        reports = {
            backend: json.loads((args.output_root / backend / "metrics.json").read_text()) for backend in commands
        }
        for backend, report in reports.items():
            if not report["throughput"]["steady_state_inference"].get("complete"):
                raise RuntimeError(
                    f"{backend}: incomplete MoH sampler timing; inspect metrics.json and restart/instrument the worker"
                )
        metrics = {
            "success_rate_percent": ("accuracy", "overall", "success_rate_percent"),
            "measured_request_count": ("request_timing_summary", "measured_request_count"),
            "client_preprocess_mean_ms": ("request_timing_summary", "steady_state", "client_preprocess_ms", "mean"),
            "client_ttfa_mean_ms": ("request_timing_summary", "steady_state", "client_ttfa_ms", "mean"),
            "client_ttfa_p95_ms": ("request_timing_summary", "steady_state", "client_ttfa_ms", "p95"),
            "client_last_ready_mean_ms": (
                "request_timing_summary",
                "steady_state",
                "client_last_ready_action_ms",
                "mean",
            ),
            "client_full_response_mean_ms": ("request_timing_summary", "steady_state", "http_total_ms", "mean"),
            "client_full_response_p95_ms": ("request_timing_summary", "steady_state", "http_total_ms", "p95"),
            "server_ttfa_mean_ms": ("request_timing_summary", "steady_state", "server_ttfa_ms", "mean"),
            "server_full_chunk_mean_ms": ("request_timing_summary", "steady_state", "server_full_chunk_ms", "mean"),
            "model_infer_mean_ms": ("request_timing_summary", "steady_state", "policy_infer_ms", "mean"),
            "model_infer_p95_ms": ("request_timing_summary", "steady_state", "policy_infer_ms", "p95"),
            "model_chunks_per_s": ("throughput", "steady_state_inference", "chunks_per_s"),
            "model_actions_per_s": ("throughput", "steady_state_inference", "actions_per_s"),
            "generated_active_actions_per_s": ("throughput", "steady_state_http", "active_actions_per_s"),
            "generated_sustained_actions_per_s": ("throughput", "steady_state_http", "sustained_actions_per_s"),
            "closed_loop_executed_actions_per_s": ("throughput", "evaluation_wall", "executed_actions_per_s"),
        }
        comparison = {}
        for name, path in metrics.items():
            comparison[name] = {}
            for backend, report in reports.items():
                value = report
                for key in path:
                    value = value.get(key) if isinstance(value, dict) else None
                comparison[name][backend] = value
        (args.output_root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
