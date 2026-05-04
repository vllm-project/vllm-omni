#!/usr/bin/env python3
"""Universal TTS benchmark CLI for vllm-omni.

Runs ``vllm bench serve --omni`` with model-aware defaults loaded from
``model_configs.yaml``.  Supports Qwen3-TTS, VoxCPM2, and any future TTS
model registered in the config file -- no code changes needed to add models.

Usage::

    python benchmarks/tts/bench_tts.py \\
        --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
        --task voice_clone \\
        --locale en \\
        --concurrency 1 4 \\
        --num-prompts 20 \\
        --dataset-path /path/to/seed-tts-eval \\
        --host localhost --port 8000

See ``--help`` for full option list.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _vllm_omni_bin() -> str:
    """Return the vllm-omni (or vllm) binary co-located with the current Python."""
    bin_dir = Path(sys.executable).parent
    for candidate in ("vllm-omni", "vllm"):
        p = bin_dir / candidate
        if p.is_file():
            return str(p)
    return "vllm-omni"  # fall back and let the shell resolve it


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_CONFIGS = _SCRIPT_DIR / "model_configs.yaml"

# Maps task name to the dataset_name used with vllm bench serve
_TASK_TO_DATASET: dict[str, str] = {
    "voice_clone": "seed-tts",
    "default_voice": "seed-tts-text",
    "voice_design": "seed-tts-design",
}

# Default design dataset path (bundled with the repo)
_DEFAULT_DESIGN_DATASET_PATH = str(_REPO_ROOT / "benchmarks" / "build_dataset" / "seed_tts_design")


def load_model_configs(path: Path) -> dict[str, Any]:
    """Load model registry from YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


def build_bench_args(
    *,
    host: str,
    port: int,
    model: str,
    task: str,
    model_cfg: dict[str, Any],
    locale: str,
    num_prompts: int,
    concurrency: int | None,
    dataset_path: str | None,
    wer_eval: bool,
    output_dir: str | None,
    result_filename: str | None,
    extra_cli_args: list[str],
) -> list[str]:
    """Build the ``vllm bench serve --omni`` command for one (task, concurrency) run."""
    dataset_name = _TASK_TO_DATASET[task]
    backend: str = model_cfg["backend"]
    endpoint: str = model_cfg["endpoint"]
    task_extra_body: dict[str, Any] = (model_cfg.get("task_extra_body") or {}).get(task) or {}

    # Resolve dataset path
    if dataset_path:
        resolved_dataset_path = dataset_path
    elif task == "voice_design":
        resolved_dataset_path = _DEFAULT_DESIGN_DATASET_PATH
    else:
        resolved_dataset_path = None

    cmd = [
        _vllm_omni_bin(),
        "bench",
        "serve",
        "--omni",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--backend",
        backend,
        "--endpoint",
        endpoint,
        "--dataset-name",
        dataset_name,
        "--num-prompts",
        str(num_prompts),
        "--num-warmups",
        "2",
        "--percentile-metrics",
        "ttft,e2el,audio_rtf,audio_ttfp,audio_duration",
    ]

    if resolved_dataset_path:
        cmd += ["--dataset-path", resolved_dataset_path]

    if locale:
        cmd += ["--seed-tts-locale", locale]

    if task_extra_body:
        cmd += ["--extra-body", json.dumps(task_extra_body, separators=(",", ":"))]

    if concurrency is not None:
        cmd += ["--max-concurrency", str(concurrency), "--request-rate", "inf"]

    if wer_eval:
        cmd.append("--seed-tts-wer-eval")

    if output_dir or result_filename:
        out_dir = output_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        cmd += ["--save-result", "--result-dir", out_dir]
        if result_filename:
            cmd += ["--result-filename", result_filename]

    cmd += extra_cli_args
    return cmd


def run_one_benchmark(cmd: list[str]) -> dict[str, Any] | None:
    """Run a single benchmark subprocess and return parsed JSON result if available."""
    print(f"\n{'=' * 60}")
    print("Running:", " ".join(cmd))
    print("=" * 60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[bench_tts] WARNING: benchmark exited with code {result.returncode}")
        return None
    # If --save-result was used, find the result file
    try:
        result_dir_idx = cmd.index("--result-dir")
        result_dir = Path(cmd[result_dir_idx + 1])
        if "--result-filename" in cmd:
            fname_idx = cmd.index("--result-filename")
            result_file = result_dir / cmd[fname_idx + 1]
        else:
            # find most recently modified json
            jsons = sorted(result_dir.glob("result_*.json"), key=lambda p: p.stat().st_mtime)
            result_file = jsons[-1] if jsons else None
        if result_file and result_file.is_file():
            return json.loads(result_file.read_text(encoding="utf-8"))
    except (ValueError, IndexError, OSError):
        pass
    return None


def print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print a unified metrics table across all (task, concurrency) runs."""
    if not results:
        return
    header = (
        f"{'Task':<16} {'Concurrency':>11} {'RTF mean':>10} "
        f"{'TTFP (ms)':>10} {'Throughput':>12} {'WER':>7} {'SIM':>7} {'UTMOS':>7}"
    )
    print(f"\n{'=' * len(header)}")
    print("BENCHMARK SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        task = r.get("_task", "?")
        conc = r.get("_concurrency", "?")
        rtf = r.get("mean_audio_rtf", float("nan"))
        ttfp = r.get("mean_audio_ttfp_ms", float("nan"))
        throughput = r.get("audio_throughput", float("nan"))
        wer = r.get("seed_tts_mean_wer", float("nan"))
        sim = r.get("seed_tts_mean_sim", float("nan"))
        utmos = r.get("seed_tts_mean_utmos", float("nan"))

        def fmt(v: float, digits: int = 3) -> str:
            return f"{v:.{digits}f}" if not math.isnan(v) else "  n/a"

        print(
            f"{task:<16} {str(conc):>11} {fmt(rtf):>10} {fmt(ttfp, 0):>10} "
            f"{fmt(throughput):>12} {fmt(wer):>7} {fmt(sim):>7} {fmt(utmos):>7}"
        )
    print("=" * len(header))


def main() -> None:
    """Entry point for the universal TTS benchmark CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)"
    )
    parser.add_argument("--task", default="all", help="Task type: voice_clone | default_voice | voice_design | all")
    parser.add_argument("--locale", default="en", choices=["en", "zh"])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4], metavar="N")
    parser.add_argument(
        "--num-prompts",
        type=int,
        nargs="+",
        default=[20],
        metavar="N",
        help="Number of prompts per run. If one value, applied to all concurrency levels.",
    )
    parser.add_argument(
        "--dataset-path", default=None, help="Root of seed-tts-eval dataset (required for voice_clone/default_voice)"
    )
    parser.add_argument("--wer-eval", action="store_true", help="Enable WER/SIM/UTMOS quality eval")
    parser.add_argument("--output-dir", default=None, help="Directory to save result JSON files")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-configs", default=str(_DEFAULT_MODEL_CONFIGS), help="Path to model_configs.yaml")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed directly to vllm bench serve")
    args = parser.parse_args()

    model_configs = load_model_configs(Path(args.model_configs))
    if args.model not in model_configs:
        known = "\n  ".join(model_configs.keys())
        print(f"[bench_tts] ERROR: model '{args.model}' not in model_configs.yaml.\nKnown models:\n  {known}")
        sys.exit(1)

    model_cfg = model_configs[args.model]
    supported_tasks: list[str] = model_cfg.get("supported_tasks", [])

    tasks_to_run: list[str]
    if args.task == "all":
        tasks_to_run = supported_tasks
    elif args.task in supported_tasks:
        tasks_to_run = [args.task]
    else:
        print(
            f"[bench_tts] ERROR: task '{args.task}' not supported by {args.model}.\nSupported tasks: {supported_tasks}"
        )
        sys.exit(1)

    # Align num_prompts list with concurrency list
    num_prompts_list: list[int] = args.num_prompts
    if len(num_prompts_list) == 1:
        num_prompts_list = num_prompts_list * len(args.concurrency)
    elif len(num_prompts_list) != len(args.concurrency):
        print(
            f"[bench_tts] ERROR: --num-prompts ({len(num_prompts_list)} values) must be "
            f"length 1 or match --concurrency ({len(args.concurrency)} values)."
        )
        sys.exit(1)

    all_results: list[dict[str, Any]] = []

    for task in tasks_to_run:
        for concurrency, num_prompts in zip(args.concurrency, num_prompts_list):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_filename = f"bench_tts_{args.model.replace('/', '_')}_{task}_c{concurrency}_{ts}.json"
            cmd = build_bench_args(
                host=args.host,
                port=args.port,
                model=args.model,
                task=task,
                model_cfg=model_cfg,
                locale=args.locale,
                num_prompts=num_prompts,
                concurrency=concurrency,
                dataset_path=args.dataset_path,
                wer_eval=args.wer_eval,
                output_dir=args.output_dir,
                result_filename=result_filename,
                extra_cli_args=args.extra or [],
            )
            result = run_one_benchmark(cmd)
            if result is not None:
                result["_task"] = task
                result["_concurrency"] = concurrency
                all_results.append(result)
                # Persist the metadata so plot_results.py can pick it up.
                if args.output_dir and result_filename:
                    result_path = Path(args.output_dir) / result_filename
                    if result_path.is_file():
                        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
