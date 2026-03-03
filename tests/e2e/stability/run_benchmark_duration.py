#!/usr/bin/env python3
"""
长稳用例：在指定时长内循环运行 benchmark，向已启动的 vLLM-Omni 服务持续发送请求。

benchmark 自身只支持按请求数（--num-prompts）或按 QPS 发一批请求，不支持「跑满 N 分钟」。
本脚本通过循环调用 vllm bench serve --omni，在达到目标时长前不断发 batch，用于长稳压测。

用法示例（需先启动 serve）:
  python run_benchmark_duration.py --duration-sec 300 --host 127.0.0.1 --port 8080 \\
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct --dataset-name random \\
    --request-rate 1 --num-prompts-per-batch 20 --random-input-len 100 --random-output-len 50
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# 保证能 import tests.conftest（OmniServer 等）
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_benchmark_args(
    host: str,
    port: int,
    model: str,
    dataset_name: str,
    request_rate: float,
    num_prompts: int,
    extra: dict,
) -> list[str]:
    """拼出 vllm bench serve --omni 的参数列表（不含 duration 相关）。"""
    args = [
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--backend",
        "openai-chat-omni",
        "--endpoint",
        "/v1/chat/completions",
        "--dataset-name",
        dataset_name,
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(request_rate),
        "--save-result",
    ]
    for k, v in extra.items():
        if v is None:
            continue
        key = k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                args.append(f"--{key}")
        elif isinstance(v, dict):
            args.append(f"--{key}")
            args.append(json.dumps(v, ensure_ascii=False, separators=(",", ":")))
        else:
            args.append(f"--{key}")
            args.append(str(v))
    return args


def run_one_benchmark_batch(
    duration_sec: float,
    host: str,
    port: int,
    model: str,
    dataset_name: str,
    request_rate: float,
    num_prompts_per_batch: int,
    extra_args: dict,
    result_dir: Path,
    batch_index: int,
) -> tuple[dict, float]:
    """跑一轮 benchmark（一批请求），返回 (result_dict, elapsed_sec)。"""
    result_filename = (
        f"stability_batch_{batch_index}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    extra_with_filename = {**extra_args, "result_filename": result_filename}
    args = _build_benchmark_args(
        host=host,
        port=port,
        model=model,
        dataset_name=dataset_name,
        request_rate=request_rate,
        num_prompts=num_prompts_per_batch,
        extra=extra_with_filename,
    )
    args.extend(["--result-dir", str(result_dir)])
    cmd = ["vllm", "bench", "serve", "--omni"] + args

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(60, duration_sec + 300),
        cwd=REPO_ROOT,
    )
    elapsed = time.perf_counter() - start

    result_path = result_dir / result_filename
    if not result_path.exists():
        return {
            "completed": 0,
            "failed": num_prompts_per_batch,
            "error": proc.stderr or proc.stdout or f"exit code {proc.returncode}",
        }, elapsed

    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)
    return result, elapsed


def run_benchmark_duration(
    duration_sec: float,
    host: str,
    port: int,
    model: str,
    dataset_name: str = "random",
    request_rate: float = 1.0,
    num_prompts_per_batch: int = 20,
    result_dir: Path | None = None,
    **extra_benchmark_args,
) -> dict:
    """在指定时长内循环跑 benchmark，返回汇总结果。

    Args:
        duration_sec: 目标运行时长（秒），至少会跑满一个 batch。
        host: 服务 host。
        port: 服务 port。
        model: 模型名。
        dataset_name: benchmark 数据集名，如 random / random-mm。
        request_rate: 每批的 QPS（--request-rate）。
        num_prompts_per_batch: 每批请求数（--num-prompts）。
        result_dir: 每轮结果 JSON 的目录；默认当前目录下的 stability_bench_result。
        **extra_benchmark_args: 其它 benchmark 参数，如 random_input_len, random_output_len, ignore_eos。

    Returns:
        汇总 dict：total_duration_sec, total_completed, total_failed, batches_run, batch_results 等。
    """
    if result_dir is None:
        result_dir = Path.cwd() / "stability_bench_result"
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    start_wall = time.perf_counter()
    total_completed = 0
    total_failed = 0
    batches_run = 0
    batch_results = []

    while True:
        batch_index = batches_run + 1
        result, elapsed = run_one_benchmark_batch(
            duration_sec=duration_sec,
            host=host,
            port=port,
            model=model,
            dataset_name=dataset_name,
            request_rate=request_rate,
            num_prompts_per_batch=num_prompts_per_batch,
            extra_args=extra_benchmark_args,
            result_dir=result_dir,
            batch_index=batch_index,
        )
        batches_run += 1
        completed = result.get("completed", 0)
        failed = result.get("failed", 0)
        total_completed += completed
        total_failed += failed
        batch_results.append(
            {
                "batch": batch_index,
                "completed": completed,
                "failed": failed,
                "elapsed_sec": round(elapsed, 2),
            }
        )

        total_elapsed = time.perf_counter() - start_wall
        if total_elapsed >= duration_sec:
            break

    total_duration = time.perf_counter() - start_wall
    summary = {
        "total_duration_sec": round(total_duration, 2),
        "total_completed": total_completed,
        "total_failed": total_failed,
        "batches_run": batches_run,
        "batch_results": batch_results,
    }
    summary_path = result_dir / "stability_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="在指定时长内循环运行 benchmark，用于长稳压测。"
    )
    parser.add_argument("--duration-sec", type=float, required=True, help="目标运行时长（秒）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务 host")
    parser.add_argument("--port", type=int, required=True, help="服务 port")
    parser.add_argument("--model", type=str, required=True, help="模型名")
    parser.add_argument("--dataset-name", type=str, default="random", help="benchmark 数据集名")
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="每批的 QPS（--request-rate）",
    )
    parser.add_argument(
        "--num-prompts-per-batch",
        type=int,
        default=20,
        help="每批请求数（--num-prompts）",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="结果目录，默认 ./stability_bench_result",
    )
    # 常用 benchmark 透传
    parser.add_argument("--random-input-len", type=int, default=None)
    parser.add_argument("--random-output-len", type=int, default=None)
    parser.add_argument("--random-range-ratio", type=float, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    extra = {}
    if args.random_input_len is not None:
        extra["random_input_len"] = args.random_input_len
    if args.random_output_len is not None:
        extra["random_output_len"] = args.random_output_len
    if args.random_range_ratio is not None:
        extra["random_range_ratio"] = args.random_range_ratio
    if args.ignore_eos:
        extra["ignore_eos"] = True
    if args.max_concurrency is not None:
        extra["max_concurrency"] = args.max_concurrency

    result_dir = Path(args.result_dir) if args.result_dir else None
    summary = run_benchmark_duration(
        duration_sec=args.duration_sec,
        host=args.host,
        port=args.port,
        model=args.model,
        dataset_name=args.dataset_name,
        request_rate=args.request_rate,
        num_prompts_per_batch=args.num_prompts_per_batch,
        result_dir=result_dir,
        **extra,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # 长稳用例可要求：无失败或失败率低于阈值
    if summary["total_failed"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
