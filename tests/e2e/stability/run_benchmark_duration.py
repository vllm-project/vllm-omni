#!/usr/bin/env python3
"""
长稳用例：在指定时长内运行 benchmark，向已启动的 vLLM-Omni 服务发请求。

与 perf 一致：支持 --request-rate（发送速率）和 --max-concurrency（并发数）。
通过环境变量 VLLM_BENCH_MAX_DURATION_SEC 让 benchmark 在超过指定时长后不再发起新请求，
已发出的请求会等其完成再结束。

用法示例（需先启动 serve）:
  python run_benchmark_duration.py --duration-sec 300 --host 127.0.0.1 --port 8080 \\
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct --dataset-name random \\
    --request-rate 1 --max-concurrency 4 --random-input-len 100 --random-output-len 50
"""
from __future__ import annotations

import argparse
import json
import math
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
    """拼出 vllm bench serve --omni 的参数列表。"""
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
        "inf" if request_rate == float("inf") else str(request_rate),
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


def _estimate_num_prompts(duration_sec: float, request_rate: float) -> int:
    """估计要准备的请求数，保证在 duration_sec 内有足够请求可发；benchmark 内会按 max_duration 提前停止发起新请求。"""
    if request_rate == float("inf") or request_rate <= 0:
        return max(500, int(duration_sec * 50))
    return max(100, int(math.ceil(duration_sec * request_rate * 2)))


def run_benchmark_duration(
    duration_sec: float,
    host: str,
    port: int,
    model: str,
    dataset_name: str = "random",
    request_rate: float = 1.0,
    result_dir: Path | None = None,
    **extra_benchmark_args,
) -> dict:
    """在指定时长内跑一次 benchmark（与 perf 同逻辑：request-rate + max-concurrency），超过 duration_sec 后不再发新请求。

    通过环境变量 VLLM_BENCH_MAX_DURATION_SEC 传入时长，benchmark 内部在发起每个新请求前检查时间，超时则停止发起、等待已发出请求完成。

    Args:
        duration_sec: 目标运行时长（秒）；超过后不再发起新请求。
        host: 服务 host。
        port: 服务 port。
        model: 模型名。
        dataset_name: benchmark 数据集名，如 random / random-mm。
        request_rate: 发送速率（--request-rate），与 perf 一致；可为 inf 表示尽快发。
        result_dir: 结果 JSON 目录；默认当前目录下的 stability_bench_result。
        **extra_benchmark_args: 其它 benchmark 参数，如 max_concurrency, random_input_len, ignore_eos。

    Returns:
        汇总 dict：total_duration_sec, total_completed, total_failed, batch_results 等。
    """
    if result_dir is None:
        result_dir = Path.cwd() / "stability_bench_result"
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    num_prompts = _estimate_num_prompts(duration_sec, request_rate)
    result_filename = f"stability_duration_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    extra_with_filename = {**extra_benchmark_args, "result_filename": result_filename}
    args = _build_benchmark_args(
        host=host,
        port=port,
        model=model,
        dataset_name=dataset_name,
        request_rate=request_rate,
        num_prompts=num_prompts,
        extra=extra_with_filename,
    )
    args.extend(["--result-dir", str(result_dir)])
    cmd = ["vllm", "bench", "serve", "--omni"] + args

    env = os.environ.copy()
    env["VLLM_BENCH_MAX_DURATION_SEC"] = str(duration_sec)

    start_wall = time.perf_counter()
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=max(60, int(duration_sec) + 600),
        cwd=REPO_ROOT,
    )
    total_duration = time.perf_counter() - start_wall

    result_path = result_dir / result_filename
    if not result_path.exists():
        summary = {
            "total_duration_sec": round(total_duration, 2),
            "total_completed": 0,
            "total_failed": num_prompts,
            "batch_results": [{"error": proc.stderr or proc.stdout or f"exit {proc.returncode}"}],
        }
    else:
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
        summary = {
            "total_duration_sec": round(result.get("duration", total_duration), 2),
            "total_completed": result.get("completed", 0),
            "total_failed": result.get("failed", 0),
            "batch_results": [
                {
                    "completed": result.get("completed", 0),
                    "failed": result.get("failed", 0),
                    "elapsed_sec": round(result.get("duration", total_duration), 2),
                }
            ],
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
        help="发送速率（--request-rate），与 perf 一致；可用 inf 表示尽快发",
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
