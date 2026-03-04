"""
长稳 Benchmark：在指定时长内按 request-rate 或 max-concurrency 向已启动的服务发请求，
超过 duration_sec 后不再发起新请求。

通过「小 batch + while 循环 + 超时判断」实现时长控制，不修改 benchmark 内部逻辑：
每轮复用 tests.perf.scripts.run_benchmark 跑一次 benchmark，跑完检查是否已超时，超时则停止并汇总结果。

用法：
  - 由 pytest test_benchmark_stability.py 调用；
  - 或单独运行：先启动服务，再执行
    python run_benchmark_duration.py --duration 300 --request-rate 1 --host localhost --port 8000 ...
    或
    python run_benchmark_duration.py --duration 300 --max-concurrency 4 --host localhost --port 8000 ...
"""
import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.perf.scripts.run_benchmark import run_benchmark


# 每轮 benchmark 的 prompt 数量，轮次间根据是否超时决定是否继续
NUM_PROMPTS_PER_BATCH = 20


def _build_base_args(params: dict[str, Any], host: str, port: int) -> list[str]:
    """从 params 构建 vllm bench serve 的通用参数（与 perf run_benchmark 的 args 形式一致）。"""
    exclude = {"request_rate", "max_concurrency", "num_prompts", "baseline", "duration_sec"}
    args = ["--host", host, "--port", str(port)]
    for key, value in params.items():
        if key in exclude or value is None:
            continue
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool) and value:
            args.append(arg_name)
        elif isinstance(value, dict):
            args.extend([arg_name, json.dumps(value, ensure_ascii=False, separators=(",", ":"))])
        elif not isinstance(value, bool):
            args.extend([arg_name, str(value)])
    return args


def _run_one_benchmark_batch(
    host: str,
    port: int,
    params: dict[str, Any],
    num_prompts: int,
    request_rate: float | None,
    max_concurrency: int | None,
    result_dir: str,
    batch_index: int,
) -> dict[str, Any]:
    """跑一轮 benchmark（固定 num_prompts），复用 perf 的 run_benchmark。"""
    base = _build_base_args(params, host, port)
    if request_rate is not None:
        args = base + ["--request-rate", str(request_rate), "--num-prompts", str(num_prompts)]
        flow = request_rate
    else:
        args = base + [
            "--max-concurrency", str(max_concurrency),
            "--num-prompts", str(num_prompts),
            "--request-rate", "inf",
        ]
        flow = max_concurrency

    dataset_name = params.get("dataset_name", "random")
    old_benchmark_dir = os.environ.get("BENCHMARK_DIR")
    try:
        os.environ["BENCHMARK_DIR"] = result_dir
        result = run_benchmark(
            args=args,
            test_name="stability",
            flow=flow,
            dataset_name=dataset_name,
            num_prompt=num_prompts,
        )
        return result
    except (FileNotFoundError, OSError):
        return {"completed": 0, "failed": 0, "duration": 0.0}
    finally:
        if old_benchmark_dir is not None:
            os.environ["BENCHMARK_DIR"] = old_benchmark_dir
        elif "BENCHMARK_DIR" in os.environ:
            os.environ.pop("BENCHMARK_DIR")


def _merge_batch_results(batch_results: list[dict[str, Any]], total_duration_sec: float) -> dict[str, Any]:
    """合并多轮 benchmark 结果，用于断言 completed/failed 等。"""
    if not batch_results:
        return {"completed": 0, "failed": 0, "duration": total_duration_sec}

    completed = sum(r.get("completed", 0) for r in batch_results)
    failed = sum(r.get("failed", 0) for r in batch_results)
    total_input_tokens = sum(r.get("total_input_tokens", 0) for r in batch_results)
    total_output_tokens = sum(r.get("total_output_tokens", 0) for r in batch_results)
    total_audio_duration_s = sum(r.get("total_audio_duration_s", 0.0) for r in batch_results)
    total_audio_frames = sum(r.get("total_audio_frames", 0) for r in batch_results)

    dur = float(total_duration_sec) or 0.0
    request_throughput = completed / dur if dur > 0 else 0.0
    output_throughput = total_output_tokens / dur if dur > 0 else 0.0
    total_token_throughput = (total_input_tokens + total_output_tokens) / dur if dur > 0 else 0.0
    audio_throughput = total_audio_duration_s / dur if dur > 0 else 0.0

    merged: dict[str, Any] = {
        "completed": completed,
        "failed": failed,
        "duration": total_duration_sec,
        "errors": [],
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": request_throughput,
        "output_throughput": output_throughput,
        "total_token_throughput": total_token_throughput,
        "total_audio_duration_s": total_audio_duration_s,
        "total_audio_frames": total_audio_frames,
        "audio_throughput": audio_throughput,
    }
    for r in batch_results:
        merged["errors"].extend(r.get("errors") or [])
    return merged


def _print_merged_report(result: dict[str, Any]) -> None:
    """以与单轮 benchmark 相同风格打印汇总结果。"""
    fmt = "{:<40} {:<10}"

    completed = result.get("completed", 0)
    failed = result.get("failed", 0)
    duration = float(result.get("duration", 0.0) or 0.0)
    total_input_tokens = result.get("total_input_tokens", 0)
    total_output_tokens = result.get("total_output_tokens", 0)
    request_throughput = result.get("request_throughput", 0.0)
    output_throughput = result.get("output_throughput", 0.0)
    total_token_throughput = result.get("total_token_throughput", 0.0)
    total_audio_duration_s = result.get("total_audio_duration_s", 0.0)
    total_audio_frames = result.get("total_audio_frames", 0)
    audio_throughput = result.get("audio_throughput", 0.0)

    print("\n ============ Serving Benchmark Result (Merged) ===========")
    print(fmt.format("Successful requests:", completed))
    print(fmt.format("Failed requests:", failed))
    print(fmt.format("Benchmark duration (s):", f"{duration:.2f}"))

    print(" ================== Text Result (Merged) ==================")
    print(fmt.format("Total input tokens:", total_input_tokens))
    print(fmt.format("Total generated tokens:", total_output_tokens))
    print(fmt.format("Output token throughput (tok/s):", f"{output_throughput:.2f}"))
    print(fmt.format("Total Token throughput (tok/s):", f"{total_token_throughput:.2f}"))

    print(" ================== Audio Result (Merged) =================")
    print(fmt.format("Total audio duration generated(s):", f"{total_audio_duration_s:.2f}"))
    print(fmt.format("Total audio frames generated:", total_audio_frames))
    print(fmt.format("Audio throughput(audio duration/s):", f"{audio_throughput:.2f}"))
    print(" ==========================================================\n")


def run_stability_benchmark(
    host: str,
    port: int,
    duration_sec: int | float,
    params: dict[str, Any],
    *,
    request_rate: float | None = None,
    max_concurrency: int | None = None,
    result_filename: str | None = None,
    result_dir: str = "./",
    num_prompts_per_batch: int = NUM_PROMPTS_PER_BATCH,
) -> dict[str, Any]:
    """
    在指定时长内跑 benchmark：每轮用较小 num_prompts 跑一次，循环直到超时后停止并汇总结果。
    不修改 benchmark 内部逻辑，不依赖环境变量限时。

    :param host: 服务 host
    :param port: 服务 port
    :param duration_sec: 目标运行秒数，超过后不再发起新一轮
    :param params: benchmark 其它参数（dataset_name, random_input_len 等）
    :param request_rate: 每秒请求数；与 max_concurrency 二选一
    :param max_concurrency: 最大并发数；与 request_rate 二选一
    :param result_filename: 最终结果 JSON 文件名（可选，用于保存合并结果）
    :param result_dir: 结果目录
    :param num_prompts_per_batch: 每轮请求数，默认 20
    :return: 合并后的 benchmark 结果字典（completed, failed, duration, errors 等）
    """
    if (request_rate is None) == (max_concurrency is None):
        raise ValueError("必须且仅能指定 request_rate 或 max_concurrency 之一")

    start_time = time.perf_counter()
    batch_results: list[dict[str, Any]] = []
    batch_index = 0

    while True:
        if (time.perf_counter() - start_time) >= duration_sec:
            break
        result = _run_one_benchmark_batch(
            host=host,
            port=port,
            params=params,
            num_prompts=num_prompts_per_batch,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
            result_dir=result_dir,
            batch_index=batch_index,
        )
        batch_results.append(result)
        batch_index += 1
        if (time.perf_counter() - start_time) >= duration_sec:
            break

    total_duration = time.perf_counter() - start_time
    merged = _merge_batch_results(batch_results, total_duration)

    _print_merged_report(merged)

    if result_filename and result_dir:
        result_path = Path(result_dir) / result_filename
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="长稳 Benchmark：在指定时长内按 request-rate 或 max-concurrency 发请求。"
    )
    parser.add_argument("--duration", type=float, required=True, help="最长运行秒数，超过后不再发新请求")
    parser.add_argument("--request-rate", type=float, default=None, help="每秒请求数（与 --max-concurrency 二选一）")
    parser.add_argument("--max-concurrency", type=int, default=None, help="最大并发数（与 --request-rate 二选一）")
    parser.add_argument("--host", type=str, default="localhost", help="服务 host")
    parser.add_argument("--port", type=int, required=True, help="服务 port")
    parser.add_argument("--dataset-name", type=str, default="random", help="dataset 名称")
    parser.add_argument("--random-input-len", type=int, default=2500)
    parser.add_argument("--random-output-len", type=int, default=900)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--result-dir", type=str, default="./")
    parser.add_argument("--percentile-metrics", type=str, default="ttft,tpot,itl,e2el,audio_rtf,audio_ttfp,audio_duration")
    parser.add_argument(
        "--num-prompts-per-batch",
        type=int,
        default=NUM_PROMPTS_PER_BATCH,
        help=f"每轮 benchmark 的请求数，轮次间根据是否超时决定是否继续（默认 {NUM_PROMPTS_PER_BATCH}）",
    )
    args = parser.parse_args()

    if (args.request_rate is None) == (args.max_concurrency is None):
        parser.error("必须且仅能指定 --request-rate 或 --max-concurrency 之一")

    params = {
        "dataset_name": args.dataset_name,
        "random_input_len": args.random_input_len,
        "random_output_len": args.random_output_len,
        "ignore_eos": args.ignore_eos,
        "percentile-metrics": args.percentile_metrics,
    }
    result = run_stability_benchmark(
        host=args.host,
        port=args.port,
        duration_sec=args.duration,
        params=params,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        result_dir=args.result_dir,
        num_prompts_per_batch=args.num_prompts_per_batch,
    )
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
