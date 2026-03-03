"""
长稳 Benchmark：在指定时长内按 request-rate 或 max-concurrency 向已启动的服务发请求，
超过 duration_sec 后不再发起新请求，已发出的请求会等其完成。

用法：
  - 由 pytest test_benchmark_stability.py 调用；
  - 或单独运行：先启动服务，再执行
    python run_benchmark_duration.py --duration 300 --request-rate 1 --host localhost --port 8000 ...
    或
    python run_benchmark_duration.py --duration 300 --max-concurrency 4 --host localhost --port 8000 ...
"""
import os
import subprocess
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any


# Benchmark 通过该环境变量限制运行时长（秒），超过后不再发起新请求
VLLM_BENCH_MAX_DURATION_ENV = "VLLM_BENCH_MAX_DURATION_SEC"


def _build_base_args(params: dict[str, Any], host: str, port: int) -> list[str]:
    """从 params 构建 vllm bench serve 的通用参数（不含 request-rate/num-prompts/max-concurrency）。"""
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
) -> dict[str, Any]:
    """
    在指定时长内跑 benchmark：支持按 request-rate 或 max-concurrency 发请求，
    超过 duration_sec 后不再发起新请求。

    :param host: 服务 host
    :param port: 服务 port
    :param duration_sec: 最长运行秒数，超过后不再发新请求
    :param params: benchmark 其它参数（dataset_name, random_input_len 等）
    :param request_rate: 每秒请求数；与 max_concurrency 二选一
    :param max_concurrency: 最大并发数；与 request_rate 二选一
    :param result_filename: 结果 JSON 文件名（可选）
    :param result_dir: 结果目录
    :return: benchmark 结果字典
    """
    if (request_rate is None) == (max_concurrency is None):
        raise ValueError("必须且仅能指定 request_rate 或 max_concurrency 之一")

    # 确保有足够多的 num_prompts，由时长限制实际发送量
    if request_rate is not None:
        num_prompts = max(1000, int(duration_sec * request_rate * 2))
    else:
        num_prompts = max(1000, int(duration_sec * 10))

    base = _build_base_args(params, host, port)
    base.extend(["--num-prompts", str(num_prompts)])

    if request_rate is not None:
        base.extend(["--request-rate", str(request_rate)])
    else:
        base.extend(["--max-concurrency", str(max_concurrency), "--request-rate", "inf"])

    if not result_filename:
        result_filename = f"result_stability_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    result_path = Path(result_dir) / result_filename

    command = (
        ["vllm", "bench", "serve", "--omni"]
        + base
        + [
            "--backend", "openai-chat-omni",
            "--endpoint", "/v1/chat/completions",
            "--save-result",
            "--result-filename", result_filename,
            "--result-dir", result_dir,
        ]
    )

    env = os.environ.copy()
    env[VLLM_BENCH_MAX_DURATION_ENV] = str(int(duration_sec))

    proc = subprocess.run(
        command,
        env=env,
        capture_output=False,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vllm bench serve 退出码: {proc.returncode}")

    if not result_path.exists():
        return {"completed": 0, "failed": 0, "duration": duration_sec}

    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


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
    )
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
