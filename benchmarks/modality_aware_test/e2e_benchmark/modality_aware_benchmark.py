import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np

EXPECTED_OUTPUT_TOKENS = 10


@dataclass
class Metric:
    request_id: int
    modality: str
    arrival_time: float
    ttft: float = 0
    end_time: float = 0
    success: bool = False
    output_tokens: int = 0
    raw_chunks: list[bytes] = field(default_factory=list)
    error_msg: str = ""


@dataclass
class DualResult:
    baseline: Metric
    optimized: Metric


class DualOmniBenchmark:
    def __init__(
        self,
        url_base: str,
        url_opt: str,
        qps: float,
        duration: int,
        weights: dict,
        scenario_name: str,
        data_dir: str,
        output_dir: str,
        csv_file: str,
        log_file: str,
    ):
        self.url_base = url_base
        self.url_opt = url_opt
        self.qps = qps
        self.duration = duration
        self.weights = weights
        self.scenario_name = scenario_name
        self.data_dir = data_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, csv_file)
        self.log_path = os.path.join(output_dir, log_file)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("Timestamp,Service,RequestID,Modality,Tokens,Status,ErrorOrContent\n")

        self.results: list[Any] = []

        self.dataset_pool = {k: [] for k in weights.keys()}
        self._load_all_datasets()

    async def _warmup_single(self, session: aiohttp.ClientSession, url: str, payload: dict):
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as response:
                if response.status == 200:
                    async for _ in response.content:
                        pass
                else:
                    print(f"⚠️ Warmup warning: {url} returned {response.status}")
        except Exception as e:
            print(f"⚠️ Warmup error for {url}: {e}")

    async def warmup(self, session):
        print("🔥 Warming up servers...")

        warmup_payloads = []
        for modality in self.weights.keys():
            if self.dataset_pool.get(modality):
                warmup_payloads.append(self._build_openai_payload(modality))

        if not warmup_payloads:
            warmup_payloads.append(
                {
                    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    "messages": [{"role": "user", "content": "Hello, warmup request."}],
                    "stream": True,
                    "sampling_params_list": [
                        {
                            "temperature": 0,
                            "max_tokens": EXPECTED_OUTPUT_TOKENS,
                            "ignore_eos": True,
                            "stop_token_ids": [],
                            "stop": [],
                        }
                    ],
                }
            )

        tasks = []
        for payload in warmup_payloads:
            tasks.append(self._warmup_single(session, self.url_base, payload))
            tasks.append(self._warmup_single(session, self.url_opt, payload))

        await asyncio.gather(*tasks)
        print(f"✅ Warmup done ({len(warmup_payloads)} modalities × 2 servers)")

    def _load_all_datasets(self):
        for modality in self.dataset_pool.keys():
            jsonl_path = f"{self.data_dir}/{modality}/metadata.jsonl"
            if not os.path.exists(jsonl_path):
                print(f"⚠️ Warning: Dataset for {modality} not found at {jsonl_path}")
                continue

            with open(jsonl_path) as f:
                for line in f:
                    item = json.loads(line)
                    if modality != "text":
                        item["file_abs_path"] = os.path.join(f"{self.data_dir}/{modality}/files", item["file"])
                    self.dataset_pool[modality].append(item)
            print(f"✅ Loaded {len(self.dataset_pool[modality])} samples for {modality}")

    def _build_openai_payload(self, modality: str) -> dict[str, Any]:
        if not self.dataset_pool[modality]:
            raise ValueError(f"Dataset for modality '{modality}' is empty. Please check data loading.")
        sample = random.choice(self.dataset_pool[modality])
        prompt_text = sample["prompt"]

        content = [{"type": "text", "text": prompt_text}]

        if modality != "text":
            file_url = f"file://{sample['file_abs_path']}"
            key_map = {"image": "image_url", "video": "video_url", "audio": "audio_url"}
            content.append({"type": key_map[modality], key_map[modality]: {"url": file_url}})

        return {
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "sampling_params_list": [
                {
                    "temperature": 0,
                    "max_tokens": EXPECTED_OUTPUT_TOKENS,
                    "ignore_eos": True,
                    "stop_token_ids": [],
                    "stop": [],
                }
            ],
        }

    async def _call_api(self, session: aiohttp.ClientSession, url: str, payload: dict, metric: Metric):
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=600)
            ) as response:
                if response.status == 200:
                    chunks_append = metric.raw_chunks.append
                    stream_finished = False

                    async for chunk in response.content:
                        chunks_append(chunk)
                        if b"[DONE]" in chunk:
                            stream_finished = True

                    metric.end_time = time.perf_counter()

                    if stream_finished:
                        metric.success = True
                        metric.output_tokens = EXPECTED_OUTPUT_TOKENS
                    else:
                        metric.success = False
                        metric.output_tokens = 0
                        metric.error_msg = "Stream Truncated (No [DONE] signal received)"
                else:
                    try:
                        error_text = await response.text()
                    except Exception:
                        error_text = "<Unreadable Error Body>"

                    metric.success = False
                    metric.error_msg = f"HTTP {response.status}: {error_text[:200]}"
                    print(f"Server Error {url}: {response.status}")

        except Exception as e:
            metric.success = False
            metric.error_msg = f"Network/Client Error: {repr(e)}"
            print(f"Connection Error {url}: {e}")

    async def send_dual_request(self, session: aiohttp.ClientSession, request_id: int):
        modality = random.choices(list(self.weights.keys()), weights=list(self.weights.values()))[0]
        payload = self._build_openai_payload(modality)

        arrival_time = time.perf_counter()
        m_base = Metric(request_id, modality, arrival_time)
        m_opt = Metric(request_id, modality, arrival_time)

        await asyncio.gather(
            self._call_api(session, self.url_base, payload, m_base),
            self._call_api(session, self.url_opt, payload, m_opt),
        )

        self.results.append(DualResult(baseline=m_base, optimized=m_opt))

    async def run(self):
        conn = aiohttp.TCPConnector(limit=2000)
        async with aiohttp.ClientSession(connector=conn) as session:
            await self.warmup(session)
            start_time = time.perf_counter()
            tasks = []
            req_id = 0

            print(f"🚀 Dual Benchmark Started | QPS={self.qps} | Duration={self.duration}s")
            while (time.perf_counter() - start_time) < self.duration:
                wait_time = np.random.exponential(1.0 / self.qps)
                await asyncio.sleep(wait_time)

                req_id += 1
                tasks.append(asyncio.create_task(self.send_dual_request(session, req_id)))

            await asyncio.gather(*tasks)
            self.report(time.perf_counter() - start_time)

    def _log_problematic_requests(self, metrics: list[Metric], service_name: str):
        failed_requests = [m for m in metrics if not m.success]
        problem_count = len(failed_requests)

        if problem_count > 0:
            print(f"\n⚠️  WARNING [{service_name}]: Found {problem_count} problematic requests!")
            print(f"    - FAILED (Network/HTTP Errors): {len(failed_requests)}")

            with open(self.log_path, "a", encoding="utf-8") as f:
                for m in failed_requests:
                    clean_msg = json.dumps(m.error_msg)
                    log_line = (
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')},{service_name},"
                        f"{m.request_id},{m.modality},0,FAILED,{clean_msg}\n"
                    )
                    f.write(log_line)
            print(f"    -> Detailed error logs written to {self.log_path}")
        else:
            print(
                f"✅ [{service_name}] All {len(metrics)} requests were successful"
                f"and generated {EXPECTED_OUTPUT_TOKENS} tokens."
            )

    def report(self, actual_duration: float):
        base_metrics = [r.baseline for r in self.results]
        opt_metrics = [r.optimized for r in self.results]

        self._log_problematic_requests(base_metrics, "Baseline")
        self._log_problematic_requests(opt_metrics, "Optimized")

        def calc_metrics(metrics: list[Metric]):
            if not metrics:
                return 0.0, 0.0
            success_metrics = [m for m in metrics if m.success]
            total_tokens = sum(m.output_tokens for m in success_metrics)

            tps = total_tokens / actual_duration
            error_rate = 1.0 - (len(success_metrics) / len(metrics))
            return tps, error_rate

        tps_b, err_b = calc_metrics(base_metrics)
        tps_o, err_o = calc_metrics(opt_metrics)

        print("\n" + "█" * 60)
        print(f"📊 FINAL REPORT | QPS: {self.qps} | Total Requests: {len(self.results)}")
        print("-" * 60)
        print(f"{'Metric':<25} | {'Baseline':<15} | {'Optimized':<15}")
        print("-" * 60)
        print(f"{'Token Throughput (tok/s)':<25} | {tps_b:<15.2f} | {tps_o:<15.2f}")
        print(f"{'Error Rate':<25} | {err_b:<15.2%} | {err_o:<15.2%}")
        print("-" * 60)

        header = "Scenario,QPS,Baseline_TPS,Optimized_TPS,Baseline_ErrorRate,Optimized_ErrorRate\n"
        file_exists = os.path.exists(self.csv_path)

        with open(self.csv_path, "a") as f:
            if not file_exists:
                f.write(header)
            f.write(f"{self.scenario_name},{self.qps},{tps_b:.2f},{tps_o:.2f},{err_b:.4f},{err_o:.4f}\n")
        print(f"✅ Data appended to {self.csv_path}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True, help="Name of the test scenario")
    parser.add_argument("--qps", type=float, required=True)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True, help="Directory for all outputs")
    parser.add_argument("--csv_file", type=str, default="benchmark_results.csv", help="CSV filename for metrics")
    parser.add_argument("--log_file", type=str, default="error_packets.log", help="Log filename for error details")

    args = parser.parse_args()

    URL_BASE = "http://vllm-baseline-service:8000/v1/chat/completions"
    URL_OPT = "http://vllm-opt-service:8000/v1/chat/completions"

    try:
        weights = json.loads(args.weights)
        benchmark = DualOmniBenchmark(
            URL_BASE,
            URL_OPT,
            args.qps,
            args.duration,
            weights,
            scenario_name=args.scenario,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            csv_file=args.csv_file,
            log_file=args.log_file,
        )
        asyncio.run(benchmark.run())
    except Exception as e:
        print(f"Benchmark Failed: {e}")
        import sys

        sys.exit(1)
