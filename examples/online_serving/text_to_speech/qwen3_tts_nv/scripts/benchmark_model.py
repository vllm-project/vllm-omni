# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""Benchmark Qwen3-TTS NV *talker only* via a single-stage AsyncOmni engine.

Runs only the NV AR talker (Qwen3TTSTalkerForConditionalGenerationNv)
without code2wav, producing codec tokens as output.  Measures TTFT,
per-token inter-token latency (ITL), end-to-end latency, and throughput
under configurable concurrency.

Reads texts from a file (one utterance per line, optionally tab-separated
with text in the second column) and runs concurrent requests through the
AsyncOmni engine.

Usage:
    # Basic benchmark with default prompts
    python benchmark_qwen3_tts_talker.py \\
        --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
        --num-requests 50

    # From a text file with concurrency sweep
    python benchmark_qwen3_tts_talker.py \\
        --model /path/to/checkpoint \\
        --text-file texts.txt \\
        --num-requests 100 \\
        --concurrency 1 4 8

    # With torch profiler on the final run
    python benchmark_qwen3_tts_talker.py \\
        --model /path/to/checkpoint \\
        --num-requests 20 --concurrency 1 --profile

    # Save JSON results
    python benchmark_qwen3_tts_talker.py \\
        --model /path/to/checkpoint \\
        --text-file texts.txt \\
        --num-requests 100 --concurrency 1 4 \\
        --result-dir results/
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import asyncio
import json
import logging
import random
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here on the mountain.",
    "Please remember to bring your identification documents to the appointment tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time and visit ancient civilizations?",
    "The restaurant on the corner serves the best pasta I have ever tasted in my entire life.",
    "After the meeting, we should discuss the quarterly results and plan for the next phase.",
    "Learning a new language takes patience, practice, and a genuine curiosity about other cultures.",
    "The train leaves at half past seven, so we need to arrive at the station before then.",
    "Could you please turn down the music a little bit, I'm trying to concentrate on my work.",
    "It was a dark and stormy night when the old lighthouse keeper heard a knock at the door.",
]


# ---------------------------------------------------------------------------
#  Stage config generation
# ---------------------------------------------------------------------------

def _build_talker_only_stage_config(
    max_num_seqs: int = 1,
    profile: bool = False,
    torch_profiler_dir: str = "./profiler_traces",
    with_stack: bool = False,
    record_shapes: bool = False,
    gpu_memory_utilization: float = 0.5,
    max_model_len: int = 4096,
    max_num_batched_tokens: int = 512,
    enforce_eager: bool = False,
    max_new_tokens: int = 2048,
    distributed_executor_backend: str = "mp",
) -> dict:
    """Build a single-stage YAML dict containing only the NV AR talker."""
    engine_args: dict[str, Any] = {
        "model_stage": "qwen3_tts",
        "max_num_seqs": max_num_seqs,
        "model_arch": "Qwen3TTSTalkerForConditionalGenerationNv",
        "worker_type": "ar",
        "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARAsyncScheduler",
        "enforce_eager": enforce_eager,
        "trust_remote_code": True,
        "async_scheduling": True,
        "enable_prefix_caching": False,
        "engine_output_type": "audio",
        "gpu_memory_utilization": gpu_memory_utilization,
        # "uni" runs the worker in-process (no shm_broadcast IPC); use "mp"
        # only when TP/PP > 1 or you actually need a separate worker process.
        "distributed_executor_backend": distributed_executor_backend,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_model_len": max_model_len,
    }

    if profile:
        engine_args["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": os.path.abspath(torch_profiler_dir),
            "torch_profiler_with_stack": with_stack,
            "torch_profiler_record_shapes": record_shapes,
        }

    cfg = {
        "stage_args": [
            {
                "stage_id": 0,
                "stage_type": "llm",
                "is_comprehension": True,
                "final_output": True,
                "final_output_type": "audio",
                "runtime": {"devices": "0"},
                "engine_args": engine_args,
                "default_sampling_params": {
                    "temperature": 0.9,
                    "top_k": 50,
                    "max_tokens": max_new_tokens,
                    "seed": 42,
                    "detokenize": False,
                    "repetition_penalty": 1.05,
                    "stop_token_ids": [2150],
                },
            }
        ],
    }
    return cfg


def _write_temp_stage_config(cfg: dict) -> str:
    """Write stage config dict to a temp YAML file, return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="talker_nv_bench_", delete=False,
    )
    yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    logger.info("Wrote single-stage config to %s", tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
#  Prompt construction
# ---------------------------------------------------------------------------

def _estimate_prompt_len(
    additional_information: dict[str, Any],
    model_name: str,
    _cache: dict[str, Any] = {},
) -> int:
    """Estimate prompt_token_ids placeholder length for the NV talker."""
    try:
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.qwen3_tts_nv.qwen3_tts_talker_nv import (
            Qwen3TTSTalkerForConditionalGenerationNv,
        )

        if model_name not in _cache:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side="left",
            )
            hf_cfg = Qwen3TTSConfig.from_pretrained(
                model_name, trust_remote_code=True,
            )
            _cache[model_name] = (tok, getattr(hf_cfg, "talker_config", None))

        tok, tcfg = _cache[model_name]
        task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]

        return Qwen3TTSTalkerForConditionalGenerationNv.estimate_prompt_len_from_additional_information(
            additional_information=additional_information,
            task_type=task_type,
            tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
            codec_language_id=getattr(tcfg, "codec_language_id", None),
            spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
        )
    except Exception as exc:
        logger.warning("Prompt length estimation failed, using 2048: %s", exc)
        return 2048


def build_input(
    text: str,
    speaker: str,
    language: str,
    model_name: str,
) -> dict:
    """Build an engine input dict from text + speaker + language."""
    additional_information = {
        "task_type": ["CustomVoice"],
        "text": [text],
        "language": [language],
        "speaker": [speaker],
    }
    ph_len = _estimate_prompt_len(additional_information, model_name)
    return {
        "prompt_token_ids": [0] * ph_len,
        "additional_information": additional_information,
    }


# ---------------------------------------------------------------------------
#  Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    success: bool = False
    text: str = ""
    prompt_len: int = 0
    num_generated: int = 0
    steps: int = 0
    ttft_s: float = 0.0
    e2e_s: float = 0.0
    inter_token_latencies: list = field(default_factory=list)
    error: str = ""


@dataclass
class BenchmarkResult:
    config_name: str = ""
    concurrency: int = 0
    num_requests: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFT
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    # E2E
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    # ITL (inter-token latency, excluding first token)
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    # Throughput
    total_tokens: int = 0
    mean_tokens_per_request: float = 0.0
    token_throughput: float = 0.0
    request_throughput: float = 0.0
    per_request: list = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------

async def run_one_request(omni, prompt: dict, request_id: str) -> RequestResult:
    """Submit one TTS request and collect outputs with per-token timing.

    AsyncOmni coerces sampling params to ``RequestOutputKind.DELTA`` when no
    explicit ``sampling_params`` are passed (since #2911). In DELTA mode,
    ``CompletionOutput.token_ids`` only holds the *new* tokens for the
    current step, so ``len(token_ids)`` cannot be used as a cumulative
    counter. The omni output processor always stores the cumulative list on
    ``cumulative_token_ids``; we use that to detect new tokens and to time
    inter-token latencies.
    """
    result = RequestResult()
    t_start = time.perf_counter()
    t_last_token = None
    prev_num_tokens = 0

    try:
        async for stage_output in omni.generate(prompt, request_id=request_id):
            now = time.perf_counter()
            ro = stage_output.request_output
            result.steps += 1

            cur_num_tokens = prev_num_tokens
            if hasattr(ro, "outputs") and ro.outputs:
                out0 = ro.outputs[0]
                cum_ids = getattr(out0, "cumulative_token_ids", None)
                if cum_ids is not None:
                    cur_num_tokens = len(cum_ids)
                else:
                    cur_num_tokens = len(getattr(out0, "token_ids", []) or [])

            if cur_num_tokens > prev_num_tokens:
                if t_last_token is None:
                    result.ttft_s = now - t_start
                else:
                    result.inter_token_latencies.append(now - t_last_token)
                t_last_token = now
                prev_num_tokens = cur_num_tokens

        t_end = time.perf_counter()
        result.e2e_s = t_end - t_start
        result.num_generated = prev_num_tokens
        result.success = True

        if result.ttft_s == 0.0 and result.steps > 0:
            result.ttft_s = t_end - t_start

    except Exception as exc:
        result.e2e_s = time.perf_counter() - t_start
        result.error = str(exc)
        logger.error("Request %s failed: %s", request_id, exc)

    return result


# ---------------------------------------------------------------------------
#  Worker / concurrency
# ---------------------------------------------------------------------------

async def worker(
    worker_id: int,
    omni,
    texts: list[str],
    model_name: str,
    speaker: str,
    language: str,
    results: list[RequestResult],
    counter: dict,
    lock: asyncio.Lock,
):
    """Persistent async worker that picks texts until the quota is exhausted."""
    while True:
        async with lock:
            if counter["remaining"] <= 0:
                break
            counter["remaining"] -= 1
            idx = counter["issued"]
            counter["issued"] += 1

        text = texts[idx % len(texts)]
        request_id = f"bench-nv-w{worker_id}-{uuid.uuid4().hex[:8]}"

        prompt = build_input(
            text=text,
            speaker=speaker,
            language=language,
            model_name=model_name,
        )

        result = await run_one_request(omni, prompt, request_id)
        result.text = text
        result.prompt_len = len(prompt["prompt_token_ids"])

        async with lock:
            results.append(result)
            done = len(results)

        if done % 10 == 0 or done == counter["total"]:
            logger.info("  progress: %d / %d", done, counter["total"])


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def _pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) > 0 else 0.0


def compute_and_print_metrics(
    results: list[RequestResult],
    duration: float,
    concurrency: int,
    num_requests: int,
) -> BenchmarkResult:
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        concurrency=concurrency,
        num_requests=num_requests,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )

    if not successful:
        print("ERROR: No requests completed successfully.")
        return bench

    ttfts = [r.ttft_s * 1000 for r in successful]
    e2es = [r.e2e_s * 1000 for r in successful]
    all_itls = []
    for r in successful:
        all_itls.extend([t * 1000 for t in r.inter_token_latencies])
    gen_tokens = [r.num_generated for r in successful]

    bench.mean_ttft_ms = float(np.mean(ttfts))
    bench.median_ttft_ms = float(np.median(ttfts))
    bench.p95_ttft_ms = _pct(ttfts, 95)
    bench.p99_ttft_ms = _pct(ttfts, 99)

    bench.mean_e2e_ms = float(np.mean(e2es))
    bench.median_e2e_ms = float(np.median(e2es))
    bench.p95_e2e_ms = _pct(e2es, 95)
    bench.p99_e2e_ms = _pct(e2es, 99)

    if all_itls:
        bench.mean_itl_ms = float(np.mean(all_itls))
        bench.median_itl_ms = float(np.median(all_itls))
        bench.p95_itl_ms = _pct(all_itls, 95)
        bench.p99_itl_ms = _pct(all_itls, 99)

    bench.total_tokens = sum(gen_tokens)
    bench.mean_tokens_per_request = float(np.mean(gen_tokens))
    bench.token_throughput = bench.total_tokens / duration if duration > 0 else 0.0
    bench.request_throughput = len(successful) / duration if duration > 0 else 0.0

    bench.per_request = [
        {
            "ttft_ms": r.ttft_s * 1000,
            "e2e_ms": r.e2e_s * 1000,
            "num_generated": r.num_generated,
            "steps": r.steps,
            "prompt_len": r.prompt_len,
            "mean_itl_ms": float(np.mean([t * 1000 for t in r.inter_token_latencies]))
            if r.inter_token_latencies else 0.0,
            "text": r.text,
        }
        for r in successful
    ]

    W = 56
    print(f"\n{'=' * W}")
    print(f"{'Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<42}{bench.completed}")
    print(f"{'Failed requests:':<42}{bench.failed}")
    print(f"{'Concurrency:':<42}{concurrency}")
    print(f"{'Wall-clock duration (s):':<42}{duration:.2f}")
    print(f"{'Request throughput (req/s):':<42}{bench.request_throughput:.2f}")

    print(f"\n{'-' * W}")
    print(f"{'Time to First Token (TTFT)':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean  (ms):':<42}{bench.mean_ttft_ms:.2f}")
    print(f"{'Median (ms):':<42}{bench.median_ttft_ms:.2f}")
    print(f"{'P95   (ms):':<42}{bench.p95_ttft_ms:.2f}")
    print(f"{'P99   (ms):':<42}{bench.p99_ttft_ms:.2f}")

    print(f"\n{'-' * W}")
    print(f"{'End-to-End Latency (E2E)':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean  (ms):':<42}{bench.mean_e2e_ms:.2f}")
    print(f"{'Median (ms):':<42}{bench.median_e2e_ms:.2f}")
    print(f"{'P95   (ms):':<42}{bench.p95_e2e_ms:.2f}")
    print(f"{'P99   (ms):':<42}{bench.p99_e2e_ms:.2f}")

    print(f"\n{'-' * W}")
    print(f"{'Inter-Token Latency (ITL)':^{W}}")
    print(f"{'-' * W}")
    if all_itls:
        print(f"{'Mean  (ms):':<42}{bench.mean_itl_ms:.2f}")
        print(f"{'Median (ms):':<42}{bench.median_itl_ms:.2f}")
        print(f"{'P95   (ms):':<42}{bench.p95_itl_ms:.2f}")
        print(f"{'P99   (ms):':<42}{bench.p99_itl_ms:.2f}")
    else:
        print(f"{'(no inter-token data)':^{W}}")

    print(f"\n{'-' * W}")
    print(f"{'Token Throughput':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Total tokens generated:':<42}{bench.total_tokens}")
    print(f"{'Mean tokens / request:':<42}{bench.mean_tokens_per_request:.1f}")
    print(f"{'Token throughput (tok/s):':<42}{bench.token_throughput:.2f}")
    print(f"{'=' * W}\n")

    if failed:
        print(f"  First {min(3, len(failed))} errors:")
        for r in failed[:3]:
            print(f"    {r.error[:200]}")

    return bench


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

async def main(args):
    from vllm_omni import AsyncOmni

    model_name = args.model

    # ── Load texts ────────────────────────────────────────────────────────
    if args.text_file:
        path = Path(args.text_file)
        if not path.exists():
            print(f"ERROR: text file not found: {path}")
            return
        raw_lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        texts = []
        for line in raw_lines:
            if "\t" in line:
                texts.append(line.split("\t", 1)[1].strip())
            else:
                texts.append(line)
        texts = [t for t in texts if t]
        logger.info("Loaded %d texts from %s", len(texts), path)
    else:
        texts = DEFAULT_PROMPTS
        logger.info("Using %d default prompts", len(texts))

    if not texts:
        print("ERROR: no texts available.")
        return

    max_concurrency = max(args.concurrency)

    # ── Build stage config ────────────────────────────────────────────────
    stage_cfg = _build_talker_only_stage_config(
        max_num_seqs=max_concurrency,
        profile=args.profile,
        torch_profiler_dir=args.torch_profiler_dir,
        with_stack=args.with_stack,
        record_shapes=args.record_shapes,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        max_new_tokens=args.max_new_tokens,
        distributed_executor_backend=args.distributed_executor_backend,
    )
    tmp_config_path = _write_temp_stage_config(stage_cfg)

    try:
        logger.info("Creating AsyncOmni engine (talker only) for %s ...", model_name)
        omni = AsyncOmni(
            model=model_name,
            stage_configs_path=tmp_config_path,
            log_stats=args.log_stats,
            stage_init_timeout=args.stage_init_timeout,
        )
        logger.info("Engine ready (single stage: talker).")

        all_bench_results = []

        for concurrency in args.concurrency:
            logger.info(
                "═══ concurrency=%d  requests=%d ═══",
                concurrency, args.num_requests,
            )

            # ── Warmup ────────────────────────────────────────────────────
            warmup_count = 0 if args.no_warmup else args.num_warmups * concurrency
            if warmup_count > 0:
                logger.info("Warming up with %d requests (concurrency=%d)...",
                            warmup_count, concurrency)
                warmup_results: list[RequestResult] = []
                warmup_counter = {
                    "remaining": warmup_count,
                    "issued": 0,
                    "total": warmup_count,
                }
                warmup_lock = asyncio.Lock()
                warmup_tasks = [
                    asyncio.create_task(worker(
                        worker_id=i,
                        omni=omni,
                        texts=texts,
                        model_name=model_name,
                        speaker=args.speaker,
                        language=args.language,
                        results=warmup_results,
                        counter=warmup_counter,
                        lock=warmup_lock,
                    ))
                    for i in range(concurrency)
                ]
                await asyncio.gather(*warmup_tasks)
                warmup_ok = sum(1 for r in warmup_results if r.success)
                logger.info("Warmup done: %d / %d succeeded.", warmup_ok, warmup_count)

            # ── Benchmark run ─────────────────────────────────────────────
            logger.info("Starting benchmark run (%d requests, concurrency=%d)...",
                        args.num_requests, concurrency)

            bench_results: list[RequestResult] = []
            counter = {
                "remaining": args.num_requests,
                "issued": 0,
                "total": args.num_requests,
            }
            lock = asyncio.Lock()

            if args.profile:
                logger.info("Starting profiler ...")
                await omni.start_profile(
                    profile_prefix=args.profile_prefix,
                    stages=[0],
                )

            start_time = time.perf_counter()
            try:
                tasks = [
                    asyncio.create_task(worker(
                        worker_id=i,
                        omni=omni,
                        texts=texts,
                        model_name=model_name,
                        speaker=args.speaker,
                        language=args.language,
                        results=bench_results,
                        counter=counter,
                        lock=lock,
                    ))
                    for i in range(concurrency)
                ]
                await asyncio.gather(*tasks)
            finally:
                if args.profile:
                    logger.info("Stopping profiler ...")
                    await omni.stop_profile(stages=[0])

            duration = time.perf_counter() - start_time

            bench = compute_and_print_metrics(
                bench_results, duration, concurrency, args.num_requests,
            )
            bench.config_name = args.config_name
            all_bench_results.append(asdict(bench))

        # ── Save results ──────────────────────────────────────────────────
        if args.result_dir:
            result_dir = Path(args.result_dir)
            result_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = result_dir / f"bench_talker_nv_{args.config_name}_{timestamp}.json"
            with open(result_file, "w") as f:
                json.dump(all_bench_results, f, indent=2)
            logger.info("Results saved to %s", result_file)

        omni.shutdown()
    finally:
        os.unlink(tmp_config_path)

    logger.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-TTS NV talker (AR stage only) via AsyncOmni",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    model = parser.add_argument_group("model / input")
    model.add_argument(
        "--model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Model name or path",
    )
    model.add_argument(
        "--text-file", type=str, default=None,
        help="Path to text file (one utterance per line, optionally "
             "tab-separated with text in 2nd column)",
    )
    model.add_argument("--speaker", type=str, default="aiden")
    model.add_argument("--language", type=str, default="English")
    model.add_argument(
        "--max-new-tokens", type=int, default=2048,
        help="Max sampling tokens per request (passed via "
             "default_sampling_params.max_tokens)",
    )

    bench = parser.add_argument_group("benchmark")
    bench.add_argument(
        "-c", "--concurrency", type=int, nargs="+", default=[1],
        help="Concurrency levels to test (space-separated, default: 1)",
    )
    bench.add_argument(
        "-n", "--num-requests", type=int, default=50,
        help="Total number of requests per concurrency level (default: 50)",
    )
    bench.add_argument(
        "--num-warmups", type=int, default=3,
        help="Warmup rounds per concurrency level "
             "(total warmup = concurrency * this, default: 3)",
    )
    bench.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    bench.add_argument(
        "--config-name", type=str, default="talker_nv",
        help="Label for this run (used in result filenames)",
    )
    bench.add_argument(
        "--result-dir", type=str, default=None,
        help="Directory to save JSON results",
    )

    engine = parser.add_argument_group("engine")
    engine.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    engine.add_argument("--max-model-len", type=int, default=4096)
    engine.add_argument("--max-num-batched-tokens", type=int, default=4096)
    engine.add_argument("--enforce-eager", action="store_true")
    engine.add_argument("--stage-init-timeout", type=int, default=300)
    engine.add_argument("--log-stats", action="store_true", default=False)
    engine.add_argument(
        "--distributed-executor-backend", type=str, default="uni",
        choices=["uni", "mp", "ray"],
        help="vLLM executor backend. 'uni' runs the worker in-process and "
             "avoids the shm_broadcast IPC round-trips on every "
             "execute_model/sample_tokens call (recommended for TP=1, "
             "single GPU). Default: uni.",
    )

    prof = parser.add_argument_group("profiling")
    prof.add_argument(
        "--profile", action="store_true",
        help="Enable torch profiler during the benchmark run",
    )
    prof.add_argument("--profile-prefix", type=str, default=None,
                      help="Prefix for profiler trace filenames")
    prof.add_argument("--torch-profiler-dir", type=str, default="./profiler_traces",
                      help="Directory for torch profiler traces")
    prof.add_argument("--with-stack", action="store_true",
                      help="Record Python call stacks in profiler")
    prof.add_argument("--record-shapes", action="store_true",
                      help="Record tensor shapes in profiler")

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
