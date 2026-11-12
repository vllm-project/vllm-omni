# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matched Qwen3-Omni turn benchmark: cache off, local prefix, or Mooncake.

Run each mode in a fresh process with identical arguments and a running Store.
Client timings use AsyncOmni's streaming API, excluding HTTP transport. Output
digests and memory snapshots are computed after each timed wave. There are no
tensor dumps, save barriers, or profiler hooks in the measured path.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

import numpy as np
import psutil
import pynvml
import torch
from PIL import Image

from tests.engine import kv_offload_helpers as helpers
from vllm_omni.worker.gpu_ar_worker import GPUARWorker

REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


class MeasuredGPUARWorker(GPUARWorker):
    """Count restored rows in memory; collect them only between timed waves."""

    def init_device(self):
        from vllm_omni.distributed.kv_transfer.mooncake.connector import OmniMooncakeStoreConnector

        self._benchmark_restores = []
        capture_step = OmniMooncakeStoreConnector.capture_step

        def observe(connector, req_id, attempt_id, start_token, prompt_len, tensors):
            merged = capture_step(connector, req_id, attempt_id, start_token, prompt_len, tensors)
            if merged is not None:
                inserted = merged["embedding"].shape[0] - tensors["embedding"].shape[0]
                if inserted > 0:
                    self._benchmark_restores.append((req_id, inserted))
            return merged

        OmniMooncakeStoreConnector.capture_step = observe
        super().init_device()

    def mooncake_benchmark_snapshot(self):
        process = psutil.Process()
        memory = process.memory_full_info()
        restored, self._benchmark_restores = self._benchmark_restores, []
        return {
            "pid": process.pid,
            "rss_bytes": memory.rss,
            "uss_bytes": memory.uss,
            "kv_capacity_tokens": self.cache_config.num_gpu_blocks * self.cache_config.block_size,
            "restored_rows": restored,
        }


def deployment(mode, store_config, namespace, max_concurrency, kv_bytes):
    thinker = {
        "use_v2_model_runner": True,
        "worker_cls": "benchmarks.qwen3_omni.mooncake_cache_benchmark.MeasuredGPUARWorker",
        "tensor_parallel_size": 1,
        "max_num_seqs": max_concurrency,
        "max_model_len": 1024,
        "max_num_batched_tokens": 256,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": mode == "local",
        "block_size": 16,
        "gpu_memory_utilization": 0.85,
        "kv_cache_memory_bytes": kv_bytes,
        "revision": REVISION,
    }
    if mode == "mooncake":
        thinker["omni_kv_config"] = {
            "kv_store_config": {
                "mooncake_config": {
                    "config_file": str(Path(store_config).resolve()),
                    "namespace": namespace,
                    "model_revision": REVISION,
                }
            }
        }
    return helpers.stage_overrides(
        lmcache=False,
        thinker_extra=thinker,
        downstream_extra={
            "1": {"max_num_seqs": max_concurrency, "gpu_memory_utilization": 0.3, "revision": REVISION},
            "2": {"max_num_seqs": max_concurrency, "gpu_memory_utilization": 0.2, "revision": REVISION},
        },
    )


def prompt(document, partial=False):
    text = helpers.prompts(1)[0]["prompt"].replace("user\n", f"user\nDocument {document}. ", 1)
    if partial:
        text = text.replace("Fact 12:", "Additional facts follow. Fact 12:", 1)
    return {"prompt": text}


def media_prompt(document, modality, changed=False):
    result = prompt(document)
    if modality == "image":
        tag = "<|vision_start|><|image_pad|><|vision_end|>"
        media = Image.new("RGB", (112, 112), "blue" if changed else "red")
    else:
        tag = "<|audio_start|><|audio_pad|><|audio_end|>"
        samples = np.arange(16000, dtype=np.float32) / 16000
        media = (0.1 * np.sin(2 * np.pi * (660 if changed else 440) * samples), 16000)
    result["prompt"] = result["prompt"].replace("user\n", f"user\n{tag}", 1)
    result["multi_modal_data"] = {modality: media}
    return result


def input_digest(item):
    digest = hashlib.sha256(item["prompt"].encode())
    for name, value in item.get("multi_modal_data", {}).items():
        digest.update(name.encode())
        digest.update(value.tobytes() if isinstance(value, Image.Image) else value[0].tobytes())
    return digest.hexdigest()


async def request(omni, text, request_id):
    start = time.perf_counter()
    result = {"request_id": request_id, "ttft_ms": None, "first_audio_ms": None, "cached_tokens": 0}
    tokens, audio, text_parts = [], [], []
    async for out in omni.generate(text, request_id=request_id, output_modalities=["text", "audio"]):
        if out.error:
            raise RuntimeError(out.error)
        elapsed = (time.perf_counter() - start) * 1000
        if out.stage_id == 0:
            result["cached_tokens"] = max(result["cached_tokens"], out.num_cached_tokens or 0)
            result["prompt_tokens"] = len(out.prompt_token_ids or []) or result.get("prompt_tokens", 0)
            for completion in out.outputs:
                if completion.token_ids:
                    if result["ttft_ms"] is None:
                        result["ttft_ms"] = elapsed
                    tokens.extend(completion.token_ids)
                    text_parts.append(completion.text)
        if out.final_output_type == "audio":
            value = out.multimodal_output.get("audio")
            if isinstance(value, torch.Tensor) and value.numel():
                if result["first_audio_ms"] is None:
                    result["first_audio_ms"] = elapsed
                audio.append(value)
        if isinstance(out.metrics, dict):
            for stage, metrics in out.metrics.get("stage_metrics", {}).items():
                result.setdefault("stage_metrics", {})[stage] = metrics
    result["e2e_ms"] = (time.perf_counter() - start) * 1000
    result["text"] = "".join(text_parts)
    result["token_ids"] = tokens
    return result, audio


def finish_result(result, audio):
    # Hashing and CPU materialization run after the entire timed wave ends.
    waveform = torch.cat([piece.detach().cpu().float().reshape(-1) for piece in audio]) if audio else torch.empty(0)
    result["audio_samples"] = waveform.numel()
    result["audio_sha256"] = hashlib.sha256(waveform.numpy().tobytes()).hexdigest()
    result["audio_duration_s"] = waveform.numel() / 24000
    result["rtf"] = result["e2e_ms"] / 1000 / result["audio_duration_s"] if waveform.numel() else None
    result["success"] = bool(result["token_ids"] and waveform.numel())
    return result


def memory_snapshot(store_metrics_url):
    gpus = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpus.append({"index": index, "used_bytes": memory.used, "total_bytes": memory.total})
    result = {"gpus": gpus}
    if store_metrics_url:
        with urlopen(store_metrics_url, timeout=5) as response:
            metrics = response.read().decode()
        result["store_metrics"] = {
            name: float(value)
            for line in metrics.splitlines()
            if line and not line.startswith("#")
            for name, value in [line.rsplit(" ", 1)]
            if any(word in name.lower() for word in ("memory", "bytes", "size", "capacity", "object"))
        }
    return result


def summarize(waves):
    summary = {}
    for wave in waves:
        key = f"concurrency_{wave['concurrency']}/{wave['case']}"
        entry = summary.setdefault(key, {"requests": [], "throughput_req_s": []})
        entry["requests"].extend(wave["requests"])
        entry["throughput_req_s"].append(len(wave["requests"]) / wave["wall_seconds"])
    for entry in summary.values():
        requests = entry.pop("requests")
        entry["count"] = len(requests)
        entry["failures"] = sum(not request["success"] for request in requests)
        entry["cached_tokens"] = [request["cached_tokens"] for request in requests]
        for metric in ("ttft_ms", "first_audio_ms", "e2e_ms", "rtf", "throughput_req_s"):
            values = entry.pop(metric) if metric in entry else [request[metric] for request in requests]
            values = [value for value in values if value is not None]
            entry[metric] = (
                {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                }
                if values
                else None
            )
    return summary


def compare_reports(paths):
    reports = [json.loads(Path(path).read_text()) for path in paths]
    baselines = [report for report in reports if report["mode"] == "off"]
    if not baselines:
        raise ValueError("Comparison requires at least one cache-off report")
    expected, baseline_differences, differences, missing = {}, [], [], []

    def observations(report):
        for wave in report["waves"]:
            for digest, request in zip(wave["input_sha256"], wave["requests"], strict=True):
                yield wave, digest, request

    def restored_rows(snapshot):
        if isinstance(snapshot, list):
            return sum(restored_rows(item) for item in snapshot)
        return sum(count for _, count in snapshot.get("restored_rows", [])) if isinstance(snapshot, dict) else 0

    for report in baselines:
        for wave, digest, request in observations(report):
            value = request["token_ids"], request["audio_sha256"], request["audio_samples"]
            if digest in expected and value != expected[digest]:
                baseline_differences.append({"case": wave["case"], "request_id": request["request_id"]})
            expected.setdefault(digest, value)
    hits, invalid_reports, media_checks = [], [], []
    for report in reports:
        requests = [request for _, _, request in observations(report)]
        if not report.get("completed") or not requests or not all(request.get("success") for request in requests):
            invalid_reports.append(
                {
                    "mode": report["mode"],
                    "completed": bool(report.get("completed")),
                    "observations": len(requests),
                    "failed_requests": [request["request_id"] for request in requests if not request.get("success")],
                }
            )
        for wave, digest, request in observations(report):
            if report["mode"] != "off":
                if digest not in expected:
                    missing.append({"mode": report["mode"], "request_id": request["request_id"]})
                else:
                    tokens, waveform, count = expected[digest]
                    if (request["token_ids"], request["audio_sha256"], request["audio_samples"]) != expected[digest]:
                        differences.append(
                            {
                                "mode": report["mode"],
                                "case": wave["case"],
                                "request_id": request["request_id"],
                                "tokens_equal": request["token_ids"] == tokens,
                                "audio_equal": request["audio_sha256"] == waveform,
                                "audio_samples_equal": request["audio_samples"] == count,
                            }
                        )
            cached = request["cached_tokens"]
            complete = (request.get("prompt_tokens", 0) - 1) // 16 * 16
            hits.append(
                {
                    "mode": report["mode"],
                    "case": wave["case"],
                    "request_id": request["request_id"],
                    "cached_tokens": cached,
                    "observed": "full" if complete > 0 and cached >= complete else "partial" if cached else "miss",
                }
            )
        for wave in report["waves"]:
            if report["mode"] == "off" or not wave.get("changed_media"):
                continue
            unchanged_cached = max(
                (
                    request["cached_tokens"]
                    for candidate, _, request in observations(report)
                    if candidate["case"] == f"{wave['media']}_full"
                ),
                default=0,
            )
            changed_cached = sum(request["cached_tokens"] for request in wave["requests"])
            changed_restored = restored_rows(wave["thinker"])
            media_checks.append(
                {
                    "mode": report["mode"],
                    "media": wave["media"],
                    "unchanged_cached_tokens": unchanged_cached,
                    "changed_cached_tokens": changed_cached,
                    "changed_restored_rows": changed_restored,
                    "passed": unchanged_cached > 0 and changed_cached == changed_restored == 0,
                }
            )
    return {
        "modes": sorted({report["mode"] for report in reports}),
        "baseline_repeatability_differences": baseline_differences,
        "output_differences": differences,
        "missing_baselines": missing,
        "invalid_reports": invalid_reports,
        "media_invalidation_checks": media_checks,
        "media_invalidation_passed": all(check["passed"] for check in media_checks) if media_checks else None,
        "observed_cache_hits": hits,
        "exact_outputs_passed": not baseline_differences and not differences and not missing and not invalid_reports,
        "scope": "Client tokens and audio; source tensors/Talker inputs require the separate acceptance test",
    }


async def run(args):
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    namespace = f"m1-performance-{uuid4().hex}"
    overrides = deployment(args.mode, args.store_config, namespace, max(args.concurrency), args.kv_mib * 1024**2)
    report = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "mode": args.mode,
        "unsupported_mode": "local prefix + Mooncake (two-level cooperation)",
        "arguments": vars(args),
        "model_revision": REVISION,
        "namespace": namespace,
        "deployment": overrides,
        "packages": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")},
        "measurement": "AsyncOmni client stream; no HTTP; memory snapshots between waves, not peak memory",
        "waves": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    omni = AsyncOmni(
        model=args.model,
        revision=REVISION,
        deploy_config="qwen3_omni_moe_mrv2.yaml",
        stage_overrides=overrides,
        trust_remote_code=True,
        stage_init_timeout=1800,
        init_timeout=1800,
        log_stats=False,
    )
    try:
        for index in range(args.warmup):
            await request(omni, prompt(f"warmup-{index}"), f"warmup-{index}")

        report["initial_thinker"] = await omni.collective_rpc("mooncake_benchmark_snapshot", stage_ids=[0])

        async def wave(case, repeat, concurrency, documents, partial=False, media=None, changed=False):
            requests = [
                media_prompt(document, media, changed) if media else prompt(document, partial=partial)
                for document in documents
            ]
            input_hashes = [input_digest(item) for item in requests]
            start = time.perf_counter()
            results = await asyncio.gather(
                *[request(omni, item, f"{case}-{repeat}-{concurrency}-{index}") for index, item in enumerate(requests)]
            )
            wall = time.perf_counter() - start
            row = {
                "case": case,
                "repeat": repeat,
                "concurrency": concurrency,
                "documents": documents,
                "input_sha256": input_hashes,
                "media": media,
                "changed_media": changed,
                "wall_seconds": wall,
                "requests": [finish_result(result, audio) for result, audio in results],
                "memory": memory_snapshot(args.store_metrics_url),
                "thinker": await omni.collective_rpc("mooncake_benchmark_snapshot", stage_ids=[0]),
            }
            report["waves"].append(row)
            report["summary"] = summarize(report["waves"])
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                args.mode,
                case,
                repeat,
                concurrency,
                round(wall, 3),
                "cached_tokens",
                [item["cached_tokens"] for item in row["requests"]],
                flush=True,
            )

        for concurrency in args.concurrency:
            for repeat in range(args.repeats):
                documents = [f"c{concurrency}-r{repeat}-d{index}" for index in range(concurrency)]
                await wave("cold", repeat, concurrency, documents)
                await wave("full", repeat, concurrency, documents)
                if concurrency > 1:
                    for document in documents:
                        await wave("serial_reference", repeat, 1, [document])
                await wave("partial", repeat, concurrency, documents, partial=True)
                # Each noise document diverges before the first complete block.
                # Their combined KV size exceeds the fixed local pool. Actual
                # cached-token counts, rather than the case label, prove eviction.
                for index in range(0, args.overflow_documents, concurrency):
                    noise = [f"overflow-c{concurrency}-r{repeat}-d{j}" for j in range(index, index + concurrency)]
                    await wave("overflow_fill", repeat, concurrency, noise)
                await wave("after_overflow", repeat, concurrency, documents)
        for media in args.media:
            documents = [f"media-{media}"]
            await wave(f"{media}_cold", 0, 1, documents, media=media)
            await wave(f"{media}_full", 0, 1, documents, media=media)
            await wave(f"{media}_changed", 0, 1, documents, media=media, changed=True)
        report["completed"] = True
    finally:
        omni.shutdown()
        output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "local", "mooncake"))
    parser.add_argument("--compare", nargs="+", metavar="REPORT_JSON")
    parser.add_argument("--model", default=os.environ.get("MOONCAKE_TEST_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct"))
    parser.add_argument("--store-config", default=os.environ.get("MOONCAKE_CONFIG_PATH"))
    parser.add_argument("--store-metrics-url")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--overflow-documents", type=int, default=4)
    parser.add_argument("--kv-mib", type=int, default=128)
    parser.add_argument("--media", nargs="*", choices=("image", "audio"), default=[])
    args = parser.parse_args()
    if args.compare:
        Path(args.output).write_text(json.dumps(compare_reports(args.compare), indent=2) + "\n")
        return
    if not args.mode:
        parser.error("Specify --mode or --compare")
    if args.mode == "mooncake" and not args.store_config:
        parser.error("Mooncake mode requires --store-config or MOONCAKE_CONFIG_PATH")
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        parser.error("Set VLLM_BATCH_INVARIANT=1 before starting the process")
    pynvml.nvmlInit()
    try:
        asyncio.run(run(args))
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
