# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Finite, recorded full-step client using the frozen repository request backend."""

import asyncio
import base64
import hashlib
import io
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import aiohttp
from PIL import Image


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class TaggedSession:
    def __init__(self, inner, phase, case):
        self.inner, self.phase, self.case = inner, phase, case

    def post(self, *args, **kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.update({"X-Qwen-Phase": self.phase, "X-Qwen-Case": self.case, "X-Request-Id": self.case})
        return self.inner.post(*args, headers=headers, **kwargs)


def image_evidence(body, output, expected_shape):
    images = []
    for choice in body.get("choices", []):
        content = choice.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            url = item.get("image_url", {}).get("url", "")
            assert url.startswith("data:image/")
            raw = base64.b64decode(url.split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                assert list(image.size) == list(expected_shape), (
                    f"Requested image shape {expected_shape}, observed {list(image.size)}"
                )
                rgb = image.convert("RGB")
                pixels = rgb.tobytes()
                assert min(pixels) != max(pixels), "Degenerate image output"
            name = f"image-{len(images)}.bin"
            (output / name).write_bytes(raw)
            images.append(
                {
                    "path": name,
                    "encoded_sha256": hashlib.sha256(raw).hexdigest(),
                    "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
                    "shape": list(expected_shape),
                    "channels": 3,
                }
            )
    assert len(images) == 1, "One complete image per request is required"
    return images


async def run_phase(
    source, packet, out, phase, rows, concurrency, endpoints, model, workload, request_timeout, duration_seconds=None
):
    sys.path.insert(0, str(source / "benchmarks/diffusion"))
    from backends import RequestFuncInput, async_request_chat_completions

    directory = out / phase
    directory.mkdir()
    requests = []
    for i, row in enumerate(rows):
        references = [
            str((packet / "inputs" / name).resolve()) for name in row.get("references", workload["references"])
        ]
        request = RequestFuncInput(
            prompt=row.get("prompt", workload["prompt"]),
            api_url=endpoints[i % len(endpoints)] + "/v1/chat/completions",
            model=model,
            width=workload["width"],
            height=workload["height"],
            num_inference_steps=workload["steps"],
            seed=row["seed"],
            image_paths=references or None,
            request_id=f"{phase}-{i:04d}",
            extra_body={
                "true_cfg_scale": workload["true_cfg_scale"],
                "negative_prompt": workload.get("negative_prompt", ""),
                "return_stage_metrics": True,
            },
        )
        requests.append(request)
    if duration_seconds is None:
        save(directory / "requests.json", [asdict(r) for r in requests])
    semaphore = asyncio.Semaphore(concurrency)
    active = peak = 0
    responses = []
    timing = {}
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0, limit_per_host=0),
        timeout=aiohttp.ClientTimeout(total=request_timeout, connect=60, sock_connect=60),
    ) as session:

        async def execute(request):
            nonlocal active, peak
            async with semaphore:
                active += 1
                peak = max(peak, active)
                try:
                    return await async_request_chat_completions(
                        request, TaggedSession(session, phase, request.request_id)
                    )
                finally:
                    active -= 1

        # Serialization to disk and pixel validation happen after all HTTP waits, outside serving timing.
        started = time.perf_counter()
        completion_times = {}
        if duration_seconds is None:
            responses = await asyncio.gather(*(execute(request) for request in requests))
        else:
            templates = requests
            requests = []
            completed = {}
            stop_admission = False

            async def worker():
                nonlocal stop_admission
                while not stop_admission and time.perf_counter() - started < duration_seconds:
                    index = len(requests)
                    request = replace(
                        templates[index % len(templates)],
                        request_id=f"{phase}-{index:06d}",
                        api_url=endpoints[index % len(endpoints)] + "/v1/chat/completions",
                    )
                    requests.append(request)
                    completed[index] = await execute(request)
                    completion_times[index] = time.perf_counter() - started
                    if not completed[index].success:
                        stop_admission = True

            await asyncio.gather(*(worker() for _ in range(concurrency)))
            responses = [completed[i] for i in range(len(requests))]
        timing = {
            "started_at_unix": time.time() - (time.perf_counter() - started),
            "duration_seconds": time.perf_counter() - started,
            "configured_concurrency": concurrency,
            "peak_client_outstanding": peak,
            "http_connector_limit": 0,
            "http_connector_limit_per_host": 0,
            "request_timeout_seconds": request_timeout,
            "arrival_mode": "closed_loop_duration" if duration_seconds is not None else "finite_batch",
            "admission_duration_seconds": duration_seconds,
        }
    if duration_seconds is not None:
        save(directory / "requests.json", [asdict(r) for r in requests])
    failures = []
    valid_latencies = []
    valid_indices = []
    for i, (request, response) in enumerate(zip(requests, responses, strict=True)):
        case = directory / f"case-{i:04d}"
        case.mkdir()
        record = asdict(response)
        save(case / "response.json", record)
        proof = {
            "case": request.request_id,
            "seed": request.seed,
            "prompt": request.prompt,
            "response_id": response.response_body.get("id") if isinstance(response.response_body, dict) else None,
            "reference_sha256": [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in request.image_paths or []],
            "reported_stage_durations": response.stage_durations,
            "success": response.success,
            "latency_seconds": response.latency,
            "images": [],
        }
        if response.success:
            try:
                proof["images"] = image_evidence(response.response_body, case, [workload["width"], workload["height"]])
                valid_latencies.append(response.latency)
                valid_indices.append(i)
            except Exception as error:
                proof["image_validation_error"] = str(error)
                failures.append({"case": request.request_id, "error": str(error)})
        else:
            failures.append({"case": request.request_id, "error": response.error})
        save(case / "case-proof.json", proof)
    latencies = sorted(valid_latencies)

    def percentile(q):
        if not latencies:
            return None
        position = (len(latencies) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(latencies) - 1)
        return latencies[lower] + (latencies[upper] - latencies[lower]) * (position - lower)

    summary = {
        "phase": phase,
        "count": len(requests),
        "all_full_step_images_valid": not failures,
        "failures": failures,
        "timing": timing,
        "completed": len(requests) - len(failures),
        "throughput_images_per_second": (len(requests) - len(failures)) / timing["duration_seconds"],
        "latency_p50_seconds": percentile(0.50),
        "latency_p95_seconds": percentile(0.95),
        "latency_p99_seconds": percentile(0.99),
        "tail_statistics": "DESCRIPTIVE_ONLY",
        "measurement_includes": "the original request backend, image input serialization and HTTP generation",
        "disk_and_pixel_audit": "AFTER_ALL_RESPONSES_OUTSIDE_MEASURED_WINDOW",
    }
    if duration_seconds is not None:
        inside = sum(completion_times[i] <= duration_seconds for i in valid_indices)
        summary.update(
            successful_completions_within_admission_window=inside,
            steady_window_images_per_second=inside / duration_seconds,
            drain_seconds=max(0, timing["duration_seconds"] - duration_seconds),
        )
    save(directory / "phase-summary.json", summary)
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Measure complete images across independent HTTP server replicas.")
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if min(args.concurrency, args.requests, args.width, args.height, args.steps) <= 0 or args.warmup < 0:
        parser.error("Counts and dimensions must be positive; warmup may be zero.")
    args.output.mkdir(parents=True, exist_ok=False)
    workload = {
        "prompt": args.prompt,
        "references": [],
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "true_cfg_scale": args.cfg,
        "negative_prompt": "",
    }

    async def benchmark():
        for phase, count in (
            ("first_validation", len(args.endpoint)),
            ("warmup", args.warmup),
            ("quality", 16),
            ("measured", args.requests),
        ):
            if count == 0:
                continue
            summary = await run_phase(
                args.source_dir,
                args.output,
                args.output,
                phase,
                [{"seed": args.seed + i} for i in range(count)],
                args.concurrency,
                args.endpoint,
                args.model,
                workload,
                args.timeout,
            )
            if not summary["all_full_step_images_valid"]:
                raise RuntimeError(f"{phase}: incomplete requests or invalid images; see recorded failures")
        print(json.dumps(summary, indent=2))

    asyncio.run(benchmark())


if __name__ == "__main__":
    main()
