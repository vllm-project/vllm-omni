# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev

import httpx
import numpy as np


def request_video(
    client: httpx.Client,
    base_url: str,
    fields: dict[str, str],
    poll_interval: float,
    timeout: float,
) -> tuple[dict[str, object], bytes]:
    started = time.perf_counter()
    response = client.post(base_url + "/v1/videos", files={key: (None, value) for key, value in fields.items()})
    response.raise_for_status()
    job = response.json()
    job_id = job["id"]
    deadline = time.monotonic() + timeout
    while job["status"] in ("queued", "in_progress"):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Video job {job_id} exceeded {timeout} seconds")
        time.sleep(poll_interval)
        response = client.get(base_url + "/v1/videos/" + job_id)
        response.raise_for_status()
        job = response.json()
    if job["status"] != "completed":
        raise RuntimeError(json.dumps(job))
    response = client.get(base_url + "/v1/videos/" + job_id + "/content")
    response.raise_for_status()
    content = response.content
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "job_id": job_id,
        "latency_ms": elapsed_ms,
        "mp4_bytes": len(content),
        "stage_durations": job.get("stage_durations"),
        "peak_memory_mb": job.get("peak_memory_mb"),
    }, content


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure serial video jobs from POST through complete MP4 receipt.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--request-json", type=Path, required=True, help="JSON object of multipart string fields.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-first", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--requests", type=int, default=34)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0 or args.poll_interval <= 0 or args.timeout <= 0:
        parser.error("requests, poll-interval and timeout must be positive; warmup must be nonnegative")
    fields = json.loads(args.request_json.read_text())
    if not isinstance(fields, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in fields.items()):
        parser.error("request-json must contain a JSON object of string fields")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with args.output.with_suffix(".jsonl").open("x") as log, httpx.Client(timeout=args.timeout) as client:
        for index in range(args.warmup + args.requests):
            record, content = request_video(client, args.base_url.rstrip("/"), fields, args.poll_interval, args.timeout)
            record["warmup"] = index < args.warmup
            record["index"] = index
            log.write(json.dumps(record) + "\n")
            log.flush()
            if index >= args.warmup:
                records.append(record)
            # Save and decode outside the timed interval.
            if index == args.warmup and args.save_first is not None:
                args.save_first.write_bytes(content)
            print(json.dumps(record), flush=True)
    latencies = np.asarray([record["latency_ms"] for record in records], dtype=np.float64)
    result = {
        "boundary": "POST start through complete MP4 bytes received; file writes and decode excluded",
        "base_url": args.base_url,
        "input": fields,
        "concurrency": 1,
        "warmup": args.warmup,
        "requests": args.requests,
        "poll_interval_ms": args.poll_interval * 1000,
        "mean_ms": mean(latencies),
        "stddev_ms": pstdev(latencies),
        "percentiles_ms": {str(p): float(np.percentile(latencies, p)) for p in (50, 95, 99)},
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "records": records,
    }
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
