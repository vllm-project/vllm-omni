#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Closed-loop H3 image HTTP benchmark. Never retries generation POSTs."""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from PIL import Image, ImageStat

PHASES = {"c1-b1": 1, "c1-b2": 1, "c1-b3": 1, "c2": 2, "c4": 4, "smoke": 1}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def strip_payload(obj):
    if isinstance(obj, dict):
        return {k: strip_payload(v) for k, v in obj.items() if k not in {"b64_json", "url"}}
    if isinstance(obj, list):
        return [strip_payload(v) for v in obj]
    return obj


def validate_png(blob, size):
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        if image.format != "PNG" or image.size != size:
            raise ValueError(f"Expected PNG {size}, received {image.format} {image.size}")
        extrema = image.convert("RGB").getextrema()
        if all(low == high for low, high in extrema):
            raise ValueError("Image has constant RGB pixels")
        return {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "channel_extrema": extrema,
            "channel_stddev": ImageStat.Stat(image.convert("RGB")).stddev,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
        }


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


async def campaign(args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists():
        raise SystemExit("Use a fresh output directory: config.json already exists")
    for folder in ("images", "request-state"):
        (root / folder).mkdir()
    config = vars(args).copy()
    config.update(
        started_utc=stamp(),
        latency_definition=(
            "POST start to full HTTP response-body receipt; includes queueing, generation, encoding "
            "and base64 transfer; excludes JSON/base64 decode and file write"
        ),
        throughput_definition=(
            "Successful measured images / measured phase wall time including validation and persistence; "
            "excludes phase warmups"
        ),
        retry_policy="Never retry POST, including ambiguous timeouts",
        p95_note="Exploratory with small samples; no p99 claim",
    )
    scale = max(1.0, math.sqrt(args.server_min_generation_area / (args.width * args.height)))
    canvas = (
        [max(32, round(value * scale / 32) * 32) for value in (args.width, args.height)]
        if args.server_min_generation_area
        else [args.width, args.height]
    )
    config.update(
        requested_output_dimensions=[args.width, args.height],
        expected_png_dimensions=[args.width, args.height],
        internal_generation_canvas=canvas,
        canvas_metadata_source=(
            "Computed from recorded server min_generation_area and its nearest-32px grid policy; "
            "model resizes canvas to requested PNG dimensions. Metadata only; no request override."
        ),
        sampling_schedule=f"{args.steps}-step PDD checkpoint schedule",
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    write_json(root / "config.json", config)
    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    if args.num_prompts > len(corpus):
        raise ValueError("Requested more prompts than corpus contains")
    corpus = corpus[: args.num_prompts]
    (root / "corpus.jsonl").write_text("".join(json.dumps(p) + "\n" for p in corpus))
    records, phases = [], []
    state = {"status": "running", "started_utc": stamp(), "phases": phases}
    write_json(root / "state.json", state)

    async def request(session, phase, prompt, warmup, index):
        key = f"{phase}-{'warmup' if warmup else 'measured'}-{index:03d}-{prompt['id']}"
        payload = {
            "model": args.model,
            "prompt": prompt["prompt"],
            "n": 1,
            "size": f"{args.width}x{args.height}",
            "seed": prompt["seed"],
            "generator_device": "cpu",
            "num_inference_steps": args.steps,
            "response_format": "b64_json",
            "output_format": "png",
        }
        record = {
            "key": key,
            "phase": phase,
            "concurrency": PHASES[phase],
            "warmup": warmup,
            "prompt_id": prompt["id"],
            "seed": prompt["seed"],
            "started_utc": stamp(),
            "start_unix_s": time.time(),
            "status": "intent",
            "request": payload,
        }
        path = root / "request-state" / (key + ".json")
        write_json(path, record)
        start = time.perf_counter()
        try:
            async with session.post(args.base_url.rstrip("/") + "/v1/images/generations", json=payload) as response:
                blob = await response.read()
                received = time.perf_counter()
                record.update(
                    http_status=response.status,
                    response_bytes=len(blob),
                    delivery_latency_s=received - start,
                    received_utc=stamp(),
                    response_headers={
                        k: v for k, v in response.headers.items() if k.lower() in {"content-type", "x-request-id"}
                    },
                )
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {blob[:2000].decode(errors='replace')}")
                data = json.loads(blob)
                record["response_metadata"] = strip_payload(data)
                if len(data.get("data", [])) != 1:
                    raise ValueError("Expected exactly one output image")
                raw = base64.b64decode(data["data"][0]["b64_json"], validate=True)
                image_info = validate_png(raw, (args.width, args.height))
                destination = root / "images" / (key + ".png")
                destination.write_bytes(raw)
                record.update(
                    status="success",
                    image_path=str(destination.relative_to(root)),
                    validation=image_info,
                    decode_validate_persist_s=time.perf_counter() - received,
                    validated_delivery_latency_s=time.perf_counter() - start,
                )
        except Exception as error:
            record.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                elapsed_s=time.perf_counter() - start,
                recovery_note=(
                    "No automatic retry. A timeout or transport failure may leave a generation running on the server."
                ),
            )
        record["ended_utc"] = stamp()
        record["end_unix_s"] = time.time()
        write_json(path, record)
        with (root / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        records.append(record)
        print(f"{stamp()} {key} {record['status']} latency={record.get('delivery_latency_s', 0):.3f}s", flush=True)
        return record

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=8)) as session:
            for phase in args.phases:
                canonical_index = list(PHASES).index(phase)
                order = corpus.copy()
                random.Random(args.order_seed + canonical_index).shuffle(order)
                state["active_phase"] = phase
                write_json(root / "state.json", state)
                for i in range(args.warmups):
                    record = await request(session, phase, order[i % len(order)], True, i)
                    if record["status"] != "success":
                        raise RuntimeError("Warmup failed; benchmark stopped without retries")
                window_start_unix = time.time()
                begin = time.perf_counter()
                queue = list(enumerate(order))
                phase_failed = False

                async def worker():
                    nonlocal phase_failed
                    while queue and not phase_failed:
                        i, prompt = queue.pop(0)
                        result = await request(session, phase, prompt, False, i)
                        if result["status"] != "success":
                            phase_failed = True

                await asyncio.gather(*(worker() for _ in range(PHASES[phase])))
                phases.append(
                    {
                        "phase": phase,
                        "concurrency": PHASES[phase],
                        "start_unix_s": window_start_unix,
                        "end_unix_s": time.time(),
                        "wall_s": time.perf_counter() - begin,
                    }
                )
                write_json(root / "state.json", state)
                if phase_failed:
                    raise RuntimeError("Measured request failed; campaign stopped without retries")
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
    state.update(
        ended_utc=stamp(),
        successes=sum(r["status"] == "success" for r in records),
        failures=sum(r["status"] != "success" for r in records),
    )
    write_json(root / "state.json", state)
    report = {
        "status": state["status"],
        "total_outputs": len(records),
        "valid_outputs": state["successes"],
        "measured_successes": sum(r["status"] == "success" and not r["warmup"] for r in records),
        "measured_failures": sum(r["status"] != "success" and not r["warmup"] for r in records),
        "groups": [],
        "phases": phases,
        "config": config,
    }
    lines = [
        f"MiniMax-H3-Image text-to-image benchmark: {state['status']}",
        f"Measured successes: {report['measured_successes']}; failures: {report['measured_failures']}",
        f"Validated PNG outputs: {report['valid_outputs']}/{report['total_outputs']}",
        f"Requested/output PNG: {args.width}x{args.height}; internal generation canvas: {canvas[0]}x{canvas[1]} "
        f"(server min_generation_area={args.server_min_generation_area}).",
        f"Sampling: {args.steps}-step PDD checkpoint schedule; GPUs: recorded separately in provenance.",
        "Latency includes full HTTP body receipt; throughput includes validation/persistence. Warmups excluded.",
        "Concurrency | images | mean s | median s | exploratory p95 s | images/hour",
    ]
    for concurrency in sorted({p["concurrency"] for p in phases}):
        members = [
            r for r in records if not r["warmup"] and r["concurrency"] == concurrency and r["status"] == "success"
        ]
        if not members:
            continue
        latencies = [r["delivery_latency_s"] for r in members]
        wall = sum(p["wall_s"] for p in phases if p["concurrency"] == concurrency)
        group = {
            "concurrency": concurrency,
            "successes": len(members),
            "measured_wall_s": wall,
            "mean_latency_s": statistics.mean(latencies),
            "median_latency_s": statistics.median(latencies),
            "p95_latency_s": percentile(latencies, 0.95),
            "images_per_hour": len(members) / wall * 3600,
            "mean_validated_delivery_s": statistics.mean(r["validated_delivery_latency_s"] for r in members),
        }
        report["groups"].append(group)
        lines.append(
            f"{concurrency} | {len(members)} | {group['mean_latency_s']:.3f} | {group['median_latency_s']:.3f} | "
            f"{group['p95_latency_s']:.3f} | {group['images_per_hour']:.2f}"
        )
    write_json(root / "report.json", report)
    (root / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0 if state["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="/data/hf_hub_cache/MiniMax-H3-Image")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).with_name("h3-image-prompts.jsonl"))
    parser.add_argument("--phases", nargs="+", choices=list(PHASES), default=list(PHASES)[:5])
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--server-min-generation-area",
        type=int,
        default=4194304,
        help="Recorded server policy for canvas metadata only; not sent to API. Use 0 if server disables it.",
    )
    parser.add_argument("--order-seed", type=int, default=20261105)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if len(set(args.phases)) != len(args.phases):
        parser.error("Phases must be unique to preserve request states and images")
    if (
        args.num_prompts < 1
        or args.warmups < 0
        or args.server_min_generation_area < 0
        or min(args.width, args.height, args.steps, args.timeout) <= 0
    ):
        parser.error("Invalid count, dimensions, steps, or timeout")
    return asyncio.run(campaign(args))


if __name__ == "__main__":
    raise SystemExit(main())
