# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Closed-loop whole-clip HTTP benchmark with retained failures and full decoding.

Example:
  python benchmarks/diffusion/benchmark_seedvr2_highc.py --manifest corpus.csv \
    --base-url http://127.0.0.1:8098 --concurrency 32 --requests 128 \
    --metadata run-metadata.json --output evidence/c32-a2-r1

metadata JSON must contain arm, base_sha, source_sha, topology, round and phase.
Input reads and input preflight precede timing. Output decoding uses a separate,
bounded process pool; both HTTP delivery and fully decoded throughput are reported.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path


def decode_video(path: str, expected: dict) -> dict:
    import av

    result = {"valid": False, "error_type": None, "error_text": None}
    try:
        with av.open(path) as container:
            if len(container.streams.video) != 1:
                raise ValueError("expected exactly one video stream")
            stream = container.streams.video[0]
            stream.thread_count = 1
            fps = Fraction(stream.average_rate)
            frames = 0
            previous = None
            for frame in container.decode(stream):
                if (frame.width, frame.height) != (expected["output_width"], expected["output_height"]):
                    raise ValueError("decoded frame geometry mismatch")
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("missing frame timestamp")
                timestamp = frame.pts * frame.time_base
                if (previous is None and timestamp != 0) or (previous is not None and timestamp <= previous):
                    raise ValueError("timestamps must start at zero and strictly increase")
                frame.to_ndarray(format="rgb24")  # actually decode every frame
                previous = timestamp
                frames += 1
            if frames != expected["input_frames"]:
                raise ValueError(f"frame count {frames} != {expected['input_frames']}")
            if fps != Fraction(expected["fps_num"], expected["fps_den"]):
                raise ValueError(f"FPS mismatch: {fps}")
            result.update(
                valid=True,
                output_frames=frames,
                output_width=expected["output_width"],
                output_height=expected["output_height"],
                fps_num=fps.numerator,
                fps_den=fps.denominator,
                all_frames_decodable=True,
                pts_valid=True,
                expected_shape_matched=True,
            )
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error_text=str(exc))
    result["t_decode_done_mono"] = time.perf_counter()
    return result


def load_manifest(path: Path):
    integer_fields = [
        "input_width",
        "input_height",
        "input_frames",
        "fps_num",
        "fps_den",
        "output_width",
        "output_height",
        "seed",
    ]
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError("empty manifest")
    seen = set()
    for row in rows:
        if row["sample_id"] in seen:
            raise ValueError("duplicate sample_id")
        seen.add(row["sample_id"])
        for field in integer_fields:
            row[field] = int(row[field])
        if any(row[field] <= 0 for field in integer_fields if field != "seed"):
            raise ValueError("non-positive media dimension/count/FPS")
        input_path = Path(row["input_path"])
        if not input_path.is_absolute():
            input_path = path.parent / input_path
        body = input_path.read_bytes()
        if hashlib.sha256(body).hexdigest() != row["input_sha256"]:
            raise ValueError(f"input hash mismatch: {row['sample_id']}")
        expected = dict(row, output_width=row["input_width"], output_height=row["input_height"])
        check = decode_video(str(input_path), expected)
        if not check["valid"]:
            raise ValueError(f"input preflight failed: {row['sample_id']}: {check}")
        row["input_path"] = str(input_path.resolve())
        row["_body"] = body
    return rows


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    x = (len(ordered) - 1) * p
    lo = int(x)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (x - lo)


async def run(args, samples, metadata):
    import aiohttp

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(
        json.dumps(
            dict(
                metadata,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                concurrency=args.concurrency,
                requests=args.requests,
                base_urls=args.base_url,
                timeout=args.timeout,
                decode_workers=args.decode_workers,
                input_reads_timed=False,
                manifest=[{k: v for k, v in x.items() if k != "_body"} for x in samples],
            ),
            indent=2,
        )
    )
    records = [None] * args.requests
    decode_futures = []
    active = [0] * len(args.base_url)
    peak = active.copy()
    peak_total = 0
    next_index = 0
    loop = asyncio.get_running_loop()
    # No hidden HTTP connection limit (aiohttp defaults to 100).
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    with ProcessPoolExecutor(
        max_workers=args.decode_workers, mp_context=multiprocessing.get_context("spawn")
    ) as decoders:
        # Start processes before opening client sockets / timed requests.
        await asyncio.gather(*[loop.run_in_executor(decoders, time.perf_counter) for _ in range(args.decode_workers)])
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

            async def worker():
                nonlocal next_index, peak_total
                while next_index < args.requests:
                    index = next_index
                    next_index += 1
                    replica = index % len(args.base_url)
                    # Advance the input after a full replica round so every
                    # replica sees A -> B -> A, even for two inputs/two replicas.
                    sample = samples[(index // len(args.base_url)) % len(samples)]
                    record = {k: v for k, v in sample.items() if k != "_body"}
                    record.update(
                        scheduled_index=index,
                        replica_id=replica,
                        valid=False,
                        http_status=None,
                        error_type=None,
                        error_text=None,
                    )
                    form = aiohttp.FormData()
                    form.add_field("prompt", " ")
                    form.add_field("input_references", sample["_body"], filename="input.mp4", content_type="video/mp4")
                    for key, value in dict(
                        size=f"{sample['output_width']}x{sample['output_height']}",
                        seed=sample["seed"],
                        num_inference_steps=1,
                        guidance_scale=1,
                        extra_params=json.dumps({"color_correction_method": sample["color_method"]}),
                    ).items():
                        form.add_field(key, str(value))
                    active[replica] += 1
                    peak[replica] = max(peak[replica], active[replica])
                    peak_total = max(peak_total, sum(active))
                    record["t_submit_mono"] = time.perf_counter()
                    body = b""
                    chunks = []
                    try:
                        async with session.post(
                            args.base_url[replica].rstrip("/") + "/v1/videos/sync",
                            data=form,
                            headers={"x-seedvr2-phase": str(metadata["phase"]), "x-seedvr2-case": str(index)},
                        ) as response:
                            record.update(
                                t_headers_mono=time.perf_counter(),
                                http_status=response.status,
                                headers=dict(response.headers),
                                server_request_id=response.headers.get("X-Request-Id"),
                            )
                            async for chunk in response.content.iter_chunked(1 << 20):
                                chunks.append(chunk)
                            body = b"".join(chunks)
                            record["t_body_done_mono"] = time.perf_counter()
                            if response.status != 200:
                                record.update(
                                    error_type=f"HTTP_{response.status}",
                                    error_text=body[:2000].decode(errors="replace"),
                                )
                    except Exception as exc:
                        record.update(error_type=type(exc).__name__, error_text=str(exc))
                    finally:
                        if chunks and not body:
                            body = b"".join(chunks)
                        record["t_http_terminal_mono"] = time.perf_counter()
                        active[replica] -= 1
                    path = output / f"{index:06d}.response"
                    path.write_bytes(body)
                    record.update(
                        response_bytes=len(body), output_sha256=hashlib.sha256(body).hexdigest(), output_path=path.name
                    )
                    records[index] = record
                    with (output / "http-ledger.jsonl").open("a") as ledger:
                        ledger.write(json.dumps(dict(metadata, **record)) + "\n")
                    if record["http_status"] == 200 and record["error_type"] is None:
                        future = loop.run_in_executor(decoders, decode_video, str(path), sample)
                        decode_futures.append((index, future))
                    # HTTP completion admits the next request. Decode backlog
                    # remains separately bounded by the fixed CPU pool.

            await asyncio.gather(*[worker() for _ in range(min(args.concurrency, args.requests))])
        for index, future in decode_futures:
            try:
                records[index].update(await future)
            except Exception as exc:
                records[index].update(
                    valid=False,
                    error_type=type(exc).__name__,
                    error_text=str(exc),
                    t_decode_done_mono=time.perf_counter(),
                )
    for record in records:
        record["t_terminal_mono"] = record.get("t_decode_done_mono", record["t_http_terminal_mono"])
    start = min(r["t_submit_mono"] for r in records)
    http_end = max(r["t_http_terminal_mono"] for r in records)
    decode_end = max(r["t_terminal_mono"] for r in records)
    valid = [r for r in records if r["valid"]]
    frames = sum(r["output_frames"] for r in valid)
    video_seconds = sum(r["output_frames"] * r["fps_den"] / r["fps_num"] for r in valid)
    latencies = [r["t_body_done_mono"] - r["t_submit_mono"] for r in valid]
    summary = dict(
        requests=len(records),
        valid=len(valid),
        failed=len(records) - len(valid),
        peak_http_inflight=peak_total,
        peak_per_replica=peak,
        http_wall_seconds=http_end - start,
        decoded_wall_seconds=decode_end - start,
        valid_frames_http_per_second=frames / (http_end - start),
        valid_frames_available_per_second=frames / (decode_end - start),
        valid_clips_http_per_second=len(valid) / (http_end - start),
        valid_video_seconds_http_per_second=video_seconds / (http_end - start),
        successful_latency_sample_count=len(latencies),
        http_latency_p50=percentile(latencies, 0.5),
        http_latency_p95=percentile(latencies, 0.95),
        http_latency_p99=percentile(latencies, 0.99),
    )
    with (output / "requests.jsonl").open("w") as ledger:
        for record in records:
            ledger.write(json.dumps(dict(metadata, **record)) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--base-url", action="append", required=True, help="repeat for independent replicas; fixed index round robin"
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--decode-workers", type=int, default=2)
    args = parser.parse_args()
    if min(args.concurrency, args.requests, args.decode_workers, args.timeout) <= 0:
        parser.error("counts and timeout must be positive")
    metadata = json.loads(args.metadata.read_text())
    for key in ["arm", "base_sha", "source_sha", "topology", "round", "phase"]:
        if key not in metadata:
            parser.error(f"missing run metadata: {key}")
    result = asyncio.run(run(args, load_manifest(args.manifest), metadata))
    return int(result["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
