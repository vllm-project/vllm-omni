#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate public speech streaming under overlapping reference-conditioning load."""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import time
import wave
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import numpy as np
import soundfile as sf
from prometheus_client.parser import text_string_to_metric_families
from tqdm import tqdm

SAMPLE_RATE = 24000
TEXT = (
    "Request number {index}. The river moves quietly past the old stone bridge today. "
    "In the morning, people walk along the bank and listen to the birds in the trees. "
    "A small boat carries fresh bread to the market across the water. By evening, "
    "the streets become quiet again, and warm lights shine through the windows of the houses."
)


def main():
    """Run synchronized waves and persist all request outcomes, including failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("reference", type=Path, help="Speech recording, or directory of distinct recordings")
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8091/v1/audio/speech")
    parser.add_argument("--transport", choices=("http", "websocket"), default="http")
    parser.add_argument("--metrics-url", help="Defaults to the speech URL origin plus /metrics")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument(
        "--stage0-sequences-per-request",
        type=int,
        choices=(1, 2),
        default=1,
        help="Use 2 for Chatterbox Original with CFG enabled; scheduler gauges count raw sequences",
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument(
        "--save-audio", action="store_true", help="Save the first request of each wave as WAV after timing"
    )
    args = parser.parse_args()
    if min(args.concurrency, args.cycles, args.max_new_tokens, args.timeout) <= 0:
        parser.error("concurrency, cycles, token cap and timeout must be positive")
    references = prepare_references(args.reference, args.concurrency * args.cycles)
    report = asyncio.run(run(args, references))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_representative_audio(report, args.output)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logging.basicConfig(level=logging.INFO)
    logging.info("Saved %s; passed=%s", args.output, report["passed"])
    raise SystemExit(0 if report["passed"] else 1)


def save_representative_audio(report, output):
    """Write retained PCM only after every timed workload has completed."""
    for workload in report["workloads"]:
        for row in workload["requests"]:
            pcm = row.pop("_audio_pcm", None)
            if pcm is None:
                continue
            path = output.parent / f"{output.stem}-{workload['name']}-cycle{row['cycle']}.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(row["channels"])
                audio.setsampwidth(2)
                audio.setframerate(row["sample_rate"])
                audio.writeframes(pcm)
            row["audio_path"] = str(path.resolve())


def prepare_references(source, count):
    """Encode real speech; deterministic gain variants exercise distinct cache keys."""
    files = sorted(source.glob("*.wav")) if source.is_dir() else [source]
    if source.is_dir() and len(files) < count:
        raise ValueError(f"Reference directory requires at least {count} WAV recordings")
    references = []
    for index in range(count):
        wav, rate = sf.read(files[index] if source.is_dir() else source, dtype="float32", always_2d=True)
        if wav.shape[0] <= rate * 5 or not np.isfinite(wav).all() or not np.any(wav):
            raise ValueError("References must contain more than five seconds of finite, nonzero speech")
        wav = wav.mean(axis=1)
        if not source.is_dir():
            # Real speech with a different amplitude envelope, not different URLs for identical audio.
            phase = np.arange(len(wav), dtype=np.float32) / rate
            wav = wav * (0.8 + 0.1 * np.sin(phase * (index + 1) / count))
        buffer = io.BytesIO()
        sf.write(buffer, wav, rate, format="WAV", subtype="FLOAT")
        data = buffer.getvalue()
        references.append(
            ("data:audio/wav;base64," + base64.b64encode(data).decode(), hashlib.sha256(data).hexdigest())
        )
    if len({digest for _, digest in references}) != count:
        raise ValueError("Cold workload requires distinct reference waveforms")
    return references


async def request(session, url, payload, gate, state, reference_hash, retain_audio=False):
    """Measure latency from synchronized submission to first PCM bytes and EOF."""
    await gate.wait()
    started = time.perf_counter()
    state["active"] += 1
    state["peak"] = max(state["peak"], state["active"])
    result = {"input": payload["input"], "seed": payload["seed"], "reference_sha256": reference_hash, "ok": False}
    data = bytearray()
    arrivals = []
    playback_deficit = 0.0
    try:
        async with session.post(url, json=payload) as response:
            result["status"] = response.status
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}: {(await response.text())[:500]}")
            result["sample_rate"] = int(response.headers.get("X-Audio-Sample-Rate", "0"))
            result["channels"] = int(response.headers.get("X-Audio-Channels", "0"))
            async for chunk in response.content.iter_any():
                if chunk:
                    arrived = time.perf_counter() - started
                    if not data:
                        result["ttfa_seconds"] = arrived
                    # Mono 24 kHz int16 PCM, verified below before reporting.
                    playback_deficit = max(
                        playback_deficit, arrived - result["ttfa_seconds"] - len(data) / (SAMPLE_RATE * 2)
                    )
                    arrivals.append({"seconds": arrived, "pcm_bytes": len(chunk)})
                    data.extend(chunk)
            result.update(validate_pcm(data, result["sample_rate"], result["channels"]))
            result["playback_deficit_seconds"] = playback_deficit
            result["chunk_arrivals"] = arrivals
            result["ok"] = True
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
        result["error"] = str(error) or type(error).__name__
    finally:
        result["e2e_seconds"] = time.perf_counter() - started
        state["active"] -= 1
    if result["ok"]:
        result["rtf"] = result["e2e_seconds"] / result["audio_seconds"]
    if retain_audio and result["ok"]:
        result["_audio_pcm"] = bytes(data)
    return result


async def websocket_request(session, url, payload, gate, state, reference_hash, retain_audio=False):
    """Measure one sentence-segmented session, preserving each segment outcome."""
    await gate.wait()
    started = time.perf_counter()
    state["active"] += 1
    state["peak"] = max(state["peak"], state["active"])
    result = {
        "input": payload["input"],
        "seed": payload["seed"],
        "reference_sha256": reference_hash,
        "ok": False,
        "segments": [],
    }
    pcm = bytearray()
    segment = None
    segment_pcm = bytearray()
    try:
        async with session.ws_connect(url, max_msg_size=16 * 1024 * 1024) as websocket:
            config = {
                key: payload[key] for key in ("model", "voice", "seed", "max_new_tokens", "ref_audio") if key in payload
            }
            config.update(type="session.config", response_format="pcm", stream_audio=True, split_granularity="sentence")
            await websocket.send_json(config)
            first, boundary, remainder = payload["input"].partition(". ")
            chunks = [first + ". ", remainder] if boundary else [first]
            for chunk in chunks:
                await websocket.send_json({"type": "input.text", "text": chunk})
            await websocket.send_json({"type": "input.done"})
            while True:
                message = await asyncio.wait_for(websocket.receive(), timeout=session.timeout.total)
                if message.type == aiohttp.WSMsgType.BINARY:
                    if segment is None:
                        raise ValueError("PCM arrived without audio.start")
                    if message.data:
                        now = time.perf_counter() - started
                        result.setdefault("ttfa_seconds", now)
                        segment.setdefault("ttfa_seconds", now)
                        segment_pcm.extend(message.data)
                elif message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)
                    kind = event.get("type")
                    if kind == "error":
                        raise ValueError(event.get("message", "WebSocket server error"))
                    if kind == "audio.start":
                        if segment is not None or event.get("sentence_index") != len(result["segments"]):
                            raise ValueError("Unexpected sentence ordering")
                        segment = {
                            "sentence_index": event["sentence_index"],
                            "input": event.get("sentence_text"),
                            "sample_rate": event.get("sample_rate", 0),
                            "channels": event.get("channels", 0),
                            "ok": False,
                        }
                        result["segments"].append(segment)
                        segment_pcm.clear()
                    elif kind == "audio.done":
                        if segment is None or event.get("sentence_index") != segment["sentence_index"]:
                            raise ValueError("Unmatched audio.done")
                        if event.get("error"):
                            raise ValueError("Sentence generation failed")
                        segment.update(validate_pcm(segment_pcm, segment["sample_rate"], segment["channels"]))
                        segment.update(ok=True, e2e_seconds=time.perf_counter() - started)
                        pcm.extend(segment_pcm)
                        segment = None
                    elif kind == "session.done":
                        if segment is not None or event.get("total_sentences") != len(result["segments"]):
                            raise ValueError("Incomplete sentence flush")
                        if " ".join(s["input"].strip() for s in result["segments"]) != payload["input"].strip():
                            raise ValueError("Missing sentence audio")
                        result.update(validate_pcm(pcm, SAMPLE_RATE, 1))
                        result.update(ok=True, sample_rate=SAMPLE_RATE, channels=1)
                        await websocket.send_json({"type": "session.close"})
                        break
                else:
                    result["websocket_message_type"] = message.type.name
                    result["websocket_close_code"] = (
                        int(message.data) if message.type == aiohttp.WSMsgType.CLOSE else websocket.close_code
                    )
                    result["websocket_close_reason"] = str(message.extra or "")
                    failure = websocket.exception()
                    if failure is not None or message.type == aiohttp.WSMsgType.ERROR:
                        result["websocket_exception"] = str(failure or message.data)
                    raise ValueError(
                        "WebSocket closed before session.done: "
                        f"type={message.type.name}, code={result['websocket_close_code']}, "
                        f"reason={result['websocket_close_reason']!r}, "
                        f"exception={result.get('websocket_exception', '')!r}"
                    )
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
        result["error"] = str(error) or type(error).__name__
        if segment is not None:
            segment["error"] = result["error"]
    finally:
        result["e2e_seconds"] = time.perf_counter() - started
        state["active"] -= 1
    if retain_audio and result["ok"]:
        result["_audio_pcm"] = bytes(pcm)
    return result


def validate_pcm(data, sample_rate, channels):
    """Reject empty, silent, malformed or incorrectly declared PCM streams."""
    if sample_rate != SAMPLE_RATE or channels != 1 or not data or len(data) % 2:
        raise ValueError("Expected nonempty mono s16le PCM at 24000 Hz")
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
    if not np.isfinite(samples).all() or not np.any(samples):
        raise ValueError("Audio must be finite and nonzero")
    return {
        "audio_seconds": len(samples) / sample_rate,
        "audio_sha256": hashlib.sha256(data).hexdigest(),
        "rms": float(np.sqrt(np.mean(samples**2))),
        "bytes": len(data),
    }


async def run(args, references):
    """Exercise cold distinct, repeated-reference warm, and built-in voice waves."""
    report = {
        "model": args.model,
        "transport": getattr(args, "transport", "http"),
        "url": args.url,
        "concurrency": args.concurrency,
        "stage0_sequences_per_request": args.stage0_sequences_per_request,
        "stage0_capacity_target": args.concurrency * args.stage0_sequences_per_request,
        "cycles": args.cycles,
        "max_new_tokens": args.max_new_tokens,
        "reference_source": str(args.reference),
        "reference_mode": "recordings" if args.reference.is_dir() else "deterministic_envelope_variants",
        "isolation_limit": "Distinct text/seed and PCM hashes detect exact replay, not semantic voice leakage.",
        "server_active_sequences": "Polled stage gauges count sequences; CFG may use two sequences per user request.",
        "workloads": [],
    }
    request_method = websocket_request if getattr(args, "transport", "http") == "websocket" else request
    parts = urlsplit(args.url)
    metrics_url = getattr(args, "metrics_url", None) or urlunsplit(
        (parts.scheme.replace("ws", "http", 1), parts.netloc, "/metrics", "", "")
    )
    report["metrics_url"] = metrics_url
    connector = aiohttp.TCPConnector(limit=args.concurrency + 1)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        for workload in ("cold_distinct_reference", "warm_repeated_reference", "builtin_voice"):
            rows = []
            state = {"active": 0, "peak": 0}
            metrics = {"samples": 0, "errors": [], "peaks": {}, "waves": []}
            if workload == "warm_repeated_reference":
                gate = asyncio.Event()
                gate.set()
                prime = {
                    "model": args.model,
                    "input": "Prime the repeated reference cache.",
                    "seed": 12345,
                    "voice": "default",
                    "ref_audio": references[0][0],
                    "response_format": "pcm",
                    "sample_rate": SAMPLE_RATE,
                    "stream": True,
                    "stream_format": "audio",
                    "max_new_tokens": args.max_new_tokens,
                }
                report["warmup"] = await request_method(session, args.url, prime, gate, state, references[0][1])
            started = time.perf_counter()
            for cycle in tqdm(range(args.cycles), desc=workload):
                gate = asyncio.Event()
                tasks = []
                for slot in range(args.concurrency):
                    index = cycle * args.concurrency + slot
                    payload = {
                        "model": args.model,
                        "input": TEXT.format(index=index),
                        "voice": "default",
                        "seed": index,
                        "response_format": "pcm",
                        "sample_rate": SAMPLE_RATE,
                        "stream": True,
                        "stream_format": "audio",
                        "max_new_tokens": args.max_new_tokens,
                    }
                    digest = None
                    if workload != "builtin_voice":
                        payload["ref_audio"], digest = references[index if workload == "cold_distinct_reference" else 0]
                    tasks.append(
                        asyncio.create_task(
                            request_method(
                                session,
                                args.url,
                                payload,
                                gate,
                                state,
                                digest,
                                retain_audio=getattr(args, "save_audio", False) and slot == 0,
                            )
                        )
                    )
                await asyncio.sleep(0)
                gate.set()
                stopped = asyncio.Event()
                wave_metrics = {"cycle": cycle, "samples": 0, "errors": [], "peaks": {}}
                poller = asyncio.create_task(poll_metrics(session, metrics_url, stopped, wave_metrics))
                try:
                    wave = await asyncio.gather(*tasks)
                finally:
                    stopped.set()
                    await poller
                metrics["waves"].append(wave_metrics)
                metrics["samples"] += wave_metrics["samples"]
                metrics["errors"].extend(wave_metrics["errors"])
                for name, value in wave_metrics["peaks"].items():
                    metrics["peaks"][name] = max(metrics["peaks"].get(name, 0), value)
                for row in wave:
                    row["cycle"] = cycle
                rows.extend(wave)
            elapsed = time.perf_counter() - started
            metrics["available"] = metrics["samples"] > 0 and all(
                key in metrics["peaks"]
                for key in ("stage_0_running", "stage_0_waiting", "stage_1_running", "stage_1_waiting")
            )
            summary = summarize(rows, elapsed)
            summary.update(
                {
                    "name": workload,
                    "elapsed_seconds": elapsed,
                    "peak_outstanding_clients": state["peak"],
                    "requests": rows,
                    "server_metrics": metrics,
                    "capacity_confirmed": metrics["available"]
                    and all(
                        wave["peaks"].get("stage_0_running", 0) >= report["stage0_capacity_target"]
                        for wave in metrics["waves"]
                    ),
                }
            )
            if getattr(args, "transport", "http") == "websocket":
                segments = [segment for row in rows for segment in row["segments"]]
                summary["segments"] = summarize(segments, elapsed)
            report["workloads"].append(summary)
    report["passed"] = report["warmup"]["ok"] and all(
        w["errors"] == 0 and w["peak_outstanding_clients"] == args.concurrency and w["duplicate_audio_hashes"] == 0
        for w in report["workloads"]
    )
    report["functional_passed"] = report["passed"]
    report["capacity_confirmed"] = report["functional_passed"] and all(
        workload["capacity_confirmed"] for workload in report["workloads"]
    )
    return report


async def poll_metrics(session, url, stopped, result):
    """Sample scheduler gauges every quarter second without occupying speech connections."""
    while not stopped.is_set():
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    raise ValueError(f"Metrics HTTP {response.status}")
                gauges = parse_scheduler_metrics(await response.text())
                if not gauges:
                    raise ValueError("No scheduler gauges exposed")
                result["samples"] += 1
                for name, value in gauges.items():
                    result["peaks"][name] = max(result["peaks"].get(name, 0), value)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            message = str(error) or type(error).__name__
            if message not in result["errors"]:
                result["errors"].append(message)
        try:
            await asyncio.wait_for(stopped.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            pass


def parse_scheduler_metrics(text):
    """Sum replica gauges per stage using the official Prometheus text parser."""
    gauges = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name not in {
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
                "vllm_omni:num_requests_running",
            }:
                continue
            if not np.isfinite(sample.value) or sample.value < 0:
                raise ValueError("Scheduler gauges must be finite and nonnegative")
            if sample.name == "vllm_omni:num_requests_running":
                key = "omni_running"
            else:
                stage = sample.labels.get("stage")
                if stage not in {"0", "1"}:
                    continue
                key = f"stage_{stage}_{sample.name.rsplit('_', 1)[-1]}"
            gauges[key] = gauges.get(key, 0) + sample.value
    return gauges


def summarize(rows, elapsed):
    """Report tails only for valid audio and retain failures in the denominator."""
    good = [row for row in rows if row["ok"]]
    result = {
        "completed": len(good),
        "errors": len(rows) - len(good),
        "request_throughput_per_second": len(good) / elapsed,
        "audio_throughput_seconds_per_second": sum(row["audio_seconds"] for row in good) / elapsed,
        "duplicate_audio_hashes": len(good) - len({row["audio_sha256"] for row in good}),
    }
    for metric in ("ttfa_seconds", "e2e_seconds", "rtf", "playback_deficit_seconds"):
        # Playback timing is currently captured only for HTTP PCM responses.
        values = [row[metric] for row in good if metric in row]
        result[metric] = (
            dict(zip(("p50", "p95", "p99", "max"), map(float, np.percentile(values, [50, 95, 99, 100]))))
            if values
            else None
        )
    return result


if __name__ == "__main__":
    main()
