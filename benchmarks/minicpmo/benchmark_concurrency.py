#!/usr/bin/env python3
"""Run MiniCPM-o 4.5 online concurrency checks with trace attribution."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from vllm_omni.metrics.concurrency_trace import build_summary


def _audio_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _request_payload(
    model: str,
    reference_audio: str | None,
    request_index: int,
    max_tokens: int,
) -> dict[str, Any]:
    system_content: str | list[dict[str, Any]]
    if reference_audio is None:
        system_content = "You are a helpful assistant. Reply in text and speech."
    else:
        system_content = [
            {"type": "text", "text": "Use the voice in the audio prompt to synthesize new content."},
            {"type": "audio_url", "audio_url": {"url": reference_audio}},
            {"type": "text", "text": "You are a helpful assistant with the above voice style."},
        ]
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": f"Reply with exactly this short sentence: Hello from request {request_index}.",
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"use_tts_template": True},
    }


def _waveform_info(response: dict[str, Any]) -> tuple[int, int]:
    for choice in response.get("choices", []):
        audio = (choice.get("message") or {}).get("audio") or {}
        encoded = audio.get("data")
        if not encoded:
            continue
        raw_audio = base64.b64decode(encoded)
        try:
            with wave.open(io.BytesIO(raw_audio), "rb") as wav_file:
                return len(raw_audio), wav_file.getnframes()
        except (EOFError, wave.Error):
            return len(raw_audio), 0
    return 0, 0


def _send_request(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float,
    output_path: Path,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - benchmark endpoint is explicit
            status_code = response.status
            body = response.read()
    except HTTPError as exc:
        status_code = exc.code
        body = exc.read()
    latency_s = time.perf_counter() - started_at

    output_path.write_bytes(body)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {}
    audio_bytes, audio_frames = _waveform_info(parsed)
    return {
        "request_index": payload["messages"][1]["content"].rsplit(" ", 1)[-1].rstrip("."),
        "request_id": parsed.get("id"),
        "status_code": status_code,
        "latency_s": latency_s,
        "audio_bytes": audio_bytes,
        "audio_frames": audio_frames,
        "has_text": any(bool((choice.get("message") or {}).get("content")) for choice in parsed.get("choices", [])),
        "ok": status_code == 200 and audio_bytes > 44 and audio_frames > 0,
    }


def _trace_slice(trace_path: Path, start_line: int) -> list[dict[str, Any]]:
    stage_configs: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            record = json.loads(line)
            if record.get("event") == "stage_config" and record.get("stage_id") is not None:
                stage_configs[str(record["stage_id"])] = record
            if line_number > start_line:
                records.append(record)
    return [*stage_configs.values(), *records]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8099/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-audio", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    request_count = args.requests or args.concurrency
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.trace_path.open(encoding="utf-8") as trace_file:
        trace_start_line = sum(1 for _ in trace_file)
    reference_audio = _audio_data_url(args.reference_audio) if args.reference_audio is not None else None
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"

    wall_started_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _send_request,
                endpoint,
                _request_payload(args.model, reference_audio, request_index, args.max_tokens),
                args.timeout_s,
                args.output_dir / f"response_{request_index}.json",
            ): request_index
            for request_index in range(request_count)
        }
        for future in as_completed(futures):
            results.append(future.result())
    wall_time_s = time.perf_counter() - wall_started_at
    time.sleep(1.0)

    records = _trace_slice(args.trace_path, trace_start_line)
    tts_completions = [
        record for record in records if record.get("event") == "tts_slot_completed" and record.get("outcome") == "ok"
    ]
    for result in results:
        response_id = result.get("request_id")
        matches = [
            record
            for record in tts_completions
            if response_id is not None and str(record.get("request_id", "")).startswith(f"{response_id}-")
        ]
        result["trace_tts_matches"] = len(matches)
        result["trace_waveform_samples"] = [int(record.get("waveform_samples", 0)) for record in matches]
        result["trace_attributed"] = len(matches) == 1 and result["trace_waveform_samples"][0] > 0
        result["ok"] = result["ok"] and result["trace_attributed"]
    with (args.output_dir / "trace.jsonl").open("w", encoding="utf-8") as trace_file:
        for record in records:
            trace_file.write(json.dumps(record, sort_keys=True) + "\n")
    summary = build_summary(records)
    summary["client"] = {
        "concurrency": args.concurrency,
        "requests": request_count,
        "successful_requests": sum(result["ok"] for result in results),
        "wall_time_s": wall_time_s,
        "completed_requests_per_s": request_count / wall_time_s if wall_time_s > 0 else 0.0,
        "results": sorted(results, key=lambda result: int(result["request_index"])),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["client"], indent=2, sort_keys=True))
    return 0 if summary["client"]["successful_requests"] == request_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
