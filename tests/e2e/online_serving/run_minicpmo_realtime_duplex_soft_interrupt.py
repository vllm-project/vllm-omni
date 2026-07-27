"""Validate MiniCPM-o Realtime duplex soft-interrupt delta streaming.

This E2E driver runs the public ``realtime_duplex_demo.py`` against a live
duplex backend. Arbitrary audio defaults to the model-policy lifecycle contract.
The stronger response-required mode binds the two-response contract to a known
input checksum and expected second response.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import sys
import uuid
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_PATH = REPO_ROOT / "examples/online_serving/minicpmo/realtime_duplex_demo.py"
AUDIO_DELTA_EVENTS = {"response.audio.delta", "response.output_audio.delta"}
TRANSCRIPT_DELTA_EVENTS = {
    "response.audio_transcript.delta",
    "response.output_text.delta",
}


def _canonical_path(path: str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _validate_input_sha256(path: Path, expected: str) -> str:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected.lower():
        raise ValueError(f"input WAV SHA256 mismatch: expected {expected.lower()}, got {actual}")
    return actual


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_events(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if isinstance(event, dict):
            events.append(event)
    return events


def _event_response_id(event: dict[str, object]) -> str | None:
    response_id = event.get("response_id")
    if isinstance(response_id, str):
        return response_id
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    return None


def _event_received_at_s(event: dict[str, object]) -> float | None:
    received_at = event.get("_client_received_at_s")
    if isinstance(received_at, int | float):
        return float(received_at)
    return None


def _first_event_index(events: list[dict[str, object]], event_type: str) -> int | None:
    for index, event in enumerate(events):
        if event.get("type") == event_type:
            return index
    return None


def _event_counts(events: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("type"))
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _response_created_ids(events: list[dict[str, object]]) -> list[str]:
    response_ids: list[str] = []
    for event in events:
        if event.get("type") != "response.created":
            continue
        response_id = _event_response_id(event)
        if response_id and response_id not in response_ids:
            response_ids.append(response_id)
    return response_ids


def _audio_delta_duration_ms(event: dict[str, object]) -> float | None:
    encoded = event.get("delta") or event.get("audio")
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        raw = base64.b64decode(encoded)
    except ValueError:
        return None
    fmt = str(event.get("format") or event.get("audio_format") or "pcm16").lower()
    sample_rate = event.get("sample_rate_hz") or event.get("sample_rate") or 24_000
    if not isinstance(sample_rate, int | float) or sample_rate <= 0:
        return None
    if "wav" in fmt:
        try:
            with wave.open(io.BytesIO(raw), "rb") as wav_file:
                return round((wav_file.getnframes() * 1000.0) / wav_file.getframerate(), 3)
        except (EOFError, wave.Error):
            return None
    bytes_per_sample = 4 if "f32" in fmt else 2
    return round((len(raw) * 1000.0) / (bytes_per_sample * float(sample_rate)), 3)


def _normalize_text(text: str) -> str:
    return "".join(text.split())


def _compact_sequence(events: list[dict[str, object]]) -> list[str]:
    sequence: list[str] = []
    for event in events:
        event_type = event.get("type")
        response_id = _event_response_id(event)
        suffix = f" {response_id[-8:]}" if response_id else ""
        if event_type == "response.listen":
            sequence.append("listen")
        elif event_type == "response.created":
            sequence.append(f"response.created{suffix}")
        elif event_type == "response.speak":
            sequence.append(f"speak{suffix}")
        elif event_type in AUDIO_DELTA_EVENTS:
            sequence.append(f"audio.delta{suffix}")
        elif event_type in TRANSCRIPT_DELTA_EVENTS:
            delta = str(event.get("delta") or "")
            sequence.append(f"text.delta{suffix}> {delta}")
        elif event_type == "response.done":
            sequence.append(f"response.done{suffix}")
        elif event_type == "input_audio_buffer.committed":
            sequence.append("input.committed")
    return sequence


def _response_summary(
    events: list[dict[str, object]],
    response_id: str,
    *,
    t0: float | None,
) -> dict[str, object]:
    created_indices = [
        index
        for index, event in enumerate(events)
        if event.get("type") == "response.created" and _event_response_id(event) == response_id
    ]
    done_indices = [
        index
        for index, event in enumerate(events)
        if event.get("type") == "response.done" and _event_response_id(event) == response_id
    ]
    audio_indices = [
        index
        for index, event in enumerate(events)
        if event.get("type") in AUDIO_DELTA_EVENTS and _event_response_id(event) == response_id
    ]
    transcript_indices = [
        index
        for index, event in enumerate(events)
        if event.get("type") in TRANSCRIPT_DELTA_EVENTS and _event_response_id(event) == response_id
    ]
    done_index = done_indices[0] if done_indices else None
    audio_before_done_ok = done_index is not None and all(index < done_index for index in audio_indices)
    stale_audio_count = sum(index > done_index for index in audio_indices) if done_index is not None else 0

    def offset_ms(index: int) -> float | None:
        if t0 is None:
            return None
        received_at = _event_received_at_s(events[index])
        return round((received_at - t0) * 1000.0, 3) if received_at is not None else None

    return {
        "response_id": response_id,
        "created_index": created_indices[0] if created_indices else None,
        "done_indices": done_indices,
        "created_offset_ms": offset_ms(created_indices[0]) if created_indices else None,
        "done_offset_ms": offset_ms(done_indices[0]) if done_indices else None,
        "audio_delta_count": len(audio_indices),
        "transcript_delta_count": len(transcript_indices),
        "audio_delta_offsets_ms": [offset_ms(index) for index in audio_indices],
        "audio_duration_ms": [_audio_delta_duration_ms(events[index]) for index in audio_indices],
        "transcript": "".join(str(events[index].get("delta") or "") for index in transcript_indices),
        "one_done": len(done_indices) == 1,
        "audio_before_done_ok": audio_before_done_ok,
        "stale_audio_count": stale_audio_count,
    }


def summarize_artifacts(
    *,
    output_dir: Path,
    validation_mode: str,
    min_responses: int,
    min_audio_deltas_per_response: int,
    expect_second_response_substring: str | None,
) -> dict[str, object]:
    if validation_mode not in {"model-policy", "response-required"}:
        raise ValueError(f"unsupported validation mode: {validation_mode}")
    result = _read_json(output_dir / "result.json")
    events = _read_events(output_dir / "events.jsonl")
    response_ids = _response_created_ids(events)
    times = [_event_received_at_s(event) for event in events]
    clean_times = [received_at for received_at in times if received_at is not None]
    t0 = min(clean_times) if clean_times else None
    response_summaries = [_response_summary(events, response_id, t0=t0) for response_id in response_ids]
    commit_index = _first_event_index(events, "input_audio_buffer.committed")
    first_created_index = next(
        (summary["created_index"] for summary in response_summaries if isinstance(summary.get("created_index"), int)),
        None,
    )
    first_done_index = (
        response_summaries[0]["done_indices"][0]
        if response_summaries and response_summaries[0]["done_indices"]
        else None
    )
    second_created_index = (
        response_summaries[1]["created_index"]
        if len(response_summaries) >= 2 and isinstance(response_summaries[1].get("created_index"), int)
        else None
    )
    last_done_index = next(
        (summary["done_indices"][0] for summary in reversed(response_summaries) if summary.get("done_indices")),
        None,
    )
    listen_indices = [index for index, event in enumerate(events) if event.get("type") == "response.listen"]
    effective_min_responses = min_responses if validation_mode == "response-required" else 1
    enough_responses = len(response_summaries) >= effective_min_responses
    response_lifecycle_ok = enough_responses and all(bool(summary.get("one_done")) for summary in response_summaries)
    multi_delta_ok = enough_responses and all(
        int(summary.get("audio_delta_count", 0)) >= min_audio_deltas_per_response for summary in response_summaries
    )
    response_audio_contract_ok = enough_responses and all(
        summary.get("audio_before_done_ok") is True and int(summary.get("stale_audio_count", 0)) == 0
        for summary in response_summaries
    )
    response_before_final_commit = (
        first_created_index is not None and commit_index is not None and first_created_index < commit_index
    )
    second_response_before_final_commit = (
        second_created_index is not None and commit_index is not None and second_created_index < commit_index
    )
    listen_before_first_response = first_created_index is not None and any(
        index < first_created_index for index in listen_indices
    )
    listen_between_responses = (
        first_done_index is not None
        and second_created_index is not None
        and any(first_done_index < index < second_created_index for index in listen_indices)
    )
    listen_after_last_done = last_done_index is not None and any(index > last_done_index for index in listen_indices)
    listen_after_response_before_commit = (
        last_done_index is not None
        and commit_index is not None
        and any(last_done_index < index < commit_index for index in listen_indices)
    )
    final_listen_after_commit = commit_index is not None and any(index > commit_index for index in listen_indices)
    transcript = "".join(str(summary.get("transcript") or "") for summary in response_summaries)
    second_response_transcript = (
        str(response_summaries[1].get("transcript") or "") if len(response_summaries) >= 2 else ""
    )
    second_response_transcript_expectation_ok = not expect_second_response_substring or _normalize_text(
        expect_second_response_substring
    ) in _normalize_text(second_response_transcript)
    error_events = [
        event
        for event in events
        if event.get("type") == "error"
        or str(event.get("type") or "").endswith(".cancelled")
        or str(event.get("type") or "").endswith(".truncated")
    ]
    cancelled_count = sum(
        1
        for event in events
        if event.get("type") == "response.done"
        and isinstance(event.get("response"), dict)
        and event["response"].get("status") == "cancelled"
    )
    result_ok = result.get("ok") is True
    common_contract_ok = bool(
        result_ok
        and not error_events
        and cancelled_count == 0
        and enough_responses
        and response_lifecycle_ok
        and multi_delta_ok
        and response_audio_contract_ok
        and response_before_final_commit
        and listen_after_response_before_commit
        and final_listen_after_commit
    )
    mode_contract_ok = validation_mode == "model-policy" or (
        second_response_before_final_commit
        and listen_before_first_response
        and listen_between_responses
        and listen_after_last_done
        and second_response_transcript_expectation_ok
    )
    ok = common_contract_ok and mode_contract_ok
    return {
        "ok": ok,
        "result_ok": result_ok,
        "validation_mode": validation_mode,
        "event_counts": _event_counts(events),
        "response_ids": response_ids,
        "response_summaries": response_summaries,
        "min_responses": min_responses,
        "effective_min_responses": effective_min_responses,
        "min_audio_deltas_per_response": min_audio_deltas_per_response,
        "enough_responses": enough_responses,
        "response_lifecycle_ok": response_lifecycle_ok,
        "multi_delta_ok": multi_delta_ok,
        "response_audio_contract_ok": response_audio_contract_ok,
        "response_before_final_commit": response_before_final_commit,
        "second_response_before_final_commit": second_response_before_final_commit,
        "listen_before_first_response": listen_before_first_response,
        "listen_between_responses": listen_between_responses,
        "listen_after_last_done": listen_after_last_done,
        "listen_after_response_before_commit": listen_after_response_before_commit,
        "final_listen_after_commit": final_listen_after_commit,
        "transcript": transcript,
        "second_response_transcript": second_response_transcript,
        "expect_second_response_substring": expect_second_response_substring,
        "second_response_transcript_expectation_ok": second_response_transcript_expectation_ok,
        "error_count": len(error_events),
        "cancelled_count": cancelled_count,
        "compact_sequence": _compact_sequence(events),
        "output_dir": str(output_dir),
    }


async def run_soft_interrupt(args: argparse.Namespace) -> dict[str, object]:
    input_wav = _canonical_path(args.input_wav)
    output_dir = _canonical_path(args.output_dir)
    if not input_wav.is_file():
        raise ValueError(f"input WAV does not exist: {input_wav}")
    input_sha256 = hashlib.sha256(input_wav.read_bytes()).hexdigest()
    if args.input_sha256:
        input_sha256 = _validate_input_sha256(input_wav, args.input_sha256)
    command = [
        sys.executable,
        str(DEMO_PATH),
        "--url",
        args.url,
        "--model",
        args.model,
        "--input-wav",
        str(input_wav),
        "--output-dir",
        str(output_dir),
        "--chunk-ms",
        str(args.chunk_ms),
        "--timeout-s",
        str(args.timeout_s),
        "--session-id",
        f"duplex-soft-interrupt-{uuid.uuid4().hex}",
    ]
    command.extend(["--ref-audio", str(_canonical_path(args.ref_audio))])
    if args.require_audio:
        command.append("--require-audio")
    if args.no_realtime_pacing:
        command.append("--no-realtime-pacing")

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=args.timeout_s + 30.0,
        )
    except TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        return {
            "ok": False,
            "returncode": -1,
            "stdout_tail": stdout_bytes.decode(errors="replace")[-4000:],
            "stderr_tail": (stderr_bytes.decode(errors="replace") + "\nclient timeout")[-4000:],
            "output_dir": str(output_dir),
        }

    summary = summarize_artifacts(
        output_dir=output_dir,
        validation_mode=args.validation_mode,
        min_responses=args.min_responses,
        min_audio_deltas_per_response=args.min_audio_deltas_per_response,
        expect_second_response_substring=args.expect_second_response_substring,
    )
    summary.update(
        {
            "returncode": process.returncode if process.returncode is not None else -1,
            "input_wav": str(input_wav),
            "input_sha256": input_sha256,
            "stdout_tail": stdout_bytes.decode(errors="replace")[-4000:],
            "stderr_tail": stderr_bytes.decode(errors="replace")[-4000:],
        }
    )
    summary["ok"] = bool(summary["ok"] and summary["returncode"] == 0)
    summary_output = Path(args.summary_output) if args.summary_output else output_dir / "summary.json"
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--require-audio", action="store_true")
    parser.add_argument("--no-realtime-pacing", action="store_true")
    parser.add_argument(
        "--validation-mode",
        choices=("model-policy", "response-required"),
        default="model-policy",
    )
    parser.add_argument("--min-responses", type=int, default=2)
    parser.add_argument("--min-audio-deltas-per-response", type=int, default=2)
    parser.add_argument("--input-sha256")
    parser.add_argument("--expect-second-response-substring")
    args = parser.parse_args()
    if args.validation_mode == "response-required" and args.min_responses < 2:
        parser.error("--min-responses must be at least 2")
    if args.validation_mode == "response-required" and not args.input_sha256:
        parser.error("--input-sha256 is required in response-required mode")
    if args.validation_mode == "response-required" and not args.expect_second_response_substring:
        parser.error("--expect-second-response-substring is required in response-required mode")
    if args.min_audio_deltas_per_response < 1:
        parser.error("--min-audio-deltas-per-response must be positive")
    return args


def main() -> None:
    result = asyncio.run(run_soft_interrupt(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
