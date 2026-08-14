"""Run two MiniCPM-o Realtime duplex demo processes concurrently.

This E2E driver covers the same process boundary users exercise manually:
two independent ``realtime_duplex_demo.py`` invocations, two distinct input
WAVs, and two distinct output directories.
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


def _canonical_path(path: str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_pair_args(args: argparse.Namespace) -> dict[str, object]:
    input_a = _canonical_path(args.input_wav_a)
    input_b = _canonical_path(args.input_wav_b)
    output_a = _canonical_path(args.output_dir_a)
    output_b = _canonical_path(args.output_dir_b)
    if output_a == output_b:
        raise ValueError("output directories must be different")
    if not input_a.is_file():
        raise ValueError(f"input WAV does not exist: {input_a}")
    if not input_b.is_file():
        raise ValueError(f"input WAV does not exist: {input_b}")
    sha_a = _sha256_file(input_a)
    sha_b = _sha256_file(input_b)
    if sha_a == sha_b:
        raise ValueError("input WAV files must have different content")
    return {
        "input_wav_a": str(input_a),
        "input_wav_b": str(input_b),
        "input_sha256_a": sha_a,
        "input_sha256_b": sha_b,
        "output_dir_a": str(output_a),
        "output_dir_b": str(output_b),
    }


def _event_response_id(event: dict[str, object]) -> str | None:
    response_id = event.get("response_id")
    if isinstance(response_id, str):
        return response_id
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    return None


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


def _response_ids_from_artifacts(result: dict[str, object], events: list[dict[str, object]]) -> list[str]:
    response_ids = result.get("response_ids")
    if isinstance(response_ids, list):
        ids = [item for item in response_ids if isinstance(item, str)]
        if ids:
            return ids
    seen: list[str] = []
    for event in events:
        response_id = _event_response_id(event)
        if response_id is not None and response_id not in seen:
            seen.append(response_id)
    return seen


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
            with wave.open(io.BytesIO(raw), "rb") as wav:
                return round((wav.getnframes() * 1000.0) / wav.getframerate(), 3)
        except (EOFError, wave.Error):
            return None
    bytes_per_sample = 4 if "f32" in fmt else 2
    return round((len(raw) * 1000.0) / (bytes_per_sample * float(sample_rate)), 3)


def _audio_delta_received_at_s(event: dict[str, object]) -> float | None:
    raw = event.get("_client_received_at_s")
    if isinstance(raw, int | float):
        return float(raw)
    return None


def _session_audio_contract_ok(
    *,
    audio_delta_count: int,
    delta_response_ids: list[str],
    response_ids: list[str],
    first_audio_delta_before_done: bool,
) -> bool:
    if audio_delta_count <= 0 or not response_ids:
        return False
    if len(delta_response_ids) != audio_delta_count:
        return False
    delta_id_set = set(delta_response_ids)
    if len(delta_id_set) != 1:
        return False
    return first_audio_delta_before_done and next(iter(delta_id_set)) in set(response_ids)


def _summarize_session(
    *,
    label: str,
    input_wav: str,
    output_dir: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> dict[str, object]:
    output_path = Path(output_dir)
    result = _read_json(output_path / "result.json")
    events = _read_events(output_path / "events.jsonl")
    audio_delta_indices = [index for index, event in enumerate(events) if event.get("type") in AUDIO_DELTA_EVENTS]
    done_indices = [index for index, event in enumerate(events) if event.get("type") == "response.done"]
    response_ids = _response_ids_from_artifacts(result, events)
    delta_response_ids = [
        response_id
        for response_id in (_event_response_id(events[index]) for index in audio_delta_indices)
        if response_id is not None
    ]
    first_audio_delta_before_done = bool(audio_delta_indices and done_indices) and min(audio_delta_indices) < min(
        done_indices
    )
    audio_delta_events = [events[index] for index in audio_delta_indices]
    first_received_at_s = next(
        (
            received_at
            for received_at in (_audio_delta_received_at_s(event) for event in audio_delta_events)
            if received_at is not None
        ),
        None,
    )
    audio_delta_timings = []
    for event in audio_delta_events:
        received_at_s = _audio_delta_received_at_s(event)
        audio_delta_timings.append(
            {
                "response_id": _event_response_id(event),
                "received_at_s": received_at_s,
                "arrival_offset_ms": (
                    round((received_at_s - first_received_at_s) * 1000.0, 3)
                    if received_at_s is not None and first_received_at_s is not None
                    else None
                ),
                "audio_duration_ms": _audio_delta_duration_ms(event),
            }
        )
    all_audio_deltas_same_response = (
        len(delta_response_ids) == len(audio_delta_indices) and len(set(delta_response_ids)) == 1
    )
    response_id_nonempty = bool(response_ids)
    response_audio_contract_ok = _session_audio_contract_ok(
        audio_delta_count=len(audio_delta_indices),
        delta_response_ids=delta_response_ids,
        response_ids=response_ids,
        first_audio_delta_before_done=first_audio_delta_before_done,
    )
    return {
        "label": label,
        "input_wav": input_wav,
        "output_dir": output_dir,
        "returncode": returncode,
        "ok": returncode == 0 and result.get("ok") is True and response_audio_contract_ok,
        "result_ok": result.get("ok"),
        "response_ids": response_ids,
        "response_id_nonempty": response_id_nonempty,
        "audio_bytes": result.get("audio_bytes", 0),
        "audio_delta_count": len(audio_delta_indices),
        "audio_delta_response_ids": delta_response_ids,
        "audio_delta_timings": audio_delta_timings,
        "all_audio_deltas_same_response": all_audio_deltas_same_response,
        "first_audio_delta_before_done": first_audio_delta_before_done,
        "response_audio_contract_ok": response_audio_contract_ok,
        "event_count": len(events),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def _identity_isolation_ok(sessions: list[dict[str, object]]) -> bool:
    seen: set[str] = set()
    for session in sessions:
        ids = {response_id for response_id in session.get("response_ids", []) if isinstance(response_id, str)}
        if not ids:
            return False
        if ids & seen:
            return False
        seen.update(ids)
    return True


async def _run_demo_process(
    *,
    label: str,
    args: argparse.Namespace,
    input_wav: str,
    output_dir: str,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(DEMO_PATH),
        "--url",
        args.url,
        "--model",
        args.model,
        "--input-wav",
        input_wav,
        "--output-dir",
        output_dir,
        "--chunk-ms",
        str(args.chunk_ms),
        "--timeout-s",
        str(args.timeout_s),
        "--session-id",
        f"duplex-pair-{label}-{uuid.uuid4().hex}",
    ]
    if args.no_realtime_pacing:
        command.append("--no-realtime-pacing")
    if args.require_audio:
        command.append("--require-audio")
    command.extend(["--ref-audio", str(args.ref_audio)])

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=args.timeout_s + 15.0,
        )
    except TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        return _summarize_session(
            label=label,
            input_wav=input_wav,
            output_dir=output_dir,
            returncode=-1,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=(stderr_bytes.decode(errors="replace") + "\nclient timeout"),
        )
    return _summarize_session(
        label=label,
        input_wav=input_wav,
        output_dir=output_dir,
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=stdout_bytes.decode(errors="replace"),
        stderr=stderr_bytes.decode(errors="replace"),
    )


async def run_pair(args: argparse.Namespace) -> dict[str, object]:
    validated = _validate_pair_args(args)
    Path(validated["output_dir_a"]).mkdir(parents=True, exist_ok=True)
    Path(validated["output_dir_b"]).mkdir(parents=True, exist_ok=True)

    sessions = await asyncio.gather(
        _run_demo_process(
            label="a",
            args=args,
            input_wav=str(validated["input_wav_a"]),
            output_dir=str(validated["output_dir_a"]),
        ),
        _run_demo_process(
            label="b",
            args=args,
            input_wav=str(validated["input_wav_b"]),
            output_dir=str(validated["output_dir_b"]),
        ),
    )
    identity_isolation_ok = _identity_isolation_ok(sessions)
    min_audio_delta_ok = all(
        int(session.get("audio_delta_count", 0)) >= args.min_audio_deltas_per_session for session in sessions
    )
    response_audio_contract_ok = all(session.get("response_audio_contract_ok") is True for session in sessions)
    result = {
        "ok": (
            all(session.get("ok") is True for session in sessions)
            and identity_isolation_ok
            and min_audio_delta_ok
            and response_audio_contract_ok
        ),
        "input_wavs_distinct": True,
        "output_dirs_distinct": True,
        "identity_isolation_ok": identity_isolation_ok,
        "min_audio_deltas_per_session": args.min_audio_deltas_per_session,
        "min_audio_delta_ok": min_audio_delta_ok,
        "response_audio_contract_ok": response_audio_contract_ok,
        "inputs": {
            "a": validated["input_wav_a"],
            "b": validated["input_wav_b"],
            "sha256_a": validated["input_sha256_a"],
            "sha256_b": validated["input_sha256_b"],
        },
        "sessions": sessions,
    }
    summary_output = (
        Path(args.summary_output) if args.summary_output else Path(validated["output_dir_a"]).parent / "summary.json"
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--input-wav-a", required=True)
    parser.add_argument("--input-wav-b", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--output-dir-a", required=True)
    parser.add_argument("--output-dir-b", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--no-realtime-pacing", action="store_true")
    parser.add_argument("--require-audio", action="store_true")
    parser.add_argument("--min-audio-deltas-per-session", type=int, default=2)
    args = parser.parse_args()
    if args.min_audio_deltas_per_session < 0:
        parser.error("--min-audio-deltas-per-session must be non-negative")
    return args


def main() -> None:
    result = asyncio.run(run_pair(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
