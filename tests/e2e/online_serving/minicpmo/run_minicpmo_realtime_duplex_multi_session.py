"""Concurrent MiniCPM-o Realtime duplex and resumable-session E2E driver."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import websockets

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from minicpmo_realtime_duplex_scenarios import (  # noqa: E402
    _ref_audio_data_url,
    _url_with_model,
    run_demo,
)


def _response_ids(result: dict[str, object]) -> set[str]:
    values = result.get("completed_response_ids")
    return {value for value in values if isinstance(value, str)} if isinstance(values, list) else set()


def _error_code(event: dict[str, object]) -> str | None:
    error = event.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return error["code"]
    code = event.get("code")
    return code if isinstance(code, str) else None


def _validate_identity_isolation(results: list[dict[str, object]]) -> bool:
    seen: set[str] = set()
    for result in results:
        current = _response_ids(result)
        if current & seen:
            return False
        seen.update(current)
    return True


def _validate_semantic_isolation(
    results: list[dict[str, object]],
    *,
    input_wavs: list[str],
    expected_tokens: list[str],
) -> bool:
    if not input_wavs:
        return True
    if len(results) != len(input_wavs):
        return False
    input_hashes = [hashlib.sha256(Path(path).read_bytes()).digest() for path in input_wavs]
    if len(set(input_hashes)) != len(input_hashes):
        return False
    if not expected_tokens:
        return True
    if len(expected_tokens) != len(results):
        return False
    for result, expected_token in zip(results, expected_tokens, strict=True):
        details = result.get("transcript_integrity")
        transcripts = (
            [str(item.get("transcript", "")) for item in details if isinstance(item, dict)]
            if isinstance(details, list)
            else []
        )
        if expected_token not in "".join(transcripts):
            return False
    return True


async def _receive_until(ws, event_type: str, *, timeout_s: float) -> tuple[dict[str, object], list[dict[str, object]]]:
    async def receive() -> tuple[dict[str, object], list[dict[str, object]]]:
        events: list[dict[str, object]] = []
        while True:
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if not isinstance(event, dict):
                continue
            events.append(event)
            if event.get("type") == event_type:
                return event, events

    return await asyncio.wait_for(receive(), timeout=timeout_s)


async def _open_admission_session(
    args: argparse.Namespace,
    session_id: str,
) -> tuple[object, dict[str, object]]:
    url = _url_with_model(
        args.url,
        args.model,
        autostart=False if getattr(args, "ref_audio", None) else None,
        session_id=session_id,
    )
    ws = await websockets.connect(url, max_size=64 * 1024 * 1024)
    await ws.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "session_id": session_id,
                    "model": args.model,
                    "modalities": ["audio", "text"],
                    "extra_body": {},
                    **({"ref_audio": _ref_audio_data_url(args.ref_audio)} if getattr(args, "ref_audio", None) else {}),
                },
            }
        )
    )
    created, _ = await _receive_until(ws, "session.created", timeout_s=args.timeout_s)
    return ws, created


async def _close_admission_session(ws, *, timeout_s: float) -> None:
    await ws.close()


async def _admission_probe(args: argparse.Namespace, *, limit: int) -> dict[str, object]:
    if limit < 1:
        raise ValueError("admission limit must be positive")
    prefix = f"admission-{uuid.uuid4().hex}"
    accepted: list[tuple[object, dict[str, object]]] = []
    replacement: tuple[object, dict[str, object]] | None = None
    overflow_code = None
    try:
        for index in range(limit):
            accepted.append(await _open_admission_session(args, f"{prefix}-accepted-{index}"))

        overflow_id = f"{prefix}-overflow"
        overflow_url = _url_with_model(
            args.url,
            args.model,
            autostart=False if getattr(args, "ref_audio", None) else None,
            session_id=overflow_id,
        )
        async with websockets.connect(overflow_url, max_size=64 * 1024 * 1024) as overflow:
            await overflow.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "session_id": overflow_id,
                            "model": args.model,
                            "modalities": ["audio", "text"],
                            "extra_body": {},
                            **(
                                {"ref_audio": _ref_audio_data_url(args.ref_audio)}
                                if getattr(args, "ref_audio", None)
                                else {}
                            ),
                        },
                    }
                )
            )
            error, _ = await _receive_until(overflow, "error", timeout_s=args.timeout_s)
            overflow_code = _error_code(error)

        first_ws, _ = accepted.pop(0)
        await _close_admission_session(first_ws, timeout_s=args.timeout_s)
        replacement = await _open_admission_session(args, f"{prefix}-replacement")

        first_capabilities = accepted[0][1].get("session") if accepted else replacement[1].get("session")
        capabilities = first_capabilities.get("capabilities") if isinstance(first_capabilities, dict) else None
        advertised_multi = capabilities.get("supports_multi_session") if isinstance(capabilities, dict) else None
        admission_mode = capabilities.get("session_admission_mode") if isinstance(capabilities, dict) else None
        return {
            "ok": (
                overflow_code == "resource_exhausted"
                and replacement[1].get("type") == "session.created"
                and admission_mode == "engine_managed"
            ),
            "configured_limit": limit,
            "accepted_before_rejection": limit,
            "overflow_error_code": overflow_code,
            "replacement_accepted": True,
            "advertised_multi_session": advertised_multi,
            "session_admission_mode": admission_mode,
        }
    finally:
        cleanup = list(accepted)
        if replacement is not None:
            cleanup.append(replacement)
        for ws, _ in cleanup:
            try:
                await _close_admission_session(ws, timeout_s=args.timeout_s)
            except Exception:
                await ws.close()


def _demo_args(args: argparse.Namespace, index: int) -> SimpleNamespace:
    input_wav = args.session_input_wav[index] if args.session_input_wav else args.input_wav
    return SimpleNamespace(
        url=args.url,
        model=args.model,
        session_id=f"multi-{index}-{uuid.uuid4().hex}",
        input_wav=input_wav,
        ref_audio=args.ref_audio,
        turn_input_wav=list(args.turn_input_wav),
        output_dir=str(Path(args.output_dir) / f"session_{index:02d}"),
        output_audio_format="pcm16",
        chunk_ms=args.chunk_ms,
        realtime_input=args.realtime_input,
        first_turn_ms=args.first_turn_ms,
        turn_duration_ms=list(args.turn_duration_ms),
        first_turn_transcript=f"session {index} input",
        omit_transcript_hints=True,
        validation_mode="response-required",
        temperature=args.temperature,
        scenario="sequential",
        require_audio=args.response_required,
        require_distinct_inputs=False,
        expect_empty_turn=[],
        short_ack_ms=350,
        turns=args.turns,
        timeout_s=args.timeout_s,
    )


async def run_multi_session(args: argparse.Namespace) -> dict[str, object]:
    if args.sessions < 1:
        raise ValueError("--sessions must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lifecycle_result = await run_lifecycle_probes(args)

    session_results = await asyncio.gather(
        *(run_demo(_demo_args(args, index)) for index in range(args.sessions)),
        return_exceptions=True,
    )
    failures = [repr(result) for result in session_results if isinstance(result, BaseException)]
    completed = [result for result in session_results if isinstance(result, dict)]
    identity_isolation_ok = _validate_identity_isolation(completed)
    semantic_isolation_ok = _validate_semantic_isolation(
        completed,
        input_wavs=list(args.session_input_wav),
        expected_tokens=list(args.session_expected_token),
    )
    result = {
        "ok": (
            not failures
            and len(completed) == args.sessions
            and all(item.get("ok") is True for item in completed)
            and identity_isolation_ok
            and semantic_isolation_ok
            and lifecycle_result["ok"] is True
        ),
        "session_count": args.sessions,
        "identity_isolation_ok": identity_isolation_ok,
        "semantic_isolation_ok": semantic_isolation_ok,
        "expiry": lifecycle_result["expiry"],
        "admission": lifecycle_result["admission"],
        "failures": failures,
        "sessions": completed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


async def run_lifecycle_probes(args: argparse.Namespace) -> dict[str, object]:
    """Run admission probes without requiring model output."""
    if args.sessions < 1:
        raise ValueError("--sessions must be positive")
    admission_result = (
        await _admission_probe(args, limit=args.verify_admission_limit)
        if args.verify_admission_limit is not None
        else None
    )
    return {
        "ok": admission_result is None or admission_result.get("ok") is True,
        "expiry": None,
        "admission": admission_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8113/v1/realtime?duplex=1")
    parser.add_argument("--base-url", help="Deprecated alias; /v1/realtime is appended when supplied.")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--ref-audio", help="Optional WAV used as the MiniCPM-o voice prompt for every session.")
    parser.add_argument("--session-input-wav", action="append", default=[])
    parser.add_argument("--session-expected-token", action="append", default=[])
    parser.add_argument("--turn-input-wav", action="append", default=[])
    parser.add_argument("--output-dir", default="/tmp/minicpmo_pr3907_multi_session_e2e")
    parser.add_argument("--realtime-input", action="store_true")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--first-turn-ms", type=int, default=1400)
    parser.add_argument("--turn-duration-ms", type=int, action="append", default=[])
    parser.add_argument("--response-required", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--verify-admission-limit", type=int)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    if args.base_url:
        args.url = args.base_url.rstrip("/") + "/v1/realtime?duplex=1"
    if args.session_input_wav and len(args.session_input_wav) != args.sessions:
        parser.error("provide exactly one --session-input-wav per session")
    if args.session_expected_token and len(args.session_expected_token) != args.sessions:
        parser.error("provide exactly one --session-expected-token per session")
    if args.session_expected_token and not args.session_input_wav:
        parser.error("--session-expected-token requires --session-input-wav")
    return args


def main() -> None:
    result = asyncio.run(run_multi_session(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
