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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets
from websockets.exceptions import ConnectionClosed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from minicpmo_realtime_duplex_scenarios import _url_with_model, run_demo  # noqa: E402


def _with_resume_mode(url: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["resume"] = "1"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _response_ids(result: dict[str, object]) -> set[str]:
    values = result.get("completed_response_ids")
    return {value for value in values if isinstance(value, str)} if isinstance(values, list) else set()


def _event_session_id(event: dict[str, object]) -> str | None:
    session_id = event.get("session_id")
    if isinstance(session_id, str):
        return session_id
    inner = event.get("event")
    if isinstance(inner, dict):
        nested_session_id = inner.get("session_id")
        if isinstance(nested_session_id, str):
            return nested_session_id
    return None


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
    url = _url_with_model(args.url, args.model, session_id=session_id)
    ws = await websockets.connect(url, max_size=64 * 1024 * 1024)
    await ws.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "session_id": session_id,
                    "model": args.model,
                    "modalities": ["audio", "text"],
                    "extra_body": {"minicpmo45_native_duplex": True},
                },
            }
        )
    )
    created, _ = await _receive_until(ws, "session.created", timeout_s=args.timeout_s)
    return ws, created


async def _close_admission_session(ws, *, timeout_s: float) -> None:
    await ws.send(json.dumps({"type": "session.close"}))
    await _receive_until(ws, "session.closed", timeout_s=timeout_s)
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
        overflow_url = _url_with_model(args.url, args.model, session_id=overflow_id)
        async with websockets.connect(overflow_url, max_size=64 * 1024 * 1024) as overflow:
            await overflow.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "session_id": overflow_id,
                            "model": args.model,
                            "modalities": ["audio", "text"],
                            "extra_body": {"minicpmo45_native_duplex": True},
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


async def _resume_probe(
    args: argparse.Namespace,
    *,
    session_id: str,
    expect_expired: bool = False,
) -> dict[str, object]:
    url = _url_with_model(args.url, args.model, session_id=session_id)
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as first:
        await first.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "model": args.model,
                        "modalities": ["audio", "text"],
                        "extra_body": {"minicpmo45_native_duplex": True},
                    },
                }
            )
        )
        created, first_events = await _receive_until(first, "session.created", timeout_s=args.timeout_s)
        token = created.get("resume_token")
        incarnation = created.get("incarnation")
        generation = created.get("attachment_generation")
        if not isinstance(token, str) or not isinstance(incarnation, int):
            raise RuntimeError("session.created omitted resumable credentials")
        last_seq = max(
            (
                event.get("server_event_seq", 0)
                for event in first_events
                if isinstance(event.get("server_event_seq"), int)
            ),
            default=0,
        )

    delay_s = args.expire_after_s if expect_expired else args.resume_after_ms / 1000
    if delay_s > 0:
        await asyncio.sleep(delay_s)

    resume_url = _with_resume_mode(url)
    async with websockets.connect(resume_url, max_size=64 * 1024 * 1024) as second:
        await second.send(
            json.dumps(
                {
                    "type": "session.resume",
                    "session_id": session_id,
                    "incarnation": incarnation,
                    "resume_token": token,
                    "last_received_server_event_seq": last_seq,
                }
            )
        )
        if expect_expired:
            error, error_events = await _receive_until(second, "error", timeout_s=args.timeout_s)
            error_payload = error.get("error")
            code = error_payload.get("code") if isinstance(error_payload, dict) else error.get("code")
            return {
                "ok": code == "session_resume_expired",
                "session_id": session_id,
                "expired": True,
                "error_code": code,
                "event_count": len(error_events),
            }
        resumed, replay = await _receive_until(second, "session.resumed", timeout_s=args.timeout_s)
        rotated = resumed.get("resume_token")
        if not isinstance(rotated, str) or rotated == token:
            raise RuntimeError("session.resume did not rotate the resume token")
        await second.send(json.dumps({"type": "session.heartbeat"}))
        heartbeat, heartbeat_events = await _receive_until(
            second,
            "session.heartbeat_ack",
            timeout_s=args.timeout_s,
        )
        await second.send(json.dumps({"type": "session.close"}))
        closed, close_events = await _receive_until(second, "session.closed", timeout_s=args.timeout_s)

    replay_sequences = [event["server_event_seq"] for event in replay if isinstance(event.get("server_event_seq"), int)]
    return {
        "ok": (
            resumed.get("session_id") == session_id
            and isinstance(generation, int)
            and resumed.get("attachment_generation") == generation + 1
            and _event_session_id(heartbeat) == session_id
            and _event_session_id(closed) == session_id
            and replay_sequences == sorted(replay_sequences)
        ),
        "session_id": session_id,
        "resumed_session_id": resumed.get("session_id"),
        "heartbeat_session_id": _event_session_id(heartbeat),
        "closed_session_id": _event_session_id(closed),
        "initial_attachment_generation": generation,
        "resumed_attachment_generation": resumed.get("attachment_generation"),
        "replayed_event_count": len(replay_sequences),
        "replayed_event_sequences": replay_sequences,
        "heartbeat_event_count": len(heartbeat_events),
        "close_event_count": len(close_events),
        "token_rotated": True,
    }


async def _takeover_probe(
    args: argparse.Namespace,
    *,
    session_id: str,
) -> dict[str, object]:
    url = _url_with_model(args.url, args.model, session_id=session_id)
    first = await websockets.connect(url, max_size=64 * 1024 * 1024)
    second = None
    try:
        await first.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "model": args.model,
                        "modalities": ["audio", "text"],
                        "extra_body": {"minicpmo45_native_duplex": True},
                    },
                }
            )
        )
        created, first_events = await _receive_until(first, "session.created", timeout_s=args.timeout_s)
        token = created.get("resume_token")
        incarnation = created.get("incarnation")
        generation = created.get("attachment_generation")
        if not isinstance(token, str) or not isinstance(incarnation, int) or not isinstance(generation, int):
            raise RuntimeError("session.created omitted takeover credentials")
        last_seq = max(
            (
                event.get("server_event_seq", 0)
                for event in first_events
                if isinstance(event.get("server_event_seq"), int)
            ),
            default=0,
        )

        second = await websockets.connect(_with_resume_mode(url), max_size=64 * 1024 * 1024)
        await second.send(
            json.dumps(
                {
                    "type": "session.resume",
                    "session_id": session_id,
                    "incarnation": incarnation,
                    "resume_token": token,
                    "last_received_server_event_seq": last_seq,
                }
            )
        )
        resumed, replay_events = await _receive_until(second, "session.resumed", timeout_s=args.timeout_s)
        replaced, replaced_events = await _receive_until(first, "session.replaced", timeout_s=args.timeout_s)
        await asyncio.wait_for(first.wait_closed(), timeout=args.timeout_s)

        rejected_old_writes = 0
        for _ in range(4):
            try:
                await first.send(json.dumps({"type": "session.heartbeat"}))
            except ConnectionClosed:
                rejected_old_writes += 1

        await second.send(json.dumps({"type": "session.heartbeat"}))
        heartbeat, heartbeat_events = await _receive_until(
            second,
            "session.heartbeat_ack",
            timeout_s=args.timeout_s,
        )
        await second.send(json.dumps({"type": "session.close"}))
        closed, close_events = await _receive_until(second, "session.closed", timeout_s=args.timeout_s)
        rotated_token = resumed.get("resume_token")
        return {
            "ok": (
                resumed.get("session_id") == session_id
                and resumed.get("attachment_generation") == generation + 1
                and isinstance(rotated_token, str)
                and rotated_token != token
                and _event_session_id(replaced) == session_id
                and replaced.get("attachment_generation") == generation
                and rejected_old_writes == 4
                and _event_session_id(heartbeat) == session_id
                and _event_session_id(closed) == session_id
            ),
            "session_id": session_id,
            "initial_attachment_generation": generation,
            "resumed_attachment_generation": resumed.get("attachment_generation"),
            "replaced_attachment_generation": replaced.get("attachment_generation"),
            "token_rotated": isinstance(rotated_token, str) and rotated_token != token,
            "old_attachment_closed": True,
            "rejected_old_writes": rejected_old_writes,
            "replay_event_count": len(replay_events),
            "replaced_event_count": len(replaced_events),
            "heartbeat_event_count": len(heartbeat_events),
            "close_event_count": len(close_events),
        }
    finally:
        if second is not None:
            await second.close()
        await first.close()


def _demo_args(args: argparse.Namespace, index: int) -> SimpleNamespace:
    validation_mode = "response-required" if args.response_required else "model-policy"
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
        validation_mode=validation_mode,
        temperature=args.temperature,
        scenario="sequential",
        require_audio=args.response_required,
        require_distinct_inputs=False,
        expect_empty_turn=[],
        short_ack_ms=350,
        turns=args.turns,
        timeout_s=args.timeout_s,
        model_policy_settle_ms=args.model_policy_settle_ms,
    )


async def run_multi_session(args: argparse.Namespace) -> dict[str, object]:
    if args.sessions < 1:
        raise ValueError("--sessions must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_result = None
    if args.disconnect_session_index is not None:
        if not 0 <= args.disconnect_session_index < args.sessions:
            raise ValueError("--disconnect-session-index is outside the session range")
        resume_result = await _resume_probe(
            args,
            session_id=f"resume-{args.disconnect_session_index}-{uuid.uuid4().hex}",
        )
    takeover_result = None
    if args.takeover_session_index is not None:
        if not 0 <= args.takeover_session_index < args.sessions:
            raise ValueError("--takeover-session-index is outside the session range")
        takeover_result = await _takeover_probe(
            args,
            session_id=f"takeover-{args.takeover_session_index}-{uuid.uuid4().hex}",
        )
    expiry_result = None
    if args.expire_session_index is not None:
        if not 0 <= args.expire_session_index < args.sessions:
            raise ValueError("--expire-session-index is outside the session range")
        expiry_result = await _resume_probe(
            args,
            session_id=f"expire-{args.expire_session_index}-{uuid.uuid4().hex}",
            expect_expired=True,
        )
    admission_result = (
        await _admission_probe(args, limit=args.verify_admission_limit)
        if args.verify_admission_limit is not None
        else None
    )

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
            and (resume_result is None or resume_result.get("ok") is True)
            and (takeover_result is None or takeover_result.get("ok") is True)
            and (expiry_result is None or expiry_result.get("ok") is True)
            and (admission_result is None or admission_result.get("ok") is True)
        ),
        "session_count": args.sessions,
        "identity_isolation_ok": identity_isolation_ok,
        "semantic_isolation_ok": semantic_isolation_ok,
        "resume": resume_result,
        "takeover": takeover_result,
        "expiry": expiry_result,
        "admission": admission_result,
        "failures": failures,
        "sessions": completed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


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
    parser.add_argument("--disconnect-session-index", type=int)
    parser.add_argument("--takeover-session-index", type=int)
    parser.add_argument("--resume-after-ms", type=int, default=1000)
    parser.add_argument("--expire-session-index", type=int)
    parser.add_argument("--expire-after-s", type=float, default=6.0)
    parser.add_argument("--verify-admission-limit", type=int)
    parser.add_argument("--model-policy-settle-ms", type=int, default=600)
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
