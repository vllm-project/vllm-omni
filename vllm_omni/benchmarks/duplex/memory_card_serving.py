# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Replay synthetic execution plans over the existing public duplex client.

Committed text imports and actual generated/playback turns are distinct events.
CPU-only edits and calibrated heard-text fixtures have no serving equivalent.
Gold, pressure checks and semantic scoring remain with the evaluator.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shlex
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from vllm_omni.benchmarks.duplex.memory_card_replay import (
    MemoryReplayVariant,
    load_memory_replay_cases,
    memory_replay_variants,
)
from vllm_omni.benchmarks.duplex_session_metrics import collect_duplex_session_metrics
from vllm_omni.clients.duplex import (
    AudioFormat,
    DuplexClient,
    DuplexClientError,
    DuplexEvent,
    DuplexProtocolError,
    EventCollector,
    SessionConfig,
    wait_for_condition,
)

# These are execution mappings, not claims about extraction or model quality.
SERVING_EVENT_MAP = {
    "message": "conversation.item.create -> conversation.item.created",
    "delete": "conversation.item.delete -> conversation.item.deleted",
    "idle": "fixed wall-clock delay; no extraction polling",
    "resume_live_session": "public-client reconnect -> session.resumed",
    "live_response": "actual response audio -> playback.ack; optional item.truncate",
}
_ARMS = ("budget_only", "l2", "l2_l3")
_TEXT_DELTAS = ("response.output_audio_transcript.delta", "response.output_text.delta", "response.text.delta")
ReplayStatus = Literal["completed", "unsupported", "failed"]


@dataclass(frozen=True)
class ServingConfirmedEvent:
    event_index: int
    event_type: str


@dataclass(frozen=True)
class ServingReplayResult:
    case_id: str
    query_index: int
    timing: str
    source_sha256: str
    status: ReplayStatus
    confirmed_events: tuple[ServingConfirmedEvent, ...] = ()
    error_code: str | None = None
    query_response_id: str | None = None
    output_text: str = field(default="", repr=False)
    query_timing: dict[str, object] = field(default_factory=dict)
    query_metrics: list[dict[str, object]] = field(default_factory=list)
    session_metrics: dict[str, object] = field(default_factory=dict)


class ServingManifest(TypedDict):
    schema_version: int
    event_map: dict[str, str]
    compatible_variants: int
    repeats: int
    planned_trials: int
    variants: list[dict[str, object]]
    arms: list[dict[str, str]]
    limits: list[str]


class _ReplayError(Exception):
    """Only fixed, content-free error categories leave the replay adapter."""


def _result(
    variant: MemoryReplayVariant,
    status: ReplayStatus,
    *,
    error_code: str | None = None,
    confirmed_events: tuple[ServingConfirmedEvent, ...] = (),
) -> ServingReplayResult:
    return ServingReplayResult(
        variant.case_id,
        variant.query_index,
        variant.timing,
        variant.source_sha256,
        status,
        error_code=error_code,
        confirmed_events=confirmed_events,
    )


def _supported(variant: MemoryReplayVariant) -> bool:
    return all(event.event_type in SERVING_EVENT_MAP for event in variant.events)


async def _exchange(
    client: DuplexClient, payload: dict[str, object], expected_type: str, timeout_s: float
) -> DuplexEvent:
    # Subscribe before sending: acknowledgements can arrive before send returns.
    waiter = asyncio.create_task(client.wait_for(expected_type, timeout_s=timeout_s))
    try:
        await asyncio.sleep(0)
        await client.send(payload)
        return await waiter
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


async def _create_item(client: DuplexClient, item_id: object, role: object, text: object, timeout_s: float) -> None:
    if (
        not isinstance(item_id, str)
        or not 1 <= len(item_id) <= 128
        or not isinstance(role, str)
        or role not in {"user", "assistant"}
        or not isinstance(text, str)
        or not 1 <= len(text) <= 16384
    ):
        raise _ReplayError("invalid_message")
    ack = await _exchange(
        client,
        {
            "type": "conversation.item.create",
            "item": {
                "id": item_id,
                "type": "message",
                "role": role,
                "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
            },
        },
        "conversation.item.created",
        timeout_s,
    )
    item = ack.raw.get("item")
    if not isinstance(item, dict) or item.get("id") != item_id:
        raise _ReplayError("ack_mismatch")


async def _response(client: DuplexClient, collector: EventCollector, timeout_s: float) -> str:
    start = len(collector.events)
    ack = await _exchange(client, {"type": "response.create"}, "response.created", timeout_s)
    response_id = ack.response_id
    if response_id is None:
        raise _ReplayError("missing_response_id")

    def finished() -> bool:
        for event in collector.events[start:]:
            if event.get("type") == "error":
                raise _ReplayError("protocol_error")
            if event.get("type") == "response.done" and collector.response_id(event) == response_id:
                response = event.get("response")
                if not isinstance(response, dict) or response.get("status") != "completed":
                    raise _ReplayError("response_failed")
                return True
        return False

    # The response may already be complete when response.created wakes us.
    await wait_for_condition(finished, timeout_s=timeout_s, label="memory replay response")
    return response_id


async def _playback(
    client: DuplexClient, collector: EventCollector, payload: Mapping[str, object], timeout_s: float
) -> None:
    if set(payload) != {"type", "played_audio_ms", "truncate"}:
        raise _ReplayError("invalid_live_response")
    played, truncate = payload["played_audio_ms"], payload["truncate"]
    if type(played) is not int or not 0 <= played <= 600000 or type(truncate) is not bool:
        raise _ReplayError("invalid_live_response")
    response_id = await _response(client, collector, timeout_s)
    audio = collector.audio_bytes(response_id)
    if not audio:
        raise _ReplayError("missing_audio")
    audio_format = AudioFormat(client.config.output_audio.encoding, collector.output_sample_rate_hz)
    if played > audio_format.duration_ms(len(audio)):
        raise _ReplayError("playback_out_of_range")
    item_id = None
    for raw in collector.events:
        item = raw.get("item")
        if (
            raw.get("type") == "response.output_item.added"
            and collector.response_id(raw) == response_id
            and isinstance(item, dict)
            and isinstance(item.get("id"), str)
        ):
            item_id = item["id"]
            break
    if item_id is None:
        raise _ReplayError("missing_audio_item")
    ack = await _exchange(
        client,
        {
            "type": "playback.ack",
            "response_id": response_id,
            "item_id": item_id,
            "played_ms": played,
            "committed_ms": played,
        },
        "playback.acknowledged",
        timeout_s,
    )
    details = ack.raw.get("event")
    if not isinstance(details, dict) or details.get("item_id") != item_id or details.get("committed_ms") != played:
        raise _ReplayError("ack_mismatch")
    if truncate:
        ack = await _exchange(
            client,
            {"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0, "audio_end_ms": played},
            "conversation.item.truncated",
            timeout_s,
        )
        if ack.raw.get("item_id") != item_id or ack.raw.get("audio_end_ms") != played:
            raise _ReplayError("ack_mismatch")


async def replay_serving_case(
    client: DuplexClient, variant: MemoryReplayVariant, collector: EventCollector, *, timeout_s: float = 90.0
) -> ServingReplayResult:
    """Execute one trial inside one open session with an active collector.

    Use replay_serving_trials to own fresh-session and collector lifetimes.
    A live_response is a separate serving event, never a heard_text fixture.
    """
    if not _supported(variant):
        return _result(variant, "unsupported", error_code="unsupported_event")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("invalid_replay_timeout")
    if client.config.extra_body.get("auto_response", client.config.auto_response) is not False:
        return _result(variant, "failed", error_code="auto_response_must_be_disabled")
    confirmed = []
    stream_start = time.monotonic()
    try:
        for index, event in enumerate(variant.events):
            payload, kind = event.payload, event.event_type
            ack_type = None
            if kind == "message":
                await _create_item(client, payload["item_id"], payload["role"], payload["text"], timeout_s)
                ack_type = "conversation.item.created"
            elif kind == "delete":
                ack = await _exchange(
                    client,
                    {"type": "conversation.item.delete", "item_id": payload["item_id"]},
                    "conversation.item.deleted",
                    timeout_s,
                )
                if ack.raw.get("item_id") != payload["item_id"]:
                    raise _ReplayError("ack_mismatch")
                ack_type = "conversation.item.deleted"
            elif kind == "idle":
                duration_ms = payload["duration_ms"]
                if type(duration_ms) is not int or not 0 <= duration_ms <= 600000:
                    raise _ReplayError("invalid_idle")
                await asyncio.sleep(duration_ms / 1000)
            elif kind == "resume_live_session":
                session_id = client.session_id
                if client._ws is None:
                    raise _ReplayError("missing_transport")
                waiter = asyncio.create_task(client.wait_for("session.resumed", timeout_s=timeout_s))
                try:
                    await asyncio.sleep(0)
                    await client._ws.close()
                    await waiter
                finally:
                    waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)
                if client.session_id != session_id:
                    raise _ReplayError("session_replaced")
                ack_type = "session.resumed"
            elif kind == "live_response":
                await _playback(client, collector, payload, timeout_s)
                ack_type = "conversation.item.truncated" if payload["truncate"] else "playback.acknowledged"
            if ack_type is not None:
                confirmed.append(ServingConfirmedEvent(index, ack_type))
        query_start = time.monotonic()
        item_ids = {event.payload.get("item_id") for event in variant.events}
        query_id, suffix = "memory-query", 0
        while query_id in item_ids:
            suffix += 1
            query_id = f"memory-query-{suffix}"
        await _create_item(client, query_id, "user", variant.query_text, timeout_s)
        response_id = await _response(client, collector, timeout_s)
        query_collector = EventCollector()
        for raw, received in zip(collector.events, collector.event_received_at_s, strict=True):
            if received >= query_start and collector.response_id(raw) == response_id:
                query_collector.add(raw, received_at_s=received)
        query_bundle = collect_duplex_session_metrics(
            query_collector, stream_start=query_start, session_id=client.session_id
        )
        session_bundle = collect_duplex_session_metrics(
            collector, stream_start=stream_start, session_id=client.session_id
        )
        text_times = [
            received
            for raw, received in zip(query_collector.events, query_collector.event_received_at_s, strict=True)
            if raw.get("type") in _TEXT_DELTAS and isinstance(raw.get("delta"), str) and raw["delta"]
        ]
        first_audio = query_collector.first_received_at("response.output_audio.delta")
        timing: dict[str, object] = {
            "origin": "query_item_send_client_monotonic",
            "query_sent_to_first_text_ms": (text_times[0] - query_start) * 1000 if text_times else None,
            "query_sent_to_first_audio_ms": (first_audio - query_start) * 1000 if first_audio is not None else None,
        }
        return replace(
            _result(variant, "completed"),
            confirmed_events=tuple(confirmed),
            query_response_id=response_id,
            output_text=collector.response_text(response_id),
            query_timing=timing,
            query_metrics=query_bundle.request_metrics,
            session_metrics=session_bundle.session_metrics,
        )
    except _ReplayError as exc:
        reason = str(exc)
    except (asyncio.TimeoutError, TimeoutError):
        reason = "timeout"
    except DuplexProtocolError:
        reason = "protocol_error"
    except DuplexClientError:
        reason = "connection_error"
    return _result(variant, "failed", confirmed_events=tuple(confirmed), error_code=reason)


async def replay_serving_trials(
    variants: Sequence[MemoryReplayVariant], *, client_factory: Callable[[], DuplexClient], timeout_s: float = 90.0
) -> tuple[ServingReplayResult, ...]:
    """Every query/timing trial gets a new client, session and event collector."""
    results = []
    for variant in variants:
        if not _supported(variant):
            results.append(_result(variant, "unsupported", error_code="unsupported_event"))
            continue
        client, collector = client_factory(), EventCollector()
        client.attach_event_collector(collector)
        task = asyncio.create_task(collector.consume(client))
        try:
            await asyncio.sleep(0)
            async with client:
                result = await replay_serving_case(client, variant, collector, timeout_s=timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            result = _result(variant, "failed", error_code="timeout")
        except DuplexProtocolError:
            result = _result(variant, "failed", error_code="protocol_error")
        except DuplexClientError:
            result = _result(variant, "failed", error_code="connection_error")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        results.append(result)
    return tuple(results)


def serving_manifest(
    variants: Sequence[MemoryReplayVariant], *, deployment_configs: Mapping[str, Path]
) -> ServingManifest:
    """One execution-only timeline for all arms; settings stay deployment-owned."""
    if set(deployment_configs) != set(_ARMS):
        raise ValueError("invalid_serving_arms")
    rows: list[dict[str, object]] = [
        {
            "case_id": v.case_id,
            "query_index": v.query_index,
            "timing": v.timing,
            "source_sha256": v.source_sha256,
            "supported": _supported(v),
            "query_text": v.query_text,
            "events": [{"type": e.event_type, **e.payload} for e in v.events],
        }
        for v in variants
    ]
    timeline = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    arms = []
    for name in _ARMS:
        arms.append(
            {
                "name": name,
                **_deployment_record(deployment_configs[name]),
                "timeline_sha256": timeline,
            }
        )
    count = sum(_supported(v) for v in variants)
    return {
        "schema_version": 1,
        "event_map": SERVING_EVENT_MAP,
        "compatible_variants": count,
        "repeats": 3,
        "planned_trials": count * 3 * 3,
        "variants": rows,
        "arms": arms,
        "limits": [
            "protocol compatibility does not establish pressure or extraction quality",
            "edit and calibrated assistant_playback fixtures remain CPU-only",
            "live_response is actual generated audio, not an equivalent fixture",
        ],
    }


def _deployment_record(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        raw = source.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("deployment_config_too_large")
    return {"deployment_config": str(path.resolve()), "deployment_sha256": hashlib.sha256(raw).hexdigest()}


def _write_report(path: Path, report: object) -> None:
    # Never overwrite a prior trial, and keep collected output private by default.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


async def _run_command(
    args: argparse.Namespace, variants: Sequence[MemoryReplayVariant]
) -> tuple[dict[str, object], int]:
    token = os.environ.get("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {token}"} if token else None

    def factory() -> DuplexClient:
        return DuplexClient(
            args.url,
            model=args.model,
            extra_headers=headers,
            config=SessionConfig(auto_response=False, max_output_tokens=args.max_output_tokens),
        )

    rows: list[dict[str, object]] = []
    failed, completed = False, False
    for repeat in range(args.repeats):
        results = await replay_serving_trials(variants, client_factory=factory, timeout_s=args.timeout_s)
        failed |= any(result.status == "failed" for result in results)
        completed |= any(result.status == "completed" for result in results)
        rows.extend({"repeat_index": repeat, **asdict(result)} for result in results)
    report = {
        "schema_version": 1,
        "arm": args.arm,
        "model": args.model,
        **_deployment_record(args.deployment_config),
        "trials": rows,
        "limits": [
            "deployment hash identifies the supplied file, not an attestation of the running server",
            "pressure and semantic scoring require the separate evaluator",
            "raw events and resume credentials are not exported",
        ],
    }
    return report, int(failed or not completed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="export shared timeline and separate deployment commands")
    run = commands.add_parser("run", help="replay against one already running deployment")
    for command in (prepare, run):
        command.add_argument("--cases", type=Path, required=True)
        command.add_argument("--url", required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--output", type=Path, required=True)
    for name in ("budget-only", "l2", "l2-l3"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--arm", choices=_ARMS, required=True)
    run.add_argument("--deployment-config", type=Path, required=True)
    run.add_argument("--repeats", type=int, choices=range(1, 21), default=3)
    run.add_argument("--timeout-s", type=float, default=90)
    run.add_argument("--max-output-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    url = urlsplit(args.url)
    if (
        not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or (url.scheme != "wss" and not (url.scheme == "ws" and url.hostname in {"localhost", "127.0.0.1", "::1"}))
    ):
        parser.error("use wss, or local-loopback ws; credentials belong in OPENAI_API_KEY")
    try:
        variants = memory_replay_variants(load_memory_replay_cases(args.cases))
        if args.command == "prepare":
            configs = {name: getattr(args, name) for name in _ARMS}
            manifest = serving_manifest(variants, deployment_configs=configs)
            for row in manifest["arms"]:
                row["launch_command"] = shlex.join(
                    [
                        "vllm",
                        "serve",
                        args.model,
                        "--omni",
                        "--deploy-config",
                        row["deployment_config"],
                        "--port",
                        str(url.port or 8091),
                    ]
                )
                row["replay_command"] = shlex.join(
                    [
                        "python",
                        "-m",
                        "vllm_omni.benchmarks.duplex.memory_card_serving",
                        "run",
                        "--cases",
                        str(args.cases.resolve()),
                        "--url",
                        args.url,
                        "--model",
                        args.model,
                        "--arm",
                        row["name"],
                        "--deployment-config",
                        row["deployment_config"],
                        "--output",
                        str(args.output.parent / (row["name"] + "-results.json")),
                    ]
                )
            _write_report(args.output, manifest)
            return 0
        if (
            not math.isfinite(args.timeout_s)
            or not 0 < args.timeout_s <= 3600
            or not 1 <= args.max_output_tokens <= 8192
        ):
            parser.error("invalid timeout or output token limit")
        # Validate the selected config before sending any protocol commands.
        _deployment_record(args.deployment_config)
        report, exit_code = asyncio.run(_run_command(args, variants))
        _write_report(args.output, report)
        return exit_code
    except (OSError, ValueError):
        parser.exit(2, "Invalid replay input or output path; prior reports are never overwritten.\n")


if __name__ == "__main__":
    raise SystemExit(main())
