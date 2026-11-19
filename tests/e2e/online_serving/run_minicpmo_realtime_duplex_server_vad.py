# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
from pathlib import Path

from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    read_pcm16_wav,
    wait_for,
)


def _events(events: list[dict[str, object]], response_id: str, kind: str) -> list[dict[str, object]]:
    return [e for e in events if e.get("type") == kind and RealtimeEventCollector.response_id(e) == response_id]


async def run_server_vad_interrupt(args) -> dict[str, object]:
    initial = read_pcm16_wav(Path(args.input_wav))
    interrupt = read_pcm16_wav(Path(args.interrupt_wav))[: PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * 47 // 10]
    client = RealtimeDuplexClient(build_realtime_url(args.url, args.model, autostart=False))

    async def until(predicate, label: str) -> None:
        await wait_for(predicate, timeout_s=args.timeout_s, label=label)

    try:
        await client.__aenter__()
        await client.configure(
            args.model,
            ref_audio="data:audio/wav;base64," + base64.b64encode(Path(args.ref_audio).read_bytes()).decode(),
            turn_detection={"type": "server_vad", "interrupt_response": True},
            timeout_s=args.timeout_s,
        )
        await client.stream_pcm16(initial + bytes(16_000 * 2 * 1600 // 1000), chunk_ms=200, realtime=True)
        await client.commit()
        await until(lambda: any(e.get("type") == "response.created" for e in client.events.events), "response.created")
        target_id = RealtimeEventCollector.response_id(
            next(e for e in client.events.events if e.get("type") == "response.created")
        )
        assert target_id
        await until(
            lambda: _events(client.events.events, target_id, "response.output_audio.delta"),
            "response.output_audio.delta",
        )
        cursor = len(client.events.events)
        await client.stream_pcm16(interrupt + bytes(16_000 * 2 * 800 // 1000), chunk_ms=args.chunk_ms, realtime=True)
        await until(lambda: _events(client.events.events, target_id, "response.done"), "cancelled response.done")
        await client.commit()
        await until(
            lambda: any(
                e.get("type") == "response.done" and RealtimeEventCollector.response_id(e) != target_id
                for e in client.events.events[cursor:]
            ),
            "follow-up response.done",
        )
        events = client.events.events
    finally:
        try:
            await client.close_session(timeout_s=args.timeout_s)
        finally:
            await client.__aexit__(None, None, None)

    done = _events(events, target_id, "response.done")
    terminal = done[0] if len(done) == 1 else None
    trailing = events[events.index(terminal) + 1 :] if terminal is not None else events
    stale = any(
        e.get("type") == "response.output_audio.delta" and RealtimeEventCollector.response_id(e) == target_id
        for e in trailing
    )
    followup_audio = any(
        e.get("type") == "response.output_audio.delta" and RealtimeEventCollector.response_id(e) != target_id
        for e in events[cursor:]
    )
    ok = (
        terminal is not None
        and isinstance(terminal.get("response"), dict)
        and terminal["response"].get("status") == "cancelled"
        and terminal["response"].get("status_details", {}).get("reason") == "turn_detected"
        and not stale
        and sum(e.get("type") == "input_audio_buffer.speech_started" for e in events[cursor:]) == 1
        and sum(e.get("type") == "input_audio_buffer.speech_stopped" for e in events[cursor:]) == 1
        and any(e.get("type") == "input_audio_buffer.committed" for e in events[cursor:])
        and followup_audio
        and not any(e.get("type") == "error" for e in events)
    )
    if ok:
        return {"ok": ok}

    origin_s = events[0].get("_client_received_at_s")

    def event_position(index: int) -> dict[str, object]:
        event = events[index]
        received_s = event.get("_client_received_at_s")
        relative_s = None
        if isinstance(origin_s, int | float) and isinstance(received_s, int | float):
            elapsed_s = received_s - origin_s
            if -float("inf") < elapsed_s < float("inf"):
                relative_s = elapsed_s
        kind = event.get("type")
        return {
            "index": index,
            "type": kind if isinstance(kind, str) else None,
            "response_id": RealtimeEventCollector.response_id(event),
            "received_after_first_s": relative_s,
        }

    terminals = []
    audio_indices: dict[str | None, list[int]] = {}
    errors = []
    for index, event in enumerate(events):
        kind = event.get("type")
        response_id = RealtimeEventCollector.response_id(event)
        if kind == "response.done" and response_id == target_id:
            response = event.get("response")
            status = response.get("status") if isinstance(response, dict) else None
            details = response.get("status_details") if isinstance(response, dict) else None
            reason = details.get("reason") if isinstance(details, dict) else None
            terminals.append(
                {
                    **event_position(index),
                    "response_is_dict": isinstance(response, dict),
                    "status": status if isinstance(status, str) else None,
                    "reason": reason if isinstance(reason, str) else None,
                }
            )
        if kind == "response.output_audio.delta":
            audio_indices.setdefault(response_id, []).append(index)
        elif kind == "error":
            error = event.get("error")
            code = error.get("code") if isinstance(error, dict) else None
            errors.append({**event_position(index), "code": code if isinstance(code, str) else None})

    control_types = {
        "response.created",
        "response.done",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "audio.cancelled",
        "output_audio_buffer.cleared",
        "session.closed",
    }
    return {
        "ok": ok,
        "diagnostics": {
            "target_response_id": target_id,
            "interrupt_cursor": cursor,
            "event_count": len(events),
            "target_done_count": len(done),
            "target_terminals": terminals,
            "stale_target_audio": stale,
            "stale_target_audio_count": sum(
                e.get("type") == "response.output_audio.delta" and RealtimeEventCollector.response_id(e) == target_id
                for e in trailing
            ),
            "speech_started_after_cursor": sum(
                e.get("type") == "input_audio_buffer.speech_started" for e in events[cursor:]
            ),
            "speech_stopped_after_cursor": sum(
                e.get("type") == "input_audio_buffer.speech_stopped" for e in events[cursor:]
            ),
            "committed_after_cursor": sum(e.get("type") == "input_audio_buffer.committed" for e in events[cursor:]),
            "followup_audio": followup_audio,
            "followup_audio_count": sum(
                e.get("type") == "response.output_audio.delta" and RealtimeEventCollector.response_id(e) != target_id
                for e in events[cursor:]
            ),
            "error_count": len(errors),
            "errors": errors,
            "control_event_order": [
                event_position(i)
                for i, e in enumerate(events)
                if isinstance(e.get("type"), str) and e.get("type") in control_types
            ],
            "audio_events": [
                {
                    "response_id": response_id,
                    "count": len(indices),
                    "first": event_position(indices[0]),
                    "last": event_position(indices[-1]),
                }
                for response_id, indices in audio_indices.items()
            ],
        },
    }
