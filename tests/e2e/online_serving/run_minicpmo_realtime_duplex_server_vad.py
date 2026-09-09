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
    session_id = f"server-vad-hard-interrupt-{id(args)}"
    client = RealtimeDuplexClient(build_realtime_url(args.url, args.model, autostart=False, session_id=session_id))

    async def until(predicate, label: str) -> None:
        await wait_for(predicate, timeout_s=args.timeout_s, label=label)

    try:
        await client.__aenter__()
        await client.configure(
            args.model,
            ref_audio="data:audio/wav;base64," + base64.b64encode(Path(args.ref_audio).read_bytes()).decode(),
            session_id=session_id,
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
        await until(lambda: _events(client.events.events, target_id, "response.audio.delta"), "response.audio.delta")
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
        e.get("type") == "response.audio.delta" and RealtimeEventCollector.response_id(e) == target_id for e in trailing
    )
    followup_audio = any(
        e.get("type") == "response.audio.delta" and RealtimeEventCollector.response_id(e) != target_id
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
    return {"ok": ok}
