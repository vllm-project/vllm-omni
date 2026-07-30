"""Minimal single-input MiniCPM-o 4.5 Realtime duplex demo.

Run this after starting the duplex server. Strict lifecycle, overlap, and
multi-session validation lives under ``tests/e2e/online_serving``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.experimental.fullduplex.client import (  # noqa: E402
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    read_pcm16_wav,
    wait_for,
    write_pcm16_wav,
)


class _StreamingOutputWriter:
    """Persist and report output deltas as the client receives them."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.audio_chunk_dir = output_dir / "audio_chunks"
        self.audio_chunk_paths: list[Path] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_chunk_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "output.pcm").write_bytes(b"")

    def handle(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        if event_type in {
            "response.audio_transcript.delta",
            "response.output_text.delta",
        }:
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                print(delta, end="", file=sys.stderr, flush=True)
            return
        if event_type != "response.audio.delta":
            return

        delta = event.get("delta") or event.get("audio")
        if not isinstance(delta, str) or not delta:
            return
        try:
            pcm16 = base64.b64decode(delta)
        except ValueError:
            return
        if not pcm16:
            return

        chunk_index = len(self.audio_chunk_paths) + 1
        chunk_path = self.audio_chunk_dir / f"chunk_{chunk_index:04d}.wav"
        sample_rate_hz = event.get("sample_rate_hz")
        if not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
            sample_rate_hz = 24_000
        with (self.output_dir / "output.pcm").open("ab") as output_pcm:
            output_pcm.write(pcm16)
        write_pcm16_wav(chunk_path, pcm16, sample_rate_hz=sample_rate_hz)
        self.audio_chunk_paths.append(chunk_path)
        print(
            f"\n[audio chunk {chunk_index}: {len(pcm16)} bytes -> {chunk_path}]",
            file=sys.stderr,
            flush=True,
        )


class _StreamingEventCollector(RealtimeEventCollector):
    def __init__(self, writer: _StreamingOutputWriter) -> None:
        super().__init__()
        self._writer = writer

    def add(self, event: dict[str, object], *, received_at_s: float | None = None) -> None:
        super().add(event, received_at_s=received_at_s)
        self._writer.handle(self.events[-1])


def _input_committed_index(
    events: list[dict[str, object]],
    after_index: int,
) -> int | None:
    for index, event in enumerate(events[max(after_index, 0) :], start=max(after_index, 0)):
        if event.get("type") == "input_audio_buffer.committed":
            return index
    return None


def _post_commit_model_decision(
    events: list[dict[str, object]],
    committed_index: int | None,
) -> str | None:
    if committed_index is None:
        return None
    for event in events[committed_index + 1 :]:
        event_type = event.get("type")
        if event_type == "response.listen":
            return "listen"
        if event_type == "response.done":
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "cancelled":
                return "speak"
    return None


def _latest_model_decision(
    events: list[dict[str, object]],
    after_index: int,
) -> str | None:
    decision: str | None = None
    for event in events[max(after_index, 0) :]:
        event_type = event.get("type")
        if event_type == "response.listen":
            decision = "listen"
        elif event_type == "response.done":
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "cancelled":
                decision = "speak"
    return decision


def _chunk_period_ms(events: list[dict[str, object]]) -> int:
    for event in reversed(events):
        session = event.get("session")
        if not isinstance(session, dict):
            continue
        capabilities = session.get("capabilities")
        if not isinstance(capabilities, dict):
            continue
        chunk_period_ms = capabilities.get("chunk_period_ms")
        if isinstance(chunk_period_ms, int) and chunk_period_ms > 0:
            return chunk_period_ms
    return 1000


def _has_residual_model_unit(pcm16: bytes, *, chunk_period_ms: int) -> bool:
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_period_ms // 1000
    return bool(unit_bytes > 0 and len(pcm16) % unit_bytes)


def _response_in_progress(events: list[dict[str, object]]) -> bool:
    return sum(event.get("type") == "response.created" for event in events) > sum(
        event.get("type") == "response.done" for event in events
    )


def _event_count_after(
    events: list[dict[str, object]],
    event_type: str,
    index: int | None,
) -> int:
    if index is None:
        return 0
    return sum(event.get("type") == event_type for event in events[index + 1 :])


def _ref_audio_data_url(path: str | None) -> str | None:
    if path is None:
        return None
    ref_path = Path(path).expanduser()
    return "data:audio/wav;base64," + base64.b64encode(ref_path.read_bytes()).decode("ascii")


async def run_demo(args: argparse.Namespace) -> dict[str, object]:
    input_pcm16 = read_pcm16_wav(Path(args.input_wav))
    if not input_pcm16:
        raise ValueError("input WAV has no audio")

    output_dir = Path(args.output_dir)
    stream_writer = _StreamingOutputWriter(output_dir)

    url = build_realtime_url(
        args.url,
        args.model,
        autostart=False if args.ref_audio else None,
        session_id=args.session_id,
    )
    client = RealtimeDuplexClient(url)
    client.events = _StreamingEventCollector(stream_writer)
    async with client:
        await client.configure(
            args.model,
            ref_audio=_ref_audio_data_url(args.ref_audio),
            session_id=args.session_id,
            timeout_s=args.timeout_s,
        )
        stream_event_cursor = len(client.events.events)
        await client.stream_pcm16(
            input_pcm16,
            chunk_ms=args.chunk_ms,
            realtime=not args.no_realtime_pacing,
        )
        commit_event_cursor = len(client.events.events)
        stream_decision = _latest_model_decision(client.events.events, stream_event_cursor)
        input_has_residual_model_unit = _has_residual_model_unit(
            input_pcm16,
            chunk_period_ms=_chunk_period_ms(client.events.events),
        )
        wait_for_post_commit_decision = False
        commit_sent_at_s = time.monotonic()
        await client.commit()
        wait_error: str | None = None
        committed_index: int | None = None
        post_commit_decision: str | None = None
        try:
            await wait_for(
                lambda: _input_committed_index(client.events.events, commit_event_cursor) is not None,
                timeout_s=args.timeout_s,
                label="input_audio_buffer.committed",
            )
            committed_index = _input_committed_index(client.events.events, commit_event_cursor)
            stream_decision = _latest_model_decision(client.events.events[: committed_index + 1], stream_event_cursor)
            wait_for_post_commit_decision = input_has_residual_model_unit or _response_in_progress(
                client.events.events[: committed_index + 1]
            )
            if wait_for_post_commit_decision:
                await wait_for(
                    lambda: _post_commit_model_decision(client.events.events, committed_index) is not None,
                    timeout_s=args.timeout_s,
                    label="post-commit model decision or response drain",
                )
                post_commit_decision = _post_commit_model_decision(client.events.events, committed_index)
        except TimeoutError as exc:
            wait_error = str(exc)
        await client.acknowledge_playback()
        close_error: str | None = None
        try:
            await client.close_session(timeout_s=args.timeout_s)
        except TimeoutError as exc:
            close_error = str(exc)

        audio = client.events.audio_bytes()
        first_text_at_s = client.events.first_received_at(
            "response.audio_transcript.delta",
            "response.output_text.delta",
            after_s=commit_sent_at_s,
        )
        first_audio_at_s = client.events.first_received_at(
            "response.audio.delta",
            after_s=commit_sent_at_s,
        )
        response_created_at_s = client.events.first_received_at(
            "response.created",
            after_s=commit_sent_at_s,
        )
        response_done_at_s = client.events.first_received_at(
            "response.done",
            after_s=commit_sent_at_s,
        )
        audio_duration_s = len(audio) / (client.events.output_sample_rate_hz * 2)
        response_generation_s = (
            response_done_at_s - response_created_at_s
            if response_done_at_s is not None and response_created_at_s is not None
            else None
        )
        transcript_deltas = [
            str(event.get("delta", ""))
            for event in client.events.events
            if event.get("type")
            in {
                "response.audio_transcript.delta",
                "response.output_text.delta",
            }
        ]
        response_id = client.events.response_ids[0] if client.events.response_ids else None
        timing = client.events.timing_summary(
            after_s=commit_sent_at_s,
            input_committed_at_s=commit_sent_at_s,
            response_id=response_id,
        )
        errors = client.events.errors()
        if wait_error:
            errors.append({"type": "client.timeout", "message": wait_error})
        if close_error:
            errors.append({"type": "client.timeout", "message": close_error})
        model_decision = post_commit_decision or stream_decision
        (output_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in client.events.events),
            encoding="utf-8",
        )
        (output_dir / "output.pcm").write_bytes(audio)
        if audio:
            write_pcm16_wav(
                output_dir / "output.wav",
                audio,
                sample_rate_hz=client.events.output_sample_rate_hz,
            )

        result = {
            "ok": (
                client.events.count("session.created") > 0
                and client.events.count("session.closed") > 0
                and not errors
                and model_decision is not None
                and (bool(audio) or not args.require_audio)
            ),
            "model_decision": model_decision,
            "post_commit": {
                "input_committed_event_index": committed_index,
                "decision": post_commit_decision,
                "decision_required": wait_for_post_commit_decision,
                "input_had_residual_model_unit": input_has_residual_model_unit,
                "response_listen_count": _event_count_after(
                    client.events.events,
                    "response.listen",
                    committed_index,
                ),
                "response_done_count": _event_count_after(
                    client.events.events,
                    "response.done",
                    committed_index,
                ),
            },
            "audio_bytes": len(audio),
            "audio_chunk_count": len(stream_writer.audio_chunk_paths),
            "audio_chunk_files": [str(path) for path in stream_writer.audio_chunk_paths],
            "output_sample_rate_hz": client.events.output_sample_rate_hz,
            "latency": {
                "ttft_ms": (
                    round((first_text_at_s - commit_sent_at_s) * 1000, 2) if first_text_at_s is not None else None
                ),
                "ttfp_ms": (
                    round((first_audio_at_s - commit_sent_at_s) * 1000, 2) if first_audio_at_s is not None else None
                ),
                "rtf": (
                    round(response_generation_s / audio_duration_s, 4)
                    if response_generation_s is not None and audio_duration_s > 0
                    else None
                ),
                "response_generation_ms": (
                    round(response_generation_s * 1000, 2) if response_generation_s is not None else None
                ),
                "text_stream_ms": (
                    round((response_done_at_s - first_text_at_s) * 1000, 2)
                    if response_done_at_s is not None and first_text_at_s is not None
                    else None
                ),
                "transcript_delta_count": len(transcript_deltas),
                "audio_duration_s": round(audio_duration_s, 3),
                "measurement_origin": "input_audio_buffer.commit send",
            },
            "timing": timing,
            "response_ids": client.events.response_ids,
            "transcript": "".join(transcript_deltas),
            "errors": errors,
            "output_dir": str(output_dir),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--session-id")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument(
        "--ref-audio",
        required=True,
        help=(
            "Reference WAV for the MiniCPM-o duplex assistant voice. "
            "This demo matches the official flow by always providing a reference audio clip."
        ),
    )
    parser.add_argument("--output-dir", default="/tmp/minicpmo_realtime_duplex_demo")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--no-realtime-pacing", action="store_true")
    parser.add_argument("--require-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = asyncio.run(run_demo(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
