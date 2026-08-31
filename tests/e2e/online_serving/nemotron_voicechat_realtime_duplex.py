# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import asyncio
import base64
import json
import math
import uuid
import wave
from collections.abc import Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from scipy.signal import resample_poly

from vllm_omni.experimental.fullduplex.client import RealtimeDuplexClient, wait_for, write_pcm16_wav

INPUT_SAMPLE_RATE_HZ = 16_000
OUTPUT_SAMPLE_RATE_HZ = 22_050
FRAME_SAMPLES = 1_280
FRAME_PERIOD_S = FRAME_SAMPLES / INPUT_SAMPLE_RATE_HZ
DEFAULT_FUNCTION_TOOLS = [
    {
        "type": "function",
        "name": "generate_random_number",
        "description": "Generate a random integer between min and max (inclusive).",
        "parameters": {
            "type": "object",
            "properties": {
                "min": {"type": "integer", "description": "Minimum value (inclusive)"},
                "max": {"type": "integer", "description": "Maximum value (inclusive)"},
            },
            "required": ["min", "max"],
        },
    }
]
DEFAULT_INSTRUCTIONS = "You are NVIDIA Voice Chat. Answer briefly. Start by greeting the user."
DEFAULT_FUNCTION_INSTRUCTIONS = (
    "You are NVIDIA Voice Chat. If the user's request matches an available tool, "
    "you MUST call that tool instead of answering from your own knowledge. "
    "Use only argument values spoken by the user and never invent missing values."
)


def _url(base_url: str, model: str, session_id: str) -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(duplex="1", model=model, autostart="0", session_id=session_id)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _read_wav(path: Path, *, input_channel: int = 0) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    dtypes = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}
    if width not in dtypes:
        raise ValueError(f"unsupported input sample width: {width}")
    pcm = np.frombuffer(raw, dtype=dtypes[width]).astype(np.float32)
    pcm = (pcm - 128.0) / 128.0 if width == 1 else pcm / float(1 << (width * 8 - 1))
    if not 0 <= input_channel < channels:
        raise ValueError(f"input channel {input_channel} is outside WAV channel count {channels}")
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, input_channel]
    if source_rate != INPUT_SAMPLE_RATE_HZ:
        divisor = math.gcd(source_rate, INPUT_SAMPLE_RATE_HZ)
        pcm = resample_poly(pcm, up=INPUT_SAMPLE_RATE_HZ // divisor, down=source_rate // divisor)
    return np.ascontiguousarray(pcm, dtype="<f4")


async def _stream(client: RealtimeDuplexClient, pcm: np.ndarray, *, max_frames: int | None, realtime: bool) -> int:
    count = math.ceil(pcm.size / FRAME_SAMPLES)
    if max_frames is not None:
        count = min(count, max_frames)
    for seq in range(count):
        frame = pcm[seq * FRAME_SAMPLES : (seq + 1) * FRAME_SAMPLES]
        frame = np.pad(frame, (0, FRAME_SAMPLES - frame.size)).astype("<f4")
        await client.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode("ascii"),
                "format": "pcm_f32le",
                "sample_rate_hz": INPUT_SAMPLE_RATE_HZ,
                "duration_ms": 80,
                "audio_end_ms": (seq + 1) * 80,
            }
        )
        if realtime:
            await asyncio.sleep(FRAME_PERIOD_S)
    return count


def _events(client: RealtimeDuplexClient, event_type: str) -> list[dict[str, object]]:
    return [event for event in client.events.events if event.get("type") == event_type]


async def _return_function_output_when_ready(
    client: RealtimeDuplexClient,
    *,
    output: str,
    timeout_s: float,
) -> tuple[str, int]:
    await wait_for(
        lambda: bool(client.events.errors()) or client.events.count("response.function_call_arguments.done") > 0,
        timeout_s=timeout_s,
        label="function call to execute",
    )
    if client.events.errors():
        raise AssertionError(f"function call failed before tool execution: {client.events.errors()}")
    function_done = _events(client, "response.function_call_arguments.done")[-1]
    call_id = function_done.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise AssertionError(f"completed function call has no call_id: {function_done}")
    event_count_before_output = len(client.events.events)
    await client.send(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        }
    )
    return call_id, event_count_before_output


def _write_events(path: Path, client: RealtimeDuplexClient) -> None:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in client.events.events),
        encoding="utf-8",
    )


@asynccontextmanager
async def _managed_client(client: RealtimeDuplexClient, *, timeout_s: float):
    async with client:
        try:
            yield
        finally:
            if client.events.count("session.created") and not client.events.count("session.closed"):
                with suppress(Exception):
                    await client.close_session(timeout_s=min(timeout_s, 30.0))


async def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.expect_function_call and args.instructions == DEFAULT_INSTRUCTIONS:
        instructions = DEFAULT_FUNCTION_INSTRUCTIONS
    else:
        instructions = args.instructions
    tools = DEFAULT_FUNCTION_TOOLS if args.expect_function_call else None
    session_id = f"nemotron-voicechat-{uuid.uuid4().hex}"
    client = RealtimeDuplexClient(_url(args.url, args.model, session_id))
    async with _managed_client(client, timeout_s=args.timeout_s):
        session_payload: dict[str, object] = {
            "session_id": session_id,
            "model": args.model,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm_f32le",
            "output_audio_format": "pcm16",
            "instructions": instructions,
            "idle_timeout_s": args.timeout_s,
            "turn_detection": None,
            "extra_body": {"auto_response": True},
        }
        if tools is not None:
            session_payload["tools"] = tools
        await client.send({"type": "session.update", "session": session_payload})
        await wait_for(
            lambda: client.events.count("session.created") > 0 or bool(client.events.errors()),
            timeout_s=args.timeout_s,
            label="session.created",
        )
        if client.events.errors():
            raise AssertionError(f"session setup failed: {client.events.errors()}")
        created = _events(client, "session.created")[-1]
        session = created.get("session")
        capabilities = session.get("capabilities") if isinstance(session, dict) else None
        expected = dict(
            implementation_level="model_native_duplex",
            chunk_period_ms=80,
            supports_core_resumable_request=True,
            supports_core_kv_lease=False,
            supports_multi_session=False,
        )
        if not isinstance(capabilities, dict) or any(capabilities.get(key) != value for key, value in expected.items()):
            raise AssertionError(f"unexpected capabilities: {capabilities}")

        function_output_task = (
            asyncio.create_task(
                _return_function_output_when_ready(
                    client,
                    output=args.function_output,
                    timeout_s=args.timeout_s,
                )
            )
            if args.function_output is not None
            else None
        )
        pcm = _read_wav(Path(args.input_wav), input_channel=args.input_channel)
        frame_count = await _stream(client, pcm, max_frames=args.max_frames, realtime=not args.no_realtime)
        completed_responses_at_commit = client.events.count("response.done")
        await client.send({"type": "input_audio_buffer.commit", "final": True})
        await wait_for(
            lambda: (
                bool(client.events.errors())
                or (
                    client.events.count("response.function_call_arguments.done") > 0
                    if args.expect_function_call
                    else (
                        client.events.count("response.audio.delta") >= args.minimum_audio_chunks
                        and (client.events.count("response.done") > completed_responses_at_commit)
                    )
                )
            ),
            timeout_s=args.timeout_s,
            label="model output",
        )
        await asyncio.sleep(args.drain_s)
        if client.events.errors():
            raise AssertionError(f"Realtime session emitted errors: {client.events.errors()}")
        done_events = _events(client, "response.done")
        if not args.expect_function_call:
            response = done_events[-1].get("response") if done_events else None
            status = response.get("status") if isinstance(response, dict) else None
            if status != "completed":
                raise AssertionError(f"response did not complete successfully: {done_events[-1:]}")

        function_events = [
            event for event in client.events.events if str(event.get("type", "")).startswith("response.function_call")
        ]
        function_items = [
            event
            for event in _events(client, "response.output_item.done")
            if isinstance(event.get("item"), dict) and event["item"].get("type") == "function_call"
        ]
        if args.expect_function_call and not any(
            event.get("type") == "response.function_call_arguments.done" for event in function_events
        ):
            raise AssertionError(f"no completed function call: {function_events}")
        if args.expect_function_call:
            matching_items = [
                event for event in function_items if event["item"].get("name") == args.expected_function_name
            ]
            if not matching_items:
                raise AssertionError(f"expected {args.expected_function_name!r}, got {function_items}")
            function_item = matching_items[-1]["item"]
            try:
                function_arguments = json.loads(str(function_item.get("arguments", "")))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"function arguments are not JSON: {function_item}") from exc
            if args.expected_function_arguments is not None:
                expected_arguments = json.loads(args.expected_function_arguments)
                if function_arguments != expected_arguments:
                    raise AssertionError(
                        f"function arguments differ: expected={expected_arguments!r}, actual={function_arguments!r}"
                    )

            if args.function_output is not None:
                call_id = function_item.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise AssertionError(f"function item has no call_id: {function_item}")
                assert function_output_task is not None
                returned_call_id, event_count_before_output = await function_output_task
                assert returned_call_id == call_id

                def tool_result_completed() -> bool:
                    later = client.events.events[event_count_before_output:]
                    transcript = "".join(
                        str(event.get("delta", ""))
                        for event in later
                        if event.get("type") == "response.audio_transcript.delta"
                    ).lower()
                    return (
                        any(event.get("type") == "response.audio.delta" for event in later)
                        and any(event.get("type") == "response.done" for event in later)
                        and (args.expected_post_tool_text is None or args.expected_post_tool_text.lower() in transcript)
                    )

                await wait_for(
                    lambda: bool(client.events.errors()) or tool_result_completed(),
                    timeout_s=args.timeout_s,
                    label="completed response after function output",
                )
                if client.events.errors():
                    raise AssertionError(f"function output failed: {client.events.errors()}")
                await asyncio.sleep(args.drain_s)

        audio = client.events.audio_bytes()
        audio_events = _events(client, "response.audio.delta")
        rates = {event.get("sample_rate_hz") for event in audio_events}
        if not args.expect_function_call and audio and rates != {OUTPUT_SAMPLE_RATE_HZ}:
            raise AssertionError(f"unexpected output sample rates: {rates}")
        if not args.expect_function_call and args.minimum_audio_chunks and not audio:
            raise AssertionError("model produced no audio")
        expected_bytes = 2 * OUTPUT_SAMPLE_RATE_HZ * expected["chunk_period_ms"] // 1000
        packet_sizes = [len(base64.b64decode(str(event.get("delta", "")), validate=True)) for event in audio_events]
        if not args.expect_function_call and any(size != expected_bytes for size in packet_sizes):
            raise AssertionError(f"audio deltas are not fixed 80 ms PCM16 packets: {packet_sizes}")
        audio_pcm = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
        audio_rms = float(np.sqrt(np.mean(np.square(audio_pcm)))) if audio_pcm.size else 0.0
        if not args.expect_function_call and audio and audio_rms < args.minimum_audio_rms:
            raise AssertionError(
                f"model output RMS {audio_rms:.6f} is below {args.minimum_audio_rms:.6f}; "
                "received packets contain only silence"
            )
    _write_events(output_dir / "events.jsonl", client)
    if audio:
        write_pcm16_wav(output_dir / "output.wav", audio, sample_rate_hz=OUTPUT_SAMPLE_RATE_HZ)
    result = {
        "ok": True,
        "session_id": session_id,
        "input_frames": frame_count,
        "capabilities": capabilities,
        "event_counts": {
            event_type: client.events.count(event_type)
            for event_type in sorted({str(event.get("type")) for event in client.events.events})
        },
        "audio_bytes": len(audio),
        "audio_rms": audio_rms,
        "function_events": function_events,
        "function_items": function_items,
        "output_dir": str(output_dir),
    }

    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8125/v1/realtime")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--input-channel", type=int, default=0)
    parser.add_argument("--output-dir", default="/tmp/nemotron-voicechat-duplex")
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--minimum-audio-chunks", type=int, default=1)
    parser.add_argument("--minimum-audio-rms", type=float, default=1e-4)
    parser.add_argument("--expect-function-call", action="store_true")
    parser.add_argument("--expected-function-name", default="generate_random_number")
    parser.add_argument("--expected-function-arguments")
    parser.add_argument("--function-output")
    parser.add_argument("--expected-post-tool-text")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--drain-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(asyncio.run(run(parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
