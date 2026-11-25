# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import asyncio
import base64
import json
import math
import wave
from collections.abc import Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from vllm_omni.clients.duplex import DuplexClient, EventCollector, wait_for_condition, write_pcm16_wav
from vllm_omni.clients.nemotron_voicechat import create_duplex_session_config

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


async def _stream(client: DuplexClient, pcm: np.ndarray, *, max_frames: int | None, realtime: bool) -> int:
    count = math.ceil(pcm.size / FRAME_SAMPLES)
    if max_frames is not None:
        count = min(count, max_frames)
    for seq in range(count):
        frame = pcm[seq * FRAME_SAMPLES : (seq + 1) * FRAME_SAMPLES]
        frame = np.pad(frame, (0, FRAME_SAMPLES - frame.size)).astype("<f4")
        await client.append_audio(frame.tobytes())
        if realtime:
            await asyncio.sleep(FRAME_PERIOD_S)
    return count


def _events(collector: EventCollector, event_type: str) -> list[dict[str, object]]:
    return [event for event in collector.events if event.get("type") == event_type]


async def _return_function_output_when_ready(
    client: DuplexClient,
    collector: EventCollector,
    *,
    output: str,
    timeout_s: float,
) -> tuple[str, int]:
    await wait_for_condition(
        lambda: bool(collector.errors()) or collector.count("response.function_call_arguments.done") > 0,
        timeout_s=timeout_s,
        label="function call to execute",
    )
    if collector.errors():
        raise AssertionError(f"function call failed before tool execution: {collector.errors()}")
    function_done = _events(collector, "response.function_call_arguments.done")[-1]
    call_id = function_done.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise AssertionError(f"completed function call has no call_id: {function_done}")
    event_count_before_output = len(collector.events)
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


def _write_events(path: Path, collector: EventCollector) -> None:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in collector.events),
        encoding="utf-8",
    )


@asynccontextmanager
async def _managed_client(client: DuplexClient, collector: EventCollector):
    consume_task = asyncio.create_task(collector.consume(client))
    # Subscribe before the handshake so the collector sees session.created.
    await asyncio.sleep(0)
    try:
        async with client:
            yield
        await consume_task
    finally:
        consume_task.cancel()
        with suppress(asyncio.CancelledError):
            await consume_task


async def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.expect_function_call and args.instructions == DEFAULT_INSTRUCTIONS:
        instructions = DEFAULT_FUNCTION_INSTRUCTIONS
    else:
        instructions = args.instructions
    tools = DEFAULT_FUNCTION_TOOLS if args.expect_function_call else None
    collector = EventCollector()
    client = DuplexClient(
        args.url,
        model=args.model,
        config=create_duplex_session_config(
            instructions=instructions,
            tools=tools,
            idle_timeout_s=args.timeout_s,
        ),
        handshake_timeout_s=args.timeout_s,
        reconnect=None,
        heartbeat_interval_s=None,
    )
    async with _managed_client(client, collector):
        session_id = client.session_id
        capabilities = client.session_info.get("capabilities")
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
                    collector,
                    output=args.function_output,
                    timeout_s=args.timeout_s,
                )
            )
            if args.function_output is not None
            else None
        )
        pcm = _read_wav(Path(args.input_wav), input_channel=args.input_channel)
        frame_count = await _stream(client, pcm, max_frames=args.max_frames, realtime=not args.no_realtime)
        completed_responses_at_commit = collector.count("response.done")
        await client.commit()
        await wait_for_condition(
            lambda: (
                bool(collector.errors())
                or (
                    collector.count("response.function_call_arguments.done") > 0
                    if args.expect_function_call
                    else (
                        collector.count("response.output_audio.delta") >= args.minimum_audio_chunks
                        and (
                            args.allow_incomplete_response
                            or collector.count("response.done") > completed_responses_at_commit
                        )
                    )
                )
            ),
            timeout_s=args.timeout_s,
            label="model output",
        )
        await asyncio.sleep(args.drain_s)
        if collector.errors():
            raise AssertionError(f"Realtime session emitted errors: {collector.errors()}")
        done_events = _events(collector, "response.done")
        if not args.expect_function_call and not args.allow_incomplete_response:
            response = done_events[-1].get("response") if done_events else None
            status = response.get("status") if isinstance(response, dict) else None
            if status != "completed":
                raise AssertionError(f"response did not complete successfully: {done_events[-1:]}")

        function_events = [
            event for event in collector.events if str(event.get("type", "")).startswith("response.function_call")
        ]
        function_items = [
            item
            for event in _events(collector, "response.output_item.done")
            if isinstance((item := event.get("item")), dict) and item.get("type") == "function_call"
        ]
        if args.expect_function_call and not any(
            event.get("type") == "response.function_call_arguments.done" for event in function_events
        ):
            raise AssertionError(f"no completed function call: {function_events}")
        if args.expect_function_call:
            matching_items = [item for item in function_items if item.get("name") == args.expected_function_name]
            if not matching_items:
                raise AssertionError(f"expected {args.expected_function_name!r}, got {function_items}")
            function_item: dict[str, object] = matching_items[-1]
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
                    later = collector.events[event_count_before_output:]
                    transcript = "".join(
                        str(event.get("delta", ""))
                        for event in later
                        if event.get("type") == "response.output_audio_transcript.delta"
                    ).lower()
                    return (
                        any(event.get("type") == "response.output_audio.delta" for event in later)
                        and any(event.get("type") == "response.done" for event in later)
                        and (args.expected_post_tool_text is None or args.expected_post_tool_text.lower() in transcript)
                    )

                await wait_for_condition(
                    lambda: bool(collector.errors()) or tool_result_completed(),
                    timeout_s=args.timeout_s,
                    label="completed response after function output",
                )
                if collector.errors():
                    raise AssertionError(f"function output failed: {collector.errors()}")
                await asyncio.sleep(args.drain_s)

        audio = collector.audio_bytes()
        audio_events = _events(collector, "response.output_audio.delta")
        rates = {event.get("sample_rate_hz") for event in audio_events}
        if not args.expect_function_call and audio and rates != {OUTPUT_SAMPLE_RATE_HZ}:
            raise AssertionError(f"unexpected output sample rates: {rates}")
        if not args.expect_function_call and args.minimum_audio_chunks and not audio:
            raise AssertionError("model produced no audio")
        expected_bytes = 2 * OUTPUT_SAMPLE_RATE_HZ * int(str(expected["chunk_period_ms"])) // 1000
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
    _write_events(output_dir / "events.jsonl", collector)
    if audio:
        write_pcm16_wav(output_dir / "output.wav", audio, sample_rate_hz=OUTPUT_SAMPLE_RATE_HZ)
    result: dict[str, object] = {
        "ok": True,
        "session_id": session_id if isinstance(session_id, str) else None,
        "input_frames": frame_count,
        "capabilities": capabilities,
        "event_counts": {
            event_type: collector.count(event_type)
            for event_type in sorted({str(event.get("type")) for event in collector.events})
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
    parser.add_argument(
        "--allow-incomplete-response",
        action="store_true",
        help=(
            "Accept a session whose responses all completed before the final "
            "commit. A realtime-paced pipeline delivers each turn's audio and "
            "response.done as the turn happens, so the strict "
            "done-after-commit gate only holds when delivery lags the frame "
            "clock; use this flag when measuring latency on fast configs."
        ),
    )
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
