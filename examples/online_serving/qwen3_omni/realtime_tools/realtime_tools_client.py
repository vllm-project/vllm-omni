"""Example client demonstrating tool calling with vLLM-Omni realtime API.

This client:
1) Connects to /v1/realtime WebSocket endpoint
2) Sends session.update with tool definitions (a useful weather tool plus
   a similarly-named *trap* tool that should never be called)
3) For each input WAV, streams audio over a single WebSocket session,
   drains events, executes any tool call inline, and waits for
   response.audio.done before moving on
4) Saves the audio response from each turn and prints a per-turn +
   end-of-session summary (audio received, tool calls, latency)

Single-turn usage:
  python realtime_tools_client.py \\
      --url ws://localhost:8091/v1/realtime \\
      --model Qwen/Qwen3-Omni-30B-A3B-Instruct \\
      --input-wav input_16k_mono.wav \\
      --output-wav tool_output.wav

Multi-turn usage (multiple WAVs over one WebSocket session — proves
conversation context is retained across turns):
  python realtime_tools_client.py \\
      --input-wav greeting.wav weather_paris.wav weather_london.wav \\
      --output-wav response.wav   # writes response_turn1.wav, _turn2.wav, ...

Suggested 3-turn flow demonstrating context retention:
  Turn 1  "Hello"                              → greeting (no tool)
  Turn 2  "What is the weather like in Paris?" → get_current_weather(Paris)
  Turn 3  "And what about London?"             → get_current_weather(London)
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^
                                                 proves prior-turn context
                                                 was retained — turn 3 is
                                                 ambiguous on its own.

The trap tool (``get_city_timezone``) has a similar signature; if the
model invokes it for a weather question, it picked a tool without
understanding the query.

Dependencies:
  pip install websockets
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Tool definitions + mock implementations
# ---------------------------------------------------------------------------

# Real tool: weather lookup keyed by city.
DEMO_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Paris"},
            },
            "required": ["location"],
        },
    },
}

# Trap tool: looks plausibly useful and shares the same `location` parameter
# but should NEVER be called for a weather question. Any invocation means the
# model picked a tool without understanding the query.
TRAP_TOOL = {
    "type": "function",
    "function": {
        "name": "get_city_timezone",
        "description": (
            "Look up the IANA timezone identifier and current UTC offset for a city. "
            "Useful for scheduling, calendar conversion, or displaying local time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Paris"},
            },
            "required": ["location"],
        },
    },
}

TOOL_DEFINITIONS = [DEMO_TOOL, TRAP_TOOL]

# Mock weather data keyed by city (case-insensitive).
MOCK_WEATHER: dict[str, dict[str, str]] = {
    "paris": {"temperature": "22°C", "condition": "sunny"},
    "london": {"temperature": "14°C", "condition": "cloudy with light rain"},
    "new york": {"temperature": "26°C", "condition": "clear"},
    "tokyo": {"temperature": "25°C", "condition": "humid"},
}
DEFAULT_WEATHER = {"temperature": "18°C", "condition": "partly cloudy"}


def _mock_weather_result(location: str) -> str:
    data = MOCK_WEATHER.get(location.strip().lower(), DEFAULT_WEATHER)
    return json.dumps({"location": location, **data})


def _mock_timezone_result(location: str) -> str:
    return json.dumps({"location": location, "timezone": "Europe/Paris", "utc_offset": "+02:00"})


# ---------------------------------------------------------------------------
# Latency tracking
# ---------------------------------------------------------------------------


@dataclass
class TurnLatency:
    """Per-turn timing measurements (all monotonic seconds)."""

    turn: int
    t_start: float = 0.0
    t_first_audio_delta: float = 0.0
    t_audio_done: float = 0.0
    t_tool_result_sent: float = 0.0
    audio_bytes_received: int = 0
    sample_rate: int = 24000

    @property
    def ttfa_s(self) -> float:
        if self.t_first_audio_delta:
            return self.t_first_audio_delta - self.t_start
        return float("nan")

    @property
    def tool_to_first_audio_s(self) -> float:
        if self.t_tool_result_sent and self.t_first_audio_delta:
            return self.t_first_audio_delta - self.t_tool_result_sent
        return float("nan")

    @property
    def total_s(self) -> float:
        if self.t_audio_done:
            return self.t_audio_done - self.t_start
        return float("nan")

    @property
    def audio_duration_s(self) -> float:
        if self.sample_rate:
            return self.audio_bytes_received / (self.sample_rate * 2)
        return 0.0


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------


def _read_wav_pcm16(path: Path) -> bytes:
    """Read a mono 16-bit 16 kHz uncompressed WAV and return its PCM frames."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Input WAV must be mono (got {wf.getnchannels()} channels).")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Input WAV must be 16-bit PCM (got sample width={wf.getsampwidth()}).")
        if wf.getframerate() != 16000:
            raise ValueError(f"Input WAV must be 16 kHz (got {wf.getframerate()} Hz).")
        if wf.getcomptype() != "NONE":
            raise ValueError(f"Input WAV must be uncompressed (got {wf.getcomptype()}).")
        return wf.readframes(wf.getnframes())


def _write_wav_pcm16(path: Path, data: bytes, sample_rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _turn_output_path(base: Path, turn_idx: int, total_turns: int) -> Path:
    """For single-turn, use --output-wav verbatim; for multi-turn, suffix it."""
    if total_turns == 1:
        return base
    return base.with_name(f"{base.stem}_turn{turn_idx + 1}{base.suffix}")


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


async def _send_audio(ws, pcm: bytes, chunk_size: int = 4096) -> None:
    for i in range(0, len(pcm), chunk_size):
        b64 = base64.b64encode(pcm[i : i + chunk_size]).decode("utf-8")
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))


async def _wait_session_created(ws) -> None:
    raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
    event = json.loads(raw)
    print(f"[{_ts()}] {event.get('type')}: id={event.get('id', '?')}")


async def _execute_tool_and_reply(
    ws, name: str, arguments_json: str, call_id: str, latency: TurnLatency, trap_calls: list[dict]
) -> None:
    """Run the mock implementation for ``name`` and send the result back."""
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        arguments = {}
    location = arguments.get("location", "")

    if name == "get_current_weather":
        result = _mock_weather_result(location)
        print(f"  tool result: {result}")
    elif name == "get_city_timezone":
        # Trap fired: log prominently. Still return plausible data so the
        # session can finish and downstream metrics stay measurable.
        trap_calls.append({"name": name, "arguments": arguments_json})
        result = _mock_timezone_result(location)
        print(f"  *** TRAP TRIGGERED *** get_city_timezone called for {location!r}")
    else:
        result = json.dumps({"error": f"unknown tool {name!r}"})
        print(f"  ERROR: unknown tool {name!r}")

    await ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call_id, "output": result},
            }
        )
    )
    latency.t_tool_result_sent = time.monotonic()


async def _run_turn(ws, label: str, latency: TurnLatency, trap_calls: list[dict]) -> tuple[bytes, int, list[dict]]:
    """Drain events until response.audio.done, executing any tool calls inline.

    Returns ``(audio_pcm, sample_rate, tool_calls_made)``. The caller is
    responsible for having already sent + committed the user audio.
    """
    audio_chunks: list[bytes] = []
    sample_rate = 24000
    tool_calls: list[dict] = []

    latency.t_start = time.monotonic()

    while True:
        msg = await ws.recv()
        event = json.loads(msg)
        etype = event.get("type")

        if etype == "response.audio.delta":
            sample_rate = event.get("sample_rate_hz", sample_rate)
            chunk = base64.b64decode(event.get("audio", ""))
            audio_chunks.append(chunk)
            latency.audio_bytes_received += len(chunk)
            latency.sample_rate = sample_rate
            if not latency.t_first_audio_delta:
                latency.t_first_audio_delta = time.monotonic()
                print(f"  [{label}] first audio delta (TTFA={latency.ttfa_s:.2f}s)")
        elif etype == "response.audio.done":
            latency.t_audio_done = time.monotonic()
            print(f"  [{label}] audio.done ({len(audio_chunks)} chunks)")
            break
        elif etype == "response.text.done":
            text = event.get("text", "")
            if text:
                print(f"  [{label}] text: {text[:120]}")
        elif etype == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            name = event.get("name")
            arguments_json = event.get("arguments", "{}")
            print(f"  [{label}] tool call: {name}({arguments_json})")
            tool_calls.append({"name": name, "arguments": arguments_json, "call_id": call_id})
            await _execute_tool_and_reply(ws, name, arguments_json, call_id, latency, trap_calls)
        elif etype == "error":
            print(f"  [{label}] ERROR: {event.get('error', event)}")
            break

    return b"".join(audio_chunks), sample_rate, tool_calls


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------


INSTRUCTIONS = (
    "You are a helpful English-speaking voice assistant. "
    "Whenever the user speaks, consider using the available tools to answer "
    "user questions if it makes sense to do so. Do not talk about things "
    "outside of the user request unless directed to do so by the user, or "
    "include extra information outside the scope of the user question."
)


async def run_client(url: str, model: str, input_wavs: list[Path], output_wav: Path) -> None:
    """Connect once, run a turn per input WAV, and print a session summary."""
    print(f"[{_ts()}] Connecting to {url}")

    trap_calls: list[dict] = []
    results: dict[int, tuple[list[dict], TurnLatency, bool]] = {}

    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        await _wait_session_created(ws)

        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "model": model,
                    "session": {"tools": TOOL_DEFINITIONS, "instructions": INSTRUCTIONS},
                }
            )
        )
        print(f"[{_ts()}] session.update sent ({len(TOOL_DEFINITIONS)} tools)")

        for turn_idx, in_wav in enumerate(input_wavs):
            label = f"turn{turn_idx + 1}"
            print(f"\n[{_ts()}] {label}: streaming {in_wav.name}")
            pcm = _read_wav_pcm16(in_wav)
            await _send_audio(ws, pcm)

            latency = TurnLatency(turn=turn_idx + 1)
            audio_pcm, sample_rate, tool_calls = await _run_turn(ws, label, latency, trap_calls)

            audio_ok = bool(audio_pcm)
            if audio_ok:
                out_path = _turn_output_path(output_wav, turn_idx, len(input_wavs))
                _write_wav_pcm16(out_path, audio_pcm, sample_rate)
                print(
                    f"  [{label}] saved {out_path.name} "
                    f"({latency.audio_duration_s:.2f}s audio, "
                    f"TTFA={latency.ttfa_s:.2f}s, total={latency.total_s:.2f}s)"
                )
            else:
                print(f"  [{label}] no audio received")

            results[turn_idx + 1] = (tool_calls, latency, audio_ok)

    _print_session_summary(results, trap_calls)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_session_summary(
    results: dict[int, tuple[list[dict], TurnLatency, bool]],
    trap_calls: list[dict],
) -> None:
    print("\n--- Session summary ---")
    for n in sorted(results):
        tool_calls, _, audio_ok = results[n]
        tool_str = ", ".join(f"{tc['name']}({tc['arguments'][:60]})" for tc in tool_calls) or "(none)"
        status = "ok" if audio_ok else "NO AUDIO"
        print(f"  Turn {n}: audio={status}  tool_calls={tool_str}")

    trap_label = "PASS (never called)" if not trap_calls else f"FAIL ({len(trap_calls)} call(s))"
    print(f"  Trap tool (get_city_timezone) unused: {trap_label}")

    print("\n--- Latency ---")
    print(f"  {'Turn':<6} {'TTFA':>8} {'Total':>8} {'Audio dur':>10} {'Tool→Audio':>12}")
    print(f"  {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 12}")
    for n in sorted(results):
        _, lat, _ = results[n]
        tool_col = f"{lat.tool_to_first_audio_s:.2f}s" if lat.t_tool_result_sent else "n/a"
        print(f"  {n:<6} {lat.ttfa_s:>7.2f}s {lat.total_s:>7.2f}s {lat.audio_duration_s:>9.2f}s {tool_col:>12}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime tool calling client")
    parser.add_argument("--url", default="ws://localhost:8091/v1/realtime", help="WebSocket URL")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model name")
    parser.add_argument(
        "--input-wav",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more input WAV files (mono, 16-bit, 16 kHz). Multiple files "
            "run as sequential turns over a single WebSocket session."
        ),
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=Path("tool_output.wav"),
        help=("Output WAV path. With multiple input WAVs, '_turn{N}' is inserted before the suffix."),
    )

    args = parser.parse_args()
    asyncio.run(run_client(args.url, args.model, args.input_wav, args.output_wav))


if __name__ == "__main__":
    main()
