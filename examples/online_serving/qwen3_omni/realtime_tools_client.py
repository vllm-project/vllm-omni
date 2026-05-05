"""Example client demonstrating tool calling with vLLM-Omni realtime API.

This client:
1) Connects to /v1/realtime WebSocket endpoint
2) Sends session.update with tool definitions
3) For each input WAV (one or more): streams audio, drains events, executes
   any tool call, and waits for response.audio.done before moving on
4) Saves the audio response from each turn

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
from pathlib import Path
from datetime import datetime

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise SystemExit(1)


def _read_wav_pcm16(path: Path) -> bytes:
    """Read WAV file and validate it's mono 16-bit PCM 16kHz."""
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        comptype = wf.getcomptype()

        if nchannels != 1:
            raise ValueError(f"Input WAV must be mono (got {nchannels} channels).")
        if sampwidth != 2:
            raise ValueError(f"Input WAV must be 16-bit PCM (got sample width={sampwidth}).")
        if framerate != 16000:
            raise ValueError(f"Input WAV must be 16kHz (got {framerate} Hz).")
        if comptype != "NONE":
            raise ValueError(f"Input WAV must be uncompressed (got {comptype}).")

        return wf.readframes(wf.getnframes())


def _write_wav_pcm16(path: Path, data: bytes, sample_rate: int = 24000):
    """Write PCM16 data to WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(data)


# Example tool implementations
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city (mock implementation)."""
    # In a real implementation, this would call a weather API
    weather_data = {
        "Paris": {"temp": 18, "condition": "Partly cloudy"},
        "London": {"temp": 15, "condition": "Rainy"},
        "New York": {"temp": 22, "condition": "Sunny"},
        "Tokyo": {"temp": 25, "condition": "Clear"},
    }

    city_data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
    temp = city_data["temp"]

    if unit == "fahrenheit":
        temp = (temp * 9/5) + 32

    return json.dumps({
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": city_data["condition"]
    })


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression (mock implementation)."""
    try:
        # Safe evaluation for simple math
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"result": result, "expression": expression})
    except Exception as e:
        return json.dumps({"error": str(e), "expression": expression})


AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
}


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name (e.g., 'Paris', 'London')"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _turn_output_path(base: Path, turn_idx: int, total_turns: int) -> Path:
    """For single-turn, use --output-wav verbatim. For multi-turn, suffix it."""
    if total_turns == 1:
        return base
    return base.with_name(f"{base.stem}_turn{turn_idx + 1}{base.suffix}")


async def _send_audio(ws, pcm: bytes, chunk_size: int = 4096) -> None:
    for i in range(0, len(pcm), chunk_size):
        b64 = base64.b64encode(pcm[i:i + chunk_size]).decode("utf-8")
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))


async def _run_turn(ws, label: str) -> tuple[bytes, int, float, float]:
    """Drain events until response.audio.done, executing any tool calls inline.

    Returns (audio_pcm, sample_rate, ttfa_s, total_s). The caller is responsible
    for having already sent + committed the user audio.
    """
    audio_chunks: list[bytes] = []
    sample_rate = 24000
    pending_tool_calls: dict[str, dict] = {}

    t_start = time.monotonic()
    t_first_audio: float | None = None

    while True:
        msg = await ws.recv()
        event = json.loads(msg)
        etype = event.get("type")

        if etype == "response.audio.delta":
            sample_rate = event.get("sample_rate_hz", sample_rate)
            chunk = base64.b64decode(event.get("audio", ""))
            audio_chunks.append(chunk)
            if t_first_audio is None:
                t_first_audio = time.monotonic()
                print(f"  [{label}] first audio delta (TTFA={t_first_audio - t_start:.2f}s)")
        elif etype == "response.audio.done":
            print(f"  [{label}] audio.done ({len(audio_chunks)} chunks)")
            break
        elif etype == "response.text.done":
            text = event.get("text", "")
            if text:
                print(f"  [{label}] text: {text[:120]}")
        elif etype == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            pending_tool_calls.setdefault(call_id, {"name": event.get("name"), "arguments": ""})
            pending_tool_calls[call_id]["arguments"] += event.get("delta", "")
        elif etype == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            name = event.get("name")
            arguments_json = event.get("arguments", "{}")
            print(f"  [{label}] tool call: {name}({arguments_json})")
            try:
                arguments = json.loads(arguments_json)
            except json.JSONDecodeError:
                arguments = {}
            if name not in AVAILABLE_TOOLS:
                print(f"  [{label}] ERROR: unknown tool {name!r}")
                break
            result = AVAILABLE_TOOLS[name](**arguments)
            print(f"  [{label}] tool result: {result}")
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call_id, "output": result},
            }))
        elif etype == "error":
            print(f"  [{label}] ERROR: {event.get('error', event)}")
            break

    total_s = time.monotonic() - t_start
    ttfa_s = (t_first_audio - t_start) if t_first_audio is not None else float("nan")
    return b"".join(audio_chunks), sample_rate, ttfa_s, total_s


async def run_client(url: str, model: str, input_wavs: list[Path], output_wav: Path):
    """Connect once, run a turn per input WAV, save audio per turn."""
    print(f"[{_ts()}] Connecting to {url}")

    async with websockets.connect(url) as ws:
        msg = await ws.recv()
        print(f"[{_ts()}] {json.loads(msg).get('type')}")

        await ws.send(json.dumps({
            "type": "session.update",
            "model": model,
            "session": {
                "tools": TOOL_DEFINITIONS,
                "instructions": (
                    "You are a helpful voice assistant with access to tools. "
                    "Use the available tools to answer user questions when it makes sense to do so."
                ),
            },
        }))
        print(f"[{_ts()}] session.update sent ({len(TOOL_DEFINITIONS)} tools)")

        for turn_idx, in_wav in enumerate(input_wavs):
            label = f"turn{turn_idx + 1}"
            print(f"\n[{_ts()}] {label}: streaming {in_wav.name}")
            pcm = _read_wav_pcm16(in_wav)
            await _send_audio(ws, pcm)

            audio_pcm, sample_rate, ttfa_s, total_s = await _run_turn(ws, label)

            if audio_pcm:
                out_path = _turn_output_path(output_wav, turn_idx, len(input_wavs))
                _write_wav_pcm16(out_path, audio_pcm, sample_rate)
                duration_s = len(audio_pcm) / (sample_rate * 2)
                print(
                    f"  [{label}] saved {out_path.name} "
                    f"({duration_s:.2f}s audio, TTFA={ttfa_s:.2f}s, total={total_s:.2f}s)"
                )
            else:
                print(f"  [{label}] no audio received")


def main():
    parser = argparse.ArgumentParser(description="Realtime tool calling client")
    parser.add_argument("--url", default="ws://localhost:8091/v1/realtime", help="WebSocket URL")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model name")
    parser.add_argument(
        "--input-wav",
        type=Path,
        nargs="+",
        required=True,
        help="One or more input WAV files (mono, 16-bit, 16kHz). Multiple files run as sequential turns over a single WebSocket session.",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=Path("tool_output.wav"),
        help="Output WAV path. With multiple input WAVs, '_turn{N}' is inserted before the suffix.",
    )

    args = parser.parse_args()

    asyncio.run(run_client(args.url, args.model, args.input_wav, args.output_wav))


if __name__ == "__main__":
    main()