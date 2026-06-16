"""Client for CSM-1B (Sesame) text-to-speech via the /v1/audio/speech endpoint.

CSM-1B is a plain text -> speech model with speaker-id conditioning. The OpenAI
``voice`` field maps to a CSM speaker id (a non-negative integer string, default
"0"); there are no named voice presets and no reference-audio voice cloning on
this path.

Examples:
    # Non-streaming WAV
    python speech_client.py --text "Hello from CSM." --output out.wav

    # Streaming PCM (24 kHz, mono, s16le); prints time-to-first-audio-byte
    python speech_client.py --text "Hello from CSM." --stream --output out.pcm

    # Cap the generation length (frames; 1 frame == 80 ms)
    python speech_client.py --text "A longer line." --max-new-tokens 64
"""

import argparse
import time

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "sesame/csm-1b"

# CSM / Mimi output audio is 24 kHz mono s16le PCM.
SAMPLE_RATE = 24000


def run_tts(args) -> None:
    """Generate speech via the /v1/audio/speech API."""
    payload: dict = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.response_format,
    }
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.seed is not None:
        payload["seed"] = args.seed

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    print(f"Model: {args.model}")
    print(f"Text: {args.text!r}  speaker={args.voice}")

    if args.stream:
        payload["stream"] = True
        payload["response_format"] = "pcm"
        output_path = args.output or "output.pcm"
        total = 0
        t0 = time.perf_counter()
        ttfa_ms = None
        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", api_url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    print(f"Error {resp.status_code}: {resp.read().decode(errors='replace')}")
                    return
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        if not chunk:
                            continue
                        if ttfa_ms is None:
                            ttfa_ms = (time.perf_counter() - t0) * 1000.0
                        total += len(chunk)
                        f.write(chunk)
        dur_s = total / (SAMPLE_RATE * 2)  # 2 bytes/sample, mono
        print(f"Streamed {total} PCM bytes (~{dur_s:.2f}s audio) -> {output_path}")
        if ttfa_ms is not None:
            print(f"Time-to-first-audio-byte (streamed TTFA): {ttfa_ms:.1f} ms")
        print(f"Play with: ffplay -f s16le -ar {SAMPLE_RATE} -ch_layout mono {output_path}")
        return

    output_path = args.output or "output.wav"
    t0 = time.perf_counter()
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(api_url, json=payload, headers=headers)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return
    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"Wrote {len(resp.content)} bytes -> {output_path} in {(time.perf_counter() - t0):.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSM-1B TTS client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", default="Hello, this is CSM speaking from vLLM Omni.")
    parser.add_argument("--voice", default="0", help="CSM speaker id (non-negative integer string)")
    parser.add_argument("--response-format", default="wav", choices=["wav", "pcm"])
    parser.add_argument("--stream", action="store_true", help="Stream PCM and report TTFA")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Frame cap (1 frame == 80 ms)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
