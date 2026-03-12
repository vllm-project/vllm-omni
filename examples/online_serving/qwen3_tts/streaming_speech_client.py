"""WebSocket client for streaming text-input TTS.

Connects to the /v1/audio/speech/stream endpoint, sends text incrementally
(simulating real-time STT output), and saves per-sentence audio files.

Usage:
    # Send full text at once
    python streaming_speech_client.py --text "Hello world. How are you? I am fine."

    # Simulate STT: send text word-by-word with delay
    python streaming_speech_client.py \
        --text "Hello world. How are you? I am fine." \
        --simulate-stt --stt-delay 0.1

    # VoiceDesign task
    python streaming_speech_client.py \
        --text "Today is a great day. The weather is nice." \
        --task-type VoiceDesign \
        --instructions "A cheerful young female voice"

    # Base task (voice cloning)
    python streaming_speech_client.py \
        --text "Hello world. How are you?" \
        --task-type Base \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of reference audio"

Requirements:
    pip install websockets
"""

import argparse
import asyncio
import json
import os

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise SystemExit(1)


async def stream_tts(
    url: str,
    text: str,
    config: dict,
    output_dir: str,
    simulate_stt: bool = False,
    stt_delay: float = 0.1,
) -> None:
    """Connect to the streaming TTS endpoint and process audio responses."""
    os.makedirs(output_dir, exist_ok=True)

    async with websockets.connect(url) as ws:
        # 1. Send session config
        config_msg = {"type": "session.config", **config}
        await ws.send(json.dumps(config_msg))
        print(f"Sent session config: {config}")

        # 2. Send text (either all at once or word-by-word)
        async def send_text():
            if simulate_stt:
                words = text.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input.text",
                                "text": chunk,
                            }
                        )
                    )
                    print(f"  Sent: {chunk!r}")
                    await asyncio.sleep(stt_delay)
            else:
                await ws.send(
                    json.dumps(
                        {
                            "type": "input.text",
                            "text": text,
                        }
                    )
                )
                print(f"Sent full text: {text!r}")

            # 3. Signal end of input
            await ws.send(json.dumps({"type": "input.done"}))
            print("Sent input.done")

        # Run sender and receiver concurrently
        sender_task = asyncio.create_task(send_text())

        response_format = config.get("response_format", "wav")
        current_sentence_index = 0
        current_chunks: list[bytes] = []

        try:
            while True:
                message = await ws.recv()

                if isinstance(message, bytes):
                    current_chunks.append(message)
                    print(f"  Received audio chunk for sentence {current_sentence_index}: {len(message)} bytes")
                else:
                    # JSON frame
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "audio.start":
                        current_sentence_index = msg["sentence_index"]
                        current_chunks = []
                        print(f"  [sentence {msg['sentence_index']}] Generating: {msg['sentence_text']!r}")
                    elif msg_type == "audio.done":
                        filename = os.path.join(
                            output_dir,
                            f"sentence_{msg['sentence_index']:03d}.{response_format}",
                        )
                        with open(filename, "wb") as f:
                            f.write(b"".join(current_chunks))
                        print(
                            f"  [sentence {msg['sentence_index']}] Done"
                            f" bytes={msg.get('total_bytes', len(b''.join(current_chunks)))}"
                            f" error={msg.get('error', False)}"
                            f" -> {filename}"
                        )
                        current_chunks = []
                    elif msg_type == "session.done":
                        print(f"\nSession complete: {msg['total_sentences']} sentence(s) generated")
                        break
                    elif msg_type == "error":
                        print(f"  ERROR: {msg['message']}")
                    else:
                        print(f"  Unknown message: {msg}")
        finally:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass  # Task cancellation is expected during shutdown

    print(f"\nAudio files saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Streaming text-input TTS client")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/v1/audio/speech/stream",
        help="WebSocket endpoint URL",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output-dir",
        default="streaming_tts_output",
        help="Directory to save audio files (default: streaming_tts_output)",
    )

    # Session config options
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--voice", default="Vivian", help="Speaker voice")
    parser.add_argument(
        "--task-type",
        default="CustomVoice",
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help="TTS task type",
    )
    parser.add_argument("--language", default="Auto", help="Language")
    parser.add_argument("--instructions", default=None, help="Voice style instructions")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "pcm", "flac", "mp3", "aac", "opus"],
        help="Audio format",
    )
    parser.add_argument(
        "--stream-audio",
        action="store_true",
        help="Receive one or more PCM chunks per sentence (requires --response-format pcm)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.25-4.0)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max tokens")

    # Base task options
    parser.add_argument("--ref-audio", default=None, help="Reference audio")
    parser.add_argument("--ref-text", default=None, help="Reference text")
    parser.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        default=False,
        help="Speaker embedding only mode",
    )

    # STT simulation
    parser.add_argument(
        "--simulate-stt",
        action="store_true",
        help="Simulate STT by sending text word-by-word",
    )
    parser.add_argument(
        "--stt-delay",
        type=float,
        default=0.1,
        help="Delay between words in STT simulation (seconds)",
    )

    args = parser.parse_args()

    # Build session config (only include non-None values)
    config = {}
    for key in [
        "model",
        "voice",
        "task_type",
        "language",
        "instructions",
        "response_format",
        "speed",
        "max_new_tokens",
        "ref_audio",
        "ref_text",
    ]:
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            config[key] = val
    if args.stream_audio:
        config["stream_audio"] = True
    if args.x_vector_only_mode:
        config["x_vector_only_mode"] = True

    asyncio.run(
        stream_tts(
            url=args.url,
            text=args.text,
            config=config,
            output_dir=args.output_dir,
            simulate_stt=args.simulate_stt,
            stt_delay=args.stt_delay,
        )
    )


if __name__ == "__main__":
    main()
