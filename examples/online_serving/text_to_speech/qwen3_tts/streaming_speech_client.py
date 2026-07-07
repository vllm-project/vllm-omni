"""WebSocket client for streaming text-input TTS.

Connects to the /v1/audio/speech/stream endpoint, sends text incrementally
(simulating real-time STT output), and saves per-sentence audio files.

Usage:
    # Send full text at once
    python streaming_speech_client.py --text "Hello world. How are you? I am fine."

    # Pick a built-in speaker for CustomVoice models
    python streaming_speech_client.py \
        --text "打开电子枪，关闭扫描，切换到低倍模式。" \
        --speaker "Serena" \
        --language "Chinese"

    # Simulate STT: send text word-by-word with delay
    python streaming_speech_client.py \
        --text "Hello world. How are you? I am fine." \
        --simulate-stt --stt-delay 0.1

    # Receive JSON sidecar chunks with word-level timestamps
    python streaming_speech_client.py \
        --text "Hello world. How are you?" \
        --stream-audio \
        --response-format pcm \
        --word-timestamps

    # Also save every received audio.chunk frame as individual PCM/WAV files
    python streaming_speech_client.py \
        --text "Hello world. How are you?" \
        --stream-audio \
        --response-format pcm \
        --word-timestamps \
        --save-chunks

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
import base64
import json
import os
import wave

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise SystemExit(1)


def save_audio_file(output_dir: str, sentence_index: int, response_format: str, chunks: list[bytes]) -> str:
    audio_bytes = b"".join(chunks)
    if response_format == "pcm":
        raw_path = os.path.join(output_dir, f"sentence_{sentence_index:03d}.pcm")
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

        wav_path = os.path.join(output_dir, f"sentence_{sentence_index:03d}.wav")
        save_pcm_wav(wav_path, audio_bytes)
        return wav_path

    filename = os.path.join(output_dir, f"sentence_{sentence_index:03d}.{response_format}")
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    return filename


def save_pcm_wav(path: str, audio_bytes: bytes, sample_rate: int = 24000) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)


def timestamp_words_text(timestamps: list[dict] | None) -> str:
    if not timestamps:
        return ""
    return " ".join(str(item.get("word", "")) for item in timestamps if item.get("word"))


async def stream_tts(
    url: str,
    text: str,
    config: dict,
    output_dir: str,
    simulate_stt: bool = False,
    stt_delay: float = 0.1,
    save_chunks: bool = False,
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
        word_timestamps = bool(config.get("word_timestamps", False))
        current_sentence_index = 0
        current_sentence_text = ""
        current_chunks: list[bytes] = []
        current_timestamps: list[dict] = []
        # Distinguish `null` (aligner failed) from `[]` (silence / no tokens):
        # the sentence is a failure only if no frame ever carried timestamps.
        alignment_frame_seen = False

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
                        current_sentence_text = msg.get("sentence_text", "")
                        current_chunks = []
                        current_timestamps = []
                        alignment_frame_seen = False
                        print(f"  [sentence {msg['sentence_index']}] Generating: {current_sentence_text!r}")
                    elif msg_type == "audio.chunk":
                        audio = base64.b64decode(msg["audio_b64"])
                        current_chunks.append(audio)
                        chunk_timestamps = msg.get("timestamps")
                        # Sentence-level alignment sends a trailing timestamp-only
                        # frame with empty audio; don't write an empty chunk file.
                        if save_chunks and audio:
                            chunk_base = os.path.join(
                                output_dir,
                                f"sentence_{msg['sentence_index']:03d}_chunk_{msg['chunk_id']:03d}",
                            )
                            chunk_pcm = f"{chunk_base}.pcm"
                            chunk_wav = f"{chunk_base}.wav"
                            chunk_json = f"{chunk_base}_timestamps.json"
                            with open(chunk_pcm, "wb") as f:
                                f.write(audio)
                            save_pcm_wav(chunk_wav, audio, int(msg.get("sample_rate", 24000)))
                            with open(chunk_json, "w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "sentence_index": msg["sentence_index"],
                                        "chunk_id": msg["chunk_id"],
                                        "chunk_start_ms": msg.get("chunk_start_ms"),
                                        "chunk_end_ms": msg.get("chunk_end_ms"),
                                        "sentence_text": current_sentence_text,
                                        "covered_text": timestamp_words_text(chunk_timestamps),
                                        "timestamps": chunk_timestamps,
                                    },
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                        if chunk_timestamps is None:
                            print(
                                f"  [sentence {msg['sentence_index']}] chunk {msg['chunk_id']}: "
                                f"{len(audio)} bytes, timestamps unavailable"
                            )
                        else:
                            alignment_frame_seen = True
                            current_timestamps.extend(chunk_timestamps)
                            print(
                                f"  [sentence {msg['sentence_index']}] chunk {msg['chunk_id']}: "
                                f"{len(audio)} bytes, {len(chunk_timestamps)} timestamp(s)"
                            )
                        if save_chunks and audio:
                            print(f"    saved chunk -> {chunk_wav}")
                    elif msg_type == "audio.done":
                        filename = save_audio_file(
                            output_dir,
                            msg["sentence_index"],
                            response_format,
                            current_chunks,
                        )
                        if word_timestamps:
                            # `null` on failure, else the accumulated list (`[]` for silence).
                            sentence_timestamps = current_timestamps if alignment_frame_seen else None
                            ts_filename = os.path.join(
                                output_dir,
                                f"sentence_{msg['sentence_index']:03d}_timestamps.json",
                            )
                            with open(ts_filename, "w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "sentence_index": msg["sentence_index"],
                                        "sentence_text": current_sentence_text,
                                        "covered_text": timestamp_words_text(sentence_timestamps),
                                        "timestamps": sentence_timestamps,
                                    },
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                        print(
                            f"  [sentence {msg['sentence_index']}] Done"
                            f" bytes={msg.get('total_bytes', len(b''.join(current_chunks)))}"
                            f" error={msg.get('error', False)}"
                            f" -> {filename}"
                        )
                        if word_timestamps:
                            print(f"  [sentence {msg['sentence_index']}] Timestamps -> {ts_filename}")
                        current_chunks = []
                        current_timestamps = []
                        alignment_frame_seen = False
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
    parser.add_argument("--speaker", default="Vivian", help="Speaker name")
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
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Receive JSON sidecar chunks with word-level timestamps (requires --stream-audio --response-format pcm)",
    )
    parser.add_argument(
        "--save-chunks",
        action="store_true",
        help="Save each audio.chunk frame as sentence_XXX_chunk_YYY.{pcm,wav,json}",
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

    # Build session config (only include non-None values).
    # Server canonical field is `voice`; `speaker` is accepted as an alias
    # (see StreamingSpeechSessionConfig.voice in protocol/audio.py).
    config = {}
    for key in [
        "model",
        "speaker",
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
    if args.word_timestamps:
        config["word_timestamps"] = True
        config["stream_audio"] = True
        config["response_format"] = "pcm"
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
            save_chunks=args.save_chunks,
        )
    )


if __name__ == "__main__":
    main()
