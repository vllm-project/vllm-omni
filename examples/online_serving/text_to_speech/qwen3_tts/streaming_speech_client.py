"""WebSocket client for streaming text-input TTS.

Connects to the /v1/audio/speech/stream endpoint, sends text incrementally
(simulating real-time STT output), and saves per-sentence audio files.

Usage:
    # Send full text at once
    python streaming_speech_client.py --text "Hello world. How are you? I am fine."

    # Several utterances over one connection: input.done flushes each of them
    # and the connection stays open, so only the first pays the handshake
    python streaming_speech_client.py \
        --text "First utterance." \
        --text "Second utterance, same connection."

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


def frame_basename(msg: dict) -> str:
    """Name a file after the utterance and sentence a frame belongs to.

    Every utterance is sentence 0, so the utterance index is what keeps the
    files of one connection from overwriting each other.
    """
    return f"utterance_{msg['utterance_index']:03d}_sentence_{msg['sentence_index']:03d}"


def save_audio_file(output_dir: str, basename: str, response_format: str, chunks: list[bytes]) -> str:
    audio_bytes = b"".join(chunks)
    if response_format == "pcm":
        raw_path = os.path.join(output_dir, f"{basename}.pcm")
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

        wav_path = os.path.join(output_dir, f"{basename}.wav")
        save_pcm_wav(wav_path, audio_bytes)
        return wav_path

    filename = os.path.join(output_dir, f"{basename}.{response_format}")
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


async def send_utterance(ws, text: str, simulate_stt: bool, stt_delay: float) -> None:
    """Send one utterance's text, then input.done to flush it."""
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

    await ws.send(json.dumps({"type": "input.done"}))
    print("Sent input.done")


async def receive_utterance(
    ws,
    output_dir: str,
    response_format: str,
    word_timestamps: bool,
    save_chunks: bool,
) -> None:
    """Consume frames until session.done marks the flushed utterance complete."""
    current_label = "utterance 0"
    current_sentence_text = ""
    current_chunks: list[bytes] = []
    current_timestamps: list[dict] = []
    # Distinguish `null` (aligner failed) from `[]` (silence / no tokens):
    # the sentence is a failure only if no frame ever carried timestamps.
    alignment_frame_seen = False

    while True:
        message = await ws.recv()

        if isinstance(message, bytes):
            current_chunks.append(message)
            print(f"  Received audio chunk for {current_label}: {len(message)} bytes")
        else:
            # JSON frame
            msg = json.loads(message)
            msg_type = msg.get("type")

            if msg_type == "audio.start":
                current_label = f"utterance {msg['utterance_index']}"
                current_sentence_text = msg.get("sentence_text", "")
                current_chunks = []
                current_timestamps = []
                alignment_frame_seen = False
                print(f"  [{current_label}] Generating: {current_sentence_text!r}")
            elif msg_type == "audio.chunk":
                audio = base64.b64decode(msg["audio_b64"])
                current_chunks.append(audio)
                chunk_timestamps = msg.get("timestamps")
                # Sentence-level alignment sends a trailing timestamp-only
                # frame with empty audio; don't write an empty chunk file.
                if save_chunks and audio:
                    chunk_base = os.path.join(
                        output_dir,
                        f"{frame_basename(msg)}_chunk_{msg['chunk_id']:03d}",
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
                                "utterance_index": msg["utterance_index"],
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
                    print(f"  [{current_label}] chunk {msg['chunk_id']}: {len(audio)} bytes, timestamps unavailable")
                else:
                    alignment_frame_seen = True
                    current_timestamps.extend(chunk_timestamps)
                    print(
                        f"  [{current_label}] chunk {msg['chunk_id']}: "
                        f"{len(audio)} bytes, {len(chunk_timestamps)} timestamp(s)"
                    )
                if save_chunks and audio:
                    print(f"    saved chunk -> {chunk_wav}")
            elif msg_type == "audio.done":
                filename = save_audio_file(
                    output_dir,
                    frame_basename(msg),
                    response_format,
                    current_chunks,
                )
                if word_timestamps:
                    # `null` on failure, else the accumulated list (`[]` for silence).
                    sentence_timestamps = current_timestamps if alignment_frame_seen else None
                    ts_filename = os.path.join(
                        output_dir,
                        f"{frame_basename(msg)}_timestamps.json",
                    )
                    with open(ts_filename, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "utterance_index": msg["utterance_index"],
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
                    f"  [{current_label}] Done"
                    f" bytes={msg.get('total_bytes', len(b''.join(current_chunks)))}"
                    f" error={msg.get('error', False)}"
                    f" -> {filename}"
                )
                if word_timestamps:
                    print(f"  [{current_label}] Timestamps -> {ts_filename}")
                current_chunks = []
                current_timestamps = []
                alignment_frame_seen = False
            elif msg_type == "session.done":
                print(f"\nUtterance {msg['utterance_index']} complete: {msg['total_sentences']} sentence(s) generated")
                return
            elif msg_type == "error":
                print(f"  ERROR: {msg['message']}")
            else:
                print(f"  Unknown message: {msg}")


async def stream_tts(
    url: str,
    texts: list[str],
    config: dict,
    output_dir: str,
    simulate_stt: bool = False,
    stt_delay: float = 0.1,
    save_chunks: bool = False,
) -> None:
    """Connect to the streaming TTS endpoint and process audio responses.

    Every text is synthesized over the same connection: input.done flushes an
    utterance without closing, so only the first one pays the handshake.
    """
    os.makedirs(output_dir, exist_ok=True)
    response_format = config.get("response_format", "wav")
    word_timestamps = bool(config.get("word_timestamps", False))

    async with websockets.connect(url) as ws:
        # 1. Send session config; it stays in effect for every utterance below.
        config_msg = {"type": "session.config", **config}
        await ws.send(json.dumps(config_msg))
        print(f"Sent session config: {config}")

        for text in texts:
            # 2. Send text (either all at once or word-by-word), then flush.
            sender_task = asyncio.create_task(send_utterance(ws, text, simulate_stt, stt_delay))
            try:
                await receive_utterance(
                    ws,
                    output_dir=output_dir,
                    response_format=response_format,
                    word_timestamps=word_timestamps,
                    save_chunks=save_chunks,
                )
            finally:
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass  # Task cancellation is expected during shutdown

        # 3. Release the connection once there is nothing left to say.
        await ws.send(json.dumps({"type": "session.close"}))
        print("Sent session.close")

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
        action="append",
        help="Text to synthesize; repeat to send several utterances over one connection",
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
            texts=args.text,
            config=config,
            output_dir=args.output_dir,
            simulate_stt=args.simulate_stt,
            stt_delay=args.stt_delay,
            save_chunks=args.save_chunks,
        )
    )


if __name__ == "__main__":
    main()
