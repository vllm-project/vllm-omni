"""Client for FunCineForge movie dubbing & TTS via /v1/audio/speech endpoint.

FunCineForge extends standard TTS with optional face embedding, dialogue
metadata, and speech type tags for cinematic dubbing scenarios.

Examples:
    # Basic voice cloning (ref_audio + ref_text required)
    python speech_client.py --text "Hello, how are you?" \\
        --ref-audio ref.wav \\
        --ref-text "A warm male narrator voice."

    # Voice cloning with a URL reference
    python speech_client.py --text "Hello world" \\
        --ref-audio https://example.com/ref.wav \\
        --ref-text "A warm male narrator voice."

    # Streaming PCM output
    python speech_client.py --text "Hello world" \\
        --ref-audio ref.wav --ref-text "A narrator." \\
        --stream --output output.pcm

    # With face embedding and dialogue metadata (cinematic dubbing)
    python speech_client.py --text "The door creaked open." \\
        --ref-audio ref.wav --ref-text "A narrator." \\
        --face-path faces.npz --speech-type "对话" --speech-len 200 \\
        --dialogue-json '[{"start":0,"duration":3,"spk":1,"gender":"男","age":"中年"}]'
"""

import argparse
import base64
import json
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "FunAudioLLM/Fun-CineForge"
DEFAULT_REF_AUDIO = "https://raw.githubusercontent.com/FunAudioLLM/FunCineForge/main/exps/data/ref.wav"
DEFAULT_REF_TEXT = (
    "A single middle-aged male speaker describes a business or "
    "construction requirement with a practical and matter-of-fact tone."
)


def encode_audio_to_base64(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }
    mime_type = mime_map.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def run_tts(args) -> None:
    ref_audio = args.ref_audio
    if ref_audio and not ref_audio.startswith(("http://", "https://", "data:")):
        ref_audio = encode_audio_to_base64(ref_audio)

    payload = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
        "ref_audio": ref_audio,
        "ref_text": args.ref_text,
    }

    if args.face_path:
        payload["face_path"] = args.face_path
    if args.speech_type:
        payload["speech_type"] = args.speech_type
    if args.speech_len is not None:
        payload["speech_len"] = args.speech_len
    if args.dialogue_json:
        payload["dialogue"] = json.loads(args.dialogue_json)

    if args.stream:
        payload["stream"] = True
        payload["response_format"] = "pcm"

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print(f"Ref audio: {args.ref_audio}")
    if args.face_path:
        print(f"Face embedding: {args.face_path}")
    if args.speech_type:
        print(f"Speech type: {args.speech_type}")
    print("Generating audio...")

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    if args.stream:
        output_path = args.output or "funcineforge_output.pcm"
        with httpx.Client(timeout=600.0) as client:
            with client.stream("POST", api_url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    print(f"Error: {resp.status_code}")
                    print(resp.read().decode())
                    return
                total_bytes = 0
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                        total_bytes += len(chunk)
                print(f"Streamed {total_bytes} bytes to: {output_path}")
    else:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(api_url, json=payload, headers=headers)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return

        output_path = args.output or "funcineforge_output.wav"
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FunCineForge movie dubbing & TTS client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--text", default="Hello, how are you?", help="Text to synthesize")
    parser.add_argument(
        "--ref-audio",
        default=DEFAULT_REF_AUDIO,
        help="Reference audio (local path or URL) for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        default=DEFAULT_REF_TEXT,
        help="Voice clue/description for the reference audio",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming (PCM output)",
    )
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio format (default: wav)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument(
        "--face-path",
        default=None,
        help="Path to face embedding npz file (cinematic dubbing)",
    )
    parser.add_argument(
        "--speech-type",
        default=None,
        choices=["旁白", "独白", "对话", "多人"],
        help="Speech style type tag",
    )
    parser.add_argument(
        "--speech-len",
        type=int,
        default=None,
        help="Target speech sequence length",
    )
    parser.add_argument(
        "--dialogue-json",
        default=None,
        help='Dialogue metadata as JSON array, e.g. \'[{"start":0,"duration":3,"spk":1,"gender":"男","age":"中年"}]\'',
    )
    args = parser.parse_args()
    run_tts(args)


if __name__ == "__main__":
    main()
