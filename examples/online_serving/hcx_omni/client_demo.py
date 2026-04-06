# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperCLOVAX-SEED-Omni-8B client demo.

Demonstrates Speech-to-Speech and Text-to-Vision via the OpenAI-compatible
HTTP API provided by vLLM-Omni.

Usage:
    # Start the server first (see run_server.sh), then:
    python client_demo.py --base-url http://localhost:8000/v1

    # With a local audio file:
    python client_demo.py --audio-file path/to/speech.wav

    # Text-to-Vision only:
    python client_demo.py --mode t2v --prompt "고양이 그림을 그려줘"
"""

import argparse
import base64
import io
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)


def encode_audio_file(path: str) -> str:
    """Base64-encode a WAV/MP3 file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def encode_audio_array(array, sample_rate: int = 16000) -> str:
    """Base64-encode a numpy audio array as WAV."""
    import numpy as np
    import scipy.io.wavfile as wav

    if not isinstance(array, np.ndarray):
        array = np.array(array)
    buf = io.BytesIO()
    wav.write(buf, sample_rate, (array * 32767).astype(np.int16))
    return base64.b64encode(buf.getvalue()).decode()


def speech_to_speech(client: OpenAI, audio_b64: str, prompt: str = "이 오디오에 무슨 내용이 있나요?"):
    """Send audio → receive text + audio."""
    print(f"\n[Speech-to-Speech] prompt: {prompt!r}")
    response = client.chat.completions.create(
        model="naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B",
        modalities=["text", "audio"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    choice = response.choices[0]
    print(f"Text response: {choice.message.content}")
    if hasattr(choice.message, "audio") and choice.message.audio:
        audio_data = base64.b64decode(choice.message.audio.data)
        out_path = Path("/tmp/hcx_omni_response.wav")
        out_path.write_bytes(audio_data)
        print(f"Audio saved to: {out_path}")
    return response


def text_to_vision(client: OpenAI, prompt: str = "귀여운 강아지 한 마리가 공원에서 뛰노는 그림을 그려줘."):
    """Send text → receive text + image."""
    print(f"\n[Text-to-Vision] prompt: {prompt!r}")
    response = client.chat.completions.create(
        model="naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B",
        modalities=["text", "image"],
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    )
    choice = response.choices[0]
    print(f"Text response: {choice.message.content}")
    if hasattr(choice.message, "image") and choice.message.image:
        img_data = base64.b64decode(choice.message.image.data)
        out_path = Path("/tmp/hcx_omni_generated.png")
        out_path.write_bytes(img_data)
        print(f"Image saved to: {out_path}")
    return response


def text_only(client: OpenAI, prompt: str = "대한민국의 수도는 어디인가요?"):
    """Pure text conversation (thinker only)."""
    print(f"\n[Text-only] prompt: {prompt!r}")
    response = client.chat.completions.create(
        model="naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B",
        modalities=["text"],
        messages=[{"role": "user", "content": prompt}],
    )
    print(f"Response: {response.choices[0].message.content}")
    return response


def main():
    parser = argparse.ArgumentParser(description="HyperCLOVAX-SEED-Omni-8B demo")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--mode",
        choices=["s2s", "t2v", "text", "all"],
        default="all",
        help="Demo mode: s2s=Speech-to-Speech, t2v=Text-to-Vision, text=Text-only",
    )
    parser.add_argument("--audio-file", default=None, help="Path to input audio file")
    parser.add_argument("--prompt", default=None, help="Text prompt override")
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)

    if args.mode in ("text", "all"):
        text_only(client, prompt=args.prompt or "대한민국의 수도는 어디인가요?")

    if args.mode in ("t2v", "all"):
        text_to_vision(client, prompt=args.prompt or "귀여운 강아지 한 마리가 공원에서 뛰노는 그림을 그려줘.")

    if args.mode in ("s2s", "all"):
        if args.audio_file:
            audio_b64 = encode_audio_file(args.audio_file)
        else:
            # Generate synthetic 1-second sine wave
            try:
                import numpy as np

                t = np.linspace(0, 1, 16000, endpoint=False)
                audio_array = np.sin(2 * np.pi * 440 * t).astype(np.float32)
                audio_b64 = encode_audio_array(audio_array)
            except ImportError:
                print("numpy not available, skipping S2S demo")
                return
        speech_to_speech(client, audio_b64, prompt=args.prompt or "이 오디오에 무슨 내용이 있나요?")


if __name__ == "__main__":
    main()
