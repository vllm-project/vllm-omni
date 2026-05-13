#!/usr/bin/env python3
"""
Stable Audio OpenAI-compatible text-to-audio client.

Usage:
    python openai_chat_client.py --prompt "The sound of a dog barking" --output output.wav
    python openai_chat_client.py --prompt "A piano playing" --steps 50 --seed 42
"""

import argparse
import base64
from pathlib import Path

import requests


def generate_audio(
    prompt: str,
    server_url: str = "http://localhost:8091",
    steps: int | None = None,
    guidance_scale: float | None = None,
    audio_start_in_s: float | None = None,
    audio_end_in_s: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> bytes | None:
    """Generate audio using the chat completions API.

    Args:
        prompt: Text description of the audio to generate
        server_url: Server URL
        steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        audio_start_in_s: Start time of audio segment in seconds
        audio_end_in_s: End time of audio segment in seconds
        seed: Random seed
        negative_prompt: Negative prompt

    Returns:
        Audio bytes (WAV format) or None if failed
    """
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": prompt}],
    }

    extra_body: dict[str, object] = {}
    if steps is not None:
        extra_body["num_inference_steps"] = steps
    if guidance_scale is not None:
        extra_body["guidance_scale"] = guidance_scale
    if audio_start_in_s is not None:
        extra_body["audio_start_in_s"] = audio_start_in_s
    if audio_end_in_s is not None:
        extra_body["audio_end_in_s"] = audio_end_in_s
    if seed is not None:
        extra_body["seed"] = seed
    if negative_prompt:
        extra_body["negative_prompt"] = negative_prompt

    if extra_body:
        payload["extra_body"] = extra_body

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            content = choices[0].get("message", {}).get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "audio_url":
                        audio_url = item.get("audio_url", {}).get("url", "")
                        if audio_url.startswith("data:audio/wav;base64,"):
                            b64_data = audio_url.split(",", 1)[1]
                            return base64.b64decode(b64_data)

        print(f"Unexpected response format: {data}")
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Stable Audio text-to-audio client")
    parser.add_argument("--prompt", "-p", default="The sound of a dog barking", help="Text prompt")
    parser.add_argument("--output", "-o", default="stable_audio_output.wav", help="Output file")
    parser.add_argument("--server", "-s", default="http://localhost:8091", help="Server URL")
    parser.add_argument("--steps", type=int, default=100, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--audio-start", type=float, default=0.0, help="Audio start time in seconds")
    parser.add_argument("--audio-end", type=float, default=10.0, help="Audio end time in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--negative", help="Negative prompt")

    args = parser.parse_args()

    print(f"Generating audio for: {args.prompt}")

    audio_bytes = generate_audio(
        prompt=args.prompt,
        server_url=args.server,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        audio_start_in_s=args.audio_start,
        audio_end_in_s=args.audio_end,
        seed=args.seed,
        negative_prompt=args.negative,
    )

    if audio_bytes:
        output_path = Path(args.output)
        output_path.write_bytes(audio_bytes)
        print(f"Audio saved to: {output_path}")
        print(f"Size: {len(audio_bytes) / 1024:.1f} KB")
    else:
        print("Failed to generate audio")
        exit(1)


if __name__ == "__main__":
    main()
