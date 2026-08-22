#!/usr/bin/env python3
"""
OpenAI-compatible client for MOSS-SoundEffect-v2.0 via /v1/audio/generate.

This script demonstrates how to use the OpenAI-compatible audio API
to generate audio from text using Moss-SoundEffectv2 models.

Examples:
    # Simple generation
    python openai_client.py --text "The sound of a cat purring"

    # With custom duration
    python openai_client.py --text "A dog barking" --audio_length 5.0

    # With all parameters
    python openai_client.py --text "Thunder and rain" \
        --audio_length 15.0 \
        --negative_prompt "Low quality" \
        --guidance_scale 5.0 \
        --num_inference_steps 10 \
        --sigma_shift 5.0 \
        --seed 42 \
        --output thunder.wav
"""

import argparse
import sys

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio with MOSS-SoundEffect-v2.0")
    parser.add_argument(
        "--api_url",
        default="http://localhost:8091/v1/audio/generate",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--text",
        default="The sound of a cat purring",
        help="Text prompt for audio generation",
    )
    parser.add_argument(
        "--audio_length",
        type=float,
        default=10.0,
        help="Returned audio duration in seconds (audio_start + audio_length must not exceed 30 seconds)",
    )
    parser.add_argument(
        "--audio_start",
        type=float,
        default=0.0,
        help="Audio start time in seconds",
    )
    parser.add_argument(
        "--negative_prompt",
        default=None,
        help="Negative prompt for classifier-free guidance",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for diffusion (higher = more adherence to prompt)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="Number of inference steps (higher = better quality, slower)",
    )
    parser.add_argument(
        "--sigma_shift",
        type=float,
        default=5.0,
        help="Flow-matching scheduler shift",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        default="moss_soundeffect_output.wav",
        help="Output file path",
    )
    parser.add_argument(
        "--response_format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio output format",
    )
    return parser.parse_args()


def generate_audio(args):
    """Generate audio using the API."""

    # Build request payload
    payload = {
        "input": args.text,
        "audio_length": args.audio_length,
        "audio_start": args.audio_start,
        "response_format": args.response_format,
    }

    # Add optional parameters
    if args.negative_prompt:
        payload["negative_prompt"] = args.negative_prompt
    if args.guidance_scale is not None:
        payload["guidance_scale"] = args.guidance_scale
    if args.num_inference_steps is not None:
        payload["num_inference_steps"] = args.num_inference_steps
    if args.sigma_shift is not None:
        payload["extra_params"] = {"sigma_shift": args.sigma_shift}
    if args.seed is not None:
        payload["seed"] = args.seed

    print(f"\n{'=' * 60}")
    print("MOSS-SoundEffect-v2.0 - Text-to-Audio Generation")
    print(f"{'=' * 60}")
    print(f"API URL: {args.api_url}")
    print(f"Prompt: {args.text}")
    print(f"Audio length: {args.audio_length}s")
    print(f"Negative prompt: {args.negative_prompt}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Sigma shift: {args.sigma_shift}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print(f"{'=' * 60}\n")

    try:
        # Make the API request
        print("Generating audio...")
        response = requests.post(
            args.api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,  # 5 minute timeout for long generations
        )

        # Check for errors
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False

        # Save the audio
        with open(args.output, "wb") as f:
            f.write(response.content)

        print(f"Audio saved to {args.output}")
        print(f"  File size: {len(response.content) / 1024:.1f} KB")
        return True

    except requests.exceptions.Timeout:
        print("Error: Request timed out. Try reducing inference steps or audio length.")
        return False
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {args.api_url}")
        print("Make sure the server is running.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    args = parse_args()
    success = generate_audio(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
