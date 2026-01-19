# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for text-to-speech generation using GLM-TTS.

This script demonstrates how to generate speech from text using
the GLM-TTS model with vLLM-Omni.

Usage:
    python glm_tts_example.py --text "Hello, this is a test of GLM-TTS."
    python glm_tts_example.py --text "Welcome to the future of text to speech." --duration 5.0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate speech with GLM-TTS.")
    parser.add_argument(
        "--model",
        default="zai-org/GLM-TTS",
        help="GLM-TTS model name or local path.",
    )
    parser.add_argument(
        "--text",
        default="Hello, this is a test of GLM-TTS text to speech synthesis.",
        help="Text to convert to speech.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic results.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Audio duration in seconds.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=8,
        help="Number of denoising steps for the flow matching sampler.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="glm_tts_output.wav",
        help="Path to save the generated audio (WAV format).",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Sample rate for output audio.",
    )
    return parser.parse_args()


def save_audio(audio_data: np.ndarray, output_path: str, sample_rate: int = 24000):
    """Save audio data to a WAV file."""
    try:
        import soundfile as sf

        sf.write(output_path, audio_data, sample_rate)
        print(f"✓ Audio saved to: {output_path}")
    except ImportError:
        print("Warning: soundfile not installed. Install with: pip install soundfile")
        # Fallback to scipy
        try:
            from scipy.io import wavfile

            wavfile.write(output_path, sample_rate, audio_data)
            print(f"✓ Audio saved to: {output_path}")
        except ImportError:
            print("Error: Neither soundfile nor scipy is installed.")
            print("Install with: pip install soundfile")


def main():
    args = parse_args()

    print(f"Loading GLM-TTS model: {args.model}")
    start_time = time.time()

    # Initialize the Omni engine with GLM-TTS
    omni = Omni(model=args.model)

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")

    print(f"\nGenerating speech for: '{args.text}'")
    print(f"Parameters:")
    print(f"  - Duration: {args.duration}s")
    print(f"  - Inference steps: {args.num_inference_steps}")
    print(f"  - Guidance scale: {args.guidance_scale}")
    print(f"  - Seed: {args.seed}")

    gen_start = time.time()

    # Generate speech tokens (mock for now - in production, LLM generates these)
    # TODO: Replace with actual LLM output when two-stage pipeline is ready
    speech_tokens = torch.randint(0, 10000, (1, 100), dtype=torch.long)
    speaker_embedding = torch.randn(1, 192)

    # Generate audio
    outputs = omni.generate(
        args.text,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        num_outputs_per_prompt=1,
        extra={
            "audio_duration_s": args.duration,
            "speech_tokens": speech_tokens,
            "speaker_embedding": speaker_embedding,
        },
    )

    gen_time = time.time() - gen_start
    print(f"✓ Audio generated in {gen_time:.2f}s")

    # Extract audio from outputs
    if outputs and len(outputs) > 0:
        first_output = outputs[0]
        if hasattr(first_output, "request_output") and first_output.request_output:
            req_out = first_output.request_output[0]
            if hasattr(req_out, "images") and len(req_out.images) > 0:
                audio = req_out.images[0]

                # Save audio
                if isinstance(audio, np.ndarray):
                    # Ensure audio is in the correct format for saving
                    if audio.ndim == 3:  # (batch, channels, samples)
                        audio = audio[0]  # Take first batch
                    if audio.ndim == 2:  # (channels, samples)
                        audio = audio.T  # Transpose to (samples, channels)

                    save_audio(audio, args.output, args.sample_rate)

                    # Print audio stats
                    duration_actual = audio.shape[0] / args.sample_rate
                    print(f"\nAudio Statistics:")
                    print(f"  - Shape: {audio.shape}")
                    print(f"  - Duration: {duration_actual:.2f}s")
                    print(f"  - Sample rate: {args.sample_rate} Hz")
                    print(f"  - Min/Max: {audio.min():.3f} / {audio.max():.3f}")
                else:
                    print(f"Error: Unexpected audio format: {type(audio)}")
            else:
                print("Error: No audio data in output")
        else:
            print("Error: Invalid output structure")
    else:
        print("Error: No outputs generated")

    total_time = time.time() - start_time
    print(f"\n✓ Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
