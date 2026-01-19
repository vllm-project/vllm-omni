# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for text-to-audio generation using Stable Audio Open.

This script demonstrates how to generate audio from text prompts using
the Stable Audio Open model with vLLM-Omni.

Usage:
    python text_to_audio.py --prompt "The sound of a dog barking"
    python text_to_audio.py --prompt "A piano playing a gentle melody" --audio_length 10.0
    python text_to_audio.py --prompt "Thunder and rain sounds" --negative_prompt "Low quality"
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audio with Stable Audio Open.")
    parser.add_argument(
        "--model",
        default="stabilityai/stable-audio-open-1.0",
        help="Stable Audio model name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="The sound of a hammer hitting a wooden surface.",
        help="Text prompt for audio generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        default="Low quality.",
        help="Negative prompt for classifier-free guidance.",
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
        default=7.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--audio_start",
        type=float,
        default=0.0,
        help="Audio start time in seconds.",
    )
    parser.add_argument(
        "--audio_length",
        type=float,
        default=10.0,
        help="Audio length in seconds (max ~47s for stable-audio-open-1.0).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="Number of denoising steps for the diffusion sampler.",
    )
    parser.add_argument(
        "--num_waveforms",
        type=int,
        default=1,
        help="Number of audio waveforms to generate for the given prompt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stable_audio_output.wav",
        help="Path to save the generated audio (WAV format).",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Sample rate for output audio (Stable Audio uses 44100 Hz).",
    )
    return parser.parse_args()


def save_audio(audio_data: np.ndarray, output_path: str, sample_rate: int = 44100):
    """Save audio data to a WAV file."""
    try:
        import soundfile as sf

        sf.write(output_path, audio_data, sample_rate)
    except ImportError:
        try:
            import scipy.io.wavfile as wav

            # Ensure audio is in the correct format for scipy
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                # Normalize to int16 range
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)
            wav.write(output_path, sample_rate, audio_data)
        except ImportError:
            raise ImportError(
                "Either 'soundfile' or 'scipy' is required to save audio files. "
                "Install with: pip install soundfile or pip install scipy"
            )


def main():
    args = parse_args()
    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"\n{'=' * 60}")
    print("Stable Audio Open - Text-to-Audio Generation")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Negative prompt: {args.negative_prompt}")
    print(f"  Audio length: {args.audio_length}s")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    # Initialize Omni with Stable Audio model
    omni = Omni(model=args.model)

    # Calculate audio end time
    audio_end_in_s = args.audio_start + args.audio_length

    # Time profiling for generation
    generation_start = time.perf_counter()

    # Generate audio
    outputs = omni.generate(
        args.prompt,
        negative_prompt=args.negative_prompt,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_waveforms,
        extra={
            "audio_start_in_s": args.audio_start,
            "audio_end_in_s": audio_end_in_s,
        },
    )

    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    print(f"Total generation time: {generation_time:.2f} seconds")

    # Process and save audio
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".wav"
    stem = output_path.stem or "stable_audio_output"

    # Extract audio from omni.generate() outputs
    if not outputs:
        raise ValueError("No output generated from omni.generate()")

    output = outputs[0]
    if not hasattr(output, "request_output") or not output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")
    request_output = output.request_output[0]
    if not hasattr(request_output, "multimodal_output"):
        raise ValueError("No multimodal_output found in request_output")

    audio = request_output.multimodal_output.get("audio")
    if audio is None:
        raise ValueError("No audio output found in request_output")

    # Handle different output formats
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().float().numpy()

    # Audio shape is typically [batch, channels, samples] or [channels, samples]
    if audio.ndim == 3:
        # [batch, channels, samples]
        if args.num_waveforms <= 1:
            audio_data = audio[0].T  # [samples, channels]
            save_audio(audio_data, str(output_path), args.sample_rate)
            print(f"Saved generated audio to {output_path}")
        else:
            for idx in range(audio.shape[0]):
                audio_data = audio[idx].T  # [samples, channels]
                save_path = output_path.parent / f"{stem}_{idx}{suffix}"
                save_audio(audio_data, str(save_path), args.sample_rate)
                print(f"Saved generated audio to {save_path}")
    elif audio.ndim == 2:
        # [channels, samples]
        audio_data = audio.T  # [samples, channels]
        save_audio(audio_data, str(output_path), args.sample_rate)
        print(f"Saved generated audio to {output_path}")
    else:
        # [samples] - mono audio
        save_audio(audio, str(output_path), args.sample_rate)
        print(f"Saved generated audio to {output_path}")

    print(f"\nGenerated {args.audio_length}s of audio at {args.sample_rate} Hz")


if __name__ == "__main__":
    main()
