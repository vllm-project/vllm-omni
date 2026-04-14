# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for text-to-audio generation using LongCat-AudioDiT.

Usage:
    python end2end.py --prompt "A calm ocean wave ambience"
    python end2end.py --audio-length 5.0 --num-inference-steps 16
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audio with LongCat-AudioDiT.")
    parser.add_argument(
        "--model",
        default="meituan-longcat/LongCat-AudioDiT-1B",
        help="LongCat-AudioDiT model name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="A calm ocean wave ambience with soft wind in the background.",
        help="Text prompt for audio generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="distorted, clipping, noisy",
        help="Negative prompt for classifier-free guidance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic results.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--audio-start",
        type=float,
        default=0.0,
        help="Audio start time in seconds.",
    )
    parser.add_argument(
        "--audio-length",
        type=float,
        default=5.0,
        help="Audio length in seconds.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=16,
        help="Number of denoising steps for the diffusion sampler.",
    )
    parser.add_argument(
        "--num-waveforms",
        type=int,
        default=1,
        help="Number of audio waveforms to generate for the given prompt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="longcat_audio_dit_output.wav",
        help="Path to save the generated audio (WAV format).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Sample rate for output audio (LongCat-AudioDiT uses 24000 Hz).",
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    return parser.parse_args()


def save_audio(audio_data: np.ndarray, output_path: str, sample_rate: int = 24000):
    """Save audio data to a WAV file."""
    try:
        import soundfile as sf

        sf.write(output_path, audio_data, sample_rate)
    except ImportError:
        try:
            import scipy.io.wavfile as wav

            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
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
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    print(f"\n{'=' * 60}")
    print("LongCat-AudioDiT - Text-to-Audio Generation")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Negative prompt: {args.negative_prompt}")
    print(f"  Audio length: {args.audio_length}s")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    omni = Omni(
        model=args.model,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
    )

    audio_end_in_s = args.audio_start + args.audio_length
    generation_start = time.perf_counter()

    outputs = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
        },
        OmniDiffusionSamplingParams(
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=args.num_waveforms,
            extra_args={
                "audio_start_in_s": args.audio_start,
                "audio_end_in_s": audio_end_in_s,
            },
        ),
    )

    generation_time = time.perf_counter() - generation_start
    print(f"Total generation time: {generation_time:.2f} seconds")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".wav"
    stem = output_path.stem or "longcat_audio_dit_output"

    if not outputs:
        raise ValueError("No output generated from omni.generate()")

    output = outputs[0]
    if not hasattr(output, "request_output") or not output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")
    request_output = output.request_output
    if not hasattr(request_output, "multimodal_output"):
        raise ValueError("No multimodal_output found in request_output")

    audio = request_output.multimodal_output.get("audio")
    if audio is None:
        raise ValueError("No audio output found in request_output")

    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().float().numpy()

    if audio.ndim == 3:
        if args.num_waveforms <= 1:
            audio_data = audio[0].T
            save_audio(audio_data, str(output_path), args.sample_rate)
            print(f"Saved generated audio to {output_path}")
        else:
            for idx in range(audio.shape[0]):
                audio_data = audio[idx].T
                save_path = output_path.parent / f"{stem}_{idx}{suffix}"
                save_audio(audio_data, str(save_path), args.sample_rate)
                print(f"Saved generated audio to {save_path}")
    elif audio.ndim == 2:
        audio_data = audio.T
        save_audio(audio_data, str(output_path), args.sample_rate)
        print(f"Saved generated audio to {output_path}")
    else:
        save_audio(audio, str(output_path), args.sample_rate)
        print(f"Saved generated audio to {output_path}")

    print(f"\nGenerated {args.audio_length}s of audio at {args.sample_rate} Hz")


if __name__ == "__main__":
    main()
