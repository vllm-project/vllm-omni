# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for text-to-speech generation using GLM-TTS.

This script demonstrates how to generate speech from text using
the GLM-TTS model with vLLM-Omni.

GLM-TTS is a two-stage model:
- Stage 0: LLM (Llama-based) generates speech tokens from text
- Stage 1: DiT (Flow matching) generates mel-spectrogram from speech tokens

Usage:
    python glm_tts_example.py --text "Hello, this is a test of GLM-TTS."
    python glm_tts_example.py --text "Welcome to the future of text to speech." --num_inference_steps 16
"""

import argparse
import os
import time

import numpy as np
import torch
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni

# Path to stage config (relative to repo root)
STAGE_CONFIG_PATH = "vllm_omni/model_executor/stage_configs/glm_tts.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate speech with GLM-TTS.")
    parser.add_argument(
        "--model",
        default="zai-org/GLM-TTS",
        help="GLM-TTS model name or local path.",
    )
    parser.add_argument(
        "--stage-config",
        default=None,
        help="Path to stage config YAML file. If not provided, uses default GLM-TTS config.",
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
        "--num_inference_steps",
        type=int,
        default=32,
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

    # Resolve stage config path
    stage_config = args.stage_config
    if stage_config is None:
        # Try to find the default config relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        stage_config = os.path.join(repo_root, STAGE_CONFIG_PATH)
        if not os.path.exists(stage_config):
            print(f"Error: Stage config not found at {stage_config}")
            print("Please provide --stage-config path or run from repository root.")
            return

    print(f"Loading GLM-TTS model: {args.model}")
    print(f"Using stage config: {stage_config}")
    start_time = time.time()

    # Initialize the Omni engine with GLM-TTS stage config
    omni = Omni(
        model=args.model,
        stage_config=stage_config,
    )

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")

    print(f"\nGenerating speech for: '{args.text}'")
    print(f"Parameters:")
    print(f"  - Inference steps: {args.num_inference_steps}")
    print(f"  - Seed: {args.seed}")

    gen_start = time.time()

    # Build prompt for GLM-TTS LLM stage
    # The LLM will generate speech tokens which are then processed by the DiT stage
    prompt = args.text

    # Sampling parameters for LLM stage (speech token generation)
    llm_sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.8,
        top_k=40,
        max_tokens=2048,
        seed=args.seed,
        detokenize=False,
        repetition_penalty=1.05,
        stop_token_ids=[151330],  # GLM_TTS_EOA_TOKEN_ID
    )

    # Sampling parameters for DiT stage (audio generation)
    dit_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,  # DiT doesn't use token generation
        seed=args.seed,
    )

    sampling_params_list = [llm_sampling_params, dit_sampling_params]

    # Generate audio
    outputs = list(omni.generate(
        [{"prompt": prompt}],
        sampling_params_list,
        py_generator=True,
    ))

    gen_time = time.time() - gen_start
    print(f"✓ Audio generated in {gen_time:.2f}s")

    # Extract audio from outputs
    for stage_output in outputs:
        if stage_output.final_output_type == "audio":
            for req_output in stage_output.request_output:
                if hasattr(req_output, "images") and len(req_output.images) > 0:
                    audio = req_output.images[0]

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

    total_time = time.time() - start_time
    print(f"\n✓ Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
