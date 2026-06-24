# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for text-to-music generation using ACE-Step 1.5.

Usage:
    python text_to_music.py --prompt "An upbeat jazz piano piece"
    python text_to_music.py \\
        --prompt "Soft piano ballad with strings" \\
        --lyrics "[verse]\\nQuiet evenings\\n[chorus]\\nFading light" \\
        --audio-duration 30.0

Until a diffusers-format ACE-Step checkpoint is hosted publicly, run
PR #13095's conversion script against
https://huggingface.co/ACE-Step/Ace-Step1.5 and point ``--model`` at the
converted directory.

"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate music with ACE-Step 1.5.")
    parser.add_argument(
        "--model",
        default="ACE-Step/Ace-Step1.5",
        help="ACE-Step diffusers-format checkpoint path (see PR #13095 conversion script).",
    )
    parser.add_argument(
        "--prompt",
        default="An upbeat jazz piano piece with a walking bass line.",
        help="Text caption describing the music style / instruments.",
    )
    parser.add_argument(
        "--lyrics",
        default="",
        help='Lyrics text. Supports structured tags like "[verse]" / "[chorus]".',
    )
    parser.add_argument(
        "--vocal-language",
        default="en",
        help="Language code for lyrics (e.g. 'en', 'zh', 'ja').",
    )
    parser.add_argument(
        "--audio-duration",
        type=float,
        default=30.0,
        help="Duration of generated music in seconds.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=8,
        help="Number of denoising steps. Turbo model is designed for 8.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="CFG scale. Turbo distills guidance into the weights — keep at 1.0.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=3.0,
        choices=[1.0, 2.0, 3.0],
        help="Flow-matching timestep shift (turbo schedules ship for 1, 2, 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic results.",
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
        default="ace_step_output.wav",
        help="Path to save the generated audio (WAV format).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Sample rate for output audio (ACE-Step uses 48000 Hz).",
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable ModelWise CPU offloading to save GPU memory.",
    )
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable LayerWise CPU offloading to save GPU memory.",
    )
    parser.add_argument(
        "--use-hsdp",
        action="store_true",
        help="Enable HSDP for ACE-Step DiT weight sharding.",
    )
    parser.add_argument(
        "--hsdp-shard-size",
        type=int,
        default=1,
        help="Number of GPUs to shard ACE-Step DiT weights across when HSDP is enabled.",
    )
    parser.add_argument(
        "--hsdp-replicate-size",
        type=int,
        default=1,
        help="Number of HSDP replica groups. Default 1 means pure sharding.",
    )
    return parser.parse_args()


def save_audio(audio_data: np.ndarray, output_path: str, sample_rate: int = 48000):
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
    print("ACE-Step 1.5 - Text-to-Music Generation")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Prompt: {args.prompt}")
    if args.lyrics:
        first_line = args.lyrics.splitlines()[0] if args.lyrics else ""
        print(f"  Lyrics (first line): {first_line}")
    print(f"  Vocal language: {args.vocal_language}")
    print(f"  Audio duration: {args.audio_duration}s")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Shift: {args.shift}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  ModelWise Offload: {'Enabled' if args.enable_cpu_offload else 'None'}")
    print(f"  LayerWise Offload: {'Enabled' if args.enable_layerwise_offload else 'None'}")
    if args.use_hsdp:
        print(f"  HSDP: enabled (shard_size={args.hsdp_shard_size}, replicate_size={args.hsdp_replicate_size})")
    else:
        print("  HSDP: disabled")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    parallel_config = DiffusionParallelConfig(
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )

    omni = Omni(
        model=args.model,
        parallel_config=parallel_config,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
    )

    generation_start = time.perf_counter()

    outputs = omni.generate(
        {
            "prompt": args.prompt,
        },
        OmniDiffusionSamplingParams(
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=args.num_waveforms,
            extra_args={
                "lyrics": args.lyrics,
                "vocal_language": args.vocal_language,
                "audio_duration": args.audio_duration,
                "shift": args.shift,
            },
        ),
    )

    generation_time = time.perf_counter() - generation_start
    print(f"Total generation time: {generation_time:.2f} seconds")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".wav"
    stem = output_path.stem or "ace_step_output"

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

    # Audio shape is typically [batch, channels, samples] or [channels, samples].
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

    print(f"\nGenerated {args.audio_duration}s of music at {args.sample_rate} Hz")


if __name__ == "__main__":
    main()
