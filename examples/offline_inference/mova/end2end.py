# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline inference example for MOVA (video + audio generation).

Usage:
    python examples/offline_inference/mova/end2end.py \
        --model /path/to/MOVA-360p \
        --prompt "a person talking and waving" \
        --ref-path reference.png \
        --output mova_output.mp4
"""

import argparse

from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Default negative prompt from upstream MOVA inference script
UPSTREAM_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for MOVA (video + audio).")
    parser.add_argument("--model", required=True, help="MOVA model checkpoint path.")
    parser.add_argument("--prompt", required=True, help="Text prompt for generation.")
    parser.add_argument("--ref-path", required=True, help="Reference image path (first frame).")
    parser.add_argument(
        "--negative-prompt",
        default=UPSTREAM_DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt (default: upstream MOVA default).",
    )
    parser.add_argument("--height", type=int, default=352, help="Video height (default: 352 for 360p).")
    parser.add_argument("--width", type=int, default=640, help="Video width (default: 640 for 360p).")
    parser.add_argument("--num-frames", type=int, default=193, help="Number of video frames (default: 193).")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Denoising steps (default: 50).")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="CFG guidance scale (default: 5.0).")
    parser.add_argument("--visual-shift", type=float, default=5.0, help="Visual sigma shift (default: 5.0).")
    parser.add_argument("--audio-shift", type=float, default=5.0, help="Audio sigma shift (default: 5.0).")
    parser.add_argument("--video-fps", type=float, default=24.0, help="Video frame rate (default: 24.0).")
    parser.add_argument("--boundary-ratio", type=float, default=0.9, help="DiT expert switch ratio (default: 0.9).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", default="mova_output.mp4", help="Output file path.")
    parser.add_argument("--dtype", type=str, default="auto", help="Model dtype (auto, bfloat16, float16, float32).")
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offload for memory-constrained environments.",
    )
    # Note: --enable-layerwise-offload is not exposed in this version.
    # Upstream MOVA's group offload is a custom implementation that needs
    # compatibility verification with vllm-omni's layerwise offload.
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load reference image
    ref_image = Image.open(args.ref_path).convert("RGB")

    # Build prompt dict
    prompt = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "multi_modal_data": {"image": [ref_image]},
    }

    # Build sampling params
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.cfg_scale,
        seed=args.seed,
        extra_args={
            "boundary_ratio": args.boundary_ratio,
            "video_fps": args.video_fps,
            "visual_shift": args.visual_shift,
            "audio_shift": args.audio_shift,
        },
    )

    # Initialize Omni engine
    omni = Omni(
        model=args.model,
        dtype=args.dtype,
        enable_cpu_offload=args.enable_cpu_offload,
    )

    # Generate
    outputs = omni.generate(prompt, sampling_params)

    # Extract output: pipeline returns (video_pil, audio_tensor) in images[0]
    output = outputs[0]
    result = output.images[0] if output.images else None
    if result is None:
        print("No output generated.")
        return

    if not (isinstance(result, tuple) and len(result) == 2):
        print(f"Unexpected output format: type={type(result)}")
        return

    video_data, audio_data = result

    # Unwrap batch dimension: postprocess_video returns [batch][frames]
    frames = video_data
    if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], list):
        frames = frames[0]

    print(f"Video: {len(frames)} frames")
    print(f"Audio: shape={getattr(audio_data, 'shape', 'N/A')}")

    # Save as MP4 with audio via ffmpeg
    _save_video_with_audio(frames, audio_data, args.output, fps=args.video_fps)
    print(f"Saved to {args.output}")


def _save_video_with_audio(
    frames: list,
    audio_data: object,
    output_path: str,
    fps: float = 24.0,
    audio_sample_rate: int = 48000,
) -> None:
    """Save video frames + audio to MP4 using ffmpeg."""
    import os
    import subprocess
    import tempfile

    import numpy as np
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save all frames as numbered images
        for i, frame in enumerate(frames):
            frame.save(os.path.join(tmpdir, f"frame_{i:06d}.png"))

        # Save audio as WAV
        audio_np = audio_data.cpu().float().numpy() if hasattr(audio_data, "cpu") else np.array(audio_data)
        if audio_np.ndim == 3:
            audio_np = audio_np.squeeze(0)
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze(0)
        wav_path = os.path.join(tmpdir, "audio.wav")
        sf.write(wav_path, audio_np, audio_sample_rate)

        # Combine with ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmpdir, "frame_%06d.png"),
            "-i",
            wav_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)


if __name__ == "__main__":
    main()
