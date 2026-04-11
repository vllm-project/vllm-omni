# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import re
import time

from PIL import Image
from vllm.multimodal.media.audio import load_audio

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for DreamID-Omni (video + audio).")
    parser.add_argument("--model", required=True, help="DreamID ckpt root directory.")
    parser.add_argument("--model-type", default="dreamid-omni", help="Model type.")
    parser.add_argument("--prompt", default=None, help="Text prompt.")

    parser.add_argument("--image-path", type=str, nargs="+", help="list of image-path")
    parser.add_argument("--audio-path", type=str, nargs="+", help="list of audio-path")
    parser.add_argument("--prompt-file", type=str, default=None, help="Text prompt in json format.")

    parser.add_argument("--height", type=int, default=704, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num-inference-steps", type=int, default=45, help="Sampling steps.")
    parser.add_argument("--solver-name", default="unipc", help="Solver name: unipc|dpm++|euler.")
    parser.add_argument("--shift", type=float, default=5.0, help="Scheduler shift.")
    parser.add_argument("--seed", type=int, default=103, help="Random seed for reproducible generation.")
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--video-negative-prompt",
        default="jitter, bad hands, blur, distortion",
        help="Negative prompt for video.",
    )
    parser.add_argument(
        "--audio-negative-prompt",
        default="robotic, muffled, echo, distorted",
        help="Negative prompt for audio.",
    )
    parser.add_argument("--output", default="dreamid_output.mp4", help="Output video path.")
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        default=False,
        help="Enable CPU offloading for diffusion models.",
    )
    return parser.parse_args()


def load_image_and_audio(image_paths, audio_paths):
    image = []
    audio = []

    for path in image_paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            image.append(img)

    for path in audio_paths:
        audio_array, sr = load_audio(path, sr=16000)
        audio_array = audio_array[int(sr * 1) : int(sr * 3)]
        audio.append(audio_array)
    return image, audio


def main() -> None:
    args = parse_args()
    if args.prompt is None and args.prompt_file is None:
        raise ValueError("Either --prompt or --prompt-file must be provided.")

    text_prompt = args.prompt
    if args.prompt_file:
        import json

        with open(args.prompt_file) as f:
            text_prompt = json.load(f)
            text_prompt = re.sub(
                r"\[SPEAKER_TIMESTAMPS_START\].*?\[SPEAKER_TIMESTAMPS_END\]", "", text_prompt, flags=re.DOTALL
            ).strip()
            text_prompt = re.sub(
                r"\[AUDIO_DESCRIPTION_START].*?\[AUDIO_DESCRIPTION_END]", "", text_prompt, flags=re.DOTALL
            ).strip()
            text_prompt = re.sub(r"\[[A-Z_]+\]", "", text_prompt)
            text_prompt = re.sub(r"\n\s*\n", "\n", text_prompt).strip()

    image, audio = load_image_and_audio(args.image_path, args.audio_path)

    prompt = {
        "prompt": text_prompt,
        "video_negative_prompt": args.video_negative_prompt,
        "audio_negative_prompt": args.audio_negative_prompt,
        "multi_modal_data": {"image": image, "audio": audio},
    }

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        extra_args={
            "solver_name": args.solver_name,
            "shift": args.shift,
        },
    )

    parallel_config = DiffusionParallelConfig(
        cfg_parallel_size=args.cfg_parallel_size,
    )

    omni = Omni(
        model=args.model,
        parallel_config=parallel_config,
        model_type=args.model_type,
        enable_cpu_offload=args.enable_cpu_offload,
    )
    start = time.perf_counter()
    outputs = omni.generate(prompt, sampling_params)
    elapsed = time.perf_counter() - start

    if not outputs:
        raise RuntimeError("No output returned from DreamID-Omni.")
    output = outputs[0].request_output
    generated_video = output.images[0][0]
    generated_audio = output.images[0][1]
    try:
        from dreamid_omni.utils.io_utils import save_video
    except Exception as e:
        raise RuntimeError(f"Failed to extract video and audio from DreamID-Omni output. Error: {e}")
    output_path = args.output
    save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
    print(f"Saved generated video to {output_path}")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
