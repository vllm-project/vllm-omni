# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NAVA audio-video generation with vLLM-Omni.")
    parser.add_argument("--model", required=True, help="Local NAVA model directory.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default=None, help="Optional first-frame image.")
    parser.add_argument("--spk-wavs", nargs="*", default=None, help="Optional speaker reference WAVs.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=37)
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", default="nava_output.pt")
    return parser.parse_args()


def _build_prompt(args: argparse.Namespace):
    multi_modal_data = {}
    if args.image:
        multi_modal_data["image"] = Image.open(args.image).convert("RGB")
    if args.spk_wavs:
        multi_modal_data["spk_wavs"] = args.spk_wavs
    if multi_modal_data:
        return {"prompt": args.prompt, "multi_modal_data": multi_modal_data}
    return args.prompt


def _build_sampling_params(args: argparse.Namespace) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        frame_rate=args.fps,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    omni = Omni(model=args.model, model_class_name="NAVAPipeline", enforce_eager=True)
    outputs = omni.generate([_build_prompt(args)], _build_sampling_params(args))
    output = outputs[0]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video": output.images,
            "multimodal_output": output.multimodal_output,
        },
        args.output,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
