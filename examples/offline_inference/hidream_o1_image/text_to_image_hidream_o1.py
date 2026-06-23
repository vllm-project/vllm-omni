# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiDream-O1-Image text-to-image offline inference (Phase 1: text-to-image
only -- no reference images / editing / personalization yet).

This is a focused script mirroring the reference repo's `inference.py`
``--model_type full|dev`` convenience (auto-selects steps/guidance/shift per
variant) rather than the fully generic `text_to_image.py`, since most of
that script's flags (cache backend, TP/SP/CFG-parallel, quantization) are
not yet supported for this model -- see the project roadmap for when they
land.
"""

import argparse
from pathlib import Path

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# (num_inference_steps, guidance_scale, shift) presets, matching the
# reference repo's --model_type full|dev recipe.
_MODEL_TYPE_PRESETS = {
    "full": {"num_inference_steps": 50, "guidance_scale": 5.0, "shift": 3.0},
    # guidance_scale=1.0 (not 0.0): mathematically equivalent to "no CFG"
    # (v_guided = v_cond when scale==1.0), and avoids OmniDiffusionRequest's
    # falsy-sentinel handling, which silently resets 0.0 -> 1.0 anyway and
    # would otherwise make this preset indistinguishable from "not specified".
    "dev": {"num_inference_steps": 28, "guidance_scale": 1.0, "shift": 1.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HiDream-O1-Image text-to-image generation.")
    parser.add_argument("--model", default="HiDream-ai/HiDream-O1-Image", help="Model name or local path.")
    parser.add_argument("--prompt", default="A cute cat sitting on a windowsill at sunset.", help="Text prompt.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height (snapped to a 32px grid).")
    parser.add_argument("--width", type=int, default=1024, help="Output image width (snapped to a 32px grid).")
    parser.add_argument(
        "--model-type",
        choices=["full", "dev"],
        default="full",
        help="full: 50 steps, guidance 5.0, shift 3.0. dev: 28 steps, guidance 0.0 (no CFG), shift 1.0.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num-inference-steps", type=int, default=None, help="Override the --model-type preset's step count."
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=None, help="Override the --model-type preset's guidance scale."
    )
    parser.add_argument("--shift", type=float, default=None, help="Override the --model-type preset's shift value.")
    parser.add_argument(
        "--scheduler",
        choices=["default", "flow_match"],
        default="default",
        help="default: FlowUniPCMultistepScheduler. flow_match: diffusers FlowMatchEulerDiscreteScheduler.",
    )
    parser.add_argument("--output", type=str, default="hidream_o1_t2i_output.png", help="Output image path.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = _MODEL_TYPE_PRESETS[args.model_type]
    num_inference_steps = args.num_inference_steps or preset["num_inference_steps"]
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else preset["guidance_scale"]
    shift = args.shift if args.shift is not None else preset["shift"]

    omni = Omni(model=args.model, model_class_name="HiDreamO1ImagePipeline", enforce_eager=args.enforce_eager)

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    sampling_params.extra_args.update({"shift": shift, "scheduler_name": args.scheduler})

    print(f"{'=' * 60}\nHiDream-O1-Image ({args.model_type})\n{'=' * 60}")
    print(f"  model            : {args.model}")
    print(f"  prompt           : {args.prompt}")
    print(f"  size             : {args.width}x{args.height}")
    print(f"  steps            : {num_inference_steps}")
    print(f"  guidance_scale   : {guidance_scale}")
    print(f"  shift            : {shift}")
    print(f"  scheduler        : {args.scheduler}")
    print(f"  seed             : {args.seed}\n")

    outputs = omni.generate(args.prompt, sampling_params_list=[sampling_params])
    if not outputs:
        raise RuntimeError("omni.generate() returned no outputs.")

    images = getattr(outputs[0], "images", None)
    if not images:
        req_out = getattr(outputs[0], "request_output", None)
        images = getattr(req_out, "images", None) if req_out is not None else None
    if not images:
        raise RuntimeError("No images found in the output.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)
    print(f"Saved generated image to {output_path}")


if __name__ == "__main__":
    main()
