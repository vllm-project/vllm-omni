# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiDream-O1-Image image editing and multi-reference personalization.

Phase 2 offline inference script — extends text_to_image_hidream_o1.py with
reference-image support for:
  - Instruction-based editing (1 reference image)
  - Subject-driven personalization (2-10+ reference images)

Usage examples:

  # Image editing (1 ref image)
  python image_edit_hidream_o1.py \\
      --model HiDream-ai/HiDream-O1-Image-Dev \\
      --model-type dev \\
      --ref-images /path/to/photo.jpg \\
      --prompt "Make the background a snowy mountain landscape" \\
      --output edited.png

  # Multi-reference personalization (3 ref images)
  python image_edit_hidream_o1.py \\
      --model HiDream-ai/HiDream-O1-Image-Dev \\
      --model-type dev \\
      --ref-images person1.jpg person2.jpg person3.jpg \\
      --prompt "The person is sitting at a cafe in Paris" \\
      --output personalized.png
"""

import argparse
from pathlib import Path

from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Preset generation parameters matching the reference repo's --model_type recipe.
_MODEL_TYPE_PRESETS = {
    "full": {"num_inference_steps": 50, "guidance_scale": 5.0, "shift": 3.0},
    # guidance_scale=1.0 (not 0.0): mathematically equivalent to no CFG and
    # avoids OmniDiffusionRequest's falsy-sentinel reset (see Phase 1 notes).
    "dev": {"num_inference_steps": 28, "guidance_scale": 1.0, "shift": 1.0},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HiDream-O1-Image editing / personalization.")
    p.add_argument("--model", default="HiDream-ai/HiDream-O1-Image-Dev",
                   help="Model name or local checkpoint path.")
    p.add_argument("--prompt", default="A person is sitting at a café in Paris.",
                   help="Editing / generation instruction.")
    p.add_argument("--ref-images", nargs="+", required=True, metavar="PATH",
                   help="One or more reference image paths. 1 = editing, 2+ = personalization.")
    p.add_argument("--height", type=int, default=1024,
                   help="Output image height (snapped to a 32-px grid).")
    p.add_argument("--width", type=int, default=1024,
                   help="Output image width (snapped to a 32-px grid).")
    p.add_argument("--model-type", choices=["full", "dev"], default="dev",
                   help="full: 50 steps, guidance 5.0, shift 3.0. dev: 28 steps, no CFG, shift 1.0.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="Override the preset step count.")
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="Override the preset guidance scale.")
    p.add_argument("--shift", type=float, default=None,
                   help="Override the preset shift value.")
    p.add_argument("--scheduler", choices=["default", "flow_match"], default="default")
    p.add_argument("--output", default="hidream_o1_edit_output.png")
    p.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = _MODEL_TYPE_PRESETS[args.model_type]
    num_steps = args.num_inference_steps or preset["num_inference_steps"]
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else preset["guidance_scale"]
    shift = args.shift if args.shift is not None else preset["shift"]

    ref_images = [Image.open(p).convert("RGB") for p in args.ref_images]
    mode = "editing" if len(ref_images) == 1 else f"personalization ({len(ref_images)} refs)"

    print(f"{'=' * 60}\nHiDream-O1-Image {mode} ({args.model_type})\n{'=' * 60}")
    print(f"  model           : {args.model}")
    print(f"  prompt          : {args.prompt}")
    print(f"  ref images      : {args.ref_images}")
    print(f"  size            : {args.width}x{args.height}")
    print(f"  steps           : {num_steps}")
    print(f"  guidance_scale  : {guidance_scale}")
    print(f"  shift           : {shift}")
    print(f"  scheduler       : {args.scheduler}")
    print(f"  seed            : {args.seed}\n")

    omni = Omni(
        model=args.model,
        model_class_name="HiDreamO1ImagePipeline",
        enforce_eager=args.enforce_eager,
    )

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )
    sampling_params.extra_args.update({"shift": shift, "scheduler_name": args.scheduler})

    # Reference images are passed via multi_modal_data in the prompt dict.
    prompt_dict = {
        "prompt": args.prompt,
        "multi_modal_data": {"image": ref_images},
    }

    outputs = omni.generate(prompt_dict, sampling_params_list=[sampling_params])
    if not outputs:
        raise RuntimeError("omni.generate() returned no outputs.")

    images = getattr(outputs[0], "images", None)
    if not images:
        req_out = getattr(outputs[0], "request_output", None)
        images = getattr(req_out, "images", None) if req_out is not None else None
    if not images:
        raise RuntimeError("No images found in the output.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path)
    print(f"Saved output to {out_path}")


if __name__ == "__main__":
    main()
