# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiDream-O1-Image layout-bbox controlled generation.

Phase 3 offline inference script — extends image_edit_hidream_o1.py with
bounding-box layout conditioning for spatially grounded generation.

Layout bboxes let you specify WHERE each subject (from the reference images)
should appear in the output. The model sees a layout image (black background
with colored boxes) alongside bordered reference images (each bordered in the
matching color), learning the spatial correspondence.

Coordinate format: xxyy order, relative [0, 1] or percentage [0, 100]:
  [x1, x2, y1, y2]  →  x1 < x2 are horizontal bounds, y1 < y2 vertical

Usage examples:

  # 2 ref images with explicit layout (person left, dog right)
  python layout_control_hidream_o1.py \\
      --model HiDream-ai/HiDream-O1-Image-Dev \\
      --model-type dev \\
      --ref-images person.jpg dog.jpg \\
      --layout-bboxes '[[0.0, 0.45, 0.1, 0.9], [0.55, 1.0, 0.1, 0.9]]' \\
      --prompt "A person and a dog sitting in a park" \\
      --output layout_output.png

  # Using a JSON file instead of inline string
  python layout_control_hidream_o1.py \\
      --model HiDream-ai/HiDream-O1-Image-Dev \\
      --ref-images subject.jpg \\
      --layout-bboxes /path/to/bboxes.json \\
      --prompt "The subject is standing in front of the Eiffel Tower" \\
      --output output.png
"""

import argparse
from pathlib import Path

from PIL import Image

from vllm_omni.diffusion.models.hidream_o1_image.utils_hidream_o1 import load_layout_bboxes
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

_MODEL_TYPE_PRESETS = {
    "full": {"num_inference_steps": 50, "guidance_scale": 5.0, "shift": 3.0},
    "dev": {"num_inference_steps": 28, "guidance_scale": 1.0, "shift": 1.0},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HiDream-O1-Image layout-bbox controlled generation.")
    p.add_argument("--model", default="HiDream-ai/HiDream-O1-Image-Dev", help="Model name or local checkpoint path.")
    p.add_argument("--prompt", default="A person and a dog sitting in a sunny park.", help="Generation instruction.")
    p.add_argument(
        "--ref-images",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Reference image paths. Each ref corresponds to one bbox entry.",
    )
    p.add_argument(
        "--layout-bboxes",
        required=True,
        metavar="JSON_OR_PATH",
        help=(
            "Bounding boxes as a JSON string or a .json file path. "
            "Format: [[x1,x2,y1,y2], ...] in xxyy order, values in [0,1] or [0,100]. "
            "One bbox per reference image, in the same order."
        ),
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--model-type", choices=["full", "dev"], default="dev")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--scheduler", choices=["default", "flow_match"], default="default")
    p.add_argument("--output", default="hidream_o1_layout_output.png")
    p.add_argument("--enforce-eager", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = _MODEL_TYPE_PRESETS[args.model_type]
    num_steps = args.num_inference_steps or preset["num_inference_steps"]
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else preset["guidance_scale"]
    shift = args.shift if args.shift is not None else preset["shift"]

    ref_images = [Image.open(p).convert("RGB") for p in args.ref_images]

    print(f"{'=' * 60}\nHiDream-O1-Image layout-bbox control ({args.model_type})\n{'=' * 60}")
    print(f"  model           : {args.model}")
    print(f"  prompt          : {args.prompt}")
    print(f"  ref images      : {args.ref_images}")
    print(f"  layout bboxes   : {args.layout_bboxes}")
    print(f"  size            : {args.width}x{args.height}")
    print(f"  steps           : {num_steps}")
    print(f"  guidance_scale  : {guidance_scale}")
    print(f"  shift           : {shift}")
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

    # Pre-load layout bboxes here (operator-controlled script) so the pipeline
    # receives a parsed list/dict, not a raw string.  The pipeline only accepts
    # inline JSON in multi_modal_data to prevent file-system access from
    # user-supplied data in online serving.
    layout_data = load_layout_bboxes(args.layout_bboxes)

    prompt_dict = {
        "prompt": args.prompt,
        "multi_modal_data": {
            "image": ref_images,
            "layout_bboxes": layout_data,
        },
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
