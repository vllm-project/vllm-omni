# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end offline inference for SenseNova-U1-8B-MoT.

Supports four modalities: text2img, img2img, img2text, text2text.

Text-to-image:
    python end2end.py --prompt "A cute cat" --think

Image-to-image (editing):
    python end2end.py --prompt "Turn this into an oil painting" \
        --image input.png --think

Image understanding:
    python end2end.py --modality img2text \
        --prompt "Describe this image" --image photo.jpg

Text chat:
    python end2end.py --modality text2text \
        --prompt "What is the capital of France?"

See README.md for more examples.
"""

import argparse
import os

from PIL import Image

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="SenseNova-U1 text-to-image / image-to-image / text / understanding via vLLM-Omni.",
    )
    parser.add_argument(
        "--model",
        default="SenseNova/SenseNova-U1-8B-MoT",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--modality",
        default="auto",
        choices=["auto", "text2img", "img2img", "img2text", "text2text"],
        help="Task modality. 'auto' infers from --image (img2img/img2text if images, else t2i/t2t).",
    )
    parser.add_argument(
        "--prompt",
        default="A cute cat sitting on a windowsill, soft natural light",
        help="Text prompt for generation or editing instruction.",
    )
    parser.add_argument(
        "--image",
        nargs="+",
        metavar="PATH",
        default=None,
        help="Input image path(s) for img2img or img2text.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory to save generated images.",
    )
    # Image dimensions
    parser.add_argument("--height", type=int, default=2048, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=2048, help="Width of generated image.")

    # Generation parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results.")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Text classifier-free guidance scale.")
    parser.add_argument(
        "--img-cfg-scale",
        type=float,
        default=1.0,
        help="Image CFG scale for img2img (1.0 = disabled). "
        "When both --cfg-scale and --img-cfg-scale differ, dual CFG is used.",
    )
    parser.add_argument(
        "--cfg-norm",
        type=str,
        default="none",
        choices=["none", "global", "channel", "cfg_zero_star"],
        help="CFG normalization mode. cfg_zero_star is only for t2i.",
    )
    parser.add_argument(
        "--timestep-shift",
        type=float,
        default=3.0,
        help="Timestep shift for flow-matching schedule.",
    )
    parser.add_argument(
        "--t-eps",
        type=float,
        default=0.02,
        help="Epsilon for flow-matching timestep schedule.",
    )

    # Text generation parameters (for text2text / img2text)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for text generation (text2text / img2text).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding for text generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for text generation.",
    )

    # Think mode
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable think mode: the model reasons about the prompt before generating the image.",
    )
    parser.add_argument(
        "--print-think",
        action="store_true",
        help="Print the generated think text to stdout.",
    )

    # Advanced
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help=(
            "Enable model-level (sequential) CPU offloading. "
            "Mutually exclusive between vision_model and language_model: "
            "only one is on GPU at a time, reducing peak VRAM at the cost "
            "of extra CPU<->GPU transfers."
        ),
    )

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def _resolve_modality(args):
    """Determine the effective modality from CLI args."""
    if args.modality != "auto":
        return args.modality
    has_images = args.image is not None
    # Default: image-producing tasks for backward compatibility
    return "img2img" if has_images else "text2img"


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    modality = _resolve_modality(args)
    is_text_output = modality in ("text2text", "img2text")
    has_images = modality in ("img2img", "img2text")

    omni = Omni(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
    )

    extra_args = {
        "cfg_scale": args.cfg_scale,
        "cfg_norm": args.cfg_norm,
        "timestep_shift": args.timestep_shift,
        "cfg_interval": (0.0, 1.0),
        "batch_size": 1,
        "think": args.think,
        "t_eps": args.t_eps,
    }
    if modality == "img2img":
        extra_args["img_cfg_scale"] = args.img_cfg_scale
    if is_text_output:
        extra_args["max_tokens"] = args.max_tokens
        extra_args["do_sample"] = args.do_sample
        extra_args["temperature"] = args.temperature

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_inference_steps=args.num_steps,
        extra_args=extra_args,
    )

    print(f"\n{'=' * 60}")
    print(f"SenseNova-U1 Configuration ({modality}):")
    print(f"  Model          : {args.model}")
    if not is_text_output:
        print(f"  Image size     : {args.width}x{args.height}")
        print(f"  Steps          : {args.num_steps}")
        print(f"  CFG scale      : {args.cfg_scale}")
    if modality == "img2img":
        print(f"  Img CFG scale  : {args.img_cfg_scale}")
    if has_images:
        print(f"  Input images   : {args.image}")
    if is_text_output:
        print(f"  Max tokens     : {args.max_tokens}")
        print(f"  Temperature    : {args.temperature}")
    print(f"  Seed           : {args.seed}")
    print(f"  Think mode     : {args.think}")
    print(f"  TP size        : {args.tensor_parallel_size}")
    print(f"{'=' * 60}\n")

    # Build prompt dict
    if has_images:
        if not args.image:
            raise ValueError(f"{modality} requires --image.")
        input_images = [Image.open(p).convert("RGB") for p in args.image]
        if is_text_output:
            prompt_dict = {
                "prompt": args.prompt,
                "multi_modal_data": {"image": input_images},
                "modalities": ["text"],
            }
        else:
            prompt_dict = {
                "prompt": args.prompt,
                "multi_modal_data": {"image": input_images},
                "modalities": ["img2img"],
            }
    elif is_text_output:
        prompt_dict = {"prompt": args.prompt, "modalities": ["text"]}
    else:
        prompt_dict = {"prompt": args.prompt, "modalities": ["image"]}

    outputs = list(
        omni.generate(
            prompts=prompt_dict,
            sampling_params_list=sampling_params,
        )
    )

    for req_output in outputs:
        custom = getattr(req_output, "_custom_output", {}) or {}
        if args.print_think and custom.get("think_text"):
            print(f"[Think]\n{custom['think_text']}\n")

        if is_text_output:
            text = custom.get("text_output", "")
            if not text:
                text = getattr(req_output, "text", "") or ""
            print(f"[Response]\n{text}")
        else:
            images = getattr(req_output, "images", None) or []
            if not images:
                print("[Warning] No images generated.")
                continue
            for j, img in enumerate(images):
                save_path = os.path.join(args.output, f"sensenova_u1_output_{j}.png")
                img.save(save_path)
                print(f"[Output] Saved {img.size[0]}x{img.size[1]} image to {save_path}")


if __name__ == "__main__":
    main()
