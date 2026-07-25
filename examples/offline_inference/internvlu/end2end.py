# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end offline inference for InternVL-U.

Supports four modalities: text2img, img2img (editing), img2text, text2text.

Text-to-image:
    python end2end.py --prompt "A cute cat"

Image editing:
    python end2end.py --prompt "Add a red scarf around the cat's neck" \
        --image input.png

Think mode (text-then-image; the model first writes a detailed description,
then generates the image conditioned on it):
    python end2end.py --prompt "A cute cat" --think

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
from pathlib import Path

from PIL import Image

import vllm_omni
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_image_to_image_prompt,
    build_text_to_image_prompt,
    get_model_class_name,
)

DEPLOY_DIR = Path(vllm_omni.__file__).parent / "deploy"


def parse_args():
    parser = argparse.ArgumentParser(
        description="InternVL-U text-to-image / editing / understanding / chat via vLLM-Omni.",
    )
    parser.add_argument(
        "--model",
        default="InternVL-U/InternVL-U",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--modality",
        default="auto",
        choices=["auto", "text2img", "img2img", "img2text", "text2text"],
        help="Task modality. 'auto' infers from --image (img2img if images, else text2img).",
    )
    parser.add_argument(
        "--prompt",
        default="A photo of an orange tabby cat sitting on a wooden table next to a cup of coffee.",
        help="Text prompt, editing instruction, or question.",
    )
    parser.add_argument(
        "--image",
        nargs="+",
        metavar="PATH",
        default=None,
        help="Reference image path(s) for img2img or img2text.",
    )
    parser.add_argument("--output", default=".", help="Output directory for generated images.")

    # Image generation parameters
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results.")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of denoising steps.")
    parser.add_argument(
        "--think",
        action="store_true",
        help=(
            "Text-then-image generation: deploys with internvlu_chat_think.yaml, "
            "which statically enables chain-of-thought before image generation."
        ),
    )

    # Text generation parameters (img2text / text2text)
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for text outputs.")

    return parser.parse_args()


def _resolve_modality(args) -> str:
    if args.modality != "auto":
        return args.modality
    return "img2img" if args.image else "text2img"


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    modality = _resolve_modality(args)
    is_text_output = modality in ("img2text", "text2text")

    omni_kwargs = {"model": args.model}
    if args.think:
        if is_text_output:
            raise ValueError("--think applies to image generation; use it with text2img or img2img.")
        # Think mode is configured statically per deployment, mirroring bagel_think.
        omni_kwargs["deploy_config"] = str(DEPLOY_DIR / "internvlu_chat_think.yaml")
    omni = Omni(**omni_kwargs)
    model_class_name = get_model_class_name(omni)

    input_images = None
    if args.image:
        input_images = [Image.open(path).convert("RGB") for path in args.image]

    if modality == "text2img":
        prompt_dict = build_text_to_image_prompt(
            model_class_name,
            args.prompt,
            negative_prompt=None,
            height=args.height,
            width=args.width,
        )
    elif modality == "img2img":
        if not input_images:
            raise ValueError("img2img requires --image.")
        prompt_dict = build_image_to_image_prompt(
            model_class_name,
            args.prompt,
            negative_prompt=None,
            input_image=input_images,
            height=args.height,
            width=args.width,
        )
    elif modality == "img2text":
        if not input_images:
            raise ValueError("img2text requires --image.")
        prompt_dict = {
            "prompt": args.prompt,
            "multi_modal_data": {"image": input_images},
            "modalities": ["text"],
        }
    else:  # text2text
        prompt_dict = {"prompt": args.prompt, "modalities": ["text"]}

    # Start from the deploy YAML's per-stage defaults so static settings
    # (e.g. the CoT sampling flag) are preserved, then override the diffusion
    # stage with the requested generation parameters.
    sampling_params_list = [
        params.clone() if hasattr(params, "clone") else params for params in (omni.default_sampling_params_list or [])
    ]
    if not is_text_output:
        diffusion_params = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            seed=args.seed,
            num_inference_steps=args.num_steps,
        )
        replaced = False
        for idx, params in enumerate(sampling_params_list):
            if isinstance(params, OmniDiffusionSamplingParams):
                sampling_params_list[idx] = diffusion_params
                replaced = True
        if not replaced:
            sampling_params_list.append(diffusion_params)
    else:
        for params in sampling_params_list:
            if hasattr(params, "max_tokens"):
                params.max_tokens = args.max_tokens

    outputs = list(omni.generate(prompts=prompt_dict, sampling_params_list=sampling_params_list))

    saved = 0
    for req_output in outputs:
        request_output = getattr(req_output, "request_output", None)
        prompt_obj = getattr(request_output, "prompt", None)
        if isinstance(prompt_obj, dict):
            cot_text = (prompt_obj.get("extra") or {}).get("ar_generated_text")
            if isinstance(cot_text, str) and cot_text.strip():
                print(f"[CoT]\n{cot_text}\n")

        if is_text_output:
            completion_outputs = getattr(request_output, "outputs", None) or []
            text = getattr(completion_outputs[0], "text", None) if completion_outputs else None
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("InternVL-U returned no text completion.")
            print(f"[Response]\n{text}")
            continue

        for image in getattr(req_output, "images", None) or []:
            save_path = os.path.join(args.output, f"internvlu_output_{saved}.png")
            image.save(save_path)
            saved += 1
            print(f"[Output] Saved {image.size[0]}x{image.size[1]} image to {save_path}")

    if not is_text_output and not saved:
        print("[Warning] No images generated.")


if __name__ == "__main__":
    main()
