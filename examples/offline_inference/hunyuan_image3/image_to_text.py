# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os

from PIL import Image

from vllm_omni.entrypoints.omni import Omni

"""
The tencent/HunyuanImage-3.0-Instruct base model is built on the Hunyuan v1 architecture, specifically the tencent/Hunyuan-A13B-Instruct model. It utilizes two tokenizer delimiter templates:

1) Pretrained template (default for gen_text mode), which concatenates system, image
   tokens, and user question WITHOUT role delimiters:
"<|startoftext|>{system_prompt}{image_tokens}{user_question}"

   Example (before image token expansion):
"<|startoftext|>You are an assistant that understands images and outputs text.<img>Describe the content of the picture."

2) Instruct template, which uses explicit role prefixes and separators.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from image using HunyuanImage-3.0-Instruct.")
    parser.add_argument(
        "--model",
        default="tencent/HunyuanImage-3.0-Instruct",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./image.png",
        help="Path to input image file (PNG, JPG, etc.).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<|startoftext|>You are an assistant that understands images and outputs text.<img>Identify the animal in this image and describe this animal's characteristics in the image.",
        help="Pretrain template prompt: <|startoftext|>{system}<img>{question}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    omni = Omni(model=args.model)

    prompt_dict = {
        "prompt": args.prompt,
        "modalities": ["text"],
    }

    # Add image input if provided
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Input image not found: {args.image}")

        input_image = Image.open(args.image).convert("RGB")
        prompt_dict["multi_modal_data"] = {"image": input_image}
        print(f"Input image size: {input_image.size}")

    prompts = [prompt_dict]
    omni_outputs = omni.generate(prompts=prompts)
    print("omni_output = " + str(omni_outputs))
