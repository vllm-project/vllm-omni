# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference: HunyuanImage-3.0-Instruct comprehension (AR-only).

Covers the two text-output modalities that have no shared task-example home:

  * ``image2text`` (i2t): describe / reason about an input image.
  * ``text2text``  (t2t): plain text chat.

The two image-generating modalities were migrated to the shared task examples
(see this directory's README):

  * text-to-image  -> examples/offline_inference/text_to_image/text_to_image.py
  * image-editing  -> examples/offline_inference/image_to_image/image_edit.py

All three paths build the AR prefill + stop tokens through the same declarative
seam, ``vllm_omni.model_extras.hunyuan_image3.build_ar_stage_inputs``, so prompt
formatting stays consistent across offline and online (OpenAI server) flows.

Example (image-to-text):
  python examples/offline_inference/hunyuan_image3/run_hunyuan_image3_understanding.py \
    --model tencent/HunyuanImage-3.0-Instruct \
    --modality image2text \
    --image /path/to/input.jpg \
    --prompt "Describe this image in detail."

Example (text-to-text):
  python examples/offline_inference/hunyuan_image3/run_hunyuan_image3_understanding.py \
    --model tencent/HunyuanImage-3.0-Instruct \
    --modality text2text \
    --prompt "Explain diffusion models in one paragraph."
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_extras.hunyuan_image3 import build_ar_stage_inputs

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_AR_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_ar.yaml")

_MODALITY_MODE = {
    "image2text": "image-to-text",
    "text2text": "text-to-text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0 comprehension (AR-only) offline inference.")
    parser.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct", help="Model name or local path.")
    parser.add_argument("--modality", default="image2text", choices=list(_MODALITY_MODE))
    parser.add_argument("--prompt", default="Describe this image in detail.", help="Text prompt / question.")
    parser.add_argument("--image", type=str, default=None, help="Input image path (required for image2text).")
    parser.add_argument(
        "--deploy-config", type=str, default=None, help="Custom deploy YAML path (defaults to AR yaml)."
    )
    parser.add_argument("--bot-task", type=str, default=None, help="Prompt mode override (e.g. think, recaption).")
    parser.add_argument("--use-system-prompt", type=str, default=None, help="System-prompt preset override.")
    parser.add_argument("--system-prompt", type=str, default=None, help="Custom system prompt.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-timeout", type=int, default=300, help="Initialization timeout in seconds.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_data = None
    num_images = 0
    if args.modality == "image2text":
        if not args.image or not os.path.exists(args.image):
            raise ValueError(f"--image is required and must exist for image2text, got: {args.image!r}")
        from PIL import Image

        image_data = Image.open(args.image).convert("RGB")
        num_images = 1

    # extra_body carries the model-specific prompt knobs; keys absent from the
    # CLI simply fall back to each task's defaults inside the helper.
    extra_body: dict[str, object] = {}
    if args.bot_task is not None:
        extra_body["bot_task"] = args.bot_task
    if args.use_system_prompt is not None:
        extra_body["use_system_prompt"] = args.use_system_prompt
    if args.system_prompt is not None:
        extra_body["system_prompt"] = args.system_prompt

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ar_inputs = build_ar_stage_inputs(
        prompt=args.prompt,
        tokenizer=tokenizer,
        extra_body=extra_body,
        num_images=num_images,
        text_output=True,
    )

    deploy_config = args.deploy_config or _DEFAULT_AR_DEPLOY_CONFIG
    omni = Omni(
        model=args.model,
        deploy_config=deploy_config,
        mode=_MODALITY_MODE[args.modality],
        # HunyuanImage-3.0 ships custom modeling code; the AR-only deploy sets
        # trust_remote_code only per-stage, so set it engine-wide here too.
        trust_remote_code=True,
        init_timeout=args.init_timeout,
        enforce_eager=args.enforce_eager,
    )

    prompt_dict: dict[str, object] = {"modalities": ar_inputs.modalities}
    if ar_inputs.prompt_token_ids is not None:
        prompt_dict["prompt_token_ids"] = ar_inputs.prompt_token_ids
    else:
        prompt_dict["prompt"] = ar_inputs.prompt
    if image_data is not None:
        prompt_dict["multi_modal_data"] = {"image": image_data}

    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        seed=int(args.seed),
        stop_token_ids=ar_inputs.stop_token_ids,
        detokenize=True,
    )

    print(f"\n{'=' * 60}")
    print("HunyuanImage-3.0 Comprehension:")
    print(f"  Model: {args.model}")
    print(f"  Modality: {args.modality} ({_MODALITY_MODE[args.modality]})")
    print(f"  Deploy config: {deploy_config}")
    if args.image:
        print(f"  Input image: {args.image}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Stop token ids: {ar_inputs.stop_token_ids}")
    print(f"{'=' * 60}\n")

    try:
        outputs = list(omni.generate([prompt_dict], [sampling_params]))
    finally:
        omni.close()

    for stage_outputs in outputs:
        ro = getattr(stage_outputs, "request_output", stage_outputs)
        text = ro.outputs[0].text if getattr(ro, "outputs", None) else str(ro)
        print(f"[answer] {text.strip()}")


if __name__ == "__main__":
    main()
