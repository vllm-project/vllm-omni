"""
Offline inference example: MammothModa2 image summarization (single AR stage).

Example:
  uv run python examples/offline_inference/mammothmodal2_preview/run_mammothmoda2_image_summary.py \
    --model /data/datasets/models-hf/MammothModa2-Preview \
    --image /path/to/input.jpg \
    --question "Please summarize the content of this image."
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image
from vllm import SamplingParams
from vllm.multimodal.image import convert_image_mode

from vllm_omni import Omni


DEFAULT_SYSTEM = "You are a helpful assistant."
DEFAULT_QUESTION = "Please summarize the content of this image."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MammothModa2 image summarization (offline, AR only).")
    parser.add_argument(
        "--model",
        type=str,
        default="/data/datasets/models-hf/MammothModa2-Preview",
        help="Path to model directory or model id.",
    )
    parser.add_argument(
        "--stage-config",
        type=str,
        default=str(Path(__file__).with_name("mammoth_moda2_image_summary.yaml")),
        help="Path to stage config yaml (single-stage AR->text).",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question/instruction for the model.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=DEFAULT_SYSTEM,
        help="System prompt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="image_summary.txt",
        help="Path to save output text.",
    )
    return parser.parse_args()


def build_prompt(system: str, question: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pil_image = Image.open(args.image)
    image_data = convert_image_mode(pil_image, "RGB")
    prompt = build_prompt(args.system, args.question)

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_config,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        sp = SamplingParams(
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=-1,
            max_tokens=int(args.max_tokens),
            seed=int(args.seed),
            detokenize=True,
        )
        outputs = omni.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_data},
                }
            ],
            [sp],
        )
    finally:
        omni.close()

    if not isinstance(outputs, list):
        outputs = [outputs]

    lines: list[str] = []
    for stage_outputs in outputs:
        req_outputs = getattr(stage_outputs, "request_output", stage_outputs)
        req_outputs = req_outputs if isinstance(req_outputs, list) else [req_outputs]
        for ro in req_outputs:
            text = ro.outputs[0].text if getattr(ro, "outputs", None) else str(ro)
            lines.append(f"request_id: {getattr(ro, 'request_id', 'unknown')}\n")
            lines.append("answer:\n")
            lines.append(text.strip() + "\n")
            lines.append("\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[OK] Saved summary to: {args.out}")


if __name__ == "__main__":
    main()

