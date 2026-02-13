# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline end-to-end example for Apertus text+image inference in vLLM-Omni.

This script exercises the full Omni pipeline:
1) text+image request input
2) Apertus-specific EMU image tokenization in the custom preprocessor
3) single-stage LLM generation
4) text output extraction
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni


def _default_stage_config_path() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "vllm_omni" / "model_executor" / "stage_configs" / "apertus.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Apertus text+image offline inference with vLLM-Omni.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HF ID for the Apertus model checkpoint.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=_default_stage_config_path(),
        help="Path to stage config YAML (defaults to vllm_omni/model_executor/stage_configs/apertus.yaml).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image briefly: <|image|>",
        help="Prompt text. Include <|image|> where the image should be injected.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Optional path to input image. If omitted, a synthetic image is used.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save structured output JSON.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Enable trust_remote_code for model loading.",
    )
    parser.add_argument(
        "--emu-checkpoint",
        type=str,
        default="BAAI/Emu3-VisionTokenizer",
        help="EMU vision tokenizer checkpoint (lmms-eval default).",
    )
    parser.add_argument(
        "--emu-device",
        type=str,
        default="cuda:0",
        help="Device for EMU vision tokenizer. Use cuda:0 to run on GPU.",
    )
    parser.add_argument(
        "--emu-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype for EMU vision tokenizer.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable Omni orchestrator stats logging.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for stage initialization in seconds.",
    )
    return parser.parse_args()


def _load_or_create_image(image_path: str | None) -> Image.Image:
    if image_path is None:
        # Synthetic RGB image fallback for a self-contained smoke run.
        return Image.new("RGB", (64, 64), color=(64, 128, 192))

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


def _ensure_image_placeholder(prompt: str) -> str:
    if "<|image|>" in prompt:
        return prompt
    return f"<|image|>\n{prompt}"


def _extract_text(outputs) -> str:
    if not outputs:
        raise ValueError("No outputs returned by omni.generate()")

    first = outputs[0]
    if not hasattr(first, "request_output") or not first.request_output:
        raise ValueError("No request_output found in Omni output")

    req_out = first.request_output[0]
    if not hasattr(req_out, "outputs") or not req_out.outputs:
        raise ValueError("No token outputs found in request_output")

    return req_out.outputs[0].text


def main() -> None:
    args = parse_args()
    image = _load_or_create_image(args.image_path)
    prompt = _ensure_image_placeholder(args.prompt)

    prompt_dict = {
        "prompt": prompt,
        "multi_modal_data": {"image": [image]},
        "mm_processor_kwargs": {
            "apertus_vq_hub": args.emu_checkpoint,
            "apertus_vision_tokenizer_device": args.emu_device,
            "apertus_vision_tokenizer_dtype": args.emu_dtype,
            "trust_remote_code": args.trust_remote_code,
        },
    }

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        trust_remote_code=args.trust_remote_code,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t0 = time.time()
    try:
        outputs = omni.generate([prompt_dict], [sampling_params])
    finally:
        omni.close()
    elapsed = time.time() - t0

    text = _extract_text(outputs)

    print("=== Apertus E2E Result ===")
    print(f"Model: {args.model}")
    print(f"Stage config: {args.stage_configs_path}")
    print(f"EMU checkpoint: {args.emu_checkpoint}")
    print(f"EMU device: {args.emu_device}")
    print(f"Elapsed: {elapsed:.2f}s")
    print("Prompt:")
    print(prompt)
    print("Generated text:")
    print(text)

    if args.output_json:
        payload = {
            "model": args.model,
            "stage_configs_path": args.stage_configs_path,
            "emu_checkpoint": args.emu_checkpoint,
            "emu_device": args.emu_device,
            "prompt": prompt,
            "generated_text": text,
            "elapsed_seconds": elapsed,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved output JSON to: {out_path}")


if __name__ == "__main__":
    main()
