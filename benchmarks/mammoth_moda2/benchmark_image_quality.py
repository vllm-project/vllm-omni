# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare cache-disabled and warm-hit MammothModa2 image outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from vllm import SamplingParams

from vllm_omni import Omni
from vllm_omni.model_extras import (
    build_text_to_image_prompt,
    get_model_class_name,
)

PROMPTS = [
    "A red cube on a white table",
    "A watercolor landscape with mountains and a blue lake",
    "A small robot reading a book in a warm library",
]
SEEDS = [17, 42, 123]
DEPLOY_CONFIGS = {
    "a": "vllm_omni/deploy/mammoth_moda2.yaml",
    "b2": "vllm_omni/deploy/mammoth_moda2_prefix_cache.yaml",
}
CLIP_MODEL = "openai/clip-vit-base-patch32"


def build_request(
    omni: Omni,
    prompt: str,
    seed: int,
) -> tuple[dict, list[SamplingParams]]:
    request = build_text_to_image_prompt(
        model_class_name=get_model_class_name(omni),
        prompt={"prompt": prompt, "modalities": ["image"]},
        height=256,
        width=256,
    )
    request["additional_information"].update(
        {
            "num_inference_steps": [50],
            "text_guidance_scale": [9.0],
            "cfg_range": [0.0, 1.0],
        }
    )
    info = request["additional_information"]
    ar_width = int(info["ar_width"][0])
    ar_height = int(info["ar_height"][0])
    return request, [
        SamplingParams(
            temperature=0.0,
            top_k=1,
            seed=seed,
            max_tokens=ar_height * (ar_width + 1) + 1,
            detokenize=False,
        ),
        SamplingParams(
            temperature=0.0,
            seed=seed,
            max_tokens=1,
            detokenize=False,
        ),
    ]


def generate(omni: Omni, request: dict, params: list[SamplingParams]) -> np.ndarray:
    output = omni.generate(
        request,
        sampling_params_list=params,
        use_tqdm=False,
    )[0]
    image = output.multimodal_output["image"]
    if isinstance(image, list):
        image = image[0]
    if not isinstance(image, torch.Tensor):
        raise TypeError(type(image))
    return image.detach().cpu().float().numpy()


def run_scenario(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    omni = Omni(
        model=args.model,
        deploy_config=DEPLOY_CONFIGS[args.scenario],
        mode="text-to-image",
    )
    records = []
    try:
        for prompt_index, prompt in enumerate(PROMPTS):
            for seed in SEEDS:
                request, params = build_request(omni, prompt, seed)
                if args.scenario == "b2":
                    generate(omni, request, params)
                image = generate(omni, request, params)
                path = args.output_dir / (f"{args.scenario}_p{prompt_index}_s{seed}.npy")
                np.save(path, image)
                records.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "seed": seed,
                        "path": str(path),
                        "shape": list(image.shape),
                    }
                )
    finally:
        omni.shutdown()
    (args.output_dir / f"{args.scenario}_manifest.json").write_text(json.dumps(records, indent=2) + "\n")


def _normalized_image(path: Path) -> np.ndarray:
    image = np.load(path).astype(np.float32)
    return np.clip(image / 2 + 0.5, 0, 1)


def _pil_image(image: np.ndarray) -> Image.Image:
    pixels = (image.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return Image.fromarray(pixels)


def compare(args: argparse.Namespace) -> None:
    from skimage.metrics import structural_similarity

    model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    rows = []
    with torch.inference_mode():
        for prompt_index, prompt in enumerate(PROMPTS):
            for seed in SEEDS:
                a = _normalized_image(args.output_dir / f"a_p{prompt_index}_s{seed}.npy")
                b2 = _normalized_image(args.output_dir / f"b2_p{prompt_index}_s{seed}.npy")
                mse = float(np.mean((a - b2) ** 2))
                inputs = processor(
                    text=[prompt],
                    images=[_pil_image(a), _pil_image(b2)],
                    return_tensors="pt",
                    padding=True,
                )
                outputs = model(**inputs)
                image_embeddings = torch.nn.functional.normalize(outputs.image_embeds.float(), dim=-1)
                text_embedding = torch.nn.functional.normalize(outputs.text_embeds.float(), dim=-1)[0]
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "seed": seed,
                        "mae": float(np.mean(np.abs(a - b2))),
                        "psnr_db": (float("inf") if mse == 0 else -10 * math.log10(mse)),
                        "ssim": float(
                            structural_similarity(
                                a,
                                b2,
                                channel_axis=0,
                                data_range=1.0,
                            )
                        ),
                        "clip_image_cosine": float((image_embeddings[0] * image_embeddings[1]).sum()),
                        "a_text_clip_cosine": float((image_embeddings[0] * text_embedding).sum()),
                        "b2_text_clip_cosine": float((image_embeddings[1] * text_embedding).sum()),
                    }
                )

    summary = {}
    for key in (
        "mae",
        "psnr_db",
        "ssim",
        "clip_image_cosine",
        "a_text_clip_cosine",
        "b2_text_clip_cosine",
    ):
        values = np.asarray([row[key] for row in rows])
        summary[key] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    summary["text_clip_delta_b2_minus_a"] = float(
        np.mean([row["b2_text_clip_cosine"] - row["a_text_clip_cosine"] for row in rows])
    )
    result = {"clip_model": CLIP_MODEL, "rows": rows, "summary": summary}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--scenario", choices=DEPLOY_CONFIGS, required=True)
    generate_parser.add_argument("--model", required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.set_defaults(func=run_scenario)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
