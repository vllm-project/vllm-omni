"""
Offline inference example for MammothModa2 Text-to-Image (T2I) generation.
This script uses the vllm_omni.Omni pipeline with a multi-stage configuration.

Workflow:
1. Stage 0 (AR): Generates visual tokens and their corresponding hidden states.
2. Stage 1 (DiT): Consumes the hidden states as conditions to perform diffusion 
   and VAE decoding to produce the final image.

Example Usage:
    uv run python examples/offline_inference/run_mammothmoda2_t2i.py \
        --model /path/to/MammothModa2-Preview \
        --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
        --out output.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from vllm.sampling_params import SamplingParams

from vllm_omni import Omni

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_t2i_generation_config(model_dir: str) -> Tuple[int, int, int]:
    """Load T2I token ranges from t2i_generation_config.json."""
    cfg_path = Path(model_dir) / "t2i_generation_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return (
        int(cfg["eol_token_id"]),
        int(cfg["visual_token_start_id"]),
        int(cfg["visual_token_end_id"]),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MammothModa2 T2I (AR -> DiT) with vLLM-Omni.")
    p.add_argument(
        "--model",
        type=str,
        default="/data/datasets/models-hf/MammothModa2-Preview",
        help="Path to the model directory.",
    )
    p.add_argument(
        "--stage-config",
        type=str,
        default="vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml",
        help="Path to the multi-stage YAML configuration.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="A stylish woman with sunglasses riding a motorcycle in NYC.",
        help="Text prompt for image generation.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height (must be a multiple of 16).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width (must be a multiple of 16).",
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of diffusion steps for the DiT stage.",
    )
    p.add_argument(
        "--text-guidance-scale",
        type=float,
        default=9.0,
        help="Classifier-Free Guidance (CFG) scale for DiT.",
    )
    p.add_argument(
        "--cfg-range",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        help="Relative step range [start, end] where CFG is active.",
    )
    p.add_argument("--out",
                   type=str,
                   default="output.png",
                   help="Path to save the generated image.")
    p.add_argument("--trust-remote-code",
                   action="store_true",
                   help="Trust remote code when loading the model.")
    return p.parse_args()


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a normalized torch tensor [-1, 1] to a PIL Image."""
    if image.ndim == 4:
        image = image[0]
    image = image.detach().to("cpu")
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8)
    image = image.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(image)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.height <= 0 or args.width <= 0:
        raise ValueError(
            f"Height and width must be positive, got {args.height}x{args.width}"
        )
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(
            f"Height and width must be multiples of 16, got {args.height}x{args.width}"
        )

    ar_height = args.height // 16
    ar_width = args.width // 16

    eol_token_id, visual_start, visual_end = load_t2i_generation_config(
        args.model)
    expected_grid_tokens = ar_height * (ar_width + 1)

    prompt = (f"<|im_start|>system\nYou are a helpful image generator.<|im_end|>\n"
              f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
              f"<|im_start|>assistant\n"
              f"<|image start|>{ar_width}*{ar_height}<|image token|>")

    logger.info("Initializing Omni pipeline...")
    omni = Omni(model=args.model,
                stage_configs_path=args.stage_config,
                trust_remote_code=args.trust_remote_code)

    try:
        ar_sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=2048,
            # +1 for generating eoi, +1 for generating hidden state of eoi
            max_tokens=max(1, expected_grid_tokens + 1 + 1),
            detokenize=False,
        )

        dit_sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            detokenize=False,
        )

        logger.info("Starting generation...")
        inputs = [{
            "prompt": prompt,
            "additional_information": {
                "omni_task": ["t2i"],
                "ar_width": [ar_width],
                "ar_height": [ar_height],
                "eol_token_id": [eol_token_id],
                "visual_token_start_id": [visual_start],
                "visual_token_end_id": [visual_end],
                "image_height": [args.height],
                "image_width": [args.width],
                "num_inference_steps": [args.num_inference_steps],
                "text_guidance_scale": [args.text_guidance_scale],
                "cfg_range": [args.cfg_range[0], args.cfg_range[1]],
            },
        }]

        outputs = omni.generate(inputs, [ar_sampling, dit_sampling])

        ro = outputs[0].request_output
        if isinstance(ro, list):
            if not ro:
                raise RuntimeError("Empty request_output from final stage.")
            ro = ro[0]

        mm = getattr(ro, "multimodal_output", None)
        if not isinstance(mm, dict) or "image" not in mm:
            raise RuntimeError(
                f"Unexpected final output payload: {type(mm)} {mm}")

        img_tensor = mm["image"]
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError(f"Expected image tensor, got {type(img_tensor)}")

        logger.info("Post-processing and saving image...")
        pil = tensor_to_pil(img_tensor)
        pil.save(args.out)
        logger.info(f"Successfully saved generated image to: {args.out}")

    except Exception as e:
        logger.exception(f"An error occurred during generation: {e}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
