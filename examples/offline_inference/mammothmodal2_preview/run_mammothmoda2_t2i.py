"""
Offline inference example for MammothModa2 Text-to-Image (T2I) generation.
This script uses the vllm_omni.Omni pipeline with a multi-stage configuration.

Workflow:
1. Stage 0 (AR): Generates visual tokens and their corresponding hidden states.
2. Stage 1 (DiT): Consumes the hidden states as conditions to perform diffusion
   and VAE decoding to produce the final image.

Example Usage:
    uv run python examples/offline_inference/run_mammothmoda2_t2i.py \
        --model path/to/MammothModa2-Preview \
        --stage-config vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml \
        --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
        --out output.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image
from vllm.sampling_params import SamplingParams

from vllm_omni import Omni
from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PATCH_SIZE = 16  # AR image grid patch size (pixels per token)


class T2IGenConfig(NamedTuple):
    eol_token_id: int
    visual_token_start_id: int
    visual_token_end_id: int
    top_k: int  # AR sampling top-k (covers the full visual generation vocabulary)
    # Qwen2.5-VL special vision tokens: <|image_pad|>, <|video_pad|>, <|vision_start|>, <|vision_end|>
    visual_ids: list[int]


def load_t2i_generation_config(model_dir: str) -> T2IGenConfig:
    """Load T2I token IDs from t2i_generation_config.json and config.json.

    Supports both local directory paths and HuggingFace Hub model IDs.
    """
    model_path = Path(model_dir)

    def _read_json(filename: str) -> dict:
        local = model_path / filename
        if local.exists():
            with local.open(encoding="utf-8") as f:
                return json.load(f)
        # Fall back to HuggingFace Hub when model_dir is a repo ID.
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to load configs from HF Hub. Install it with: pip install huggingface_hub"
            ) from exc
        cached = hf_hub_download(repo_id=model_dir, filename=filename)
        with open(cached, encoding="utf-8") as f:
            return json.load(f)

    gen_cfg = _read_json("t2i_generation_config.json")
    llm_cfg = _read_json("config.json").get("llm_config", {})

    return T2IGenConfig(
        eol_token_id=int(gen_cfg["eol_token_id"]),
        visual_token_start_id=int(gen_cfg["visual_token_start_id"]),
        visual_token_end_id=int(gen_cfg["visual_token_end_id"]),
        top_k=int(gen_cfg["top_k"]),
        visual_ids=[
            int(llm_cfg["image_token_id"]),
            int(llm_cfg["video_token_id"]),
            int(llm_cfg["vision_start_token_id"]),
            int(llm_cfg["vision_end_token_id"]),
        ],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MammothModa2 T2I (AR -> DiT) with vLLM-Omni.")
    p.add_argument("--model", type=str, required=True, help="Path to the model directory.")
    p.add_argument("--stage-config", type=str, required=True, help="Path to the multi-stage YAML configuration.")
    p.add_argument(
        "--prompt",
        type=str,
        action="append",
        default=None,
        help=(
            "Text prompt for image generation. Can be provided multiple times "
            "to generate multiple images with shared height/width/CFG settings."
        ),
    )
    p.add_argument("--height", type=int, default=1024, help="Output image height (must be a multiple of 16).")
    p.add_argument("--width", type=int, default=1024, help="Output image width (must be a multiple of 16).")
    p.add_argument("--num-inference-steps", type=int, default=50, help="Number of diffusion steps for the DiT stage.")
    p.add_argument(
        "--text-guidance-scale", type=float, default=9.0, help="Classifier-Free Guidance (CFG) scale for DiT."
    )
    p.add_argument(
        "--cfg-range",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        help="Relative step range [start, end] where CFG is active.",
    )
    p.add_argument("--out", type=str, default="output.png", help="Path to save the generated image.")
    p.add_argument("--trust-remote-code", action="store_true", help="Trust remote code when loading the model.")
    nullify_stage_engine_defaults(p)
    args = p.parse_args()
    if not args.prompt:
        args.prompt = ["A stylish woman with sunglasses riding a motorcycle in NYC."]
    return args


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a normalized torch tensor [-1, 1] to a PIL Image."""
    if image.ndim == 4:
        image = image[0]
    image = image.detach().to("cpu")
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8)
    image = image.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(image)


def _format_prompt(user_prompt: str, ar_width: int, ar_height: int) -> str:
    """Build the AR-stage prompt string including the image grid header."""
    return (
        "<|im_start|>system\nYou are a helpful image generator.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<|image start|>{ar_width}*{ar_height}<|image token|>"
    )


def _collect_images(outputs: list) -> list[torch.Tensor]:
    """Extract all image tensors produced by the final (DiT) stage."""
    images: list[torch.Tensor] = []
    for out in outputs:
        ro_item = getattr(out, "request_output", out)
        for completion in getattr(ro_item, "outputs", None) or []:
            mm = getattr(completion, "multimodal_output", None)
            if not isinstance(mm, dict) or "image" not in mm:
                raise RuntimeError(f"Missing image in multimodal output: {mm}")
            payload = mm["image"]
            for tensor in payload if isinstance(payload, list) else [payload]:
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Expected image tensor, got {type(tensor)}")
                images.append(tensor)
    return images


def _save_images(images: list[torch.Tensor], out_path: str) -> list[str]:
    """Save image tensors to disk.

    Single image: written to *out_path* exactly.
    Multiple images: suffixed as ``<base>_0<ext>``, ``<base>_1<ext>``, …
    """
    if not images:
        raise RuntimeError("No images to save.")
    base, ext = os.path.splitext(out_path)
    ext = ext or ".png"
    paths = []
    for i, tensor in enumerate(images):
        path = out_path if len(images) == 1 else f"{base}_{i}{ext}"
        tensor_to_pil(tensor).save(path)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.height <= 0 or args.width <= 0:
        raise ValueError(f"Height and width must be positive, got {args.height}x{args.width}")
    if args.height % _PATCH_SIZE != 0 or args.width % _PATCH_SIZE != 0:
        raise ValueError(f"Height and width must be multiples of {_PATCH_SIZE}, got {args.height}x{args.width}")

    ar_height = args.height // _PATCH_SIZE
    ar_width = args.width // _PATCH_SIZE
    gen_cfg = load_t2i_generation_config(args.model)
    expected_grid_tokens = ar_height * (ar_width + 1)

    logger.info("Initializing Omni pipeline...")
    omni = Omni(model=args.model, stage_configs_path=args.stage_config, trust_remote_code=args.trust_remote_code)
    try:
        ar_sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=gen_cfg.top_k,
            max_tokens=max(1, expected_grid_tokens + 1),  # +1 for hidden state of eoi
            detokenize=False,
        )
        dit_sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            detokenize=False,
        )

        additional_information = {
            "omni_task": ["t2i"],
            "ar_width": [ar_width],
            "ar_height": [ar_height],
            "eol_token_id": [gen_cfg.eol_token_id],
            "visual_token_start_id": [gen_cfg.visual_token_start_id],
            "visual_token_end_id": [gen_cfg.visual_token_end_id],
            "image_height": [args.height],
            "image_width": [args.width],
            "num_inference_steps": [args.num_inference_steps],
            "text_guidance_scale": [args.text_guidance_scale],
            "cfg_range": [args.cfg_range[0], args.cfg_range[1]],
            "visual_ids": gen_cfg.visual_ids,
        }
        inputs = [
            {
                "prompt": _format_prompt(p, ar_width, ar_height),
                "additional_information": dict(additional_information),
            }
            for p in args.prompt
        ]

        logger.info("Starting generation...")
        # omni.generate() returns a Generator; consume it to run the full pipeline.
        outputs = list(omni.generate(inputs, [ar_sampling, dit_sampling]))

        logger.info("Post-processing and saving image(s)...")
        for path in _save_images(_collect_images(outputs), args.out):
            logger.info(f"Saved: {path}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
