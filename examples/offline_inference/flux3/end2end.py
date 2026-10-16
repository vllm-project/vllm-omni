# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLUX 3 Image Klein end-to-end offline example with optional image conditioning.

Provide the path to the FLUX 3 Image Klein model checkpoint::

    python end2end.py --model /path/to/flux-3-image-klein
    python end2end.py --model /path/to/flux-3-image-klein --prompt "restyle as watercolor" --image ref.jpg
    python end2end.py --model /path/to/flux-3-image-klein --variant fp8r
    python end2end.py --model /path/to/flux-3-image-klein --variant nvfp4

Variants must exist under the chosen checkpoint root's variants/ directory.
For the base model, use /path/to/flux-3-image-klein-base.

Conditioning inputs:
    --image  optional reference image; omit for text-to-image

LoRA:
    python end2end.py --model /path/to/flux-3-image-klein-base --lora-path /path/to/lora --lora-scale 1.0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import PIL.Image
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

DEFAULT_PROMPT = "a serene alpine meadow at golden hour, wildflowers in the foreground"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX 3 Image Klein offline generation with optional image conditioning.")
    p.add_argument("--model", required=True, help="Path to the FLUX 3 Image Klein checkpoint directory.")
    p.add_argument(
        "--variant",
        default=None,
        help=(
            "Folder name under <model>/variants, for example fp8r or nvfp4. "
            "Must match an existing folder exactly. Omit to load the BF16 root package."
        ),
    )
    p.add_argument("--image", nargs="+", default=None, help="Optional conditioning image paths, in reference order.")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--height", type=int, default=None, help="Output height in pixels.")
    p.add_argument("--width", type=int, default=None, help="Output width in pixels.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive text prompt.")
    p.add_argument("--negative-prompt", default="", help="Negative text prompt (empty string = unconditional).")
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Override the package FlowMatch scheduler shift.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fp8-attention",
        action="store_true",
        help="Use FlashInfer/cuDNN FP8 DiT attention, available on SM100.",
    )
    p.add_argument(
        "--cache-backend",
        default="none",
        choices=["none", "cache_dit", "tea_cache", "mag_cache"],
        help="Cache acceleration backend. Default: none.",
    )
    p.add_argument(
        "--cache-fn",
        type=int,
        default=1,
        help="cache-dit: number of transformer blocks that run fully per cached step (Fn). "
        "Remaining blocks replay cached residuals. Higher = better quality, less speedup. Default: 1.",
    )
    p.add_argument(
        "--cache-bn",
        type=int,
        default=0,
        help="cache-dit: additional blocks run at the end of each cached step (Bn). Default: 0.",
    )
    p.add_argument(
        "--cache-warmup",
        type=int,
        default=4,
        help="cache-dit: denoising steps that run fully before caching activates. Default: 4.",
    )
    p.add_argument(
        "--cache-threshold",
        type=float,
        default=0.08,
        help="cache-dit: residual diff threshold; lower = more selective caching, higher quality. Default: 0.08.",
    )
    p.add_argument("--cache-summary", action="store_true", help="Print cache-dit hit/miss statistics after generation.")
    p.add_argument(
        "--profile",
        action="store_true",
        help="Bracket each generate() call with cudaProfilerStart/Stop for nsys --capture-range cudaProfilerApi.",
    )
    p.add_argument(
        "--lora-path",
        default=None,
        help="Path to a LoRA adapter directory (PEFT format). Applied on top of the base model.",
    )
    p.add_argument("--lora-scale", type=float, default=1.0, help="LoRA adapter scale factor (default: 1.0).")
    return p.parse_args()


def _save_image(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, list):
        img = img[0]
    if isinstance(img, torch.Tensor):
        img = img.squeeze(0).permute(1, 2, 0).clamp(-1, 1)
        img = ((img * 0.5 + 0.5) * 255).to(torch.uint8).cpu().numpy()
        PIL.Image.fromarray(img).save(path)
    else:
        img.save(path)


def _load_image(path: str) -> PIL.Image.Image:
    return PIL.Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()

    lora_request: LoRARequest | None = None
    if args.lora_path:
        lora_request = LoRARequest(
            lora_name=Path(args.lora_path).stem,
            lora_int_id=stable_lora_int_id(args.lora_path),
            lora_path=args.lora_path,
        )

    additional_config: dict[str, str] = {}
    if args.variant is not None:
        additional_config["variant"] = args.variant

    attention_kwargs = {}
    if args.fp8_attention:
        attention_kwargs["diffusion_attention_config"] = {
            "default": {"backend": "FLASHINFER_ATTN", "quant": {"dtype_qk": "fp8_e4m3"}}
        }

    omni = Omni(
        model=args.model,
        additional_config=additional_config,
        cache_backend=args.cache_backend,
        cache_config={
            "Fn_compute_blocks": args.cache_fn,
            "Bn_compute_blocks": args.cache_bn,
            "max_warmup_steps": args.cache_warmup,
            "residual_diff_threshold": args.cache_threshold,
        },
        enable_cache_dit_summary=args.cache_summary,
        lora_path=args.lora_path,
        **attention_kwargs,
    )

    output_dir = Path(args.output_dir)

    prompt_dict: dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt
    if args.image is not None:
        prompt_dict["multi_modal_data"] = {"image": [_load_image(path) for path in args.image]}

    extra: dict = {}
    if args.flow_shift is not None:
        extra["flow_shift"] = args.flow_shift

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    t0 = time.perf_counter()
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        extra_args=extra,
    )
    if lora_request is not None:
        sampling_params.lora_request = lora_request
        sampling_params.lora_scale = args.lora_scale
    outputs = omni.generate(prompt_dict, sampling_params)
    elapsed = time.perf_counter() - t0
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    result = outputs[0]
    out_path = output_dir / "image.png"
    _save_image(result.images[0] if result.images else None, out_path)
    print(f"Saved {out_path} ({elapsed:.2f}s)")

    omni.close()


if __name__ == "__main__":
    main()
