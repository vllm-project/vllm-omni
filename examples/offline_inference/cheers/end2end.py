"""
Cheers end-to-end offline inference via vLLM-Omni.

Text-to-image:
  python end2end.py --model /path/to/Cheers --prompts "a photo of a bench"
  python end2end.py --model /path/to/Cheers --txt-prompts prompts.txt

Multi-prompt from file (one prompt per line):
  python end2end.py --model /path/to/Cheers --txt-prompts prompts.txt --output ./outputs
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Cheers T2I via vLLM-Omni pipeline")
    parser.add_argument(
        "--model",
        default="ai9stars/Cheers",
        help="HuggingFace model ID or local path to Cheers model directory.",
    )
    parser.add_argument("--prompts", nargs="+", default=None, help="Input text prompts.")
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./cheers_outputs",
        help="Output directory to save images.",
    )
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps.")
    parser.add_argument("--cfg", type=float, default=9.5, help="CFG scale.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Time-shift alpha.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.cheers.pipeline_cheers import (
        CheersGenerationPipeline,
    )
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    prompts: list[str] = []
    if args.txt_prompts:
        with open(args.txt_prompts, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
        print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")
    elif args.prompts:
        prompts = args.prompts
    else:
        prompts = ["A cute cat sitting on a windowsill"]
        print(f"[Info] No prompts provided, using default: {prompts}")

    od_config = OmniDiffusionConfig(model=args.model, revision=None)
    pipeline = CheersGenerationPipeline(od_config=od_config)
    pipeline._load_pretrained_weights(pipeline.model_path)

    for i, prompt in enumerate(prompts):
        sampling_params = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            seed=args.seed + i,
            extra_args={"cfg_scale": args.cfg, "alpha": args.alpha},
        )
        req = OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
        )

        output = pipeline.forward(req)
        save_path = os.path.join(args.output, f"output_{i}.png")
        output.output.save(save_path)
        print(f"[{i + 1}/{len(prompts)}] Saved: {save_path}  prompt: {prompt[:60]}")

    print("Done.")


if __name__ == "__main__":
    main()
