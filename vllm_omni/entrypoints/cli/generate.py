# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate subcommand for vLLM-Omni.

Provides a simple CLI for text-to-image generation with diffusion models.
Currently only supports diffusion image generation; other modalities
(image edit, video, TTS) are deferred to future PRs.

Example:
  vllm generate --omni --model Qwen/Qwen-Image \\
    --prompt "a cup of coffee on the table" --output output.png
"""

import argparse
from pathlib import Path

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

DESCRIPTION = """Generate images from text prompts using diffusion models.

Currently supports text-to-image generation only.
Other parallel knobs (ulysses, ring, cfg_parallel, vae_patch_parallel)
are available in the full offline example script.

Examples:
  vllm generate --omni --model Qwen/Qwen-Image \\
    --prompt "a cup of coffee on the table" --output output.png

  vllm generate --omni --model Qwen/Qwen-Image \\
    --prompt "a sunset over the ocean" --height 512 --width 512 \\
    --num-inference-steps 30 --seed 123
"""


class OmniGenerateCommand(CLISubcommand):
    """The `generate` subcommand for the vLLM-Omni CLI."""

    name = "generate"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        import torch

        from vllm_omni.diffusion.data import DiffusionParallelConfig
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.platforms import current_omni_platform

        # 1. torch.Generator for reproducible generation
        generator = torch.Generator(
            device=current_omni_platform.device_type
        ).manual_seed(args.seed)

        # 2. Parallel config (only tensor_parallel_size exposed in MVP)
        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=args.tensor_parallel_size,
        )

        # 3. Initialize Omni
        omni_kwargs: dict = {
            "model": args.model,
            "parallel_config": parallel_config,
            "enforce_eager": args.enforce_eager,
            "enable_cpu_offload": args.enable_cpu_offload,
            "mode": "text-to-image",
        }
        if args.stage_configs_path:
            omni_kwargs["stage_configs_path"] = args.stage_configs_path
        omni = Omni(**omni_kwargs)

        # 4. Generate
        outputs = omni.generate(
            {"prompt": args.prompt, "negative_prompt": args.negative_prompt},
            OmniDiffusionSamplingParams(
                height=args.height,
                width=args.width,
                generator=generator,
                true_cfg_scale=args.cfg_scale,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_outputs_per_prompt=args.num_images,
            ),
        )

        # 5. Validate output
        if not outputs:
            raise RuntimeError("No output generated.")
        req_out = outputs[0].request_output
        if not hasattr(req_out, "images") or not req_out.images:
            raise RuntimeError("No images in output.")
        images = req_out.images

        # 6. Save images
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix or ".png"
        if len(images) == 1:
            save_path = output_path if output_path.suffix else output_path.with_suffix(suffix)
            images[0].save(save_path)
            print(f"Saved image to {save_path}")
        else:
            stem = output_path.stem
            for i, img in enumerate(images):
                save_path = output_path.parent / f"{stem}_{i}{suffix}"
                img.save(save_path)
                print(f"Saved image to {save_path}")

    def validate(self, args: argparse.Namespace) -> None:
        from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model

        if not is_diffusion_model(args.model):
            raise ValueError(
                f"'{args.model}' is not a diffusion model. "
                "'vllm generate' currently only supports text-to-image diffusion models."
            )

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            description=DESCRIPTION,
            usage="vllm generate --omni --model MODEL --prompt PROMPT [options]",
        )

        # Accept --omni so argparse doesn't reject it
        # (actual dispatch happens in main.py before subcommand parsing)
        parser.add_argument(
            "--omni",
            action="store_true",
            help=argparse.SUPPRESS,
        )

        # Required
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Diffusion model name or path.",
        )
        parser.add_argument(
            "--prompt",
            type=str,
            required=True,
            help="Text prompt for image generation.",
        )

        # Output
        parser.add_argument(
            "--output",
            type=str,
            default="output.png",
            help="Output image path (default: output.png).",
        )

        # Generation parameters
        parser.add_argument(
            "--height",
            type=int,
            default=1024,
            help="Image height (default: 1024).",
        )
        parser.add_argument(
            "--width",
            type=int,
            default=1024,
            help="Image width (default: 1024).",
        )
        parser.add_argument(
            "--num-inference-steps",
            type=int,
            default=50,
            help="Number of denoising steps (default: 50).",
        )
        parser.add_argument(
            "--guidance-scale",
            type=float,
            default=4.0,
            help="CFG guidance scale (default: 4.0).",
        )
        parser.add_argument(
            "--cfg-scale",
            type=float,
            default=4.0,
            help="True CFG scale for Qwen-Image (default: 4.0).",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed (default: 42).",
        )
        parser.add_argument(
            "--num-images",
            type=int,
            default=1,
            help="Number of images to generate (default: 1).",
        )
        parser.add_argument(
            "--negative-prompt",
            type=str,
            default=None,
            help="Negative prompt for CFG.",
        )

        # Hardware / loading
        parser.add_argument(
            "--tensor-parallel-size",
            type=int,
            default=1,
            help="Tensor parallelism size (default: 1).",
        )
        parser.add_argument(
            "--stage-configs-path",
            type=str,
            default=None,
            help="Path to stage config YAML.",
        )
        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            help="Disable torch.compile.",
        )
        parser.add_argument(
            "--enable-cpu-offload",
            action="store_true",
            help="Enable CPU offloading.",
        )

        return parser


def cmd_init() -> list[CLISubcommand]:
    return [OmniGenerateCommand()]
