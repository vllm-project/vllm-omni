"""
HunyuanImage-3.0-Instruct unified end-to-end inference script.

Supports all modalities through a single entry point:
  - text2img:  Text → AR → DiT → Image
  - img2img:   Text+Image → AR → DiT → Edited Image (IT2I)
  - img2text:  Image+Text → AR → Text description (I2T)
  - text2text: Text → AR → Text (comprehension, no image)

Usage:
    python end2end.py --modality text2img --prompts "A cute cat"
    python end2end.py --modality img2img --image-path input.png --prompts "Make it snowy"
    python end2end.py --modality img2text --image-path input.png --prompts "Describe this image"
"""

import argparse
import os
from pathlib import Path

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    _TASK_PRESETS,
    build_prompt_tokens,
    resolve_stop_token_ids,
)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniPromptType

# Default deploy configs are absolute so this example works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3.yaml")
_DEFAULT_AR_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_ar.yaml")

_MODALITY_DEFAULT_DEPLOY_CONFIG = {
    "text2img": _DEFAULT_DEPLOY_CONFIG,
    "img2img": _DEFAULT_DEPLOY_CONFIG,
    "img2text": _DEFAULT_AR_DEPLOY_CONFIG,
    "text2text": _DEFAULT_AR_DEPLOY_CONFIG,
}

_MODALITY_MODE = {
    "text2img": "text-to-image",
    "img2img": "image-editing",
    "img2text": "image-to-text",
    "text2text": "text-to-text",
}

_MODALITY_TASK_MAP = {
    "text2img": "t2i",
    "img2img": "it2i",
    "img2text": "i2t",
    "text2text": "t2t",
}


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0-Instruct end-to-end inference.")
    parser.add_argument(
        "--model",
        default="tencent/HunyuanImage-3.0-Instruct",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--modality",
        default="text2img",
        choices=["text2img", "img2img", "img2text", "text2text"],
        help="Modality mode to control stage execution.",
    )
    parser.add_argument("--prompts", nargs="+", default=None, help="Input text prompts.")
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to input image (for img2img/img2text).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory to save results.",
    )

    # Generation parameters
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization.",
    )

    # Prompt configuration
    parser.add_argument(
        "--bot-task",
        type=str,
        default="auto",
        choices=["auto", "think", "recaption", "think_recaption", "vanilla"],
        help=(
            "Prompt behavior. 'auto' selects the default for the modality; "
            "'think' adds <think>; 'recaption' adds <recaption>; "
            "'vanilla' uses the t2i pretrain template."
        ),
    )
    parser.add_argument(
        "--sys-type",
        type=str,
        default=None,
        help="Override system prompt type (e.g. en_unified, en_vanilla).",
    )

    # Omni init args
    parser.add_argument("--deploy-config", type=str, default=None, help="Custom deploy YAML path.")
    parser.add_argument("--stage-configs-path", type=str, default=None, help="Custom legacy stage config YAML path.")
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--init-timeout", type=int, default=300, help="Initialization timeout in seconds.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")

    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Determine task for prompt formatting from modality + bot behavior.
    task = _MODALITY_TASK_MAP[args.modality]
    assert task is not None
    bot_task = args.bot_task
    if bot_task != "auto":
        task = task + "_" + bot_task
    if task not in _TASK_PRESETS:
        valid_bot_tasks = {
            "text2img": ["think", "recaption", "vanilla"],
            "img2img": ["think", "recaption"],
            "img2text": ["auto"],
            "text2text": ["auto"],
        }[args.modality]
        raise ValueError(
            f"--bot-task {bot_task!r} is not supported for {args.modality}. Choose from: {valid_bot_tasks}"
        )

    if args.deploy_config is not None and args.stage_configs_path is not None:
        raise ValueError("--deploy-config and --stage-configs-path are mutually exclusive.")

    deploy_config = args.deploy_config
    stage_configs_path = args.stage_configs_path
    if deploy_config is None and stage_configs_path is None:
        deploy_config = _MODALITY_DEFAULT_DEPLOY_CONFIG[args.modality]

    # Build Omni
    omni_kwargs = {
        "model": args.model,
        "vae_use_tiling": args.vae_use_tiling,
        "log_stats": args.log_stats,
        "init_timeout": args.init_timeout,
        "enforce_eager": args.enforce_eager,
    }
    if deploy_config is not None:
        omni_kwargs["deploy_config"] = deploy_config
    else:
        omni_kwargs["stage_configs_path"] = stage_configs_path
    omni_kwargs["mode"] = _MODALITY_MODE[args.modality]

    omni = Omni(**omni_kwargs)

    # Prepare prompts
    prompts = args.prompts or ["A cute cat"]
    if not prompts:
        print("[Info] No prompts provided, using default.")
        prompts = ["A cute cat"]

    # Load image if needed
    input_image = None
    if args.modality in ("img2img", "img2text"):
        if not args.image_path or not os.path.exists(args.image_path):
            raise ValueError(f"--image-path required for {args.modality}, got: {args.image_path}")
        from PIL import Image

        input_image = Image.open(args.image_path).convert("RGB")

    # Load tokenizer for segment-wise prompt tokenization (matches HF
    # apply_chat_template byte-for-byte; see build_prompt_tokens docstring).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Format prompts
    formatted_prompts: list[OmniPromptType] = []
    for p in prompts:
        result = build_prompt_tokens(p, tokenizer, task=task, sys_type=args.sys_type)
        token_ids = result.token_ids
        effective_sys_type = result.system_prompt_type

        # `prompt_token_ids` drives the AR stage (matches HF byte-for-byte).
        # `prompt` and `use_system_prompt` are forwarded by ar2diffusion to
        # the DiT stage so the diffusion pipeline can rebuild the same
        # system prefix when constructing its model inputs.
        prompt_dict: dict = {
            "prompt_token_ids": token_ids,
            "prompt": p,
            "use_system_prompt": effective_sys_type,
        }

        if args.modality == "text2img":
            prompt_dict["modalities"] = ["image"]
        elif args.modality == "img2img":
            prompt_dict["modalities"] = ["image"]
            prompt_dict["multi_modal_data"] = {"image": input_image}
            prompt_dict["height"] = input_image.height
            prompt_dict["width"] = input_image.width
        elif args.modality == "img2text":
            prompt_dict["modalities"] = ["text"]
            prompt_dict["multi_modal_data"] = {"image": input_image}
        elif args.modality == "text2text":
            prompt_dict["modalities"] = ["text"]

        formatted_prompts.append(prompt_dict)

    # Build sampling params from defaults
    params_list = list(omni.default_sampling_params_list)

    # Override diffusion params if applicable
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    ar_stop_token_ids = resolve_stop_token_ids(task=task, bot_task=bot_task, tokenizer=tokenizer)
    assert ar_stop_token_ids is not None
    for sp in params_list:
        if isinstance(sp, OmniDiffusionSamplingParams):
            sp.num_inference_steps = args.steps
            sp.guidance_scale = args.guidance_scale
            sp.guidance_scale_provided = True
            if args.seed is not None:
                sp.seed = args.seed
            if args.modality in ("text2img",):
                sp.height = args.height
                sp.width = args.width
        elif hasattr(sp, "stop_token_ids"):
            sp.stop_token_ids = ar_stop_token_ids

    # Print configuration
    print(f"\n{'=' * 60}")
    print("HunyuanImage-3.0 Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Modality: {args.modality}")
    print(f"  Prompt task: {task}")
    print(f"  Bot task: {bot_task}")
    if deploy_config is not None:
        print(f"  Deploy config: {deploy_config}")
    else:
        print(f"  Stage config: {stage_configs_path}")
    print(f"  Num stages: {omni.num_stages}")
    if args.modality in ("text2img", "img2img"):
        print(f"  Inference steps: {args.steps}")
        print(f"  Guidance scale: {args.guidance_scale}")
        print(f"  Seed: {args.seed}")
    if args.modality == "text2img":
        print(f"  Output size: {args.width}x{args.height}")
    if args.image_path:
        print(f"  Input image: {args.image_path}")
    print(f"  Prompts: {prompts}")
    print(f"{'=' * 60}\n")

    # Generate
    omni_outputs = list(omni.generate(prompts=formatted_prompts, sampling_params_list=params_list))

    # Process outputs
    img_idx = 0
    for req_output in omni_outputs:
        # Text output (AR stage or text-only)
        ro = getattr(req_output, "request_output", None)
        txt = ""
        if ro and getattr(ro, "outputs", None):
            txt = "".join(getattr(o, "text", "") or "" for o in ro.outputs)
        if not txt:
            ar_text = getattr(req_output, "custom_output", {}).get("ar_generated_text")
            if isinstance(ar_text, list):
                txt = "\n".join(text for text in ar_text if text)
            else:
                txt = ar_text or ""
        if txt:
            print(f"[Output] Text:\n{txt}")

        # Image output (DiT stage)
        images = getattr(req_output, "images", None)
        if not images and ro and hasattr(ro, "images"):
            images = ro.images

        if images:
            for j, img in enumerate(images):
                save_path = os.path.join(args.output, f"output_{img_idx}_{j}.png")
                img.save(save_path)
                print(f"[Output] Saved image to {save_path}")
            img_idx += 1


if __name__ == "__main__":
    main()
