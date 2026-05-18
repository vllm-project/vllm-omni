"""
HunyuanImage-3.0-Instruct unified end-to-end inference script.
"""

import argparse
import json
import os
from pathlib import Path

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    MAX_IMAGES_PER_REQUEST,
    build_prompt_tokens,
    resolve_stop_token_ids,
    resolve_sys_type,
)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniPromptType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3.yaml")
_DEFAULT_AR_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_ar.yaml")

_MODALITY_TASK_MAP: dict[str, tuple[str, str | None]] = {
    "text2img": ("t2i", "think"),
    "img2img": ("it2i", "think"),
    "img2text": ("i2t", None),
    "text2text": ("t2t", None),
}

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


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0-Instruct end-to-end inference.")
    parser.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct", help="Model name or local path.")
    parser.add_argument(
        "--modality",
        default="text2img",
        choices=list(_MODALITY_TASK_MAP),
    )
    parser.add_argument("--prompts", nargs="+", default=None, help="Input text prompts.")
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Input image path(s) for img2img/img2text. Comma-separated for multi-image (up to 3).",
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory to save results.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument(
        "--bot-task",
        type=str,
        default=None,
        choices=["none", "think", "recaption", "think_recaption", "vanilla"],
        help="Override prompt mode. Default: auto from --modality.",
    )
    parser.add_argument("--sys-type", type=str, default=None, help="Override system prompt type.")
    parser.add_argument("--deploy-config", type=str, default=None, help="Custom deploy YAML path.")
    parser.add_argument("--stage-configs-path", type=str, default=None, help="Custom legacy stage config YAML path.")
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--init-timeout", type=int, default=300, help="Initialization timeout in seconds.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument(
        "--additional-config",
        type=str,
        default=None,
        help=(
            "JSON object forwarded to Omni/additional_config, for example "
            '\'{"torchair_graph_config":{"enabled":true}}\'. Different platforms may support different '
            "configs. Make sure the configs are valid for the platform you are using. "
            "Contents must be hashable."
        ),
    )

    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def parse_additional_config(raw_value: str | None) -> dict | None:
    """Parse a JSON string into an additional_config mapping."""
    if raw_value is None:
        return None

    try:
        additional_config = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --additional-config JSON: {exc}") from exc

    if additional_config is None:
        return None
    if not isinstance(additional_config, dict):
        raise ValueError(f"--additional-config must decode to a JSON object, got {type(additional_config).__name__}")
    return additional_config


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    additional_config = parse_additional_config(args.additional_config)

    task, default_bot_task = _MODALITY_TASK_MAP[args.modality]
    if args.bot_task is None:
        bot_task: str | None = default_bot_task
    elif args.bot_task == "none":
        bot_task = None
    else:
        bot_task = args.bot_task

    if args.deploy_config is not None and args.stage_configs_path is not None:
        raise ValueError("--deploy-config and --stage-configs-path are mutually exclusive.")

    deploy_config = args.deploy_config
    stage_configs_path = args.stage_configs_path
    if deploy_config is None and stage_configs_path is None:
        deploy_config = _MODALITY_DEFAULT_DEPLOY_CONFIG[args.modality]

    omni_kwargs = {
        "model": args.model,
        "vae_use_tiling": args.vae_use_tiling,
        "log_stats": args.log_stats,
        "init_timeout": args.init_timeout,
        "enforce_eager": args.enforce_eager,
        "mode": _MODALITY_MODE[args.modality],
    }

    if additional_config is not None:
        omni_kwargs["additional_config"] = additional_config
    if deploy_config is not None:
        omni_kwargs["deploy_config"] = deploy_config
    else:
        omni_kwargs["stage_configs_path"] = stage_configs_path

    omni = Omni(**omni_kwargs)

    prompts = args.prompts or ["A cute cat"]
    input_images: list = []
    if args.modality in ("img2img", "img2text"):
        if not args.image_path:
            raise ValueError(f"--image-path required for {args.modality}, got: {args.image_path}")
        from PIL import Image

        image_paths = [p.strip() for p in args.image_path.split(",") if p.strip()]
        if len(image_paths) > MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"--image-path accepts at most {MAX_IMAGES_PER_REQUEST} images for "
                f"HunyuanImage-3.0 IT2I, got {len(image_paths)}: {args.image_path}"
            )
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise ValueError(f"Image path does not exist: {image_path}")
            input_images.append(Image.open(image_path).convert("RGB"))
        if not input_images:
            raise ValueError(f"--image-path produced no usable paths: {args.image_path!r}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mm_image_payload = (input_images[0] if len(input_images) == 1 else input_images) if input_images else None

    formatted_prompts: list[OmniPromptType] = []
    for prompt in prompts:
        build_kwargs: dict = {"task": task, "bot_task": bot_task, "sys_type": args.sys_type}
        if input_images:
            build_kwargs["num_images"] = len(input_images)
        result = build_prompt_tokens(prompt, tokenizer, **build_kwargs)
        token_ids = result.token_ids
        effective_sys_type = args.sys_type or resolve_sys_type(bot_task)

        prompt_dict: dict = {
            "prompt_token_ids": token_ids,
            "prompt": prompt,
            "use_system_prompt": effective_sys_type,
        }
        if args.modality == "text2img":
            prompt_dict["modalities"] = ["image"]
        elif args.modality == "img2img":
            prompt_dict["modalities"] = ["image"]
            prompt_dict["multi_modal_data"] = {"image": mm_image_payload}
            prompt_dict["height"] = input_images[0].height
            prompt_dict["width"] = input_images[0].width
        elif args.modality == "img2text":
            prompt_dict["modalities"] = ["text"]
            prompt_dict["multi_modal_data"] = {"image": mm_image_payload}
        else:
            prompt_dict["modalities"] = ["text"]
        formatted_prompts.append(prompt_dict)

    params_list = list(omni.default_sampling_params_list)

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    ar_stop_token_ids = resolve_stop_token_ids(task=task, bot_task=bot_task, tokenizer=tokenizer)
    for sp in params_list:
        if isinstance(sp, OmniDiffusionSamplingParams):
            sp.num_inference_steps = args.steps
            sp.guidance_scale = args.guidance_scale
            sp.guidance_scale_provided = True
            if args.seed is not None:
                sp.seed = args.seed
            if args.modality == "text2img":
                sp.height = args.height
                sp.width = args.width
        elif hasattr(sp, "stop_token_ids"):
            sp.stop_token_ids = ar_stop_token_ids

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
    if additional_config is not None:
        print(f"  Additional config: {additional_config}")
    print(f"  Prompts: {prompts}")
    print(f"{'=' * 60}\n")

    omni_outputs = list(omni.generate(prompts=formatted_prompts, sampling_params_list=params_list))
    img_idx = 0
    for req_output in omni_outputs:
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
