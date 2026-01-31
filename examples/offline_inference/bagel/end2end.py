import argparse
import os
from typing import cast

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="ByteDance-Seed/BAGEL-7B-MoT",
        help="Path to merged model directory.",
    )
    parser.add_argument("--prompts", nargs="+", default=None, help="Input text prompts.")
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line (preferred).",
    )
    parser.add_argument("--prompt_type", default="text", choices=["text"])

    parser.add_argument(
        "--modality",
        default="text2img",
        choices=["text2img", "img2img", "img2text", "text2text"],
        help="Modality mode to control stage execution.",
    )

    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to input image for img2img.",
    )

    # OmniLLM init args
    parser.add_argument("--enable-stats", action="store_true", default=False)
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    parser.add_argument("--worker-backend", type=str, default="process", choices=["process", "ray"])
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--stage-configs-path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.model
    prompts: list[OmniPromptType] = []
    try:
        # Preferred: load from txt file (one prompt per line)
        if getattr(args, "txt_prompts", None) and args.prompt_type == "text":
            with open(args.txt_prompts, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
            prompts = [ln for ln in lines if ln != ""]
            print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")
        else:
            prompts = args.prompts
    except Exception as e:
        print(f"[Error] Failed to load prompts: {e}")
        raise

    if not prompts:
        # Default prompt for text2img test if none provided
        prompts = ["<|im_start|>A cute cat<|im_end|>"]
        print(f"[Info] No prompts provided, using default: {prompts}")
    omni_outputs = []

    from PIL import Image

    if args.modality == "img2img":
        from PIL import Image

        from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion

        print("[Info] Running in img2img mode (Stage 1 only)")
        client = OmniDiffusion(model=model_name)

        if args.image_path:
            if os.path.exists(args.image_path):
                loaded_image = Image.open(args.image_path).convert("RGB")
                prompts = [
                    {
                        "prompt": cast(str, p),
                        "multi_modal_data": {"image": loaded_image},
                    }
                    for p in prompts
                ]
            else:
                print(f"[Warning] Image path {args.image_path} does not exist.")

        result = client.generate(
            prompts,
            OmniDiffusionSamplingParams(
                seed=52,
                need_kv_receive=False,
                num_inference_steps=args.steps,
            ),
        )

        # Ensure result is a list for iteration
        if not isinstance(result, list):
            omni_outputs = [result]
        else:
            omni_outputs = result

    else:
        from vllm_omni.entrypoints.omni import Omni

        omni_kwargs = {}
        if args.stage_configs_path:
            omni_kwargs["stage_configs_path"] = args.stage_configs_path

        omni_kwargs.update(
            {
                "log_stats": args.enable_stats,
                "init_sleep_seconds": args.init_sleep_seconds,
                "batch_timeout": args.batch_timeout,
                "init_timeout": args.init_timeout,
                "shm_threshold_bytes": args.shm_threshold_bytes,
                "worker_backend": args.worker_backend,
                "ray_address": args.ray_address,
            }
        )

        omni = Omni(model=model_name, **omni_kwargs)

        formatted_prompts = []
        for p in args.prompts:
            if args.modality == "img2text":
                if args.image_path:
                    loaded_image = Image.open(args.image_path).convert("RGB")
                    final_prompt_text = f"<|im_start|>user\n<|image_pad|>\n{p}<|im_end|>\n<|im_start|>assistant\n"
                    prompt_dict = {
                        "prompt": final_prompt_text,
                        "multi_modal_data": {"image": loaded_image},
                        "modalities": ["text"],
                    }
                    formatted_prompts.append(prompt_dict)
            elif args.modality == "text2text":
                final_prompt_text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
                prompt_dict = {"prompt": final_prompt_text, "modalities": ["text"]}
                formatted_prompts.append(prompt_dict)
            else:
                # text2img
                final_prompt_text = f"<|im_start|>{p}<|im_end|>"
                prompt_dict = {"prompt": final_prompt_text, "modalities": ["image"]}
                formatted_prompts.append(prompt_dict)

        params_list = omni.default_sampling_params_list
        if args.modality == "text2img":
            params_list[0].max_tokens = 1  # type: ignore # The first stage is a SamplingParam (vllm)
            if len(params_list) > 1:
                params_list[1].num_inference_steps = args.steps  # type: ignore # The second stage is an OmniDiffusionSamplingParam

        omni_outputs = list(omni.generate(prompts=formatted_prompts, sampling_params_list=params_list))

    for i, req_output in enumerate(omni_outputs):
        images = getattr(req_output, "images", None)
        if not images and hasattr(req_output, "output"):
            if isinstance(req_output.output, list):
                images = req_output.output
            else:
                images = [req_output.output]

        if images:
            for j, img in enumerate(images):
                img.save(f"output_{i}_{j}.png")

        if hasattr(req_output, "request_output") and req_output.request_output:
            for stage_out in req_output.request_output:
                if hasattr(stage_out, "images") and stage_out.images:
                    for k, img in enumerate(stage_out.images):
                        save_path = f"output_{i}_stage_{getattr(stage_out, 'stage_id', '?')}_{k}.png"
                        img.save(save_path)
                        print(f"[Info] Saved stage output image to {save_path}")

    print(omni_outputs)


if __name__ == "__main__":
    main()
