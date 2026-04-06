import argparse
import os

from vllm_omni.inputs.data import OmniPromptType
from vllm_omni.model_executor.stage_input_processors.bagel import GEN_THINK_SYSTEM_PROMPT


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
    parser.add_argument("--prompt-type", default="text", choices=["text"])

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

    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory to save images.",
    )

    # OmniLLM init args
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    parser.add_argument("--worker-backend", type=str, default="process", choices=["process", "ray"])
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--stage-configs-path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")

    parser.add_argument("--cfg-text-scale", type=float, default=4.0, help="Text CFG scale (default: 4.0)")
    parser.add_argument("--cfg-img-scale", type=float, default=1.5, help="Image CFG scale (default: 1.5)")
    parser.add_argument(
        "--negative-prompt", type=str, default=None, help="Negative prompt for CFG (default: empty prompt)"
    )
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="CFG parallel size: 1=batched (single GPU), 2=parallel with 2 branches (text CFG only), 3=parallel (3 GPUs).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation.")
    parser.add_argument(
        "--cfg-interval",
        type=float,
        nargs=2,
        default=None,
        help="CFG interval [start, end] (default: pipeline default)",
    )
    parser.add_argument(
        "--cfg-renorm-type", type=str, default=None, help="CFG renorm type: global, text_channel, channel"
    )
    parser.add_argument("--cfg-renorm-min", type=float, default=None, help="CFG renorm min")
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g. 'fp8').",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="Enable thinking mode: AR stage decodes <think>...</think> planning tokens before image generation.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
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
        prompts = ["A cute cat"]
        print(f"[Info] No prompts provided, using default: {prompts}")
    omni_outputs = []

    from PIL import Image

    from vllm_omni.entrypoints.omni import Omni

    omni_kwargs = {}
    stage_configs_path = args.stage_configs_path
    if args.think and stage_configs_path is None:
        stage_configs_path = "vllm_omni/model_executor/stage_configs/bagel_think.yaml"
        print(f"[Info] Think mode enabled, using stage config: {stage_configs_path}")
    if stage_configs_path:
        omni_kwargs["stage_configs_path"] = stage_configs_path

    omni_kwargs.update(
        {
            "log_stats": args.log_stats,
            "init_sleep_seconds": args.init_sleep_seconds,
            "batch_timeout": args.batch_timeout,
            "init_timeout": args.init_timeout,
            "shm_threshold_bytes": args.shm_threshold_bytes,
            "worker_backend": args.worker_backend,
            "ray_address": args.ray_address,
            "enable_diffusion_pipeline_profiler": args.enable_diffusion_pipeline_profiler,
        }
    )
    if args.quantization:
        omni_kwargs["quantization_config"] = args.quantization

    omni = Omni(model=model_name, **omni_kwargs)

    formatted_prompts = []
    for p in prompts:
        if args.modality == "img2img":
            if not args.image_path or not os.path.exists(args.image_path):
                raise ValueError(f"img2img requires --image-path pointing to an existing file, got: {args.image_path}")
            loaded_image = Image.open(args.image_path).convert("RGB")
            think_prefix = f"<|im_start|>{GEN_THINK_SYSTEM_PROMPT}<|im_end|>" if args.think else ""
            final_prompt_text = f"{think_prefix}<|fim_middle|><|im_start|>{p}<|im_end|>"
            prompt_dict = {
                "prompt": final_prompt_text,
                "multi_modal_data": {"img2img": loaded_image},
                "modalities": ["img2img"],
            }
            if args.negative_prompt is not None:
                prompt_dict["negative_prompt"] = args.negative_prompt
            formatted_prompts.append(prompt_dict)
        elif args.modality == "img2text":
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
            think_prefix = f"<|im_start|>{GEN_THINK_SYSTEM_PROMPT}<|im_end|>" if args.think else ""
            final_prompt_text = f"{think_prefix}<|im_start|>{p}<|im_end|>"
            prompt_dict = {"prompt": final_prompt_text, "modalities": ["image"]}
            if args.negative_prompt is not None:
                prompt_dict["negative_prompt"] = args.negative_prompt
            formatted_prompts.append(prompt_dict)

    params_list = omni.default_sampling_params_list
    if args.modality in ("text2img", "img2img"):
        if len(params_list) > 1:
            diffusion_params = params_list[1]
            diffusion_params.num_inference_steps = args.steps  # type: ignore
            diffusion_params.cfg_parallel_size = args.cfg_parallel_size  # type: ignore
            if args.seed is not None:
                diffusion_params.seed = args.seed  # type: ignore
            extra = {
                "cfg_text_scale": args.cfg_text_scale,
                "cfg_img_scale": args.cfg_img_scale,
            }
            if args.cfg_interval is not None:
                extra["cfg_interval"] = tuple(args.cfg_interval)
            if args.cfg_renorm_type is not None:
                extra["cfg_renorm_type"] = args.cfg_renorm_type
            if args.cfg_renorm_min is not None:
                extra["cfg_renorm_min"] = args.cfg_renorm_min
            if args.negative_prompt is not None:
                extra["negative_prompt"] = args.negative_prompt
            diffusion_params.extra_args = extra  # type: ignore

    omni_outputs = list(omni.generate(prompts=formatted_prompts, sampling_params_list=params_list))

    img_idx = 0
    for req_output in omni_outputs:
        if args.think:
            text_output = getattr(req_output, "text", None) or getattr(req_output, "outputs", None)
            if text_output:
                if isinstance(text_output, list) and text_output:
                    for out in text_output:
                        txt = getattr(out, "text", str(out))
                        if txt:
                            print(f"[Think] {txt}")
                elif isinstance(text_output, str):
                    print(f"[Think] {text_output}")

        images = getattr(req_output, "images", None)

        if not images:
            continue

        for j, img in enumerate(images):
            save_path = os.path.join(args.output, f"output_{img_idx}_{j}.png")
            img.save(save_path)
            print(f"[Output] Saved image to {save_path}")
        img_idx += 1

    print(omni_outputs)


if __name__ == "__main__":
    main()
