import argparse
from functools import lru_cache

import gradio as gr
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}
ASPECT_RATIO_CHOICES = [f"{ratio} ({w}x{h})" for ratio, (w, h) in ASPECT_RATIOS.items()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for Qwen-Image offline inference.")
    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Diffusion model name or local path.")
    parser.add_argument(
        "--height",
        type=int,
        default=1328,
        help="Default image height (must match one of the supported presets).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1328,
        help="Default image width (must match one of the supported presets).",
    )
    parser.add_argument("--default-prompt", default="a cup of coffee on the table", help="Initial prompt shown in UI.")
    parser.add_argument("--default-seed", type=int, default=42, help="Initial seed shown in UI.")
    parser.add_argument("--default-cfg-scale", type=float, default=4.0, help="Initial CFG scale shown in UI.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Default number of denoising steps shown in the UI.",
    )
    parser.add_argument("--ip", default="127.0.0.1", help="Host/IP for Gradio `launch`.")
    parser.add_argument("--port", type=int, default=7862, help="Port for Gradio `launch`.")
    parser.add_argument("--share", action="store_true", help="Share the Gradio demo publicly.")
    args = parser.parse_args()
    args.aspect_ratio_label = next(
        (ratio for ratio, dims in ASPECT_RATIOS.items() if dims == (args.width, args.height)),
        None,
    )
    if args.aspect_ratio_label is None:
        supported = ", ".join(f"{ratio} ({w}x{h})" for ratio, (w, h) in ASPECT_RATIOS.items())
        parser.error(f"Unsupported resolution {args.width}x{args.height}. Please pick one of: {supported}.")
    return args


@lru_cache(maxsize=1)
def get_omni(model_name: str) -> Omni:
    # Enable VAE memory optimizations on NPU
    vae_use_slicing = current_omni_platform.is_npu()
    vae_use_tiling = current_omni_platform.is_npu()
    return Omni(
        model=model_name,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
    )


def build_demo(args: argparse.Namespace) -> gr.Blocks:
    omni = get_omni(args.model)

    def run_inference(
        prompt: str,
        seed_value: float,
        cfg_scale_value: float,
        resolution_choice: str,
        num_steps_value: float,
        num_images_choice: float,
    ):
        if not prompt or not prompt.strip():
            raise gr.Error("Please enter a non-empty prompt.")
        ratio_label = resolution_choice.split(" ", 1)[0]
        if ratio_label not in ASPECT_RATIOS:
            raise gr.Error(f"Unsupported aspect ratio: {ratio_label}")
        width, height = ASPECT_RATIOS[ratio_label]
        try:
            seed = int(seed_value)
            num_steps = int(num_steps_value)
            num_images = int(num_images_choice)
        except (TypeError, ValueError) as exc:
            raise gr.Error("Seed, inference steps, and number of images must be valid integers.") from exc
        if num_steps <= 0:
            raise gr.Error("Inference steps must be a positive integer.")
        if num_images not in {1, 2, 3, 4}:
            raise gr.Error("Number of images must be 1, 2, 3, or 4.")
        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
        outputs = omni.generate(
            prompt.strip(),
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                generator=generator,
                true_cfg_scale=float(cfg_scale_value),
                num_inference_steps=num_steps,
                num_outputs_per_prompt=num_images,
            ),
        )
        images_outputs = []
        for output in outputs:
            req_out = output.request_output[0]
            if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
                raise ValueError("Invalid request_output structure or missing 'images' key")
            images = req_out.images
            if not images:
                raise ValueError("No images found in request_output")
            # Extend the list with individual images (not append the entire list)
            images_outputs.extend(images)
            if len(images_outputs) >= num_images:
                break
        # Return only the requested number of images
        return images_outputs[:num_images]

    with gr.Blocks(
        title="vLLM-Omni Web Serving Demo",
        css="""
        /* Left column button width */
        .left-column button {
            width: 100%;
        }
        /* Right preview area: fixed height, hide unnecessary buttons */
        .fixed-image {
            height: 660px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .fixed-image .duplicate-button,
        .fixed-image .svelte-drgfj2 {
            display: none !important;
        }
        /* Gallery container: fill available space and center content */
        #image-gallery {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        /* Gallery grid: center horizontally and vertically, set gap */
        #image-gallery .grid {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            align-content: center;
            gap: 16px;
            width: 100%;
            height: 100%;
        }
        /* Gallery grid items: center content */
        #image-gallery .grid > div {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        /* Gallery images: limit max height, maintain aspect ratio */
        .fixed-image img {
            max-height: 660px !important;
            width: auto !important;
            object-fit: contain;
        }
        """,
    ) as demo:
        gr.Markdown("# vLLM-Omni Web Serving Demo")
        gr.Markdown(f"**Model:** {args.model}")

        with gr.Row():
            with gr.Column(scale=1, elem_classes="left-column"):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value=args.default_prompt,
                    placeholder="Describe the image you want to generate...",
                    lines=5,
                )
                seed_input = gr.Number(label="Seed", value=args.default_seed, precision=0)
                cfg_input = gr.Number(label="CFG Scale", value=args.default_cfg_scale)
                steps_input = gr.Number(
                    label="Inference Steps",
                    value=args.num_inference_steps,
                    precision=0,
                    minimum=1,
                )
                aspect_dropdown = gr.Dropdown(
                    label="Aspect Ratio (W:H)",
                    choices=ASPECT_RATIO_CHOICES,
                    value=f"{args.aspect_ratio_label} ({ASPECT_RATIOS[args.aspect_ratio_label][0]}x{ASPECT_RATIOS[args.aspect_ratio_label][1]})",
                )
                num_images = gr.Dropdown(
                    label="Number of images",
                    choices=["1", "2", "3", "4"],
                    value="1",
                )
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2, elem_classes="fixed-image"):
                gallery = gr.Gallery(
                    label="Preview",
                    columns=2,
                    rows=2,
                    height=660,
                    allow_preview=True,
                    show_label=True,
                    elem_id="image-gallery",
                )

        generate_btn.click(
            fn=run_inference,
            inputs=[prompt_input, seed_input, cfg_input, aspect_dropdown, steps_input, num_images],
            outputs=gallery,
        )

    return demo


def main():
    args = parse_args()
    demo = build_demo(args)
    demo.launch(server_name=args.ip, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
