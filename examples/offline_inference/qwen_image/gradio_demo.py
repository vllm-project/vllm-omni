import argparse
from functools import lru_cache

import gradio as gr
import torch

from vllm_omni.entrypoints.omni import Omni

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
        "--num_inference_steps",
        type=int,
        default=50,
        help="Default number of denoising steps shown in the UI.",
    )
    parser.add_argument("--ip", default="127.0.0.1", help="Host/IP for Gradio `launch`.")
    parser.add_argument("--port", type=int, default=7862, help="Port for Gradio `launch`.")
    parser.add_argument("--share", action="store_true", help="Share the Gradio demo publicly.")
    args = parser.parse_args()
    args.aspect_ratio_label = next(
        (
            ratio
            for ratio, dims in ASPECT_RATIOS.items()
            if dims == (args.width, args.height)
        ),
        None,
    )
    if args.aspect_ratio_label is None:
        supported = ", ".join(f"{ratio} ({w}x{h})" for ratio, (w, h) in ASPECT_RATIOS.items())
        parser.error(f"Unsupported resolution {args.width}x{args.height}. Please pick one of: {supported}.")
    return args


@lru_cache(maxsize=1)
def get_omni(model_name: str) -> Omni:
    return Omni(model=model_name)


def build_demo(args: argparse.Namespace) -> gr.Blocks:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    omni = get_omni(args.model)

    def run_inference(
        prompt: str,
        seed_value: float,
        cfg_scale_value: float,
        resolution_choice: str,
        num_steps_value: float,
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
        except (TypeError, ValueError) as exc:
            raise gr.Error("Seed and inference steps must be valid integers.") from exc
        if num_steps <= 0:
            raise gr.Error("Inference steps must be a positive integer.")
        generator = torch.Generator(device=device).manual_seed(seed)
        images = omni.generate(
            prompt.strip(),
            height=height,
            width=width,
            generator=generator,
            true_cfg_scale=float(cfg_scale_value),
            num_inference_steps=num_steps,
        )
        return images[0]

    with gr.Blocks(
        title="vLLM-Omni Online Serving Demo",
        css="""
        .control-panel {
            display: flex;
            gap: 12px;
            align-items: stretch;
        }
        .left-controls {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .left-controls .gradio-number,
        .left-controls .gradio-dropdown {
            margin-bottom: 0;
        }
        .prompt-panel textarea {
            min-height: 270px !important;
            height: 100%;
        }
        """,
    ) as demo:
        gr.Markdown("# vLLM-Omni Online Serving Demo")
        gr.Markdown(f"**Model:** {args.model}")

        with gr.Row(elem_classes="control-panel"):
            with gr.Column(scale=1, elem_classes="left-controls"):
                seed_input = gr.Number(label="Seed", value=args.default_seed, precision=0)
                cfg_input = gr.Number(label="CFG Scale", value=args.default_cfg_scale)
                steps_input = gr.Number(
                    label="Inference Steps",
                    value=args.num_inference_steps,
                    precision=0,
                    minimum=1,
                )
                aspect_dropdown = gr.Dropdown(
                    label="Aspect ratio (W:H)",
                    choices=ASPECT_RATIO_CHOICES,
                    value=f"{args.aspect_ratio_label} ({ASPECT_RATIOS[args.aspect_ratio_label][0]}x{ASPECT_RATIOS[args.aspect_ratio_label][1]})",
                )
            with gr.Column(scale=2, elem_classes="prompt-panel"):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value=args.default_prompt,
                    placeholder="Describe the image you want to generate...",
                    lines=6,
                )
        generate_btn = gr.Button("Generate", variant="primary")

        image_output = gr.Image(label="Generated Image", type="pil", show_download_button=True)

        generate_btn.click(
            fn=run_inference,
            inputs=[prompt_input, seed_input, cfg_input, aspect_dropdown, steps_input],
            outputs=image_output,
        )

    return demo


def main():
    args = parse_args()
    demo = build_demo(args)
    demo.launch(server_name=args.ip, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

