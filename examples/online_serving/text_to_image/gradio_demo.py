#!/usr/bin/env python3
"""
Qwen-Image Gradio Demo for online serving.

Usage:
    python gradio_demo.py [--server http://localhost:8091] [--port 7860]
"""

import argparse
import base64
from io import BytesIO

import gradio as gr
import requests
from PIL import Image


def generate_image(
    prompt: str,
    height: int,
    width: int,
    steps: int,
    cfg_scale: float,
    seed: int | None,
    negative_prompt: str,
    server_url: str,
) -> Image.Image | None:
    """Generate an image using the chat completions API."""
    messages = [{"role": "user", "content": prompt}]

    # Build extra_body with generation parameters
    extra_body = {
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "true_cfg_scale": cfg_scale,
    }
    if seed is not None and seed >= 0:
        extra_body["seed"] = seed
    if negative_prompt:
        extra_body["negative_prompt"] = negative_prompt

    # Build request payload
    payload = {"messages": messages, "extra_body": extra_body}

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list) and len(content) > 0:
            image_url = content[0].get("image_url", {}).get("url", "")
            if image_url.startswith("data:image"):
                _, b64_data = image_url.split(",", 1)
                image_bytes = base64.b64decode(b64_data)
                return Image.open(BytesIO(image_bytes))

        return None

    except Exception as e:
        print(f"Error: {e}")
        raise gr.Error(f"Generation failed: {e}")


def create_demo(server_url: str):
    """Create Gradio demo interface."""

    with gr.Blocks(title="Qwen-Image Demo") as demo:
        gr.Markdown("# Qwen-Image Online Generation")
        gr.Markdown("Generate images using Qwen-Image model")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Describe what you don't want...",
                    lines=2,
                )

                with gr.Row():
                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=64,
                    )
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=64,
                    )

                with gr.Row():
                    steps = gr.Slider(
                        label="Inference Steps",
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                    )
                    cfg_scale = gr.Slider(
                        label="True CFG Scale",
                        minimum=1.0,
                        maximum=20.0,
                        value=4.0,
                        step=0.5,
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="Random Seed (-1 for random)",
                        value=-1,
                        precision=0,
                    )

                generate_btn = gr.Button("Generate Image", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                )

        # Examples
        gr.Examples(
            examples=[
                ["A beautiful landscape painting with misty mountains", "", 1024, 1024, 50, 4.0, 42],
                ["A cute cat sitting on a windowsill with sunlight", "", 1024, 1024, 50, 4.0, 123],
                ["Cyberpunk style futuristic city with neon lights", "blurry, low quality", 1024, 768, 50, 4.0, 456],
                ["Chinese ink painting of bamboo forest with a house", "", 768, 1024, 50, 4.0, 789],
            ],
            inputs=[prompt, negative_prompt, height, width, steps, cfg_scale, seed],
        )

        generate_btn.click(
            fn=lambda p, h, w, st, c, se, n: generate_image(p, h, w, st, c, se if se >= 0 else None, n, server_url),
            inputs=[prompt, height, width, steps, cfg_scale, seed, negative_prompt],
            outputs=[output_image],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image Gradio Demo")
    parser.add_argument("--server", default="http://localhost:8091", help="Server URL")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    print(f"Connecting to server: {args.server}")
    demo = create_demo(args.server)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
