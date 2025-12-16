#!/usr/bin/env python3
"""
Qwen-Image-Edit Gradio Demo for online serving.

Usage:
    python gradio_demo.py [--server http://localhost:8092] [--port 7861]
"""

import argparse
import base64
from io import BytesIO

import gradio as gr
import requests
from PIL import Image


def edit_image(
    input_image: Image.Image,
    prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    negative_prompt: str,
    server_url: str,
) -> Image.Image | None:
    """Edit an image using the chat completions API."""
    if input_image is None:
        raise gr.Error("Please upload an image first")

    # Convert image to base64
    buffer = BytesIO()
    input_image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Build user message with text and image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }
    ]

    # Build extra_body with generation parameters
    extra_body = {
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
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
        raise gr.Error(f"Edit failed: {e}")


def create_demo(server_url: str):
    """Create Gradio demo interface."""

    with gr.Blocks(title="Qwen-Image-Edit Demo") as demo:
        gr.Markdown("# Qwen-Image-Edit Online Editing")
        gr.Markdown("Upload an image and describe the editing effect you want")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                )
                prompt = gr.Textbox(
                    label="Edit Instruction",
                    placeholder="Describe the editing effect you want...",
                    lines=2,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Describe what you don't want...",
                    lines=2,
                )

                with gr.Row():
                    steps = gr.Slider(
                        label="Inference Steps",
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (CFG)",
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="Random Seed (-1 for random)",
                        value=-1,
                        precision=0,
                    )

                edit_btn = gr.Button("Edit Image", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Edited Image",
                    type="pil",
                )

        # Examples
        gr.Examples(
            examples=[
                ["Convert this image to watercolor style"],
                ["Convert the image to black and white"],
                ["Enhance the color saturation"],
                ["Convert to cartoon style"],
                ["Add vintage filter effect"],
                ["Convert daytime to nighttime"],
                ["Convert to oil painting style"],
                ["Add dreamy blur effect"],
            ],
            inputs=[prompt],
        )

        def process_edit(img, p, st, g, se, n):
            actual_seed = se if se >= 0 else None
            return edit_image(img, p, st, g, actual_seed, n, server_url)

        edit_btn.click(
            fn=process_edit,
            inputs=[input_image, prompt, steps, guidance_scale, seed, negative_prompt],
            outputs=[output_image],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit Gradio Demo")
    parser.add_argument("--server", default="http://localhost:8092", help="Server URL")
    parser.add_argument("--port", type=int, default=7861, help="Gradio port")
    parser.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    print(f"Connecting to server: {args.server}")
    demo = create_demo(args.server)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
