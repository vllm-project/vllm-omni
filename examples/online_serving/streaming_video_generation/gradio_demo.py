#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gradio demo for `/v1/realtime/video`.

Usage:
    python gradio_demo.py --host 127.0.0.1 --port 7860

Start a vLLM-Omni server with streaming output enabled first, for example:
    vllm serve BestWishYsh/Helios-Distilled --omni --diffusion-streaming-output --port 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError:
    raise ImportError("gradio is required to run this demo. Install it with: pip install 'vllm-omni[demo]'") from None

DEFAULT_PROMPT = "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement."
VIDEO_STREAM_VIEW_HTML = Path(__file__).with_name("video-stream-view.html")
VIDEO_STREAM_VIEW_JS = Path(__file__).with_name("video-stream-view.js")


def _maybe_set(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    payload[key] = value


def _optional_int(value: int | float | None) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: int | float | None) -> float | None:
    return None if value is None else float(value)


def _build_session_start(
    *,
    model: str,
    prompt: str,
    negative_prompt: str,
    width: int | float | None,
    height: int | float | None,
    fps: int | float | None,
    num_frames: int | float | None,
    num_inference_steps: int | float | None,
    guidance_scale: int | float | None,
    guidance_scale_2: int | float | None,
    boundary_ratio: int | float | None,
    flow_shift: int | float | None,
    true_cfg_scale: int | float | None,
    seed: int | float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "session.start",
        "model": model,
        "prompt": prompt,
        "format": "m4s",
    }

    _maybe_set(payload, "negative_prompt", negative_prompt)
    _maybe_set(payload, "width", _optional_int(width))
    _maybe_set(payload, "height", _optional_int(height))
    _maybe_set(payload, "fps", _optional_int(fps))
    _maybe_set(payload, "num_frames", _optional_int(num_frames))
    _maybe_set(payload, "num_inference_steps", _optional_int(num_inference_steps))
    _maybe_set(payload, "guidance_scale", _optional_float(guidance_scale))
    _maybe_set(payload, "guidance_scale_2", _optional_float(guidance_scale_2))
    _maybe_set(payload, "boundary_ratio", _optional_float(boundary_ratio))
    _maybe_set(payload, "flow_shift", _optional_float(flow_shift))
    _maybe_set(payload, "true_cfg_scale", _optional_float(true_cfg_scale))
    _maybe_set(payload, "seed", _optional_int(seed))

    if "Helios" in model:
        payload["extra_params"] = {
            "is_enable_stage2": True,
            "pyramid_num_stages": 3,
            "pyramid_num_inference_steps_list": [1, 1, 1],
            "is_amplify_first_chunk": True,
        }

    return payload


def _set_optional_field_enabled(enabled: bool) -> Any:
    return gr.update(interactive=enabled)


def build_browser_stream_config(
    host: str,
    port: int | float,
    model: str,
    prompt: str,
    negative_prompt: str,
    width: int | float | None,
    height: int | float | None,
    fps: int | float,
    num_frames: int | float | None,
    num_inference_steps: int | float | None,
    include_guidance_scale: bool,
    guidance_scale: int | float | None,
    include_guidance_scale_2: bool,
    guidance_scale_2: int | float | None,
    include_seed: bool,
    seed: int | float | None,
    include_boundary_ratio: bool,
    boundary_ratio: int | float | None,
    include_flow_shift: bool,
    flow_shift: int | float | None,
    include_true_cfg_scale: bool,
    true_cfg_scale: int | float | None,
) -> tuple[str, Any]:
    """Build the browser-side WebSocket request config for the HTML/MSE player."""
    if not model.strip():
        raise gr.Error("Model is required.")
    if not prompt.strip():
        raise gr.Error("Prompt is required.")

    payload = _build_session_start(
        model=model.strip(),
        prompt=prompt.strip(),
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale if include_guidance_scale else None,
        guidance_scale_2=guidance_scale_2 if include_guidance_scale_2 else None,
        boundary_ratio=boundary_ratio if include_boundary_ratio else None,
        flow_shift=flow_shift if include_flow_shift else None,
        true_cfg_scale=true_cfg_scale if include_true_cfg_scale else None,
        seed=seed if include_seed else None,
    )
    config = {
        "url": f"ws://{host}:{int(port)}/v1/realtime/video",
        "payload": payload,
    }
    return json.dumps(config, ensure_ascii=False), gr.update(value="Streaming...", interactive=False)


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Streaming Video Generation") as demo:
        gr.Markdown("# Streaming Video Generation")
        gr.Markdown(
            "Connects to `WS /v1/realtime/video` in the browser and appends fMP4 chunks "
            "directly to a Media Source Extensions video player."
        )

        with gr.Row():
            with gr.Column(scale=1):
                host = gr.Textbox(label="Server Host", value="127.0.0.1")
                port = gr.Number(label="Server Port", value=8000, precision=0)
                model = gr.Textbox(label="Model", value="BestWishYsh/Helios-Distilled")
                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=3)
                negative_prompt = gr.Textbox(label="Negative Prompt", value="", lines=2)

                with gr.Row():
                    width = gr.Number(label="Width", value=640, precision=0)
                    height = gr.Number(label="Height", value=384, precision=0)

                with gr.Row():
                    fps = gr.Number(label="FPS", value=16, precision=0)
                    num_frames = gr.Number(label="Num Frames", value=132, precision=0)
                    num_inference_steps = gr.Number(label="Num Inference Steps", value=50, precision=0)

                with gr.Row():
                    with gr.Column():
                        guidance_scale = gr.Number(
                            label="Guidance Scale",
                            value=1.0,
                            interactive=True,
                            scale=4,
                        )
                        include_guidance_scale = gr.Checkbox(label="Send", value=True, scale=1, min_width=70)
                    with gr.Column():
                        guidance_scale_2 = gr.Number(
                            label="Guidance Scale 2",
                            value=None,
                            interactive=False,
                            scale=4,
                        )
                        include_guidance_scale_2 = gr.Checkbox(label="Send", value=False, scale=1, min_width=70)
                    with gr.Column():
                        true_cfg_scale = gr.Number(
                            label="True CFG Scale",
                            value=None,
                            interactive=False,
                            scale=4,
                        )
                        include_true_cfg_scale = gr.Checkbox(label="Send", value=False, scale=1, min_width=70)
                    # I want it to wrap after 3, but it auto wrap after 2. I'll rather fully delegate to auto wrapping.
                    with gr.Column():
                        boundary_ratio = gr.Number(
                            label="Boundary Ratio",
                            value=None,
                            interactive=False,
                            scale=4,
                        )
                        include_boundary_ratio = gr.Checkbox(label="Send", value=False, scale=1, min_width=70)
                    with gr.Column():
                        flow_shift = gr.Number(label="Flow Shift", value=None, interactive=False, scale=4)
                        include_flow_shift = gr.Checkbox(label="Send", value=False, scale=1, min_width=70)
                    with gr.Column():
                        seed = gr.Number(label="Seed", value=42, precision=0, interactive=True, scale=4)
                        include_seed = gr.Checkbox(label="Send", value=True, scale=1, min_width=70)

                with gr.Row():
                    start_button = gr.Button("Start", variant="primary", elem_id="streaming-video-start")
                    hidden_stream_config = gr.Textbox(value="", visible=False)

            with gr.Column(scale=1):
                gr.HTML(
                    VIDEO_STREAM_VIEW_HTML.read_text(encoding="utf-8"),
                    label="Streaming Preview",
                    js_on_load=VIDEO_STREAM_VIEW_JS.read_text(encoding="utf-8"),
                )

        inputs = [
            host,
            port,
            model,
            prompt,
            negative_prompt,
            width,
            height,
            fps,
            num_frames,
            num_inference_steps,
            include_guidance_scale,
            guidance_scale,
            include_guidance_scale_2,
            guidance_scale_2,
            include_seed,
            seed,
            include_boundary_ratio,
            boundary_ratio,
            include_flow_shift,
            flow_shift,
            include_true_cfg_scale,
            true_cfg_scale,
        ]
        for toggle, field in [
            (include_guidance_scale, guidance_scale),
            (include_guidance_scale_2, guidance_scale_2),
            (include_seed, seed),
            (include_boundary_ratio, boundary_ratio),
            (include_flow_shift, flow_shift),
            (include_true_cfg_scale, true_cfg_scale),
        ]:
            toggle.change(
                fn=_set_optional_field_enabled,
                inputs=toggle,
                outputs=field,
            )
        start_button.click(
            fn=build_browser_stream_config,
            inputs=inputs,
            outputs=[hidden_stream_config, start_button],
        ).then(
            fn=lambda config: config,
            inputs=hidden_stream_config,
            outputs=hidden_stream_config,
            js="""
            (config) => {
              if (window.vllmStreamingVideoStart) {
                window.vllmStreamingVideoStart(config);
              }
              return config;
            }
            """,
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming video generation Gradio demo")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP for gradio launch")
    parser.add_argument("--port", type=int, default=7860, help="Port for gradio launch")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = create_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
