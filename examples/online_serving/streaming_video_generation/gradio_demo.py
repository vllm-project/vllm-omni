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
import math
from pathlib import Path
from typing import Any, TypedDict

try:
    import gradio as gr
except ImportError:
    raise ImportError("gradio is required to run this demo. Install it with: pip install 'vllm-omni[demo]'") from None

DEFAULT_PROMPT = "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement."
DEFAULT_TRANSITION_CHUNKS = 3
PROMPT_UPDATE_HEADERS = ["At (s)", "Prompt", "Transition Chunks"]
PROMPT_UPDATE_VALUES = [
    [
        4,
        (
            "An underwater tornado appears and affects the ocean floor in a dramatic and chaotic scene. "
            "The water is murky, swirling violently, carrying debris and marine life into the vortex. "
            "The tropical fish on the scene all swim in panic, trying to avoid the powerful currents. "
            "The camera remains stationary, capturing the intensity of the underwater tornado as it disrupts the serene ocean floor. "
            "Close-up shot emphasizing the turbulent motion and destruction."
        ),
        3,
    ],
    [
        11,
        (
            "The swirling underwater vortex now seizes a heavy, encrusted treasure chest, its lid flapping open as it is smashed onto the ocean floor. "
            "Gold coins and silver trinkets spill out, glittering briefly in the murky water before being swept instantly into the violent funnel. "
            "The heavy wooden box tumbles end over end, colliding with floating rocks and adding to the debris field. "
            "Swirling sediment and bubbles surround the spilling fortune, highlighting the chaotic power of the storm as it ravages the seabed. "
            "Close-up shot emphasizing the turbulent motion and destruction."
        ),
        3,
    ],
]
VIDEO_STREAM_VIEW_HTML = Path(__file__).with_name("video-stream-view.html")
VIDEO_STREAM_VIEW_JS = Path(__file__).with_name("video-stream-view.js")


class ScheduledPromptUpdate(TypedDict):
    at: float
    prompt: str
    transition_chunks: int


def _maybe_set(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        payload[key] = value.strip()
    else:
        payload[key] = value


def _optional_int(value: int | float | None) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: int | float | None) -> float | None:
    return None if value is None else float(value)


def _normalize_prompt_update_row(row: list[Any] | tuple[Any, ...]) -> tuple[Any, Any, Any]:
    values = []
    for v in row[: len(PROMPT_UPDATE_HEADERS)]:
        if isinstance(v, str):
            values.append(v.strip() or None)
        elif isinstance(v, float) and math.isnan(v):
            values.append(None)
        else:
            values.append(v)
    while len(values) < len(PROMPT_UPDATE_HEADERS):
        values.append(None)
    return values[0], values[1], values[2]  # pyright: ignore[reportIndexIssue]


def _prompt_update_row_is_empty(row: list[Any] | tuple[Any, ...]) -> bool:
    at, prompt, transition = _normalize_prompt_update_row(row)
    return at is None and prompt is None and transition is None


def _parse_prompt_update_row(row: list[Any] | tuple[Any, ...], *, index: int) -> ScheduledPromptUpdate:
    at_val, prompt_val, transition_val = _normalize_prompt_update_row(row)
    row_label = f"Prompt update row {index + 1}"

    if at_val is None or prompt_val is None:
        raise gr.Error(f"{row_label} requires both At (s) and Prompt.")

    try:
        at = float(at_val)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{row_label}: At (s) must be a number.") from exc
    if at < 0:
        raise gr.Error(f"{row_label}: At (s) must be >= 0.")

    prompt = str(prompt_val).strip()
    if not prompt:
        raise gr.Error(f"{row_label}: Prompt must be a non-empty string.")

    transition_chunks = DEFAULT_TRANSITION_CHUNKS
    if transition_val is not None:
        try:
            transition_chunks = int(transition_val)
        except (TypeError, ValueError) as exc:
            raise gr.Error(f"{row_label}: Transition Chunks must be an integer.") from exc
        if transition_chunks < 0:
            raise gr.Error(f"{row_label}: Transition Chunks must be >= 0.")

    return {
        "at": at,
        "prompt": prompt,
        "transition_chunks": transition_chunks,
    }


def _parse_prompt_updates_dataframe(rows: list[list[Any]] | None) -> list[ScheduledPromptUpdate]:
    if not rows:
        return []

    # Single row: empty or complete.
    if len(rows) == 1:
        row = rows[0]
        if _prompt_update_row_is_empty(row):
            return []
        return [_parse_prompt_update_row(row, index=0)]

    updates: list[ScheduledPromptUpdate] = []
    for index, row in enumerate(rows):
        if _prompt_update_row_is_empty(row):
            raise gr.Error(f"Prompt update row {index + 1} cannot be empty when multiple rows are present.")
        updates.append(_parse_prompt_update_row(row, index=index))
    return sorted(updates, key=lambda update: update["at"])


def _build_session_start(
    *,
    model: str,
    prompt: str,
    negative_prompt: str | None,
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
    include_negative_prompt: bool,
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
    prompt_updates_df: list[list[Any]] | None,
) -> tuple[str, Any]:
    """Build the browser-side WebSocket request config for the HTML/MSE player."""
    if not model.strip():
        raise gr.Error("Model is required.")
    if not prompt.strip():
        raise gr.Error("Prompt is required.")

    payload = _build_session_start(
        model=model.strip(),
        prompt=prompt.strip(),
        negative_prompt=negative_prompt if include_negative_prompt else None,
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
    prompt_updates = _parse_prompt_updates_dataframe(prompt_updates_df)
    config = {
        "url": f"ws://{host}:{int(port)}/v1/realtime/video",
        "payload": payload,
    }
    if prompt_updates:
        config["prompt_updates"] = prompt_updates
    return json.dumps(config, ensure_ascii=False), gr.update(value="Streaming...", interactive=False)


def create_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Streaming Video Generation",
        css="textarea:disabled, input:disabled { color: var(--body-text-color-subdued); -webkit-text-fill-color: var(--body-text-color-subdued); background-color: var(--background-fill-secondary); }",
    ) as demo:
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

                with gr.Row():
                    width = gr.Number(label="Width", value=640, precision=0)
                    height = gr.Number(label="Height", value=384, precision=0)

                with gr.Row():
                    fps = gr.Number(label="FPS", value=16, precision=0)
                    num_frames = gr.Number(label="Num Frames", value=429, precision=0)
                    num_inference_steps = gr.Number(label="Num Inference Steps", value=50, precision=0)

                with gr.Accordion("Advanced Sampling", open=False):
                    with gr.Row():
                        with gr.Column():
                            negative_prompt = gr.Textbox(label="Negative Prompt", value="", lines=1, interactive=False)
                            include_negative_prompt = gr.Checkbox(label="Send", value=False)
                    with gr.Row():
                        with gr.Column(scale=0, min_width=150):
                            guidance_scale = gr.Number(
                                label="Guidance Scale",
                                value=1.0,
                                interactive=True,
                            )
                            include_guidance_scale = gr.Checkbox(label="Send", value=True)
                        with gr.Column(scale=0, min_width=150):
                            guidance_scale_2 = gr.Number(
                                label="Guidance Scale 2",
                                value=None,
                                interactive=False,
                            )
                            include_guidance_scale_2 = gr.Checkbox(label="Send", value=False)
                        with gr.Column(scale=0, min_width=150):
                            true_cfg_scale = gr.Number(
                                label="True CFG Scale",
                                value=None,
                                interactive=False,
                            )
                            include_true_cfg_scale = gr.Checkbox(label="Send", value=False)
                        with gr.Column(scale=0, min_width=150):
                            boundary_ratio = gr.Number(
                                label="Boundary Ratio",
                                value=None,
                                interactive=False,
                            )
                            include_boundary_ratio = gr.Checkbox(label="Send", value=False)
                        with gr.Column(scale=0, min_width=150):
                            flow_shift = gr.Number(label="Flow Shift", value=None, interactive=False)
                            include_flow_shift = gr.Checkbox(label="Send", value=False)
                        with gr.Column(scale=0, min_width=150):
                            seed = gr.Number(label="Seed", value=42, precision=0, interactive=True)
                            include_seed = gr.Checkbox(label="Send", value=True)

                with gr.Accordion("Prompt Updates", open=True):
                    gr.Markdown(
                        "Schedule midway text `session.interaction` messages after `video.start`. "
                        "With one row, leave it fully empty or fill both **At (s)** and **Prompt**. "
                        "With multiple rows, every row must be complete."
                    )
                    prompt_updates_df = gr.Dataframe(
                        headers=PROMPT_UPDATE_HEADERS,
                        datatype=["number", "str", "number"],
                        type="array",
                        row_count=1,
                        row_limits=(1, None),
                        column_count=3,
                        value=PROMPT_UPDATE_VALUES,
                        interactive=True,
                    )

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
            include_negative_prompt,
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
            prompt_updates_df,
        ]
        for toggle, field in [
            (include_negative_prompt, negative_prompt),
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
