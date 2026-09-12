"""Realtime video chat demo for Qwen3-Omni using FastRTC.

Streams webcam frames to a vLLM-Omni server and displays
text responses in a chat interface.

Usage:
    # 1. Start the vLLM-Omni server:
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091

    # 2. Run this demo:
    python video_chat_demo.py --port 8091

    # 3. Open http://localhost:7860 in your browser

Requirements:
    pip install fastrtc websocket-client
"""

import argparse
import base64
import io
import json
import threading

import cv2
import gradio as gr
import numpy as np
from fastrtc import Stream
from PIL import Image
from websocket import create_connection

# NOTE: Global frame buffer -- this demo is single-user only.
# FastRTC's Stream handler has no access to Gradio session state,
# so per-session isolation isn't possible with the current API.
# Bind to localhost (not 0.0.0.0) in production or use SSH tunneling.
frame_buffer: list[np.ndarray] = []
frame_lock = threading.Lock()
MAX_FRAMES = 64


def on_frame(frame, _question):
    """Buffer each webcam frame and pass through for display."""
    with frame_lock:
        frame_buffer.append(frame.copy())
        if len(frame_buffer) > MAX_FRAMES:
            frame_buffer.pop(0)
    return frame


def ask(question, chatbot, ws_url, model):
    """Send buffered frames + question to the server, stream response."""
    with frame_lock:
        frames = list(frame_buffer)

    if not frames:
        chatbot.append(
            {
                "role": "assistant",
                "content": "No frames captured. Start the webcam first.",
            }
        )
        yield chatbot
        return

    if not question.strip():
        return

    chatbot.append({"role": "user", "content": question})
    chatbot.append({"role": "assistant", "content": ""})
    yield chatbot

    # Sample up to 8 frames uniformly
    n = min(len(frames), 8)
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)

    try:
        ws = create_connection(ws_url, timeout=60)
        ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "model": model,
                    "modalities": ["text"],
                    "system_prompt": (
                        "You are a helpful vision assistant. Give concise, direct answers in 1-2 sentences."
                    ),
                }
            )
        )

        for i in indices:
            img = Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode()
            ws.send(json.dumps({"type": "video.frame", "data": b64}))

        ws.send(json.dumps({"type": "video.query", "text": question}))

        text = ""
        while True:
            msg = json.loads(ws.recv())
            if msg["type"] == "response.text.delta":
                text += msg.get("delta", "")
                chatbot[-1]["content"] = text
                yield chatbot
            elif msg["type"] == "response.text.done":
                text = msg.get("text", text)
                chatbot[-1]["content"] = text
                yield chatbot
                break
            elif msg["type"] == "response.error":
                chatbot[-1]["content"] = f"Error: {msg.get('message')}"
                yield chatbot
                break

        ws.send(json.dumps({"type": "video.done"}))
        ws.close()
    except Exception as e:
        chatbot[-1]["content"] = f"Error: {e}"
        yield chatbot
        try:
            ws.close()
        except Exception:
            pass


def build_demo(ws_url: str, model: str):
    stream = Stream(
        handler=on_frame,
        modality="video",
        mode="send-receive",
        additional_inputs=[
            gr.Textbox(visible=False, value=""),
        ],
        track_constraints={
            "width": {"ideal": 640},
            "height": {"ideal": 480},
        },
        ui_args={"full_screen": False},
    )

    with gr.Blocks(title="Qwen3-Omni Video Chat") as demo:
        gr.Markdown("# Qwen3-Omni Video Chat")
        gr.Markdown(
            "Start the webcam, type a question, and click **Ask**. Powered by [FastRTC](https://fastrtc.org) ⚡"
        )

        ws_state = gr.State(ws_url)
        model_state = gr.State(model)

        with gr.Row():
            with gr.Column(scale=1):
                stream.ui.render()
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=400,
                    type="messages",
                )
                with gr.Row():
                    question = gr.Textbox(
                        label="Question",
                        placeholder="Ask about what the camera sees...",
                        value="What do you see?",
                        scale=4,
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)

        ask_btn.click(
            ask,
            inputs=[question, chatbot, ws_state, model_state],
            outputs=[chatbot],
        )
        question.submit(
            ask,
            inputs=[question, chatbot, ws_state, model_state],
            outputs=[chatbot],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Omni video chat demo")
    parser.add_argument("--host", default="localhost", help="vLLM server host")
    parser.add_argument("--port", type=int, default=8091, help="vLLM server port")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name",
    )
    parser.add_argument("--demo-port", type=int, default=7860, help="Gradio demo port")
    args = parser.parse_args()

    ws_url = f"ws://{args.host}:{args.port}/v1/video/chat/stream"
    demo = build_demo(ws_url, args.model)
    demo.launch(server_name="127.0.0.1", server_port=args.demo_port)


if __name__ == "__main__":
    main()
