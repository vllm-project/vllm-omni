import argparse
import base64
import io
import os
import random
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from openai import OpenAI
from PIL import Image

SEED = 42

SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "sampling_params": {
            "thinker": {
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 1,
                "max_tokens": 16384,
                "detokenize": True,
                "repetition_penalty": 1.05,
                "stop_token_ids": [151645],
                "seed": SEED,
            },
            "talker": {
                "temperature": 0.9,
                "top_k": 50,
                "max_tokens": 4096,
                "seed": SEED,
                "detokenize": False,
                "repetition_penalty": 1.05,
                "stop_token_ids": [2150],
            },
            "code2wav": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 4096 * 16,
                "seed": SEED,
                "detokenize": True,
                "repetition_penalty": 1.1,
            },
        },
    },
}
# Ensure deterministic behavior across runs.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for Qwen3-Omni online inference.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name/path (should match the server model).",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8091/v1",
        help="Base URL for the vLLM API server.",
    )
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="Host/IP for gradio `launch`.",
    )
    parser.add_argument("--port", type=int, default=7861, help="Port for gradio `launch`.")
    parser.add_argument("--share", action="store_true", help="Share the Gradio demo publicly.")
    return parser.parse_args()


def build_sampling_params_dict(seed: int, model_key: str) -> list[dict]:
    """Build sampling params as dict for HTTP API mode."""
    model_conf = SUPPORTED_MODELS.get(model_key)
    if model_conf is None:
        raise ValueError(f"Unsupported model '{model_key}'")

    sampling_templates: dict[str, dict[str, Any]] = model_conf["sampling_params"]
    sampling_params: list[dict] = []
    for stage_name, template in sampling_templates.items():
        params = dict(template)
        params["seed"] = seed
        sampling_params.append(params)
    return sampling_params


def image_to_base64_data_url(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    buffered = io.BytesIO()
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"


def audio_to_base64_data_url(audio_data: tuple[np.ndarray, int]) -> str:
    """Convert audio (numpy array, sample_rate) to base64 data URL."""
    audio_np, sample_rate = audio_data
    # Convert to int16 format for WAV
    if audio_np.dtype != np.int16:
        # Normalize to [-1, 1] range if needed
        if audio_np.dtype == np.float32 or audio_np.dtype == np.float64:
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)

    # Write to WAV bytes
    buffered = io.BytesIO()
    sf.write(buffered, audio_np, sample_rate, format="WAV")
    wav_bytes = buffered.getvalue()
    wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{wav_b64}"


def video_to_base64_data_url(video_file: str) -> str:
    """Convert video file to base64 data URL."""
    video_path = Path(video_file)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    # Detect MIME type from extension
    video_path_lower = str(video_path).lower()
    if video_path_lower.endswith(".mp4"):
        mime_type = "video/mp4"
    elif video_path_lower.endswith(".webm"):
        mime_type = "video/webm"
    elif video_path_lower.endswith(".mov"):
        mime_type = "video/quicktime"
    elif video_path_lower.endswith(".avi"):
        mime_type = "video/x-msvideo"
    elif video_path_lower.endswith(".mkv"):
        mime_type = "video/x-matroska"
    else:
        mime_type = "video/mp4"

    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{video_b64}"


def process_audio_file(
    audio_file: Any | None,
) -> tuple[np.ndarray, int] | None:
    """Normalize Gradio audio input to (np.ndarray, sample_rate)."""
    if audio_file is None:
        return None

    sample_rate: int | None = None
    audio_np: np.ndarray | None = None

    def _load_from_path(path_str: str) -> tuple[np.ndarray, int] | None:
        if not path_str:
            return None
        path = Path(path_str)
        if not path.exists():
            return None
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        return data.astype(np.float32), int(sr)

    if isinstance(audio_file, tuple):
        if len(audio_file) == 2:
            first, second = audio_file
            # Case 1: (sample_rate, np.ndarray)
            if isinstance(first, (int, float)) and isinstance(second, np.ndarray):
                sample_rate = int(first)
                audio_np = second
            # Case 2: (filepath, (sample_rate, np.ndarray or list))
            elif isinstance(first, str):
                if isinstance(second, tuple) and len(second) == 2:
                    sr_candidate, data_candidate = second
                    if isinstance(sr_candidate, (int, float)) and isinstance(data_candidate, np.ndarray):
                        sample_rate = int(sr_candidate)
                        audio_np = data_candidate
                if audio_np is None:
                    loaded = _load_from_path(first)
                    if loaded is not None:
                        audio_np, sample_rate = loaded
            # Case 3: (None, (sample_rate, np.ndarray))
            elif first is None and isinstance(second, tuple) and len(second) == 2:
                sr_candidate, data_candidate = second
                if isinstance(sr_candidate, (int, float)) and isinstance(data_candidate, np.ndarray):
                    sample_rate = int(sr_candidate)
                    audio_np = data_candidate
        elif len(audio_file) == 1 and isinstance(audio_file[0], str):
            loaded = _load_from_path(audio_file[0])
            if loaded is not None:
                audio_np, sample_rate = loaded
    elif isinstance(audio_file, str):
        loaded = _load_from_path(audio_file)
        if loaded is not None:
            audio_np, sample_rate = loaded

    if audio_np is None or sample_rate is None:
        return None

    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]

    return audio_np.astype(np.float32), sample_rate


def process_image_file(image_file: Image.Image | None) -> Image.Image | None:
    """Process image file from Gradio input.

    Returns:
        PIL Image in RGB mode or None if no image provided.
    """
    if image_file is None:
        return None
    # Convert to RGB if needed
    if image_file.mode != "RGB":
        image_file = image_file.convert("RGB")
    return image_file


def run_inference_api(
    client: OpenAI,
    model: str,
    sampling_params_dict: list[dict],
    user_prompt: str,
    audio_file: tuple[str, tuple[int, np.ndarray]] | None = None,
    image_file: Image.Image | None = None,
    video_file: str | None = None,
    use_audio_in_video: bool = False,
    output_modalities: str | None = None,
    stream: bool = False,
):
    """Run inference using OpenAI API client with multimodal support."""
    if not user_prompt.strip() and not audio_file and not image_file and not video_file:
        yield "Please provide at least a text prompt or multimodal input.", None

    try:
        # Build message content list
        content_list = []

        # Process audio
        audio_data = process_audio_file(audio_file)
        if audio_data is not None:
            audio_url = audio_to_base64_data_url(audio_data)
            content_list.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_url},
                }
            )

        # Process image
        if image_file is not None:
            image_data = process_image_file(image_file)
            if image_data is not None:
                image_url = image_to_base64_data_url(image_data)
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                )

        # Process video
        mm_processor_kwargs = {}
        if video_file is not None:
            video_url = video_to_base64_data_url(video_file)
            video_content = {
                "type": "video_url",
                "video_url": {"url": video_url},
            }
            if use_audio_in_video:
                video_content["video_url"]["num_frames"] = 32  # Default max frames
                mm_processor_kwargs["use_audio_in_video"] = True
            content_list.append(video_content)

        # Add text prompt
        if user_prompt.strip():
            content_list.append(
                {
                    "type": "text",
                    "text": user_prompt,
                }
            )

        # Build messages
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, "
                            "Alibaba Group, capable of perceiving auditory and visual inputs, "
                            "as well as generating text and speech."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": content_list,
            },
        ]

        # Build extra_body
        extra_body = {
            "sampling_params_list": sampling_params_dict,
        }
        if mm_processor_kwargs:
            extra_body["mm_processor_kwargs"] = mm_processor_kwargs

        # Parse output modalities
        if output_modalities and output_modalities.strip():
            output_modalities_list = [m.strip() for m in output_modalities.split(",")]
        else:
            output_modalities_list = None

        # Call API
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            modalities=output_modalities_list,
            extra_body=extra_body,
            stream=stream,
        )

        if not stream:
            # Non-streaming mode: extract outputs and yield once
            text_outputs: list[str] = []
            audio_output = None

            for choice in chat_completion.choices:
                if choice.message.content:
                    text_outputs.append(choice.message.content)
                if choice.message.audio:
                    # Decode base64 audio
                    audio_data = base64.b64decode(choice.message.audio.data)
                    # Load audio from bytes
                    audio_np, sample_rate = sf.read(io.BytesIO(audio_data))
                    # Convert to mono if needed
                    if audio_np.ndim > 1:
                        audio_np = audio_np[:, 0]
                    audio_output = (int(sample_rate), audio_np.astype(np.float32))

            text_response = "\n\n".join(text_outputs) if text_outputs else "No text output."
            yield text_response, audio_output
        else:
            # Streaming mode: yield incremental updates
            text_content = ""
            audio_output = None

            for chunk in chat_completion:
                for choice in chunk.choices:
                    if hasattr(choice, "delta"):
                        content = getattr(choice.delta, "content", None)
                    else:
                        content = None

                    # Handle audio modality
                    if getattr(chunk, "modality", None) == "audio" and content:
                        try:
                            # Decode base64 audio
                            audio_data = base64.b64decode(content)
                            # Load audio from bytes
                            audio_np, sample_rate = sf.read(io.BytesIO(audio_data))
                            # Convert to mono if needed
                            if audio_np.ndim > 1:
                                audio_np = audio_np[:, 0]
                            audio_output = (int(sample_rate), audio_np.astype(np.float32))
                            # Yield current text and audio
                            yield text_content if text_content else "", audio_output
                        except Exception:  # pylint: disable=broad-except
                            # If audio processing fails, just yield text
                            yield text_content if text_content else "", None

                    # Handle text modality
                    elif getattr(chunk, "modality", None) == "text":
                        if content:
                            text_content += content
                            # Yield updated text content (keep existing audio if any)
                            yield text_content, audio_output

            # Final yield with accumulated text and last audio (if any)
            yield text_content if text_content else "No text output.", audio_output

    except Exception as exc:  # pylint: disable=broad-except
        error_msg = f"Inference failed: {exc}"
        yield error_msg, None


def build_interface(
    client: OpenAI,
    model: str,
    sampling_params_dict: list[dict],
):
    """Build Gradio interface for API server mode."""

    def run_inference(
        user_prompt: str,
        audio_file: tuple[str, tuple[int, np.ndarray]] | None,
        image_file: Image.Image | None,
        video_file: str | None,
        use_audio_in_video: bool,
        output_modalities: str | None = None,
        stream: bool = False,
    ):
        # Always yield from the API function to maintain consistent generator behavior
        yield from run_inference_api(
            client,
            model,
            sampling_params_dict,
            user_prompt,
            audio_file,
            image_file,
            video_file,
            use_audio_in_video,
            output_modalities,
            stream,
        )

    css = """
    .media-input-container {
        display: flex;
        gap: 10px;
    }
    .media-input-container > div {
        flex: 1;
    }
    .media-input-container .image-input,
    .media-input-container .audio-input {
        height: 300px;
    }
    .media-input-container .video-column {
        height: 300px;
        display: flex;
        flex-direction: column;
    }
    .media-input-container .video-input {
        flex: 1;
        min-height: 0;
    }
    #generate-btn button {
        width: 100%;
    }
    """

    with gr.Blocks(css=css) as demo:
        gr.Markdown("# vLLM-Omni Online Serving Demo")
        gr.Markdown(f"**Model:** {model} \n\n")

        with gr.Column():
            with gr.Row():
                input_box = gr.Textbox(
                    label="Text Prompt",
                    placeholder="For example: Describe what happens in the media inputs.",
                    lines=4,
                    scale=1,
                )
            with gr.Row(elem_classes="media-input-container"):
                image_input = gr.Image(
                    label="Image Input (optional)",
                    type="pil",
                    sources=["upload"],
                    scale=1,
                    elem_classes="image-input",
                )
                with gr.Column(scale=1, elem_classes="video-column"):
                    video_input = gr.Video(
                        label="Video Input (optional)",
                        sources=["upload"],
                        elem_classes="video-input",
                    )
                    use_audio_in_video_checkbox = gr.Checkbox(
                        label="Use audio from video",
                        value=False,
                        info="Extract the video's audio track when provided.",
                    )
                audio_input = gr.Audio(
                    label="Audio Input (optional)",
                    type="numpy",
                    sources=["upload", "microphone"],
                    scale=1,
                    elem_classes="audio-input",
                )

        with gr.Row():
            output_modalities = gr.Textbox(
                label="Output Modalities",
                value=None,
                placeholder="For example: text, image, video. Use comma to separate multiple modalities.",
                lines=1,
                scale=2,
            )
            stream_checkbox = gr.Checkbox(
                label="Stream output",
                value=False,
                info="Enable streaming to see output as it's generated.",
                scale=1,
            )

        with gr.Row():
            generate_btn = gr.Button(
                "Generate",
                variant="primary",
                size="lg",
                elem_id="generate-btn",
            )

        with gr.Row():
            text_output = gr.Textbox(label="Text Output", lines=10, scale=2)
            audio_output = gr.Audio(label="Audio Output", interactive=False, scale=1)

        generate_btn.click(
            fn=run_inference,
            inputs=[
                input_box,
                audio_input,
                image_input,
                video_input,
                use_audio_in_video_checkbox,
                output_modalities,
                stream_checkbox,
            ],
            outputs=[text_output, audio_output],
        )
        demo.queue()
    return demo


def main():
    args = parse_args()

    model_name = "/".join(args.model.split("/")[-2:])
    assert model_name in SUPPORTED_MODELS, (
        f"Unsupported model '{model_name}'. Supported models: {SUPPORTED_MODELS.keys()}"
    )

    # Initialize OpenAI client
    print(f"Connecting to API server at: {args.api_base}")
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.api_base,
    )
    print("✓ Connected to API server")

    # Build sampling params
    sampling_params_dict = build_sampling_params_dict(SEED, model_name)

    demo = build_interface(
        client,
        args.model,
        sampling_params_dict,
    )
    try:
        demo.launch(
            server_name=args.ip,
            server_port=args.port,
            share=args.share,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
