import argparse
import base64
import io
import os
import os as _os_env_toggle
import random
import signal
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

import gradio as gr
import numpy as np
import soundfile as sf
from PIL import Image
import torch
from vllm.assets.video import video_get_metadata, video_to_ndarrays

# For HTTP API mode
from openai import OpenAI

# For AsyncOmniLLM mode
from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM

# Import utils from offline inference example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../offline_inference/qwen2_5_omni"))
try:
    from utils import make_omni_prompt
except ImportError:
    # Fallback if utils doesn't exist - we'll build prompts directly
    make_omni_prompt = None

SEED = 42
ASYNC_INIT_TIMEOUT = 600

SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-Omni-7B": {
        "sampling_params": {
            "thinker": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 2048,
                "detokenize": True,
                "repetition_penalty": 1.1,
            },
            "talker": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 2048,
                "detokenize": True,
                "repetition_penalty": 1.1,
                "stop_token_ids": [8294],
            },
            "code2wav": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 2048,
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
    parser = argparse.ArgumentParser(
        description="Gradio demo for Qwen2.5-Omni online inference."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Omni-7B",
        help="Path to model (for async_omni_llm mode) or model name (for http_api mode).",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8091/v1",
        help="Base URL for vllm serve API (only used when use-api-server is set).",
    )
    parser.add_argument(
        "--use-api-server",
        action="store_true",
        help="If set, connect to an existing vLLM HTTP API server instead of running AsyncOmniLLM locally.",
    )
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="Host/IP for gradio `launch`.",
    )
    parser.add_argument(
        "--port", type=int, default=7861, help="Port for gradio `launch`."
    )
    parser.add_argument(
        "--share", action="store_true", help="Share the Gradio demo publicly."
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to custom stage configs YAML file (optional).",
    )
    return parser.parse_args()


def build_sampling_params(seed: int, model_key: str) -> list[SamplingParams]:
    """Build SamplingParams objects by reusing the dict definitions."""
    return [
        SamplingParams(**params_dict)
        for params_dict in build_sampling_params_dict(seed, model_key)
    ]


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


def create_prompt_args(base_args: argparse.Namespace) -> SimpleNamespace:
    # The prompt builder expects a minimal namespace with these attributes.
    return SimpleNamespace(
        model=base_args.model,
        prompt_type="text",
        tokenize=True,
        use_torchvision=True,
        legacy_omni_video=False,
    )


def get_system_prompt():
    """Get system prompt for HTTP API mode."""
    return {
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
    }


def process_audio_file(
    audio_file: Optional[Any],
) -> Optional[Tuple[np.ndarray, int]]:
    """Normalize Gradio audio input to (np.ndarray, sample_rate)."""
    if audio_file is None:
        return None

    sample_rate: Optional[int] = None
    audio_np: Optional[np.ndarray] = None

    def _load_from_path(path_str: str) -> Optional[Tuple[np.ndarray, int]]:
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
                    if isinstance(sr_candidate, (int, float)) and isinstance(
                        data_candidate, np.ndarray
                    ):
                        sample_rate = int(sr_candidate)
                        audio_np = data_candidate
                if audio_np is None:
                    loaded = _load_from_path(first)
                    if loaded is not None:
                        audio_np, sample_rate = loaded
            # Case 3: (None, (sample_rate, np.ndarray))
            elif first is None and isinstance(second, tuple) and len(second) == 2:
                sr_candidate, data_candidate = second
                if isinstance(sr_candidate, (int, float)) and isinstance(
                    data_candidate, np.ndarray
                ):
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


def process_image_file(image_file: Optional[Image.Image]) -> Optional[Image.Image]:
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


def process_video_file(
    video_file: Optional[str],
    enable_audio_in_video: bool = False,
    max_frames: int = 32,
) -> Optional[Tuple[np.ndarray, dict[str, Any], Optional[Tuple[np.ndarray, int]]]]:
    """Process video file and optionally extract audio track."""
    if video_file is None:
        return None

    video_path = Path(video_file)
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return None

    try:
        frames = video_to_ndarrays(str(video_path), num_frames=max_frames)
        metadata = video_get_metadata(str(video_path), num_frames=frames.shape[0])
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to decode video {video_path}: {exc}")
        return None

    audio_tuple: Optional[Tuple[np.ndarray, int]] = None
    if enable_audio_in_video:
        try:
            import librosa  # type: ignore import

            audio_signal, sampling_rate = librosa.load(str(video_path), sr=16000)
            audio_tuple = (audio_signal.astype(np.float32), sampling_rate)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to extract audio from video {video_path}: {exc}")

    return frames, metadata, audio_tuple


async def run_inference_async_omni_llm(
    omni_llm: AsyncOmniLLM,
    sampling_params: list[SamplingParams],
    prompt_args_template: SimpleNamespace,
    user_prompt: str,
    audio_file: Optional[Tuple[str, Tuple[int, np.ndarray]]] = None,
    image_file: Optional[Image.Image] = None,
    video_file: Optional[str] = None,
    use_audio_in_video: bool = False,
):
    """Run inference using AsyncOmniLLM directly with multimodal support."""
    if not user_prompt.strip() and not audio_file and not image_file and not video_file:
        return "Please provide at least a text prompt or multimodal input.", None

    try:
        # Build prompt with multimodal data
        prompt_args = SimpleNamespace(**prompt_args_template.__dict__)
        
        # Process multimodal inputs
        multi_modal_data = {}
        mm_processor_kwargs = {}
        
        # Process audio
        audio_data = process_audio_file(audio_file)
        if audio_data is not None:
            multi_modal_data["audio"] = audio_data
        
        # Process image
        image_data = process_image_file(image_file)
        if image_data is not None:
            multi_modal_data["image"] = image_data
        
        # Process video
        video_payload = process_video_file(video_file, enable_audio_in_video=use_audio_in_video)
        if video_payload is not None:
            video_frames, video_metadata, extracted_audio = video_payload
            video_entry: Any
            if video_metadata:
                video_entry = (video_frames, video_metadata)
            else:
                video_entry = video_frames
            multi_modal_data["video"] = video_entry
            if use_audio_in_video and extracted_audio is not None and "audio" not in multi_modal_data:
                multi_modal_data["audio"] = extracted_audio
                mm_processor_kwargs["use_audio_in_video"] = True
        
        # Build the prompt input
        if make_omni_prompt is not None:
            omni_prompt = make_omni_prompt(prompt_args, user_prompt)
            # Add multimodal data if present
            if multi_modal_data:
                if isinstance(omni_prompt, dict):
                    omni_prompt["multi_modal_data"] = multi_modal_data
                    if mm_processor_kwargs:
                        omni_prompt["mm_processor_kwargs"] = mm_processor_kwargs
                else:
                    # If make_omni_prompt returns a string, we need to wrap it
                    omni_prompt = {
                        "prompt": omni_prompt,
                        "multi_modal_data": multi_modal_data,
                    }
                    if mm_processor_kwargs:
                        omni_prompt["mm_processor_kwargs"] = mm_processor_kwargs
        else:
            # Fallback: build prompt directly
            default_system = (
                "You are Qwen, a virtual human developed by the Qwen Team, "
                "Alibaba Group, capable of perceiving auditory and visual inputs, "
                "as well as generating text and speech."
            )
            prompt = (
                f"<|im_start|>system\n{default_system}<|im_end|>\n"
                "<|im_start|>user\n"
            )
            if audio_data:
                prompt += "<|audio_bos|><|AUDIO|><|audio_eos|>"
            if image_data:
                prompt += "<|vision_bos|><|IMAGE|><|vision_eos|>"
            if video_payload is not None:
                prompt += "<|vision_bos|><|VIDEO|><|vision_eos|>"
            if user_prompt.strip():
                prompt += f"{user_prompt}"
            prompt += "<|im_end|>\n<|im_start|>assistant\n"
            
            omni_prompt = {
                "prompt": prompt,
                "multi_modal_data": multi_modal_data,
            }
            if mm_processor_kwargs:
                omni_prompt["mm_processor_kwargs"] = mm_processor_kwargs

        request_id = "0"
        text_outputs: list[str] = []
        audio_output = None

        async for stage_outputs in omni_llm.generate(
            prompt=omni_prompt,
            request_id=request_id,
            sampling_params_list=sampling_params,
        ):
            # stage_outputs.request_output is a RequestOutput object, not a list
            request_output = stage_outputs.request_output
            if stage_outputs.final_output_type == "text":
                if request_output.outputs:
                    for output in request_output.outputs:
                        if output.text:
                            text_outputs.append(output.text)
            elif stage_outputs.final_output_type == "audio":
                # multimodal_output is on the RequestOutput object
                # See vllm_omni/entrypoints/openai/serving_chat.py:680 for reference
                if hasattr(request_output, "multimodal_output") and request_output.multimodal_output:
                    if "audio" in request_output.multimodal_output:
                        audio_tensor = request_output.multimodal_output["audio"]
                        # Ensure audio is 1D (flatten if needed)
                        if hasattr(audio_tensor, "ndim") and audio_tensor.ndim > 1:
                            audio_tensor = audio_tensor.flatten()
                        audio_np = audio_tensor.detach().cpu().numpy()
                        audio_output = (
                            24000,  # sampling rate in Hz
                            audio_np,
                        )

        text_response = "\n\n".join(text_outputs) if text_outputs else "No text output."
        return text_response, audio_output
    except Exception as exc:  # pylint: disable=broad-except
        return f"Inference failed: {exc}", None


def run_inference_http_api(
    client: OpenAI,
    model: str,
    sampling_params_list: list[dict],
    user_prompt: str,
    audio_file: Optional[Tuple[str, Tuple[int, np.ndarray]]] = None,
    image_file: Optional[Image.Image] = None,
    video_file: Optional[str] = None,
    use_audio_in_video: bool = False,
):
    """Run inference using HTTP API (vllm serve) with multimodal support."""
    if not user_prompt.strip() and not audio_file and not image_file and not video_file:
        return "Please provide at least a text prompt or multimodal input.", None

    try:
        # Build user content array
        user_content = []
        
        # Add audio if provided
        if audio_file is not None:
            # For HTTP API, encode audio as base64 data URL
            audio_data = process_audio_file(audio_file)
            if audio_data is not None:
                audio_array, sample_rate = audio_data
                # Save to temporary file and encode as base64
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_array, sample_rate)
                    tmp_path = tmp_file.name
                    # Read back and encode
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    os.unlink(tmp_path)  # Clean up temp file
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                    user_content.append({
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}
                    })
        
        # Add image if provided
        if image_file is not None:
            image_data = process_image_file(image_file)
            if image_data is not None:
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                image_data.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
        
        # Add video if provided
        if video_file is not None:
            video_path = Path(video_file)
            if video_path.exists():
                try:
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
                    ext = video_path.suffix.lower()
                    mime_type = {
                        ".mp4": "video/mp4",
                        ".avi": "video/x-msvideo",
                        ".mov": "video/quicktime",
                        ".mkv": "video/x-matroska",
                    }.get(ext, "video/mp4")
                    user_content.append({
                        "type": "video_url",
                        "video_url": {"url": f"data:{mime_type};base64,{video_base64}"}
                    })
                except Exception:
                    user_content.append({
                        "type": "video_url",
                        "video_url": {"url": f"file://{video_path.resolve()}"}
                    })
        
        # Add text prompt
        if user_prompt.strip():
            user_content.append({
                "type": "text",
                "text": user_prompt,
            })
        
        messages = [
            get_system_prompt(),
            {
                "role": "user",
                "content": user_content,
            },
        ]

        extra_body = {
            "sampling_params_list": sampling_params_list
        }
        
        # Add mm_processor_kwargs if needed
        if use_audio_in_video:
            extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            extra_body=extra_body,
        )

        text_outputs: list[str] = []
        audio_output = None

        for choice in chat_completion.choices:
            if choice.message.audio:
                # Decode base64 audio data (already in WAV format)
                audio_data = base64.b64decode(choice.message.audio.data)
                # Use soundfile to read WAV from bytes
                audio_io = io.BytesIO(audio_data)
                audio_np, sample_rate = sf.read(audio_io)
                # Ensure mono audio
                if len(audio_np.shape) > 1:
                    audio_np = audio_np[:, 0]
                audio_output = (sample_rate, audio_np)
            elif choice.message.content:
                text_outputs.append(choice.message.content)

        text_response = "\n\n".join(text_outputs) if text_outputs else "No text output."
        return text_response, audio_output
    except Exception as exc:  # pylint: disable=broad-except
        return f"Inference failed: {exc}", None


def build_interface(
    use_api_server: bool,
    omni_llm: Optional[AsyncOmniLLM],
    sampling_params: Optional[list[SamplingParams]],
    prompt_args_template: Optional[SimpleNamespace],
    client: Optional[OpenAI],
    model: str,
    sampling_params_dict: Optional[list[dict]],
    api_base: Optional[str] = None,
):
    """Build Gradio interface based on the selected mode."""

    if not use_api_server:
        # AsyncOmniLLM mode - Gradio supports async functions directly
        async def run_inference(
            user_prompt: str,
            audio_file: Optional[Tuple[str, Tuple[int, np.ndarray]]],
            image_file: Optional[Image.Image],
            video_file: Optional[str],
            use_audio_in_video: bool,
        ):
            return await run_inference_async_omni_llm(
                omni_llm, sampling_params, prompt_args_template, user_prompt,
                audio_file, image_file, video_file, use_audio_in_video
            )

    else:
        # HTTP API mode
        def run_inference(
            user_prompt: str,
            audio_file: Optional[Tuple[str, Tuple[int, np.ndarray]]],
            image_file: Optional[Image.Image],
            video_file: Optional[str],
            use_audio_in_video: bool,
        ):
            return run_inference_http_api(
                client, model, sampling_params_dict, user_prompt,
                audio_file, image_file, video_file, use_audio_in_video
            )

    with gr.Blocks() as demo:
        gr.Markdown("# vLLM-Omni Online Serving Demo")
        info_text = f"**Model:** {model} \n\n"
        if use_api_server and api_base:
            info_text += f"**API Base:** {api_base}\n\n"
        gr.Markdown(info_text)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_box = gr.Textbox(
                    label="Text Prompt",
                    placeholder="For example: Please tell me a joke in 30 words.",
                    lines=4,
                )
                audio_input = gr.Audio(
                    label="Audio Input (optional)",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                image_input = gr.Image(
                    label="Image Input (optional)",
                    type="pil",
                    sources=["upload"],
                )
                video_input = gr.Video(
                    label="Video Input (optional)",
                    sources=["upload"],
                )
                use_audio_in_video_checkbox = gr.Checkbox(
                    label="Use audio from video",
                    value=False,
                    info="Extract and use audio track from video input",
                )
            with gr.Column(scale=1):
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                text_output = gr.Textbox(label="Text Output", lines=10)
                audio_output = gr.Audio(label="Audio Output", interactive=False)

        generate_btn.click(
            fn=run_inference,
            inputs=[input_box, audio_input, image_input, video_input, use_audio_in_video_checkbox],
            outputs=[text_output, audio_output],
        )
        demo.queue()
    return demo


def main():
    args = parse_args()
    omni_llm = None

    model_name = "/".join(args.model.split("/")[-2:])
    assert model_name in SUPPORTED_MODELS, f"Unsupported model '{model_name}'. Supported models: {SUPPORTED_MODELS.keys()}"

    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, shutting down...")
        if omni_llm is not None:
            try:
                omni_llm.shutdown()
            except Exception as e:
                print(f"Error during shutdown: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not args.use_api_server:
        # Initialize AsyncOmniLLM
        print(f"Initializing AsyncOmniLLM with model: {args.model}")
        if args.stage_configs_path:
            print(f"Using custom stage configs: {args.stage_configs_path}")
        
        sampling_params = build_sampling_params(SEED, model_name)
        omni_llm = AsyncOmniLLM(
            model=args.model,
            stage_configs_path=args.stage_configs_path,
            init_timeout=ASYNC_INIT_TIMEOUT,
        )
        print("✓ AsyncOmniLLM initialized successfully")
        prompt_args_template = create_prompt_args(args)
        client = None
        sampling_params_dict = None
    else:
        # HTTP API mode
        print(f"Using HTTP API mode with base URL: {args.api_base}")
        print(f"Please make sure vllm serve is running: vllm serve {args.model} --omni --port 8091")
        client = OpenAI(api_key="EMPTY", base_url=args.api_base)
        sampling_params_dict = build_sampling_params_dict(SEED, model_name)
        omni_llm = None
        sampling_params = None
        prompt_args_template = None

    demo = build_interface(
        args.use_api_server,
        omni_llm,
        sampling_params,
        prompt_args_template,
        client,
        args.model,
        sampling_params_dict,
        api_base=args.api_base if args.use_api_server else None,
    )
    try:
        demo.launch(
            server_name=args.ip,
            server_port=args.port,
            share=args.share,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        if omni_llm is not None:
            try:
                omni_llm.shutdown()
            except Exception as e:
                print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()

