import argparse
import os
import random
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from vllm.assets.video import video_get_metadata, video_to_ndarrays
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni

# Import utils from offline inference example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../offline_inference/qwen3_omni"))
try:
    from utils import make_omni_prompt
except ImportError:
    # Fallback if utils doesn't exist - we'll build prompts directly
    make_omni_prompt = None

SEED = 42
ASYNC_INIT_TIMEOUT = 600

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
        help="Path to local model checkpoint.",
    )
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="Host/IP for gradio `launch`.",
    )
    parser.add_argument("--port", type=int, default=7861, help="Port for gradio `launch`.")
    parser.add_argument("--share", action="store_true", help="Share the Gradio demo publicly.")
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to custom stage configs YAML file (optional).",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable statistics logging for AsyncOmni.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path prefix for AsyncOmni log files.",
    )
    parser.add_argument(
        "--init-sleep-seconds",
        type=int,
        default=30,
        help="Seconds to sleep between starting stage processes.",
    )
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Threshold in bytes for using shared memory IPC.",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=10,
        help="Batching timeout (seconds) inside each stage.",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=ASYNC_INIT_TIMEOUT,
        help="Timeout (seconds) for initializing all stages.",
    )
    parser.add_argument(
        "--worker-backend", type=str, default="multi_process", choices=["multi_process", "ray"], help="backend"
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address of the Ray cluster.",
    )
    return parser.parse_args()


def build_async_omni_cli_args(base_args: argparse.Namespace) -> argparse.Namespace:
    """Construct the minimal CLI args Namespace expected by AsyncOmni."""
    return argparse.Namespace(
        model=base_args.model,
        stage_configs_path=getattr(base_args, "stage_configs_path", None),
        log_stats=bool(getattr(base_args, "log_stats", False)),
        log_file=getattr(base_args, "log_file", None),
        init_sleep_seconds=int(getattr(base_args, "init_sleep_seconds", 30)),
        shm_threshold_bytes=int(getattr(base_args, "shm_threshold_bytes", 65536)),
        batch_timeout=int(getattr(base_args, "batch_timeout", 10)),
        init_timeout=int(getattr(base_args, "init_timeout", ASYNC_INIT_TIMEOUT)),
        worker_backend=getattr(base_args, "worker_backend", "multi_process"),
        ray_address=getattr(base_args, "ray_address", None),
    )


def build_sampling_params(seed: int, model_key: str) -> list[SamplingParams]:
    """Build SamplingParams objects by reusing the dict definitions."""
    return [SamplingParams(**params_dict) for params_dict in build_sampling_params_dict(seed, model_key)]


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


def process_video_file(
    video_file: str | None,
    enable_audio_in_video: bool = False,
    max_frames: int = 32,
) -> tuple[np.ndarray, dict[str, Any], tuple[np.ndarray, int] | None] | None:
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

    audio_tuple: tuple[np.ndarray, int] | None = None
    if enable_audio_in_video:
        try:
            import librosa  # type: ignore import

            audio_signal, sampling_rate = librosa.load(str(video_path), sr=16000)
            audio_tuple = (audio_signal.astype(np.float32), sampling_rate)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to extract audio from video {video_path}: {exc}")

    return frames, metadata, audio_tuple


async def run_inference_async_omni(
    omni: AsyncOmni,
    sampling_params: list[SamplingParams],
    prompt_args_template: SimpleNamespace,
    user_prompt: str,
    audio_file: tuple[str, tuple[int, np.ndarray]] | None = None,
    image_file: Image.Image | None = None,
    video_file: str | None = None,
    use_audio_in_video: bool = False,
    output_modalities: str | None = None,
):
    """Run inference using AsyncOmni directly with multimodal support."""
    if not user_prompt.strip() and not audio_file and not image_file and not video_file:
        return "Please provide at least a text prompt or multimodal input.", None

    if output_modalities is not None:
        output_modalities = output_modalities.split(",")

    try:
        # Build prompt with multimodal data
        prompt_args = SimpleNamespace(**prompt_args_template.__dict__)

        # Process multimodal inputs
        multi_modal_data = {}
        mm_processor_kwargs = {}

        # Process audio
        audio_data = process_audio_file(audio_file)
        if audio_data is not None:
            multi_modal_data["audio"] = [audio_data] if not isinstance(audio_data, list) else audio_data

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
                multi_modal_data["audio"] = (
                    [extracted_audio] if not isinstance(extracted_audio, list) else extracted_audio
                )
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
            prompt = f"<|im_start|>system\n{default_system}<|im_end|>\n<|im_start|>user\n"
            if audio_data:
                prompt += "<|audio_start|><|audio_pad|><|audio_end|>"
            if image_data:
                prompt += "<|vision_start|><|image_pad|><|vision_end|>"
            if video_payload is not None:
                prompt += "<|vision_start|><|video_pad|><|vision_end|>"
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

        async for stage_outputs in omni.generate(
            prompt=omni_prompt,
            request_id=request_id,
            sampling_params_list=sampling_params,
            output_modalities=output_modalities,
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
                # See vllm_omni/entrypoints/openai/serving_chat.py:733 for reference
                if hasattr(request_output, "multimodal_output") and request_output.multimodal_output:
                    if "audio" in request_output.multimodal_output:
                        audio_tensor = request_output.multimodal_output["audio"]
                        # Ensure audio is 1D (flatten if needed)
                        if hasattr(audio_tensor, "ndim") and audio_tensor.ndim > 1:
                            audio_tensor = audio_tensor.flatten()
                        # Convert BFloat16 to Float32 before numpy conversion (numpy doesn't support BFloat16)
                        audio_np = audio_tensor.float().detach().cpu().numpy()
                        audio_output = (
                            24000,  # sampling rate in Hz
                            audio_np,
                        )

        text_response = "\n\n".join(text_outputs) if text_outputs else "No text output."
        return text_response, audio_output
    except Exception as exc:  # pylint: disable=broad-except
        return f"Inference failed: {exc}", None


def build_interface(
    omni: AsyncOmni,
    sampling_params: list[SamplingParams],
    prompt_args_template: SimpleNamespace,
    model: str,
):
    """Build Gradio interface for AsyncOmni mode."""

    async def run_inference(
        user_prompt: str,
        audio_file: tuple[str, tuple[int, np.ndarray]] | None,
        image_file: Image.Image | None,
        video_file: str | None,
        use_audio_in_video: bool,
        output_modalities: str | None = None,
    ):
        return await run_inference_async_omni(
            omni,
            sampling_params,
            prompt_args_template,
            user_prompt,
            audio_file,
            image_file,
            video_file,
            use_audio_in_video,
            output_modalities,
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
                placeholder="For example: text, image, video. Use comma to separate multiple modalities.",
                lines=1,
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
            inputs=[input_box, audio_input, image_input, video_input, use_audio_in_video_checkbox, output_modalities],
            outputs=[text_output, audio_output],
        )
        demo.queue()
    return demo


def main():
    args = parse_args()
    omni = None

    model_name = "/".join(args.model.split("/")[-2:])
    assert model_name in SUPPORTED_MODELS, (
        f"Unsupported model '{model_name}'. Supported models: {SUPPORTED_MODELS.keys()}"
    )

    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, shutting down...")
        if omni is not None:
            try:
                omni.shutdown()
            except Exception as e:
                print(f"Error during shutdown: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"Initializing AsyncOmni with model: {args.model}")
    if args.stage_configs_path:
        print(f"Using custom stage configs: {args.stage_configs_path}")

    sampling_params = build_sampling_params(SEED, model_name)
    cli_args = build_async_omni_cli_args(args)
    omni = AsyncOmni(model=args.model, cli_args=cli_args)
    print("✓ AsyncOmni initialized successfully")
    prompt_args_template = create_prompt_args(args)

    demo = build_interface(
        omni,
        sampling_params,
        prompt_args_template,
        args.model,
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
        if omni is not None:
            try:
                omni.shutdown()
            except Exception as e:
                print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()
