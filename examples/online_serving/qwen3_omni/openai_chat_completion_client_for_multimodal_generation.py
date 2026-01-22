import base64
import os
from typing import NamedTuple

import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8091/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

SEED = 42


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64 format."""
    with open(file_path, "rb") as f:
        content = f.read()
        result = base64.b64encode(content).decode("utf-8")
    return result


def get_video_url_from_path(video_path: str | None) -> str:
    """Convert a video path (local file or URL) to a video URL format for the API.

    If video_path is None or empty, returns the default URL.
    If video_path is a local file path, encodes it to base64 data URL.
    If video_path is a URL, returns it as-is.
    """
    if not video_path:
        # Default video URL
        return "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

    # Check if it's a URL (starts with http:// or https://)
    if video_path.startswith(("http://", "https://")):
        return video_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Detect video MIME type from file extension
    video_path_lower = video_path.lower()
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
        # Default to mp4 if extension is unknown
        mime_type = "video/mp4"

    video_base64 = encode_base64_content_from_file(video_path)
    return f"data:{mime_type};base64,{video_base64}"


def get_image_url_from_path(image_path: str | None) -> str:
    """Convert an image path (local file or URL) to an image URL format for the API.

    If image_path is None or empty, returns the default URL.
    If image_path is a local file path, encodes it to base64 data URL.
    If image_path is a URL, returns it as-is.
    """
    if not image_path:
        # Default image URL
        return "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"

    # Check if it's a URL (starts with http:// or https://)
    if image_path.startswith(("http://", "https://")):
        return image_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Detect image MIME type from file extension
    image_path_lower = image_path.lower()
    if image_path_lower.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif image_path_lower.endswith(".png"):
        mime_type = "image/png"
    elif image_path_lower.endswith(".gif"):
        mime_type = "image/gif"
    elif image_path_lower.endswith(".webp"):
        mime_type = "image/webp"
    else:
        # Default to jpeg if extension is unknown
        mime_type = "image/jpeg"

    image_base64 = encode_base64_content_from_file(image_path)
    return f"data:{mime_type};base64,{image_base64}"


def get_audio_url_from_path(audio_path: str | None) -> str:
    """Convert an audio path (local file or URL) to an audio URL format for the API.

    If audio_path is None or empty, returns the default URL.
    If audio_path is a local file path, encodes it to base64 data URL.
    If audio_path is a URL, returns it as-is.
    """
    if not audio_path:
        # Default audio URL
        return AudioAsset("mary_had_lamb").url

    # Check if it's a URL (starts with http:// or https://)
    if audio_path.startswith(("http://", "https://")):
        return audio_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect audio MIME type from file extension
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".m4a"):
        mime_type = "audio/mp4"
    else:
        # Default to wav if extension is unknown
        mime_type = "audio/wav"

    audio_base64 = encode_base64_content_from_file(audio_path)
    return f"data:{mime_type};base64,{audio_base64}"


def get_system_prompt():
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


def get_text_query(custom_prompt: str | None = None):
    question = (
        custom_prompt or "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
    )
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{question}",
            }
        ],
    }
    return prompt


default_system = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def get_video_query(video_path: str | None = None, custom_prompt: str | None = None):
    question = custom_prompt or "Why is this video funny?"
    video_url = get_video_url_from_path(video_path)
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": video_url},
            },
            {
                "type": "text",
                "text": f"{question}",
            },
        ],
    }
    return prompt


def get_image_query(image_path: str | None = None, custom_prompt: str | None = None):
    question = custom_prompt or "What is the content of this image?"
    image_url = get_image_url_from_path(image_path)
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
            {
                "type": "text",
                "text": f"{question}",
            },
        ],
    }
    return prompt


def get_audio_query(audio_path: str | None = None, custom_prompt: str | None = None):
    question = custom_prompt or "What is the content of this audio?"
    audio_url = get_audio_url_from_path(audio_path)
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": audio_url},
            },
            {
                "type": "text",
                "text": f"{question}",
            },
        ],
    }
    return prompt


def get_mixed_modalities_query(
    video_path: str | None = None,
    image_path: str | None = None,
    audio_path: str | None = None,
    custom_prompt: str | None = None,
):
    """
    Online-friendly multimodal user message:
    - Uses URLs (or base64 data URLs) for audio / image / video.
    - Returns the OpenAI-style message dict directly (not the offline QueryResult).
    """
    question = (
        custom_prompt or "What is recited in the audio? What is the content of this image? Why is this video funny?"
    )

    audio_url = get_audio_url_from_path(audio_path)
    image_url = get_image_url_from_path(image_path)
    video_url = get_video_url_from_path(video_path)

    return {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": question},
        ],
    }


def get_multi_audios_query(custom_prompt: str | None = None):
    """
    Online-friendly two-audio comparison request.
    - Encodes both audio clips as URLs (or data URLs).
    - Returns the OpenAI-style message dict.
    """
    question = custom_prompt or "Are these two audio clips the same?"
    # Use default demo clips; you can point to your own via --audio-path if needed.
    audio_url_1 = get_audio_url_from_path(AudioAsset("winning_call").url)
    audio_url_2 = get_audio_url_from_path(AudioAsset("mary_had_lamb").url)

    return {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url_1}},
            {"type": "audio_url", "audio_url": {"url": audio_url_2}},
            {"type": "text", "text": question},
        ],
    }


def get_use_audio_in_video_query(
    video_path: str | None = None,
    audio_path: str | None = None,
    custom_prompt: str | None = None,
):
    question = custom_prompt or (
        "Describe the content of the video in details, then convert what the baby say into text."
    )
    video_url = get_video_url_from_path(video_path)
    audio_url = get_audio_url_from_path(audio_path)
    return {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "audio_url", "audio_url": {"url": audio_url}},
            {"type": "text", "text": question},
        ],
    }


query_map = {
    "text": get_text_query,
    "use_audio": get_audio_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
    "use_mixed_modalities": get_mixed_modalities_query,
    "use_multi_audios": get_multi_audios_query,
    "use_audio_in_video": get_use_audio_in_video_query,
}


def run_multimodal_generation(args) -> None:
    model_name = args.model
    thinker_sampling_params = {
        "temperature": 0.4,  # Deterministic
        "top_p": 0.9,
        "top_k": 1,
        "max_tokens": 16384,
        "repetition_penalty": 1.05,
        "stop_token_ids": [151645],  # Qwen EOS token <|im_end|>
        "seed": SEED,
    }

    # Sampling parameters for Talker stage (codec generation)
    # Stop at codec EOS token
    talker_sampling_params = {
        "temperature": 0.9,
        "top_k": 50,
        "max_tokens": 4096,
        "seed": SEED,
        "detokenize": False,
        "repetition_penalty": 1.05,
        "stop_token_ids": [2150],  # TALKER_CODEC_EOS_TOKEN_ID
    }

    # # Sampling parameters for Code2Wav stage (audio generation)
    code2wav_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 4096 * 16,
        "seed": SEED,
        "detokenize": True,
        "repetition_penalty": 1.1,
    }

    sampling_params_list = [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]

    # Get paths and custom prompt from args
    video_path = getattr(args, "video_path", None)
    image_path = getattr(args, "image_path", None)
    audio_path = getattr(args, "audio_path", None)
    custom_prompt = getattr(args, "prompt", None)

    # Get the query function and call it with appropriate parameters
    query_func = query_map[args.query_type]
    if args.query_type == "use_video":
        prompt = query_func(video_path=video_path, custom_prompt=custom_prompt)
    elif args.query_type == "use_image":
        prompt = query_func(image_path=image_path, custom_prompt=custom_prompt)
    elif args.query_type == "use_audio":
        prompt = query_func(audio_path=audio_path, custom_prompt=custom_prompt)
    elif args.query_type == "text":
        prompt = query_func(custom_prompt=custom_prompt)
    elif args.query_type == "use_audio_in_video":
        prompt = query_func(
            video_path=video_path,
            audio_path=audio_path,
            custom_prompt=custom_prompt,
        )
    else:
        prompt = query_func()

    extra_body = {
        "sampling_params_list": sampling_params_list  # Optional, it has a default setting in stage_configs of the corresponding model.
    }

    if args.query_type == "use_audio_in_video":
        extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}

    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
    else:
        output_modalities = None

    chat_completion = client.chat.completions.create(
        messages=[
            get_system_prompt(),
            prompt,
        ],
        model=model_name,
        modalities=output_modalities,
        extra_body=extra_body,
        stream=args.stream,
    )

    count = 0
    if not args.stream:
        for choice in chat_completion.choices:
            if choice.message.audio:
                audio_data = base64.b64decode(choice.message.audio.data)
                audio_file_path = f"audio_{count}.wav"
                with open(audio_file_path, "wb") as f:
                    f.write(audio_data)
                print(f"Audio saved to {audio_file_path}")
                count += 1
            elif choice.message.content:
                print("Chat completion output from text:", choice.message.content)
    else:
        printed_content = False
        for chunk in chat_completion:
            for choice in chunk.choices:
                if hasattr(choice, "delta"):
                    content = getattr(choice.delta, "content", None)
                else:
                    content = None

                if getattr(chunk, "modality", None) == "audio" and content:
                    audio_data = base64.b64decode(content)
                    audio_file_path = f"audio_{count}.wav"
                    with open(audio_file_path, "wb") as f:
                        f.write(audio_data)
                    print(f"\nAudio saved to {audio_file_path}")
                    count += 1

                elif getattr(chunk, "modality", None) == "text":
                    if not printed_content:
                        printed_content = True
                        print("\ncontent:", end="", flush=True)
                    print(content, end="", flush=True)


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="use_mixed_modalities",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model Name / Path",
    )
    parser.add_argument(
        "--video-path",
        "-v",
        type=str,
        default=None,
        help="Path to local video file or URL. If not provided and query-type is 'use_video', uses default video URL.",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="Path to local image file or URL. If not provided and query-type is 'use_image', uses default image URL.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file or URL. If not provided and query-type is 'use_audio', uses default audio URL.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Custom text prompt/question to use instead of the default prompt for the selected query type.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Output modalities to use for the prompts.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multimodal_generation(args)
