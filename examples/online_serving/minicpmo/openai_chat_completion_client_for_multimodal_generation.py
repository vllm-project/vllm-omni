# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI chat-completions client for MiniCPM-o 4.5 online serving.

Thin wrapper around the shared multimodal client helpers, with MiniCPM-specific
defaults (port 8099, system prompt, ``chat_template_kwargs.use_tts_template``).
"""

from __future__ import annotations

import base64
import concurrent.futures
import os
import sys

from openai import OpenAI

from vllm_omni.utils.tracking_parser import TrackingArgumentParser

# Reuse prompt builders / URL helpers from the shared online client.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ONLINE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ONLINE_ROOT not in sys.path:
    sys.path.insert(0, _ONLINE_ROOT)

from openai_chat_completion_client_for_multimodal_generation import (  # noqa: E402
    _build_prompt_for_query_type,
    _parse_csv_arg,
    make_audio_output_filename,
    query_map,
)

SEED = 42

default_system = (
    "You are MiniCPM-o, a helpful multimodal assistant that can "
    "understand images, audio and video, and respond in text and speech."
)


def get_system_prompt() -> dict:
    return {
        "role": "system",
        "content": [{"type": "text", "text": default_system}],
    }


def _wants_tts(modalities: list[str] | None) -> bool:
    if modalities is None:
        return True
    return "audio" in modalities


def run_multimodal_generation(args, client: OpenAI) -> None:
    model_name = args.model
    video_path = getattr(args, "video_path", None)
    image_path = getattr(args, "image_path", None)
    audio_path = getattr(args, "audio_path", None)
    custom_prompt = getattr(args, "prompt", None)

    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
    else:
        output_modalities = None

    use_tts = _wants_tts(output_modalities)
    num_concurrent_requests = args.num_concurrent_requests
    prompt_list = _parse_csv_arg(getattr(args, "prompts", None))

    request_payloads = []
    for idx in range(num_concurrent_requests):
        per_req_prompt = (
            prompt_list[idx]
            if idx < len(prompt_list)
            else (custom_prompt if idx == 0 or not prompt_list else prompt_list[-1])
        )
        prompt = _build_prompt_for_query_type(
            query_type=args.query_type,
            custom_prompt=per_req_prompt,
            video_path=video_path,
            image_path=image_path,
            audio_path=audio_path,
        )
        extra_body: dict = {
            "chat_template_kwargs": {"use_tts_template": use_tts},
        }
        if args.query_type == "use_audio_in_video":
            # MiniCPM-o 4.5 does not use Qwen's use_audio_in_video path; keep
            # the flag only if the shared query type is selected explicitly.
            extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}
        request_payloads.append({"prompt": prompt, "extra_body": extra_body})

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
        futures = [
            executor.submit(
                client.chat.completions.create,
                messages=[
                    get_system_prompt(),
                    payload["prompt"],
                ],
                model=model_name,
                modalities=output_modalities,
                extra_body=payload["extra_body"],
                stream=args.stream,
            )
            for payload in request_payloads
        ]
        chat_completions = [future.result() for future in concurrent.futures.as_completed(futures)]

    assert len(chat_completions) == num_concurrent_requests
    count = 0
    if not args.stream:
        for chat_completion in chat_completions:
            request_id = getattr(chat_completion, "id", None)
            for choice in chat_completion.choices:
                if choice.message.audio:
                    audio_data = base64.b64decode(choice.message.audio.data)
                    audio_file_path = make_audio_output_filename(request_id=request_id, index=count)
                    with open(audio_file_path, "wb") as f:
                        f.write(audio_data)
                    print(f"Audio saved to {audio_file_path}")
                    count += 1
                elif choice.message.content:
                    print("Chat completion output from text:", choice.message.content)
    else:
        printed_content = False
        for chat_completion in chat_completions:
            for chunk in chat_completion:
                for choice in chunk.choices:
                    if hasattr(choice, "delta"):
                        content = getattr(choice.delta, "content", None)
                    else:
                        content = None

                    if getattr(chunk, "modality", None) == "audio" and content:
                        audio_data = base64.b64decode(content)
                        request_id = getattr(chunk, "id", None)
                        audio_file_path = make_audio_output_filename(request_id=request_id, index=count)
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
    parser = TrackingArgumentParser(description="MiniCPM-o 4.5 online multimodal chat client")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="text",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openbmb/MiniCPM-o-4_5",
        help="Model Name / Path",
    )
    parser.add_argument(
        "--video-path",
        "-v",
        type=str,
        default=None,
        help="Path to local video file or URL.",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="Path to local image file or URL.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file or URL.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Custom text prompt/question.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Output modalities, e.g. text or text,audio.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response.",
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests to send.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8099,
        help="Port of the vLLM Omni API server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host/IP of the vLLM Omni API server.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated prompts for concurrent requests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    openai_api_base = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(api_key="EMPTY", base_url=openai_api_base)
    run_multimodal_generation(args, client)
