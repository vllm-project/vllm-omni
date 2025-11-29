import base64

import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils import FlexibleArgumentParser

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8091/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

SEED = 42


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


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


def get_text_query():
    question = "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
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


def get_mixed_modalities_query():
    question = "What is recited in the audio? What is the content of this image? Why is this video funny?"
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": AudioAsset("mary_had_lamb").url},
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
                },
            },
            {
                "type": "video_url",
                "video_url": {
                    "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"
                },
            },
            {
                "type": "text",
                "text": f"{question}",
            },
        ],
    }

    return prompt


def get_use_audio_in_video_query():
    question = "Describe the content of the video, then convert what the baby say into text."

    prompt = {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                    "num_frames": 16,
                },
            },
            {
                "type": "text",
                "text": f"{question}",
            },
        ],
    }

    return prompt


def get_multi_audios_query():
    question = "Are these two audio clips the same?"
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": AudioAsset("mary_had_lamb").url},
            },
            {
                "type": "audio_url",
                "audio_url": {"url": AudioAsset("winning_call").url},
            },
            {
                "type": "text",
                "text": f"{question}",
            },
        ],
    }
    return prompt


query_map = {
    "mixed_modalities": get_mixed_modalities_query,
    "use_audio_in_video": get_use_audio_in_video_query,
    "multi_audios": get_multi_audios_query,
    "text": get_text_query,
}


def run_multimodal_generation(args) -> None:
    model_name = "Qwen/Qwen2.5-Omni-7B"
    thinker_sampling_params = {
        "temperature": 0.0,  # Deterministic - no randomness
        "top_p": 1.0,  # Disable nucleus sampling
        "top_k": -1,  # Disable top-k sampling
        "max_tokens": 2048,
        "seed": SEED,  # Fixed seed for sampling
        "detokenize": True,
        "repetition_penalty": 1.1,
    }
    talker_sampling_params = {
        "temperature": 0.9,
        "top_p": 0.8,
        "top_k": 40,
        "max_tokens": 2048,
        "seed": SEED,  # Fixed seed for sampling
        "detokenize": True,
        "repetition_penalty": 1.05,
        "stop_token_ids": [8294],
    }
    code2wav_sampling_params = {
        "temperature": 0.0,  # Deterministic - no randomness
        "top_p": 1.0,  # Disable nucleus sampling
        "top_k": -1,  # Disable top-k sampling
        "max_tokens": 2048,
        "seed": SEED,  # Fixed seed for sampling
        "detokenize": True,
        "repetition_penalty": 1.1,
    }

    sampling_params_list = [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]

    prompt = query_map[args.query_type]()
    extra_body = {
        "sampling_params_list": sampling_params_list  # Optional, it has a default setting in stage_configs of the corresponding model.
    }

    if args.query_type == "use_audio_in_video":
        extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}

    chat_completion = client.chat.completions.create(
        messages=[
            get_system_prompt(),
            prompt,
        ],
        model=model_name,
        extra_body=extra_body,
    )

    count = 0
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


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="mixed_modalities",
        choices=query_map.keys(),
        help="Query type.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multimodal_generation(args)
