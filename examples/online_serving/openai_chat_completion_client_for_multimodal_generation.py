import base64

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8091/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

SEED = 42


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


def run_text_to_audio(model: str) -> None:
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

    chat_completion = client.chat.completions.create(
        messages=[
            get_system_prompt(),
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words.",
                    },
                ],
            },
        ],
        model=model,
        extra_body={
            "sampling_params_list": sampling_params_list
        },  # Optional, it has a default setting in stage_configs of the corresponding model.
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


if __name__ == "__main__":
    run_text_to_audio("Qwen/Qwen2.5-Omni-7B")
