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


def run_text_to_audio(model: str) -> None:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Explain the system architecture for a scalable \
                            audio generation pipeline. Answer in 15 words.",
                    },
                ],
            }
        ],
        model=model,
    )

    for choice in chat_completion.choices:
        if choice.message.content_type == "audio":
            audio_data = base64.b64decode(choice.message.content)
            with open("audio.wav", "wb") as f:
                f.write(audio_data)
        elif choice.message.content_type == "text":
            print("Chat completion output from text:", choice.message.content)


if __name__ == "__main__":
    run_text_to_audio("Qwen/Qwen2.5-Omni-7B")
