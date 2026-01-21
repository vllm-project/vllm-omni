from vllm.entrypoints.openai.protocol import (
    ChatCompletionStreamResponse,
)


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: str | None = "text"
