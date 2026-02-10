from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse, ChatCompletionStreamResponse


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: str | None = "text"
    metrics: dict[str, Any] | None = None


class OmniChatCompletionResponse(ChatCompletionResponse):
    metrics: dict[str, Any] | None = None
