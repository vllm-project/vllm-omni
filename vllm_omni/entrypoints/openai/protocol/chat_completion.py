from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse, ChatCompletionStreamResponse


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: str | None = "text"
    sample_rate_hz: int | None = None
    metrics: dict[str, Any] | None = None


class OmniChatCompletionResponse(ChatCompletionResponse):
    metrics: dict[str, Any] | None = None
