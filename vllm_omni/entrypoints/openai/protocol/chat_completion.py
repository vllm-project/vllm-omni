from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
)


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: str | None = "text"
    metrics: dict[str, Any] | None = None


class OmniChatCompletionResponse(ChatCompletionResponse):
    metrics: dict[str, Any] | None = None


class OmniChatMessage(ChatMessage):
    """ChatMessage with a sidecar for non-text model outputs.

    Single-stage AR models that also emit a non-text artifact (e.g. Alpamayo's
    sampled action trajectory, surfaced via ``OmniOutput.multimodal_outputs``)
    populate ``multimodal_output`` here so HTTP clients can read it under
    ``response.choices[*].message.multimodal_output[...]``. Values are
    JSON-serializable (tensors are converted upstream).
    """

    multimodal_output: dict[str, Any] | None = None
