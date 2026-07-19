from typing import Any

from pydantic import model_serializer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
)


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: str | None = "text"
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


class OmniChatCompletionResponse(ChatCompletionResponse):
    metrics: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    def _serialize_with_multimodal(self, handler):
        """Preserve ``OmniChatMessage.multimodal_output`` in the JSON response.

        ``ChatCompletionResponseChoice.message`` is annotated as the base
        ``ChatMessage``, so Pydantic v2 serializes each message by that schema
        and drops the extra ``multimodal_output`` carried by an
        ``OmniChatMessage``. Re-inject it from the live message objects after the
        default serialization — done here, at the protocol layer, so the serving
        code that builds the choices stays untouched.
        """
        data = handler(self)
        choices = data.get("choices")
        if not isinstance(choices, list):
            return data
        for i, choice in enumerate(self.choices):
            mo = getattr(getattr(choice, "message", None), "multimodal_output", None)
            if mo is None or i >= len(choices):
                continue
            serialized_choice = choices[i]
            if isinstance(serialized_choice, dict) and isinstance(serialized_choice.get("message"), dict):
                serialized_choice["message"]["multimodal_output"] = mo
        return data
