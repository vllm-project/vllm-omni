from typing import Optional

from vllm.entrypoints.openai.protocol import ChatCompletionStreamResponse


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: Optional[str] = "text"
