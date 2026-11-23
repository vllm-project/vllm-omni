# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

from pydantic import SerializeAsAny
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
)

from vllm_omni.entrypoints.openai.protocol.audio import AudioChunkMetadata


class OmniChatCompletionResponseStreamChoice(ChatCompletionResponseStreamChoice):
    audio_metadata: AudioChunkMetadata | None = None


class OmniChatCompletionResponseChoice(ChatCompletionResponseChoice):
    audio_metadata: AudioChunkMetadata | None = None


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    choices: list[SerializeAsAny[ChatCompletionResponseStreamChoice]]
    modality: str | None = "text"
    metrics: dict[str, Any] | None = None


class OmniChatCompletionResponse(ChatCompletionResponse):
    choices: list[SerializeAsAny[ChatCompletionResponseChoice]]
    metrics: dict[str, Any] | None = None
