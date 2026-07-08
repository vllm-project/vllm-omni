# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended protocol events for OpenAI-compatible realtime API with tool calling."""

from enum import Enum
from typing import Literal

from pydantic import Field
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class RealtimeEventType(str, Enum):
    """Wire-format event type strings for the /v1/realtime WebSocket API."""

    # Client → Server
    SESSION_UPDATE = "session.update"
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    CONVERSATION_ITEM_CREATE = "conversation.item.create"

    # Server → Client
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"

    # Item types (used inside conversation.item.create)
    FUNCTION_CALL_OUTPUT = "function_call_output"


# Additional Server -> Client Events for Tool Calling


class ResponseTextDelta(OpenAIBaseModel):
    """Incremental text response (for debugging and tool calls)"""

    type: Literal[RealtimeEventType.RESPONSE_TEXT_DELTA] = RealtimeEventType.RESPONSE_TEXT_DELTA
    delta: str


class ResponseTextDone(OpenAIBaseModel):
    """Final text response"""

    type: Literal[RealtimeEventType.RESPONSE_TEXT_DONE] = RealtimeEventType.RESPONSE_TEXT_DONE
    text: str


class ResponseFunctionCallArgumentsDelta(OpenAIBaseModel):
    """Incremental function call arguments"""

    type: Literal[RealtimeEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA] = (
        RealtimeEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA
    )
    call_id: str = Field(description="Unique ID for this function call")
    name: str = Field(description="Function name being called")
    delta: str


class ResponseFunctionCallArgumentsDone(OpenAIBaseModel):
    """Complete function call arguments"""

    type: Literal[RealtimeEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE] = (
        RealtimeEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE
    )
    call_id: str = Field(description="Unique ID for this function call")
    name: str = Field(description="Function name being called")
    arguments: str


class ResponseAudioDelta(OpenAIBaseModel):
    """Incremental audio response"""

    type: Literal[RealtimeEventType.RESPONSE_AUDIO_DELTA] = RealtimeEventType.RESPONSE_AUDIO_DELTA
    audio: str = Field(description="Base64-encoded audio chunk")
    sample_rate_hz: int = Field(default=24000, description="Audio sample rate")


class ResponseAudioDone(OpenAIBaseModel):
    """Audio response complete"""

    type: Literal[RealtimeEventType.RESPONSE_AUDIO_DONE] = RealtimeEventType.RESPONSE_AUDIO_DONE
    has_audio: bool = True


# Client -> Server Events for Tool Results


class ConversationItemFunctionCallOutput(OpenAIBaseModel):
    """Conversation item for function call results"""

    type: Literal[RealtimeEventType.FUNCTION_CALL_OUTPUT] = RealtimeEventType.FUNCTION_CALL_OUTPUT
    call_id: str = Field(description="ID of the function call this is a result for")
    output: str = Field(description="JSON-encoded function result")


class ConversationItemCreate(OpenAIBaseModel):
    """Create a new conversation item (tool result)"""

    type: Literal[RealtimeEventType.CONVERSATION_ITEM_CREATE] = RealtimeEventType.CONVERSATION_ITEM_CREATE
    item: ConversationItemFunctionCallOutput
