# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni extensions to the realtime websocket protocol.

Upstream's realtime protocol (``vllm/entrypoints/speech_to_text/realtime/protocol.py``)
is transcription-oriented: ``session.update`` carries only ``model``, and there is no
wire mechanism for tools at all. The events below add tool/function calling using the
shapes of the OpenAI realtime API, so clients that already speak it (LiveKit Agents and
similar) work without changes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel
from vllm.utils import random_uuid

# Client -> Server events


class OmniSessionUpdate(OpenAIBaseModel):
    """``session.update`` extended with tool configuration.

    Upstream only reads ``model``; the extra fields are ignored by the base
    implementation, so a client that sends them against a build without this
    support degrades to plain speech-to-speech instead of failing.
    """

    type: Literal["session.update"] = "session.update"
    model: str | None = None
    tools: list[ChatCompletionToolsParam] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] = "auto"


class FunctionCallOutputItem(OpenAIBaseModel):
    """The result of a tool call, produced by the client."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


class ConversationItemCreate(OpenAIBaseModel):
    """``conversation.item.create`` carrying a tool result."""

    type: Literal["conversation.item.create"] = "conversation.item.create"
    item: FunctionCallOutputItem


# Server -> Client events


class ResponseFunctionCallArgumentsDelta(OpenAIBaseModel):
    """Incremental arguments for a function call the model is requesting."""

    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"
    item_id: str = Field(default_factory=lambda: f"item-{random_uuid()}")
    call_id: str
    name: str | None = None
    delta: str


class ResponseFunctionCallArgumentsDone(OpenAIBaseModel):
    """Terminal event for one function call, with the complete arguments."""

    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    item_id: str = Field(default_factory=lambda: f"item-{random_uuid()}")
    call_id: str
    name: str
    arguments: str
