# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt formatting for realtime tool calling.

The realtime prompt built by ``buffer_realtime_audio`` has no system turn, so a
tool-calling session has to render the tool definitions -- and later the tool
results -- itself and feed them in as prompt context.

**These templates are Qwen-family specific.** They reproduce, turn for turn,
what the model's own chat template emits for tools (verified against
Qwen3-Omni's ``chat_template.json``): the tool block is a system turn, a tool
result is a user turn wrapped in ``<tool_response>``, and a tool call is an
assistant turn wrapped in ``<tool_call>`` -- which is also the shape the
``hermes`` tool parser reads back out of generated text.

They are rendered here rather than through ``tokenizer.apply_chat_template`` so
that the prompt stays deterministic and unit-testable, and does not depend on
whether a given checkpoint ships a tools-aware chat template. Supporting another
model family means adding its variants alongside these.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

TOOL_PREAMBLE_TEMPLATE = """<|im_start|>system


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_signatures}
</tools>

For each function call, return a json object with function name and arguments \
within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
"""

TOOL_RESULT_TEMPLATE = """<|im_start|>user
<tool_response>
{output}
</tool_response><|im_end|>
"""

ASSISTANT_TOOL_CALL_TEMPLATE = """<|im_start|>assistant
<tool_call>
{{"name": "{name}", "arguments": {arguments}}}
</tool_call><|im_end|>
"""


def render_tool_preamble(tools: Sequence[Any]) -> str:
    """Render the system turn that declares the available tools."""
    signatures = "\n".join(json.dumps(tool.model_dump(exclude_none=True), ensure_ascii=False) for tool in tools)
    return TOOL_PREAMBLE_TEMPLATE.format(tool_signatures=signatures)


def render_tool_result(output: str) -> str:
    """Render the user turn carrying a tool result."""
    return TOOL_RESULT_TEMPLATE.format(output=output)


def render_assistant_tool_call(name: str, arguments: str) -> str:
    """Render the assistant turn for a call the model made.

    ``arguments`` is embedded as the JSON object it already is: the chat template
    emits a string-valued arguments field verbatim rather than quoting it.
    """
    return ASSISTANT_TOOL_CALL_TEMPLATE.format(name=name, arguments=arguments)
