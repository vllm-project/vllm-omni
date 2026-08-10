# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Official Omni-Diffusion inference chat template."""

from collections.abc import Mapping
from typing import Any

import torch

# This mirrors Omni-Diffusion's official inference script template. Keep it in
# a small dedicated module so the model wrapper can reuse the official prompt
# behavior without embedding a long Jinja template in the main implementation.
OMNI_DIFFUSION_CHAT_TEMPLATE = (
    "\n"
    "{%- if tools %}\n"
    "    {{- '<|im_start|>system\\n' }}\n"
    "    {%- if messages[0]['role'] == 'system' %}\n"
    "        {{- messages[0]['content'] }}\n"
    "    {%- endif %}\n"
    '    {{- "\\n\\n# Tools\\n\\n'
    "You may call one or more functions to assist with the user query.\\n\\n"
    'You are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n'
    "    {%- for tool in tools %}\n"
    '        {{- "\\n" }}\n'
    "        {{- tool | tojson }}\n"
    "    {%- endfor %}\n"
    '    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name '
    "and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n"
    '{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n'
    '</tool_call><|im_end|>\\n" }}\n'
    "{%- else %}\n"
    "    {%- if messages[0]['role'] == 'system' %}\n"
    "        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
    "{%- for message in messages %}\n"
    '    {%- if (message.role == "user") or (message.role == "system" and not loop.first) '
    'or (message.role == "assistant" and not message.tool_calls) %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n"
    '    {%- elif message.role == "assistant" %}\n'
    "        {{- '<|im_start|>' + message.role }}\n"
    "        {%- if message.content %}\n"
    "            {{- '\\n' + message.content }}\n"
    "        {%- endif %}\n"
    "        {%- for tool_call in message.tool_calls %}\n"
    "            {%- if tool_call.function is defined %}\n"
    "                {%- set tool_call = tool_call.function %}\n"
    "            {%- endif %}\n"
    '            {{- \'\\n<tool_call>\\n{"name": "\' }}\n'
    "            {{- tool_call.name }}\n"
    '            {{- \'", "arguments": \' }}\n'
    "            {{- tool_call.arguments | tojson }}\n"
    "            {{- '}\\n</tool_call>' }}\n"
    "        {%- endfor %}\n"
    "        {{- '<|im_end|>\\n' }}\n"
    '    {%- elif message.role == "tool" %}\n'
    '        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}\n'
    "            {{- '<|im_start|>user' }}\n"
    "        {%- endif %}\n"
    "        {{- '\\n<tool_response>\\n' }}\n"
    "        {{- message.content }}\n"
    "        {{- '\\n</tool_response>' }}\n"
    '        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n'
    "            {{- '<|im_end|>\\n' }}\n"
    "        {%- endif %}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}\n\n"
)


def normalize_chat_template_token_ids(tokenized_output: Any) -> list[int]:
    """Normalize a tokenized chat template to one token ID sequence."""
    # Transformers v5 may return a BatchEncoding/mapping, while v4 usually
    # returns a plain token ID list.
    if isinstance(tokenized_output, Mapping):
        if "input_ids" not in tokenized_output:
            raise ValueError(
                "Expected chat template tokenization output to contain input_ids, "
                f"got keys={list(tokenized_output.keys())}."
            )
        tokenized_output = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        tokenized_output = tokenized_output.input_ids

    if isinstance(tokenized_output, torch.Tensor):
        tokenized_output = tokenized_output.detach().cpu().tolist()

    if tokenized_output and isinstance(tokenized_output[0], list):
        if len(tokenized_output) != 1:
            raise ValueError(
                f"Expected Omni-Diffusion chat template to return a single token sequence, got {len(tokenized_output)}."
            )
        tokenized_output = tokenized_output[0]

    return [int(token_id) for token_id in tokenized_output]
