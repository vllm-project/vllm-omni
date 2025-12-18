# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The request function for API endpoints."""

import io
import json
import os
import sys
import time
import traceback
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Optional, Protocol, Union
import aiohttp
from tqdm.asyncio import tqdm
from vllm.benchmarks.lib.endpoint_request_func import (async_request_openai_completions,async_request_openai_audio,
                                                       async_request_openai_embeddings, RequestFunc,
                                                       RequestFuncInput,
                                                       RequestFuncOutput,StreamedResponseHandler)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

@dataclass
class MixRequestFuncOutput(RequestFuncOutput):
    output_audio_num: int = None

async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("chat/completions", "profile")), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'.")

    content = [{"type": "text", "text": request_func_input.prompt}]
    if request_func_input.multi_modal_content:
        mm_content = request_func_input.multi_modal_content
        if isinstance(mm_content, list):
            content.extend(mm_content)
        elif isinstance(mm_content, dict):
            content.append(mm_content)
        else:
            raise TypeError(
                "multi_modal_content must be a dict or list[dict] "
                "for openai-chat"
            )
    payload = {
        "model":
        request_func_input.model_name
        if request_func_input.model_name else request_func_input.model,
        "messages": [
            {
                "role": "user",
                "content": content
            },
        ],
        "temperature":
        0.0,
        "max_completion_tokens":
        request_func_input.output_len,
        "stream":
        False
    }
    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    if request_func_input.extra_headers:
        headers |= request_func_input.extra_headers
    if request_func_input.request_id:
        headers["x-request-id"] = request_func_input.request_id

    output = MixRequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    output.ttft = 0.0
    st = time.perf_counter()
    output.start_time = st
    output.output_audio_num = 0
    try:
        async with session.post(url=api_url, json=payload,
                                headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                choices = data.get("choices")
                for choice in choices:
                    content = choice["message"].get("content")
                    output.generated_text += content or ""
                    if choice["message"].get("audio"):
                        output.output_audio_num += 1
                output.output_tokens = 0
                output.success = True
                output.latency = time.perf_counter() - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

# TODO: Add more request functions for different API protocols.
ASYNC_REQUEST_FUNCS: dict[str, RequestFunc] = {
    "vllm": async_request_openai_completions,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "openai-audio": async_request_openai_audio,
    "openai-embeddings": async_request_openai_embeddings,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k for k, v in ASYNC_REQUEST_FUNCS.items()
    if v in (async_request_openai_completions,
             async_request_openai_chat_completions)
]
