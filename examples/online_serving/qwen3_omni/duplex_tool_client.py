# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Example Qwen3-Omni duplex client that runs one local tool.

The server does not execute tools. This process listens for a completed
function call, runs ``get_current_time`` locally, and sends
``function_call_output``. Qwen is turn-commit, so the follow-up answer is a
new ``response.create`` after the tool result is accepted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

from vllm_omni.clients.duplex import DuplexClient, SessionConfig


def execute_local_tool(name: str, arguments: str) -> str:
    """Run the example's local function. Unknown names return an error string."""
    del arguments
    if name == "get_current_time":
        return datetime.now(timezone.utc).isoformat()
    return json.dumps({"error": f"unknown tool {name}"})


def qwen_tool_session_config() -> SessionConfig:
    return SessionConfig(
        auto_response=False,
        instructions="When the user asks for the time, call get_current_time.",
        extra_body={
            "realtime_tools": [
                {
                    "type": "function",
                    "name": "get_current_time",
                    "description": "Return the current UTC time.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        },
    )


def function_call_from_event(event: dict[str, object]) -> dict[str, str] | None:
    if event.get("type") != "response.output_item.done":
        return None
    item = event.get("item")
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return None
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments", "")
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        return None
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"call_id": call_id, "name": name, "arguments": arguments}


async def answer_function_calls(client: DuplexClient) -> None:
    answered: set[str] = set()
    async for event in client.events():
        call = function_call_from_event(event.raw)
        if call is None or call["call_id"] in answered:
            continue
        answered.add(call["call_id"])
        output = execute_local_tool(call["name"], call["arguments"])
        await client.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call["call_id"], "output": output},
            }
        )
        await client.send({"type": "response.create"})


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8091/v1/realtime")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    async with DuplexClient(args.url, model=args.model, config=qwen_tool_session_config()) as client:
        await answer_function_calls(client)


if __name__ == "__main__":
    asyncio.run(_main())
