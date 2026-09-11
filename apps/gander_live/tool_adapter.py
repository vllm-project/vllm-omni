# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A deliberately bounded local lookup tool, executed only on model calls."""

import asyncio
import json
from pathlib import Path

from output_audit import OutputAudit

DATA = Path(__file__).parent / "pickup.json"
TOOLS = [
    {
        "name": "task_start",
        "description": "新建后台查询任务。此演示仅支持查询本地取货暗号；name 是简短任务名称。",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    }
]


async def relay(client, backend):
    audit = OutputAudit()
    epoch = 0
    seen = set()
    jobs = set()

    async def execute(call):
        call_id = call["call_id"]
        try:
            args = json.loads(call["arguments"])
            if call["name"] != "task_start" or set(args) != {"name"} or not isinstance(args["name"], str):
                raise ValueError("Unsupported tool or arguments")
            await client.send_json({"type": "local.tool.started", "call": call})
            await backend.send(
                json.dumps(
                    {
                        "type": "input.context.append",
                        "context": {
                            "kind": "runtime_event",
                            "event_id": f"progress-{call_id}",
                            "epoch": epoch,
                            "call_id": call_id,
                            "output": {"status": "running", "progress": "正在读取 Mac 本地取货记录"},
                        },
                    },
                    ensure_ascii=False,
                )
            )
            await asyncio.sleep(2)
            record = json.loads(DATA.read_text())
            result = {
                "status": "completed",
                "answer": record["answer"],
                "source": "Mac 本地 pickup.json",
                "scope": "演示查询，不执行其他业务任务",
            }
        except Exception as exc:
            result = {"status": "failed", "error": str(exc)}
        await backend.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": f"result-{call_id}",
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    },
                },
                ensure_ascii=False,
            )
        )
        await client.send_json({"type": "local.tool.completed", "call_id": call_id, "result": result})

    try:
        async for raw in backend:
            event = json.loads(raw)
            audit.write(event)
            await client.send_text(raw)
            native = event.get("event", event)
            if isinstance(native.get("epoch"), int):
                epoch = native["epoch"]
            if event.get("type") == "response.output_item.done":
                call = event.get("item", {})
                if call.get("type") == "function_call" and call.get("call_id") not in seen:
                    seen.add(call["call_id"])
                    task = asyncio.create_task(execute(call))
                    jobs.add(task)
                    task.add_done_callback(jobs.discard)
    finally:
        audit.close()
        for task in jobs:
            task.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
