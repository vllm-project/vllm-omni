# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Receive request-scoped vLLM action callbacks in the vLLM-Omni policy process."""

import json
import threading
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from vllm_omni.entrypoints.cosmos3.action_policy_http import ActionPolicyHandler, ActionPolicyHTTPServer


# 接收和管理动作事件的通道
class ActionStreamChannel:
    def __init__(self, publish: Callable[[dict[str, Any]], None], action_count: int = 16) -> None:
        self.publish = publish
        self.action_count = action_count
        self.lock = threading.Lock()
        self.seen = np.zeros(16, dtype=bool)
        self.actions = np.zeros((16, 10), dtype=np.float32)
        self.last_step = 0

    def accept(self, event: dict[str, Any]) -> None:
        if event.get("policy_profile") != "libero10_native_nonstream_v1" or event.get("action_stream_version") != 1:
            raise ValueError("The backend did not confirm the action stream profile")
        indices = event.get("action_indices")
        step = event.get("step")
        if (
            event.get("type") != "partial"
            or event.get("total_steps") != 10
            or not isinstance(indices, list)
            or not indices
            or any(type(index) is not int or not 0 <= index < self.action_count for index in indices)
            or len(set(indices)) != len(indices)
            or type(step) is not int
            or not 1 <= step <= 10
        ):
            raise ValueError("Invalid action stream indices or denoising step")
        actions = np.asarray(event.get("actions"), dtype=np.float32)
        if actions.shape != (len(indices), 10) or not np.isfinite(actions).all():
            raise ValueError("Invalid action stream values")
        with self.lock:
            if step < self.last_step or self.seen[indices].any():
                raise ValueError("Duplicate or out-of-order action stream event")
            self.publish({**event, "actions": actions})
            self.actions[indices] = actions
            self.seen[indices] = True
            self.last_step = step

    def verify_final(self, actions: np.ndarray) -> None:
        with self.lock:
            if not self.seen[: self.action_count].all():
                raise RuntimeError(
                    "vLLM did not stream every action; use the vLLM-Omni Cosmos3 HAS pipeline and restart"
                )
            if actions.shape != (self.action_count, 10) or not np.allclose(
                self.actions[: self.action_count], actions, rtol=0, atol=1e-6
            ):
                raise RuntimeError("Streamed actions differ from the final action chunk")


# 注册通道生成token
class ActionStreamRegistry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.channels: dict[str, ActionStreamChannel] = {}

    def add(self, channel: ActionStreamChannel) -> str:
        token = uuid.uuid4().hex
        with self.lock:
            self.channels[token] = channel
        return token

    def remove(self, token: str) -> None:
        with self.lock:
            self.channels.pop(token, None)

    def accept(self, token: str, event: dict[str, Any]) -> None:
        with self.lock:
            channel = self.channels[token]
        channel.accept(event)


class RemoteActionPolicyHandler(ActionPolicyHandler):
    def do_POST(self) -> None:  # noqa: N802
        prefix = "/internal/action_events/"
        if not self.path.startswith(prefix):
            super().do_POST()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024 * 1024:
                raise ValueError("Invalid action callback Content-Length")
            event = json.loads(self.rfile.read(length))
            if not isinstance(event, dict):
                raise ValueError("Action callback must be an object")
            self.server.service.streams.accept(self.path[len(prefix) :], event)
            self._send_json(200, {"accepted": True})
        except KeyError:
            self._send_json(404, {"error": "Unknown or expired action stream"})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"error": str(exc)})


class RemoteActionPolicyHTTPServer(ActionPolicyHTTPServer):
    def __init__(self, address: tuple[str, int], service: Any) -> None:
        super().__init__(address, service)
        self.RequestHandlerClass = RemoteActionPolicyHandler
