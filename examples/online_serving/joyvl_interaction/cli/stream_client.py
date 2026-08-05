# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import requests


@dataclass
class Tick:
    index: int
    t: float
    action: str
    text: str
    skipped: bool
    latency_ms: float
    delegation: dict | None


def iter_frames(video_path: str, fps: float) -> Iterator[tuple[float, bytes]]:
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield idx / src_fps, buf.tobytes()
        idx += 1
    cap.release()


def stream(
    video_path: str,
    server: str,
    session_id: str,
    query: str | None = None,
    fps: float = 1.0,
    realtime: bool = False,
) -> Iterator[Tick]:
    """Send each sampled frame as one chat completion; yield the parsed action.

    ``query`` is attached to the first tick to arm the session; inject later
    queries by posting another frame turn with a new text part.
    """
    url = server.rstrip("/") + "/v1/chat/completions"
    requests.post(server.rstrip("/") + "/reset", json={"session_id": session_id}, timeout=10)
    for i, (t, jpeg) in enumerate(iter_frames(video_path, fps)):
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        content = [{"type": "image_url", "image_url": {"url": data_url}}]
        if i == 0 and query:
            content.insert(0, {"type": "text", "text": query})
        resp = requests.post(
            url,
            json={"session_id": session_id, "messages": [{"role": "user", "content": content}]},
            timeout=120,
        ).json()
        info = resp.get("interaction", {})
        yield Tick(
            index=info.get("frame_index", i + 1),
            t=t,
            action=info.get("action", "?"),
            text=resp["choices"][0]["message"]["content"],
            skipped=info.get("inference_skipped", False),
            latency_ms=info.get("latency_ms", 0.0),
            delegation=info.get("delegation"),
        )
        if realtime:
            time.sleep(1.0 / fps)
