# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model-neutral input delivery shared by the duplex benchmarks.

OmniInteract and Omni-DuplexEval replay a recorded clip into one duplex
session. Two properties of that session decide how the clip has to be sent:

* **Who ends a turn.** A model-native duplex model (MiniCPM-o) decides on its
  own when to speak, so the benchmark streams the clip and commits once at the
  end. A turn-based model (Qwen3-Omni) only answers a committed turn; with
  ``server_vad`` turn detection the server cuts the user's speech into turns,
  so the benchmark must not commit and must wait for every turn to settle.
* **How camera frames travel.** Audio-native models take frames on the audio
  append (``video_frames``). Models that refuse that field but accept
  ``input_image`` conversation items get the frames as items, inside the
  server's bounded image context (8 images / 4 MiB, see
  ``vllm_omni/engine/duplex/session/control.py``).

This module imports no client or server runtime code, so both benchmarks can
use it without crossing the duplex import boundary.
"""

from __future__ import annotations

import io
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence

import pybase64 as base64

TURN_DETECTION_NONE = "none"
TURN_DETECTION_SERVER_VAD = "server_vad"
TURN_DETECTION_MODES = (TURN_DETECTION_NONE, TURN_DETECTION_SERVER_VAD)

FRAME_TRANSPORT_APPEND = "append"
FRAME_TRANSPORT_IMAGE_ITEMS = "image_items"

#: Mirrors the server's image context bound on ``conversation.item.create``.
MAX_IMAGE_ITEMS = 8
_IMAGE_CONTEXT_BYTES = 4 * 1024 * 1024
#: Largest side of an image item. Frames are downscaled only when needed to
#: keep a full window inside the server's byte budget.
_IMAGE_ITEM_MAX_SIDE = 640
_IMAGE_ITEM_JPEG_QUALITY = 85

#: Trailing silence appended after the clip in ``server_vad`` mode. It must be
#: longer than the VAD's ``silence_duration_ms`` (500 ms by default) so the
#: server closes a question spoken right at the end of the clip.
SERVER_VAD_TAIL_S = 1.5


def session_turn_detection(mode: str) -> dict[str, object] | None:
    """Return the ``turn_detection`` session field for a benchmark mode."""
    if mode == TURN_DETECTION_NONE:
        return None
    if mode == TURN_DETECTION_SERVER_VAD:
        # Server defaults (threshold 0.5, 500 ms silence, create and interrupt
        # responses) are the reference Qwen3-Omni duplex configuration.
        return {"type": "server_vad"}
    raise ValueError(f"turn detection must be one of {TURN_DETECTION_MODES}, got {mode!r}")


def select_frame_transport(capabilities: Mapping[str, object] | None) -> str:
    """Pick how camera frames reach a session from its advertised capabilities.

    A session that advertises no input-modality contract keeps the historical
    ``video_frames`` append, which is what every audio-native duplex model
    accepts.
    """
    modality_keys = ("required_input_modalities", "optional_input_modalities")
    if not capabilities or not any(key in capabilities for key in modality_keys):
        return FRAME_TRANSPORT_APPEND
    accepted: set[str] = set()
    for key in modality_keys:
        value = capabilities.get(key)
        if isinstance(value, str):
            accepted.add(value)
        elif isinstance(value, Iterable):
            accepted.update(str(item) for item in value)
    if "video" in accepted:
        return FRAME_TRANSPORT_APPEND
    if capabilities.get("supports_image_input"):
        return FRAME_TRANSPORT_IMAGE_ITEMS
    raise ValueError("The duplex session accepts neither video_frames on appends nor input_image items")


def _fit_image_item(frame: str | bytes) -> str:
    """Return a bare base64 JPEG no larger than the per-item share of the budget."""
    raw = frame if isinstance(frame, bytes) else base64.b64decode(frame.split(",", 1)[-1], validate=True)
    if len(raw) * 4 // 3 <= _IMAGE_CONTEXT_BYTES // (MAX_IMAGE_ITEMS + 1):
        return base64.b64encode(raw).decode("ascii")
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        image.thumbnail((_IMAGE_ITEM_MAX_SIDE, _IMAGE_ITEM_MAX_SIDE))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=_IMAGE_ITEM_JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ImageItemWindow:
    """Send frames as ``input_image`` items, keeping at most ``MAX_IMAGE_ITEMS``.

    The server refuses an item that would push the session past its image
    context, so the oldest frame is deleted before each new one once the window
    is full. Commands on one session are processed in order, so the delete is
    applied before the create that follows it.
    """

    def __init__(
        self,
        send: Callable[[dict[str, object]], Awaitable[object]],
        *,
        item_prefix: str,
        max_items: int = MAX_IMAGE_ITEMS,
    ) -> None:
        if not 1 <= max_items <= MAX_IMAGE_ITEMS:
            raise ValueError(f"max_items must be between 1 and {MAX_IMAGE_ITEMS}")
        self._send = send
        self._prefix = item_prefix
        self._max_items = max_items
        self._live: deque[str] = deque()
        self.sent = 0

    async def push(self, frames: Sequence[str | bytes]) -> None:
        for frame in frames:
            while len(self._live) >= self._max_items:
                await self._send({"type": "conversation.item.delete", "item_id": self._live.popleft()})
            item_id = f"{self._prefix}_{self.sent}"
            await self._send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "data:image/jpeg;base64," + _fit_image_item(frame)}
                        ],
                    },
                }
            )
            self._live.append(item_id)
            self.sent += 1


def _response_id(event: Mapping[str, object]) -> str | None:
    response_id = event.get("response_id")
    if isinstance(response_id, str) and response_id:
        return response_id
    response = event.get("response")
    nested = response.get("id") if isinstance(response, Mapping) else None
    return nested if isinstance(nested, str) and nested else None


def turn_activity_count(events: Sequence[Mapping[str, object]]) -> int:
    """Count input and response events, the ones that show a turn is still moving.

    Playback acknowledgements are left out: a client that acks on a timer
    would otherwise keep a session from ever looking quiet.
    """
    return sum(
        1
        for event in events
        if isinstance(event_type := event.get("type"), str)
        and event_type.startswith(("input_audio_buffer.", "response."))
    )


def server_turns_pending(events: Sequence[Mapping[str, object]]) -> bool:
    """Whether server-detected speech or a response is still open.

    Open speech is a ``speech_started`` without its ``speech_stopped``; an open
    response is a ``response.created`` without its ``response.done``.
    """
    speaking = False
    created: set[str] = set()
    done: set[str] = set()
    for event in events:
        event_type = event.get("type")
        if event_type == "input_audio_buffer.speech_started":
            speaking = True
        elif event_type == "input_audio_buffer.speech_stopped":
            speaking = False
        elif event_type in {"response.created", "response.done"}:
            response_id = _response_id(event)
            if response_id:
                (created if event_type == "response.created" else done).add(response_id)
    return speaking or bool(created - done)
