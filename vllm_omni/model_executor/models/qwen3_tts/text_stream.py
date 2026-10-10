# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Text that reaches a Qwen3-TTS take after it starts, and its text lead.

A client that knows only part of a take's text when it starts the take names
a stream in ``additional_information["text_stream"]`` and appends the text's
token ids to it as they arrive; ``additional_information["text_lead"]`` places
the first ``k`` ids before codec BOS (the text-lead layout of a checkpoint
trained on leads). The talker appends newly arrived ids to the take's
trailing text queue at each decode step, and the scheduler holds a take whose
next text row has not arrived, so a take waits for text instead of reading
PAD before its text END.

A stream is two tmpfs files: ``<key>.ids`` (int32 little-endian token ids,
appended) and ``<key>.end``, created once every id is written.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np

STREAM_DIR = os.environ.get("VLLM_OMNI_TTS_TEXT_STREAM_DIR", "/dev/shm/qwen3-tts-text")


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def stream_key(info: Mapping[str, Any] | None) -> str | None:
    """The take's text stream, or ``None`` when its text was complete at start."""
    if not info:
        return None
    key = _first(info.get("text_stream"))
    return str(key) if key else None


def text_lead(info: Mapping[str, Any] | None) -> int:
    """How many text ids sit before codec BOS (0: the streaming layout)."""
    if not info:
        return 0
    lead = _first(info.get("text_lead"))
    return int(lead) if lead else 0


def _paths(key: str) -> tuple[str, str]:
    return os.path.join(STREAM_DIR, f"{key}.ids"), os.path.join(STREAM_DIR, f"{key}.end")


def read_ids(key: str, start: int = 0) -> tuple[np.ndarray, bool]:
    """Ids from ``start`` on, and whether the stream has ended.

    The end marker is read first: once it is seen, every id is in the file.
    """
    ids_path, end_path = _paths(key)
    ended = os.path.exists(end_path)
    try:
        with open(ids_path, "rb") as f:
            f.seek(4 * start)
            data = f.read()
    except FileNotFoundError:
        return np.empty(0, dtype=np.int32), ended
    whole = len(data) // 4 * 4
    return np.frombuffer(data[:whole], dtype="<i4").astype(np.int64), ended


def count(key: str) -> tuple[int, bool]:
    """How many ids the stream holds, and whether it has ended."""
    ids_path, end_path = _paths(key)
    ended = os.path.exists(end_path)
    try:
        size = os.path.getsize(ids_path)
    except FileNotFoundError:
        size = 0
    return size // 4, ended
