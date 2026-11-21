# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, metadata-only wire traces for the public duplex client."""

from __future__ import annotations

import json
import math
import time
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path
from typing import Literal


class DuplexTrace:
    """Record client-observed wire events without retaining media or text.

    Attach one instance to one ``DuplexClient``. Recording does no file I/O.
    Call :meth:`write_json` after recording has stopped. If offloading export
    to a worker thread, do not record concurrently: snapshots are not thread-safe.
    The oldest records are dropped when ``max_events`` is reached.
    """

    def __init__(self, *, max_events: int = 4096) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        self._events: deque[dict[str, object]] = deque(maxlen=max_events)
        self._origin_s: float | None = None
        self._total_events = 0

    def record(self, direction: Literal["send", "receive"], event: Mapping[str, object]) -> None:
        """Record a completed send or a decoded receive, before replay filtering."""
        now = time.monotonic()
        if self._origin_s is None:
            self._origin_s = now
        row: dict[str, object] = {
            "index": self._total_events,
            "elapsed_s": now - self._origin_s,
            "direction": direction,
        }
        # Copy only scalar metadata. Never retain raw payloads, nested output,
        # transcript deltas, instructions, reference audio, or resume tokens.
        # Omit oversized identifiers rather than truncate and merge distinct IDs.
        for key in ("type", "event_id", "session_id", "response_id", "item_id"):
            value = event.get(key)
            if isinstance(value, str) and len(value) <= 256:
                row[key] = value
        for key in ("server_event_seq", "played_ms", "sample_rate_hz"):
            value = event.get(key)
            if type(value) is int or (type(value) is float and math.isfinite(value)):
                row[key] = value
        for key, fields in (
            ("response", (("id", "response_id"), ("status", "response_status"))),
            ("error", (("code", "error_code"), ("event_id", "related_event_id"))),
        ):
            nested = event.get(key)
            if isinstance(nested, dict):
                for source, target in fields:
                    value = nested.get(source)
                    if isinstance(value, str) and len(value) <= 256:
                        row.setdefault(target, value)
        session = event.get("session")
        if isinstance(session, dict):
            value = session.get("id") or session.get("session_id")
            if isinstance(value, str) and len(value) <= 256:
                row.setdefault("session_id", value)
        self._events.append(row)
        self._total_events += 1

    def snapshot(self) -> dict[str, object]:
        """Return an independent, JSON-serializable diagnostic snapshot.

        Counts describe retained wire events, including resume replay. They
        are not unique-response, success-rate, or server latency metrics.
        """
        events = [dict(row) for row in self._events]
        counts = Counter(f"{row['direction']}:{row.get('type', 'unknown')}" for row in events)
        responses: dict[str, dict[str, object]] = {}
        milestones = {
            "receive:response.created": "created_s",
            "receive:response.output_audio.delta": "first_audio_s",
            "receive:response.done": "done_s",
            "send:response.cancel": "cancel_sent_s",
        }
        for row in events:
            response_id = row.get("response_id")
            if not isinstance(response_id, str):
                continue
            response = responses.setdefault(response_id, {"response_id": response_id})
            milestone = milestones.get(f"{row['direction']}:{row.get('type')}")
            if milestone is not None:
                response.setdefault(milestone, row["elapsed_s"])
            if milestone == "done_s" and "response_status" in row:
                response.setdefault("status", row["response_status"])
        return {
            "schema_version": 1,
            "clock": "client_monotonic",
            "total_events": self._total_events,
            "dropped_events": self._total_events - len(events),
            "event_counts": dict(counts),
            "responses": list(responses.values()),
            "events": events,
        }

    def write_json(self, path: str | Path) -> None:
        """Write a snapshot synchronously; file errors propagate to the caller."""
        Path(path).write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
