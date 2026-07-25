# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DuplexSamplingRow:
    """Request-local context for the experimental duplex sampling hook."""

    row_idx: int
    request_id: str
    session_id: str | None
    incarnation: int
    seq: int | None
    payload: dict[str, object] | None
    max_tokens: int | None


class DuplexSamplingHelper:
    """Tracks duplex sampling rows outside the generic AR model runner."""

    def __init__(self) -> None:
        self.active_request_ids: set[str] = set()
        self.hook_active = False

    @staticmethod
    def _is_duplex_data_plane_info(info: object) -> bool:
        duplex = info.get("duplex") if isinstance(info, dict) else None
        return isinstance(duplex, dict) and duplex.get("data_plane") is True

    @staticmethod
    def _request_intermediate_info(runner: object, req_id: str) -> dict[str, Any] | None:
        model_intermediate_buffer = getattr(runner, "model_intermediate_buffer", {})
        info = model_intermediate_buffer.get(req_id) if isinstance(model_intermediate_buffer, dict) else None
        if isinstance(info, dict):
            return info
        requests = getattr(runner, "requests", {})
        req_state = requests.get(req_id) if isinstance(requests, dict) else None
        info = getattr(req_state, "additional_information_cpu", None)
        return info if isinstance(info, dict) else None

    def refresh_active_request(self, runner: object, req_id: str) -> None:
        if self._is_duplex_data_plane_info(self._request_intermediate_info(runner, req_id)):
            self.active_request_ids.add(req_id)
        else:
            self.active_request_ids.discard(req_id)

    def update_states(self, runner: object, scheduler_output: object) -> None:
        self.active_request_ids.difference_update(str(req_id) for req_id in scheduler_output.finished_req_ids)
        for request in scheduler_output.scheduled_new_reqs:
            self.refresh_active_request(runner, str(request.req_id))

    def rows(self, runner: object) -> tuple[DuplexSamplingRow, ...]:
        rows: list[DuplexSamplingRow] = []
        req_ids = [str(req_id) for req_id in getattr(runner.input_batch, "req_ids", [])]
        requests = getattr(runner, "requests", {})
        for row_idx, req_id in enumerate(req_ids):
            if req_id not in self.active_request_ids:
                continue
            info = self._request_intermediate_info(runner, req_id)
            duplex = info.get("duplex") if isinstance(info, dict) else None
            if not isinstance(duplex, dict) or duplex.get("data_plane") is not True:
                continue
            session_id = duplex.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                session_id = None
            try:
                incarnation = int(duplex.get("incarnation", 0))
            except (TypeError, ValueError):
                incarnation = 0
            try:
                seq = int(duplex.get("seq"))
            except (TypeError, ValueError):
                seq = None
            payload = duplex.get("payload")
            if not isinstance(payload, dict):
                payload = None
            request = requests.get(req_id) if isinstance(requests, dict) else None
            sampling_params = getattr(request, "sampling_params", None)
            try:
                max_tokens = int(getattr(sampling_params, "max_tokens", 0) or 0)
            except (TypeError, ValueError):
                max_tokens = 0
            rows.append(
                DuplexSamplingRow(
                    row_idx=row_idx,
                    request_id=req_id,
                    session_id=session_id,
                    incarnation=incarnation,
                    seq=seq,
                    payload=payload,
                    max_tokens=max_tokens if max_tokens > 0 else None,
                )
            )
        return tuple(rows)

    def clear(self) -> None:
        self.active_request_ids.clear()
        self.hook_active = False


__all__ = ["DuplexSamplingHelper", "DuplexSamplingRow"]
