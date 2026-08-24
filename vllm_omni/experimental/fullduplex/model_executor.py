# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.sample.metadata import SamplingMetadata


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


class DuplexSamplingRunnerMixin:
    """Runner-side wiring for the experimental duplex sampling hook.

    GPU and NPU AR runners are siblings, not parent and child, so neither
    inherits the other's ``_sample``. Keep the hook here and have each runner
    opt in, instead of copying the GPU ``_sample`` path and dropping the
    ``prepare_duplex_sampling`` call.
    """

    def _init_duplex_sampling_state(self) -> None:
        """Reset hook resolution. Call from ``__init__``, before ``load_model``."""
        self._duplex_sampling_hook = None
        self._duplex_sampling_hook_resolved = False

    def _resolve_duplex_sampling_hook(self, *, force: bool = False):
        """Bind ``model.prepare_duplex_sampling``. Call with ``force`` after load."""
        if not force and getattr(self, "_duplex_sampling_hook_resolved", False):
            return self._duplex_sampling_hook
        candidate = getattr(getattr(self, "model", None), "prepare_duplex_sampling", None)
        self._duplex_sampling_hook = candidate if callable(candidate) else None
        self._duplex_sampling_hook_resolved = True
        if self._duplex_sampling_hook is not None and not hasattr(self, "_duplex_sampling_helper"):
            self._duplex_sampling_helper = DuplexSamplingHelper()
        return self._duplex_sampling_hook

    def _update_duplex_sampling_states(self, scheduler_output: SchedulerOutput) -> None:
        """Refresh which batch rows are duplex rows. Call from ``_update_states``."""
        if self._resolve_duplex_sampling_hook() is None:
            return
        helper = getattr(self, "_duplex_sampling_helper", None)
        if helper is not None:
            helper.update_states(self, scheduler_output)

    def _apply_duplex_sampling(self, logits: torch.Tensor, prepared_sampling_metadata: SamplingMetadata) -> None:
        """Hand the model its duplex rows. Call from ``_sample``, before the sampler."""
        prepare_duplex_sampling = self._resolve_duplex_sampling_hook()
        if prepare_duplex_sampling is None:
            return
        helper = getattr(self, "_duplex_sampling_helper", None)
        rows = helper.rows(self) if helper is not None and helper.active_request_ids else ()
        if rows or (helper is not None and helper.hook_active):
            prepare_duplex_sampling(logits, prepared_sampling_metadata, rows)
        if helper is not None:
            helper.hook_active = bool(rows)

    def _clear_duplex_sampling(self) -> None:
        helper = getattr(self, "_duplex_sampling_helper", None)
        if helper is not None:
            helper.clear()


__all__ = ["DuplexSamplingHelper", "DuplexSamplingRow", "DuplexSamplingRunnerMixin"]
