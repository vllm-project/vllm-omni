# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional stage-owned policy for streaming request windows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request, StreamingUpdate

    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler


class WindowOperation(NamedTuple):
    """An ordered worker instruction, applied before finished-request cleanup.

    Keep captures the finished request's window. Hand transfers a kept window
    to a new request. Drop discards the session's window and needs no request.
    """

    kind: Literal["keep", "hand", "drop"]
    session_id: str
    request_id: str = ""


class SchedulerWindowPolicy(Protocol):
    def prepare_append(
        self,
        scheduler: OmniARScheduler,
        request: Request,
        update: StreamingUpdate,
        *,
        segment_output_ids: list[int],
        completed_terminator: int | None,
    ) -> int | None:
        """Prepare the append, returning the projected length if it replaces the prompt."""
        ...

    def before_finish(self, requests: Mapping[str, Request], request_ids: Iterable[str], *, aborted: bool) -> None:
        """Observe live requests before upstream releases them."""
        ...

    def before_book(self, requests: Mapping[str, Request], output: SchedulerOutput) -> None:
        """Commit snapshots for the final scheduled spans before upstream books them."""
        ...

    def on_schedule(self, requests: Mapping[str, Request], output: SchedulerOutput) -> list[WindowOperation]:
        """Return this dispatch's worker operations in application order, including finish-only steps."""
        ...
