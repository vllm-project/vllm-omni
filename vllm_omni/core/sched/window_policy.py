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


class WindowCut(NamedTuple):
    """A settled input range and the generated/closing tokens to replay after it."""

    seq: int
    start: int
    rows: int
    accepted_ids: tuple[int, ...]
    trailing_ids: tuple[int, ...]


class WindowOperation(NamedTuple):
    """An ordered worker instruction, applied before finished-request cleanup.

    Keep captures a finished request. Restore binds its accepted cut to an
    identified successor. Hand and drop deliver or expire a bounded offer.
    """

    kind: Literal["keep", "hand", "drop", "restore"]
    session_id: str
    request_id: str = ""
    cut: WindowCut | None = None
    source_request_id: str = ""


class SchedulerWindowPolicy(Protocol):
    def release_session(self, session_id: str) -> None:
        """Release retained request windows on logical session teardown."""
        ...

    def superseded_by(self, request: Request, live: Mapping[str, Request]) -> list[str]:
        """Identify older response epochs fenced by this admission."""
        ...

    def prepare_admission(self, request: Request, max_prompt_len: int) -> int | None:
        """Prepare the initial prefill, or return its required length if it cannot fit."""
        ...

    def observe_progress(self, request: Request) -> None:
        """Observe settled model inputs and accepted samples before segment rollover."""
        ...

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
