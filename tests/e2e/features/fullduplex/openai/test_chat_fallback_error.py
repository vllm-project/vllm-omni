# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression test for the duplex chat-fallback error path.

``ErrorResponse`` carries its detail nested under ``error: ErrorInfo`` and has
no flat ``message`` / ``type``, so reading those raised ``AttributeError`` from
inside the error branch. The outer handler then reported *that* to the client,
losing the real cause.
"""

from __future__ import annotations

import pytest
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse

from vllm_omni.experimental.fullduplex.openai.chat_fallback import (
    ChatFallbackProjectorMixin,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Session:
    session_id = "s1"
    epoch = 0
    input_commit_seq = 0
    active_request_id = ""

    def begin_response(self) -> str:
        return "resp-1"

    def bind_request(self, request_id: str) -> None:
        self.active_request_id = request_id

    def end_response(self, **_kwargs: object) -> None:
        pass


class _ErroringChatService:
    async def create_chat_completion(self, _request: object, raw_request: object = None) -> ErrorResponse:
        return ErrorResponse(error=ErrorInfo(message="boom", type="bad_request", code=400))


class _Projector(ChatFallbackProjectorMixin):
    _chat_service = _ErroringChatService()

    @staticmethod
    def _response_created_payload(_session: _Session, response_id: str, **_kwargs: object) -> dict[str, str]:
        return {"type": "response.created", "response_id": response_id}

    @staticmethod
    def _build_chat_request(_session: _Session, _request_id: str) -> object:
        return object()


@pytest.mark.asyncio
async def test_fallback_reports_the_underlying_error() -> None:
    """The handler must surface the real message, not the crash from reading it.

    Before the fix the flat attribute access raised, the outer ``except`` caught
    it, and the client was told ``'ErrorResponse' object has no attribute
    'message'`` with code ``response_error`` instead of the real failure.
    """
    sent: list[dict] = []

    async def send_json(payload: dict) -> None:
        sent.append(payload)

    await _Projector()._run_response(_Session(), send_json)

    errors = [event for event in sent if event.get("type") == "error"]
    assert errors, f"no error event emitted; got {[event.get('type') for event in sent]}"
    assert errors[0]["error"] == "boom"
    assert errors[0]["code"] == "bad_request"
