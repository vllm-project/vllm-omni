# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from openai.types import realtime as types
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.entrypoints.openai_api.test_realtime_prompt_preflight import _make_connection
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime.connection import _ResolvedResponse
from vllm_omni.entrypoints.openai.realtime.session import ActiveResponse
from vllm_omni.tracing import RequestTrace, capture_trace_headers

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "send_done"),
    [("completed", True), ("completed", False), ("incomplete", True), ("failed", True), ("cancelled", True)],
)
async def test_realtime_session_turn_parent_and_cleanup(mocker, monkeypatch, outcome, send_done):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    engine = mocker.Mock(spec=AsyncOmni)
    engine.engine = mocker.Mock(tracer_provider=provider)
    websocket = mocker.AsyncMock(spec=WebSocket)
    websocket.headers = {
        "traceparent": "00-12345678901234567890123456789012-1234567890123456-01",
        "tracestate": "vendor=state",
    }
    websocket.receive_text.side_effect = [
        '{"type":"response.create"}',
        '{"type":"response.create"}',
        WebSocketDisconnect(),
    ]
    connection, _ = _make_connection(websocket=websocket, engine=engine)
    response = _ResolvedResponse(None, None, ["text"], 16, None, "none", None)
    response_ids = iter(("first", "second"))

    async def dispatch(_event):
        response_id = next(response_ids)
        connection.session.active_response = ActiveResponse(response_id=response_id, request_id=response_id)
        connection._response_cancel_event.clear()
        await connection._run_response(response_id, response, {"prompt_token_ids": [1]})

    async def generate(response_id, resolved, session, active, engine_input):
        request = RequestTrace(provider.get_tracer("test"), active.request_id, capture_trace_headers())
        request.end()
        if outcome == "failed":
            raise RuntimeError("private backend error")
        if outcome == "cancelled":
            connection._response_cancel_event.set()
            raise asyncio.CancelledError
        if send_done:
            await connection._send_event(
                types.ResponseDoneEvent(
                    event_id=response_id,
                    type="response.done",
                    response=connection._response_object(response_id, resolved, outcome),
                )
            )

    monkeypatch.setattr(connection, "_dispatch_event", dispatch)
    monkeypatch.setattr(connection, "_run_response_inner", generate)
    try:
        await connection.handle_connection()
        spans = exporter.get_finished_spans()
        session = next(span for span in spans if span.name == "omni.session")
        turns = [span for span in spans if span.name == "omni.turn"]
        pipelines = [span for span in spans if span.name == "omni.pipeline"]
        assert session.parent.span_id == int("1234567890123456", 16)
        assert session.context.trace_state.to_header() == "vendor=state"
        assert len(turns) == len(pipelines) == 2
        assert all(turn.parent.span_id == session.context.span_id for turn in turns)
        assert all(turn.attributes["omni.turn.outcome"] == outcome for turn in turns)
        assert all((turn.status.status_code == trace.StatusCode.ERROR) == (outcome == "failed") for turn in turns)
        assert {pipeline.parent.span_id for pipeline in pipelines} == {turn.context.span_id for turn in turns}
        assert connection.session.active_response is None
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        provider.shutdown()
