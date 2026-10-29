# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import runpy
import socket
import sys
import time
from pathlib import Path

import anyio
import httpx
import janus
import pytest
import uvicorn
from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.engine.exceptions import EngineDeadError

from tests.engine.test_orchestrator import (
    FakeOutputProcessor,
    FakeStageClient,
    _build_request_output,
    _build_stage_pools,
    _sampling_params,
)
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import StageSubmissionMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_runtime import StageRuntimeInfo
from vllm_omni.errors import DEFAULT_CLIENT_ERROR_TYPE
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.tracing import RequestTrace, TraceContextMiddleware, capture_trace_headers, create_tracer_provider

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
HEADERS = {
    "traceparent": "00-12345678901234567890123456789012-1234567890123456-01",
    "tracestate": "vendor=state",
    "baggage": "tenant=gold",
}


@pytest.fixture
def recording():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("test"), exporter
    provider.shutdown()


def test_tracing_disabled_does_not_require_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk", None)
    tracing = runpy.run_path(str(Path(__file__).parents[2] / "vllm_omni" / "tracing.py"))
    assert tracing["create_tracer_provider"](None) is None
    assert tracing["capture_trace_headers"](HEADERS) == HEADERS


@pytest.mark.parametrize(
    ("protocol", "missing_module"),
    [
        ("grpc", "opentelemetry.sdk"),
        ("grpc", "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"),
        ("http/protobuf", "opentelemetry.exporter.otlp.proto.http.trace_exporter"),
    ],
)
def test_tracing_missing_dependency_has_install_hint(monkeypatch, protocol, missing_module):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", protocol)
    monkeypatch.setitem(sys.modules, missing_module, None)
    with pytest.raises(ImportError, match=r"pip install 'vllm-omni\[tracing\]'") as exc_info:
        create_tracer_provider("http://localhost:4317")
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
    assert exc_info.value.__cause__.name == missing_module


def test_context_preserves_unsampled_and_vendor_state(recording):
    tracer, exporter = recording
    headers = dict(HEADERS, traceparent=HEADERS["traceparent"][:-2] + "00")
    request = RequestTrace(tracer, "unsampled", headers)
    request.start_stage(0, 0, "llm")
    propagated = request.headers()
    request.end()
    assert propagated["traceparent"].endswith("-00")
    assert propagated["tracestate"] == headers["tracestate"]
    assert propagated["baggage"] == headers["baggage"]
    assert not exporter.get_finished_spans()
    assert create_tracer_provider(None) is None


def test_inbound_headers_do_not_skip_active_http_span(recording):
    tracer, exporter = recording
    context = TraceContextTextMapPropagator().extract(HEADERS)
    with tracer.start_as_current_span("HTTP", context=context) as http:
        carrier = capture_trace_headers(HEADERS)
        assert carrier["baggage"] == HEADERS["baggage"]
        request = RequestTrace(tracer, "req", carrier)
        assert request.headers()["baggage"] == HEADERS["baggage"]
        request.end()
    pipeline = next(s for s in exporter.get_finished_spans() if s.name == "omni.pipeline")
    assert pipeline.parent.span_id == http.get_span_context().span_id


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancelled", "error", "dead_replica"])
async def test_orchestrator_closes_diffusion_request_and_stage(recording, outcome):
    tracer, exporter = recording
    client = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orch = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=_build_stage_pools([[client]]),
        tracer=tracer,
    )
    for index in range(2):
        await orch._handle_add_request(
            StageSubmissionMessage(
                type="add_request",
                request_id=f"req-{index}",
                prompt={"prompt": "private"},
                original_prompt={"prompt": "private"},
                output_prompt_text=None,
                sampling_params_list=[OmniDiffusionSamplingParams()],
                final_stage_id=0,
                preprocess_ms=0,
                request_timestamp=time.time(),
                enqueue_ts=time.perf_counter(),
                trace_headers=HEADERS,
            )
        )
    if outcome == "success":
        for index in range(2):
            await orch._handle_processed_outputs(
                0,
                0,
                [
                    OmniRequestOutput.from_diffusion(
                        request_id=f"req-{index}",
                        images=[],
                        final_output_type="image",
                    )
                ],
            )
    elif outcome == "error":
        for index in range(2):
            await orch._fail_request_dead_stage(f"req-{index}", 0)
    elif outcome == "dead_replica":
        await orch._handle_dead_replica(0, 0, EngineDeadError())
    else:
        await orch._cleanup_request_ids(["req-0", "req-1"], abort=True)
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    pipelines = [s for s in spans if s.name == "omni.pipeline"]
    stages = [s for s in spans if s.name == "omni.stage"]
    expected_outcome = "error" if outcome == "dead_replica" else outcome
    for pipeline in pipelines:
        assert pipeline.context.trace_id == int("12345678901234567890123456789012", 16)
        assert pipeline.attributes["omni.request.outcome"] == expected_outcome
        assert pipeline.status.is_ok == (expected_outcome != "error")
        assert len([s for s in stages if s.parent.span_id == pipeline.context.span_id]) == 1
    if expected_outcome == "error":
        assert all(s.attributes["error.type"] == "engine_dead" for s in spans)
    assert not orch.request_states
    assert all("private" not in str(s.attributes) for s in spans)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["invalid_request_error", None])
async def test_stage_error_preserves_trace_error_type(recording, error_type):
    tracer, exporter = recording
    orch = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=_build_stage_pools([[FakeStageClient(final_output=True)]]),
        tracer=tracer,
    )
    request_trace = RequestTrace(tracer, "failed", HEADERS)
    request_trace.start_stage(0, 0, "llm")
    orch.request_states["failed"] = OrchestratorRequestState(
        request_id="failed", final_stage_id=0, sampling_params_list=[_sampling_params()], trace=request_trace
    )
    await orch._handle_stage_error(0, OmniRequestOutput.from_error("failed", "private error", error_type=error_type))
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert all(span.attributes["error.type"] == (error_type or "stage_error") for span in spans)
    assert all(span.status.status_code == trace.StatusCode.ERROR for span in spans)
    assert not orch.request_states


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["validation_error", None, ""])
@pytest.mark.parametrize("release_owners", [False, True])
async def test_client_error_trace_uses_default_for_missing_type(recording, error_type, release_owners):
    tracer, exporter = recording
    orch = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=_build_stage_pools([[FakeStageClient(final_output=True)]]),
        tracer=tracer,
    )
    request_trace = RequestTrace(tracer, "failed", HEADERS)
    request_trace.start_stage(0, 0, "llm")
    orch.request_states["failed"] = OrchestratorRequestState(
        request_id="failed", final_stage_id=0, sampling_params_list=[_sampling_params()], trace=request_trace
    )
    await orch._fail_request_client_error(
        "failed", 0, "invalid input", error_type=error_type, release_owners=release_owners
    )
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert all(span.attributes["error.type"] == (error_type or DEFAULT_CLIENT_ERROR_TYPE) for span in spans)
    assert all(span.attributes["omni.request.outcome"] == "error" for span in spans)
    assert all(span.status.status_code == trace.StatusCode.ERROR for span in spans)
    assert not orch.request_states


@pytest.mark.asyncio
async def test_first_output_check_stops_after_nonempty_output(recording, mocker):
    tracer, exporter = recording
    pool = _build_stage_pools([[FakeStageClient(final_output=True, final_output_type="text")]])[0]
    check_output = mocker.spy(pool, "has_non_empty_output")
    orch = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=[pool],
        tracer=tracer,
    )
    request_trace = RequestTrace(tracer, "stream", HEADERS)
    request_trace.start_stage(0, 0, "llm")
    req_state = OrchestratorRequestState(
        request_id="stream", final_stage_id=0, sampling_params_list=[_sampling_params()], trace=request_trace
    )
    for text in ("", "first", "second"):
        await orch._route_output(0, 0, _build_request_output("stream", text=text, finished=False), req_state, None)
    request_trace.end()
    stage = next(span for span in exporter.get_finished_spans() if span.name == "omni.stage")
    assert [event.name for event in stage.events] == ["omni.stage.first_output"]
    assert check_output.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("resumable", [True, False])
async def test_streaming_update_carries_explicit_parent(resumable):
    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = janus.Queue()
    engine.default_sampling_params_list = [OmniDiffusionSamplingParams()]
    engine.stage_metadata = [StageRuntimeInfo(stage_type="diffusion", final_output=True, final_output_type="image")]
    try:
        await engine.add_streaming_update_async(
            request_id="stream",
            prompt={"prompt": "hello"},
            trace_headers=HEADERS,
            resumable=resumable,
        )
        message = engine.request_queue.sync_q.get_nowait()
        assert message.type == "streaming_update"
        assert message.trace_headers == HEADERS
    finally:
        engine.request_queue.close()
        await engine.request_queue.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["close", "shutdown"])
async def test_duplex_session_and_overlapping_turns_end(recording, terminal):
    from tests.engine.duplex.test_session_manager import Harness
    from vllm_omni.engine.duplex.config import DuplexSessionConfig
    from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
    from vllm_omni.engine.duplex.events import ResponseCreated, ResponseDone
    from vllm_omni.engine.duplex.messages import OpenDuplexSessionMessage

    tracer, exporter = recording
    async with Harness.create() as harness:
        harness.manager.tracer = tracer
        await harness.manager.open(
            OpenDuplexSessionMessage(
                control_id="open",
                session_id="session",
                session_config=DuplexSessionConfig(model="fake-model"),
                output_buffer=DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512),
                trace_headers=HEADERS,
            )
        )
        assert (await harness.result()).ok
        session = harness.session("session")
        for response_id in ("first", "second"):
            harness.manager.emit(session, [ResponseCreated(response_id=response_id)])
        harness.manager.emit(session, [ResponseDone(response_id="first", response={"status": "completed"})])
        # A duplicate admission must leave the existing session and its spans alive.
        assert not (await harness.open("session")).ok
        assert "session" in harness.manager.session_traces
        if terminal == "close":
            await harness.close("session")
        else:
            await harness.manager.shutdown()
        assert not harness.manager.session_traces
    spans = exporter.get_finished_spans()
    session_span = next(span for span in spans if span.name == "omni.session")
    assert session_span.parent.span_id == int("1234567890123456", 16)
    assert session_span.context.trace_state.to_header() == "vendor=state"
    turns = [span for span in spans if span.name == "omni.turn"]
    assert len(turns) == 2
    assert {span.attributes["omni.turn.outcome"] for span in turns} == {"completed", "cancelled"}
    assert all(span.parent.span_id == session_span.context.span_id for span in turns)


@pytest.mark.asyncio
async def test_duplex_open_propagates_context_across_executor(recording):
    from tests.engine.test_duplex_omni_engine import _engine
    from vllm_omni.engine.duplex.config import DuplexSessionConfig
    from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
    from vllm_omni.engine.duplex.messages import DuplexControlResultMessage

    tracer, _ = recording
    engine = _engine(DuplexControlResultMessage(control_id="open", operation="open", session_id="session", ok=True))
    with tracer.start_as_current_span("upstream", context=TraceContextTextMapPropagator().extract(HEADERS)) as parent:
        await engine.open_session_async(
            "session",
            DuplexSessionConfig(model="fake"),
            output_buffer=DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512),
        )
    message = engine._correlated_rpc_client.calls[0][1]
    carrier = message.trace_headers
    assert carrier["tracestate"] == "vendor=state"
    assert carrier["traceparent"].split("-")[2] == f"{parent.get_span_context().span_id:016x}"


@pytest.mark.asyncio
async def test_keepalive_uploads_keep_independent_trace_parents() -> None:
    app = FastAPI()

    @app.post("/upload")
    async def upload(request: Request) -> dict[str, str | int]:
        await request.body()
        parent = trace.get_current_span().get_span_context()
        assert request.client is not None
        return {"trace_id": f"{parent.trace_id:032x}", "port": request.client.port}

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider, exclude_spans=["send", "receive"])
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(TraceContextMiddleware(app), log_level="error", http="h11"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            with anyio.fail_after(10):
                while not server.started:
                    await asyncio.sleep(0.01)
                async with httpx.AsyncClient(trust_env=False) as client:
                    ports: set[int] = set()
                    for index, sampled in enumerate(["01", "00", "01"], start=1):
                        trace_id = f"{index:032x}"
                        response = await client.post(
                            f"http://127.0.0.1:{listener.getsockname()[1]}/upload",
                            headers={
                                "traceparent": f"00-{trace_id}-1234567890123456-{sampled}",
                                "tracestate": "vendor=state",
                            },
                            content=b"x" * (1024 * 1024),
                        )
                        assert response.status_code == 200
                        assert response.json()["trace_id"] == trace_id
                        ports.add(response.json()["port"])
                    assert len(ports) == 1
        finally:
            server.should_exit = True
            await serving
            provider.shutdown()
    spans = exporter.get_finished_spans()
    assert [span.context.trace_id for span in spans] == [1, 3]
    assert all(span.parent.span_id == int("1234567890123456", 16) for span in spans)
    assert all(span.context.trace_state.get("vendor") == "state" for span in spans)


@pytest.mark.asyncio
@pytest.mark.parametrize("segment_finished", [False, True])
@pytest.mark.parametrize("has_processed_output", [False, True])
async def test_raw_terminal_stage_trace_survives_downstream_failure(recording, segment_finished, has_processed_output):
    tracer, exporter = recording
    request_id = "raw-terminal"
    processed = [_build_request_output(request_id)] if has_processed_output else []
    orch = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=_build_stage_pools(
            [[FakeStageClient()], [FakeStageClient(final_output=True)]],
            output_processors=[FakeOutputProcessor(request_outputs=processed), FakeOutputProcessor()],
        ),
        tracer=tracer,
    )
    request_trace = RequestTrace(tracer, request_id, HEADERS)
    for stage_id in (0, 1):
        request_trace.start_stage(stage_id, 0, "llm")
    req_state = OrchestratorRequestState(
        request_id=request_id,
        final_stage_id=1,
        sampling_params_list=[_sampling_params(), _sampling_params()],
        stage_submit_ts={0: time.time(), 1: time.time()},
        skip_legacy_stage_forward=True,
        trace=request_trace,
    )
    req_state.streaming.enabled = True
    orch.request_states[request_id] = req_state
    raw = EngineCoreOutputs(
        outputs=[
            OmniEngineCoreOutput(
                request_id=request_id,
                new_token_ids=[],
                finish_reason=FinishReason.STOP,
                is_segment_finished=segment_finished,
            )
        ]
    )
    terminal_ids: set[str] = set()

    outputs = await orch._process_llm_stage_outputs(0, 0, raw, terminal_ids)
    await orch._handle_processed_outputs(0, 0, outputs)
    await orch._finish_raw_terminal_requests(0, 0, terminal_ids)
    await orch._finish_raw_terminal_requests(0, 0, terminal_ids)

    assert terminal_ids == (set() if segment_finished else {request_id})
    assert len(exporter.get_finished_spans()) == (0 if segment_finished else 1)
    assert request_id in orch.request_states

    await orch._process_llm_stage_outputs(
        1,
        0,
        EngineCoreOutputs(
            outputs=[
                OmniEngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=[],
                    finish_reason=FinishReason.ERROR,
                    stop_reason="downstream failed",
                )
            ]
        ),
        set(),
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    stages = {span.attributes["omni.stage.id"]: span for span in spans if span.name == "omni.stage"}
    pipeline = next(span for span in spans if span.name == "omni.pipeline")
    upstream = stages[0]
    assert upstream.status.status_code == (trace.StatusCode.ERROR if segment_finished else trace.StatusCode.UNSET)
    assert ("error.type" in upstream.attributes) == segment_finished
    assert [event.name for event in upstream.events] == (["omni.stage.first_output"] if has_processed_output else [])
    assert stages[1].status.status_code == trace.StatusCode.ERROR
    assert pipeline.status.status_code == trace.StatusCode.ERROR
    assert not orch.request_states
