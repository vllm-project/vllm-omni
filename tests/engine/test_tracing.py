# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import time

import janus
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from tests.engine.test_orchestrator import FakeStageClient, _build_stage_pools
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import StageSubmissionMessage
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_runtime import StageRuntimeInfo
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.tracing import RequestTrace, capture_trace_headers, create_tracer_provider

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
        request = RequestTrace(tracer, "req", carrier)
        request.end()
    pipeline = next(s for s in exporter.get_finished_spans() if s.name == "omni.pipeline")
    assert pipeline.parent.span_id == http.get_span_context().span_id


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancelled", "error"])
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
    else:
        await orch._cleanup_request_ids(["req-0", "req-1"], abort=True)
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    pipelines = [s for s in spans if s.name == "omni.pipeline"]
    stages = [s for s in spans if s.name == "omni.stage"]
    for pipeline in pipelines:
        assert pipeline.context.trace_id == int("12345678901234567890123456789012", 16)
        assert pipeline.attributes["omni.request.outcome"] == outcome
        assert pipeline.status.is_ok == (outcome != "error")
        assert len([s for s in stages if s.parent.span_id == pipeline.context.span_id]) == 1
    assert not orch.request_states
    assert all("private" not in str(s.attributes) for s in spans)


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
