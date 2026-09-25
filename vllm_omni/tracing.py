# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request and stage lifetimes observed by the Omni orchestrator."""

from collections.abc import Mapping

from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk import trace as sdk_trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

_PROPAGATOR = CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])


def capture_trace_headers(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    carrier: dict[str, str] = {}
    context = _PROPAGATOR.extract(headers) if headers is not None else None
    current = trace.get_current_span().get_span_context()
    remote = trace.get_current_span(context).get_span_context()
    # Upstream serving can pass inbound headers even inside an active HTTP span.
    if current.is_valid and (not remote.is_valid or current.trace_id == remote.trace_id):
        context = None
    _PROPAGATOR.inject(carrier, context=context)
    return carrier


def create_tracer_provider(endpoint: str | None) -> sdk_trace.TracerProvider | None:
    if not endpoint:
        return None
    import os

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    protocol = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    )
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    else:
        raise ValueError(f"Unsupported OTLP traces protocol: {protocol}")
    provider = sdk_trace.TracerProvider(
        resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "vllm-omni")})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    return provider


class RequestTrace:
    def __init__(self, tracer: trace.Tracer, request_id: str, headers: Mapping[str, str] | None) -> None:
        self.tracer = tracer
        parent = _PROPAGATOR.extract(headers or {})
        self.span = tracer.start_span("omni.pipeline", context=parent, attributes={"omni.request.id": request_id})
        self.context = trace.set_span_in_context(self.span, parent)
        self.stages: dict[int, trace.Span] = {}
        self.first_outputs: set[int] = set()
        self.finished_stages: set[int] = set()
        self.error_type: str | None = None

    def headers(self) -> dict[str, str]:
        carrier: dict[str, str] = {}
        _PROPAGATOR.inject(carrier, context=self.context)
        return carrier

    def start_stage(self, stage_id: int, replica_id: int, stage_type: str) -> None:
        if stage_id in self.stages:
            return
        self.stages[stage_id] = self.tracer.start_span(
            "omni.stage",
            context=self.context,
            attributes={"omni.stage.id": stage_id, "omni.stage.type": stage_type, "omni.replica.id": replica_id},
        )

    def output(self, stage_id: int) -> None:
        if stage_id in self.first_outputs or stage_id not in self.stages:
            return
        self.first_outputs.add(stage_id)
        self.stages[stage_id].add_event("omni.stage.first_output")

    def finish_stage(self, stage_id: int) -> None:
        if stage_id in self.finished_stages or stage_id not in self.stages:
            return
        self.finished_stages.add(stage_id)
        self.stages[stage_id].end()

    def fail(self, error_type: str) -> None:
        self.error_type = error_type
        self.span.set_attribute("error.type", error_type)
        self.span.set_status(trace.StatusCode.ERROR)
        for stage_id, span in self.stages.items():
            if stage_id not in self.finished_stages:
                span.set_attribute("error.type", error_type)
                span.set_status(trace.StatusCode.ERROR)

    def end(self, outcome: str = "success") -> None:
        if outcome == "error" and self.error_type is None:
            self.fail("stage_error")
        self.span.set_attribute("omni.request.outcome", "error" if self.error_type else outcome)
        for stage_id, span in self.stages.items():
            if stage_id not in self.finished_stages:
                span.set_attribute("omni.request.outcome", "error" if self.error_type else outcome)
                self.finish_stage(stage_id)
        self.span.end()
