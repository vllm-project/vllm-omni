# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Project initialized model declarations and serving policy into discovery.

Construction runs once after app initialization and endpoint restrictions.
Requests only read this projection and the frontend's local execution gate.
No model/config resolver, backend RPC, health probe, or generation is involved.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Literal, cast

from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute

from vllm_omni.config.endpoint_policy import EndpointRestriction
from vllm_omni.entrypoints.openai.protocol.capabilities import (
    CapabilityExecution,
    CapabilityInputProfile,
    CapabilityModel,
    CapabilityOperation,
    CapabilityRoute,
    CapabilityRouteReference,
    CapabilityStage,
    Exposure,
    OmniCapabilitiesResponse,
)
from vllm_omni.entrypoints.openai.utils import is_image_generation_stage, is_video_generation_pipeline
from vllm_omni.serving_capabilities import OPERATIONS, OperationId, ServingCapabilities

ExecutionState = Literal["unpaused", "paused", "unknown"]
RouteKey = tuple[str, str, str]

# Protocol ownership, independent of model names and output-type heuristics.
_OPERATION_ROUTES: dict[OperationId, tuple[RouteKey, ...]] = {
    "chat": (("http", "POST", "/v1/chat/completions"),),
    "image_generation": (("http", "POST", "/v1/images/generations"),),
    "image_edit": (("http", "POST", "/v1/images/edits"),),
    "video_generation": (
        ("http", "POST", "/v1/videos"),
        ("http", "POST", "/v1/videos/sync"),
        ("websocket", "WEBSOCKET", "/v1/realtime/video"),
    ),
    "speech": (("http", "POST", "/v1/audio/speech"),),
    "audio_generation": (("http", "POST", "/v1/audio/generate"),),
}

_ROUTE_HANDLERS = {
    "/v1/chat/completions": "openai_serving_chat",
    "/v1/chat/completions/batch": "openai_serving_chat_batch",
    "/v1/audio/speech": "openai_serving_speech",
    "/v1/audio/generate": "openai_serving_audio_generate",
    "/v1/videos": "openai_serving_video",
    "/v1/videos/sync": "openai_serving_video",
    "/v1/realtime/video": "openai_streaming_video_output",
    "/v1/audio/speech/batch": "openai_serving_speech",
    "/v1/audio/speech/stream": "openai_streaming_speech",
    "/v1/video/chat/stream": "openai_streaming_video",
    "/v1/duplex": "openai_serving_duplex",
    "/v1/realtime/robot/openpi": "openai_serving_realtime_robot",
}


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _stage_projection(stages: Sequence[object]) -> tuple[CapabilityStage, ...]:
    projected = []
    for stage in stages:
        stage_id = _field(stage, "stage_id")
        if stage_id is None:
            # A legacy/plugin config without an ID cannot be assigned its list
            # offset: that would invent a topology the engine did not declare.
            continue
        stage_type = _field(stage, "stage_type", "unknown")
        stage_type = getattr(stage_type, "value", stage_type)
        raw_inputs = _field(stage, "input_sources", _field(stage, "engine_input_source", ()))
        inputs = raw_inputs if isinstance(raw_inputs, Sequence) else ()
        projected.append(CapabilityStage(id=str(stage_id), type=str(stage_type), depends_on=[str(s) for s in inputs]))
    return tuple(projected)


class OmniServingCapabilities:
    """A deployment projection whose query path has no initialization work."""

    def __init__(
        self,
        app: FastAPI,
        engine_client: object,
        *,
        restrictions: tuple[EndpointRestriction, ...] = (),
        single_api_dependencies: tuple[Callable[..., object], ...] = (),
    ) -> None:
        state = app.state
        serving_models = getattr(state, "openai_serving_models", None)
        self._model_ids = tuple(path.name for path in getattr(serving_models, "base_model_paths", ()))
        stages = list(getattr(engine_client, "stage_configs", ()) or ())
        self._stages = _stage_projection(stages)
        engine = getattr(engine_client, "engine", None)
        declaration = getattr(engine, "serving_capabilities", None)
        self._declaration = declaration if isinstance(declaration, ServingCapabilities) else ServingCapabilities()
        count = getattr(state, "api_server_count", None)
        self._scope: Literal["deployment", "worker"] = "deployment" if type(count) is int and count == 1 else "worker"
        observer = getattr(engine_client, "get_execution_state", None)
        # Duplex chat uses session admission, not AsyncOmni.generate's gate.
        self._observer = (
            cast(Callable[[], Awaitable[ExecutionState]], observer)
            if callable(observer) and getattr(state, "openai_serving_duplex", None) is None
            else None
        )
        disabled = {
            ("http", method, restriction.capability.path)
            for restriction in restrictions
            for method in restriction.capability.methods
        }
        routes: dict[RouteKey, CapabilityRoute] = {}
        for route in app.routes:
            if isinstance(route, APIRoute):
                keys = [("http", method, route.path) for method in route.methods]
            elif isinstance(route, APIWebSocketRoute):
                keys = [("websocket", "WEBSOCKET", route.path)]
            else:
                continue
            if not route.path.startswith("/v1/"):
                continue
            for transport, method, path in keys:
                key = (transport, method, path)
                # Unknown extension routes may have their own admission rules.
                # Registration alone does not establish their availability.
                exposure: Exposure = "unknown"
                handler_name = _ROUTE_HANDLERS.get(path)
                if handler_name is not None:
                    exposure = "enabled" if getattr(state, handler_name, None) is not None else "disabled"
                if path in {"/v1/models", "/v1/omni/capabilities"}:
                    exposure = "enabled"
                if path in {"/v1/images/generations", "/v1/images/edits"}:
                    exposure = "enabled" if any(is_image_generation_stage(stage) for stage in stages) else "disabled"
                    if len(stages) > 1 and getattr(state, "openai_serving_chat", None) is None:
                        exposure = "disabled"
                if path in {"/v1/videos", "/v1/videos/sync", "/v1/realtime/video"}:
                    if not is_video_generation_pipeline(stages):
                        exposure = "disabled"
                if path == "/v1/duplex" and count is not None:
                    if type(count) is not int or count < 1:
                        exposure = "disabled"
                    elif count > 1:
                        config = getattr(engine_client, "duplex_session_config", None)
                        capabilities = getattr(engine_client, "duplex_capabilities", None)
                        if getattr(config, "replica_policy", None) != "least_sessions" or (
                            getattr(capabilities, "supports_session_resume", None) is not False
                        ):
                            exposure = "disabled"
                if path == "/v1/realtime":
                    duplex = getattr(state, "openai_serving_duplex", None)
                    realtime = getattr(state, "openai_realtime_handler", None)
                    chat = getattr(state, "openai_serving_chat", None)
                    ming3 = getattr(chat, "_ming3_speech_service", None)
                    if duplex is None:
                        exposure = "enabled" if realtime is not None or ming3 is not None else "disabled"
                    # Query parameters choose duplex vs turn-based sessions.
                    # Their mixed policies cannot be represented by one bool.
                dependencies = route.dependant.dependencies
                if any(dependency.call in single_api_dependencies for dependency in dependencies):
                    if (type(count) is not int or count < 1) and exposure != "disabled":
                        exposure = "unknown"
                    elif type(count) is int and count > 1:
                        exposure = "disabled"
                if key in disabled:
                    exposure = "disabled"
                routes[key] = CapabilityRoute(
                    transport=cast(Literal["http", "websocket"], transport), method=method, path=path, exposure=exposure
                )
        # An absent operation route is an explicit deployment observation.
        for operation_routes in _OPERATION_ROUTES.values():
            for transport, method, path in operation_routes:
                key = (transport, method, path)
                if key not in routes:
                    routes[key] = CapabilityRoute(
                        transport=cast(Literal["http", "websocket"], transport),
                        method=method,
                        path=path,
                        exposure="disabled",
                    )
        self._routes = routes

    async def get_capabilities(self) -> OmniCapabilitiesResponse:
        now = datetime.now(timezone.utc)
        execution_state = await self._observer() if self._observer is not None else "unknown"
        execution = CapabilityExecution(state=execution_state, scope=self._scope, observed_at=now)
        operations = {}
        for operation_id in OPERATIONS:
            declaration = self._declaration.get(operation_id)
            routes = [self._routes[key] for key in _OPERATION_ROUTES[operation_id]]
            exposures = {route.exposure for route in routes}
            exposure: Exposure = (
                "enabled" if "enabled" in exposures else "unknown" if "unknown" in exposures else "disabled"
            )
            references = [
                CapabilityRouteReference(transport=route.transport, method=route.method, path=route.path)
                for route in routes
            ]
            operations[operation_id] = CapabilityOperation(
                support=declaration.support,
                exposure=exposure,
                reason_code="endpoint_disabled"
                if exposure == "disabled" and declaration.support != "unsupported"
                else declaration.reason_code,
                profiles=[
                    CapabilityInputProfile(
                        id=profile.id,
                        support=profile.support,
                        required_inputs=list(profile.required_inputs),
                        outputs=list(profile.outputs),
                        routes=references,
                        reason_code=profile.reason_code,
                        constraints=dict(profile.constraints),
                    )
                    for profile in declaration.profiles
                ],
                execution=execution,
            )
        return OmniCapabilitiesResponse(
            observed_at=now,
            models=[
                CapabilityModel(id=model_id, stage_ids=[stage.id for stage in self._stages], operations=operations)
                for model_id in self._model_ids
            ],
            stages=list(self._stages),
            routes=[self._routes[key] for key in sorted(self._routes)],
        )
