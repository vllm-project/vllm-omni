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
from vllm.logger import init_logger

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
from vllm_omni.entrypoints.openai.serving_policy import (
    duplex_admission_error,
    get_image_stage_configs,
    get_ming3_chat_handler,
    get_route_handler,
    get_video_stage_configs,
    handler_exposure,
    image_edit_error,
    image_generation_error,
    requires_image_chat,
    single_api_exposure,
    video_generation_error,
)
from vllm_omni.request_capabilities import (
    ImageGeometry,
    InputConstraint,
    ParameterConstraint,
    RequestConstraints,
    RuntimeServingMetadata,
)
from vllm_omni.serving_capabilities import OPERATIONS, CapabilityProfile, OperationId, ServingCapabilities

ExecutionState = Literal["unpaused", "paused", "unknown"]
RouteKey = tuple[str, str, str]
logger = init_logger(__name__)

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


def _initialized_image_geometry(engine: object) -> ImageGeometry:
    """Read completed replica snapshots, including remote membership changes."""
    pools = getattr(engine, "stage_pools", ())
    if pools:
        runtime = [
            getattr(client, "serving_metadata", None)
            for pool in pools
            if getattr(getattr(pool, "stage_client", None), "final_output_type", None) == "image"
            for client in tuple(pool.clients)
            if client is not None
        ]
    else:
        runtime = [
            getattr(stage, "serving_metadata", None)
            for stage in getattr(engine, "stage_metadata", ())
            if getattr(stage, "final_output_type", None) == "image"
        ]
    geometries = [item.image_geometry if isinstance(item, RuntimeServingMetadata) else None for item in runtime]
    if geometries and geometries[0] is not None and all(item == geometries[0] for item in geometries):
        return geometries[0]
    return ImageGeometry()


def _with_geometry(profile: CapabilityInputProfile, geometry: ImageGeometry) -> CapabilityInputProfile:
    routes = []
    for route in profile.routes:
        request = route.request
        if request is not None and request.mask is not None and request.mask.support == "supported":
            request = request.model_copy(update={"mask": request.mask.model_copy(update={"geometry": geometry})})
            route = route.model_copy(update={"request": request})
        routes.append(route)
    return profile.model_copy(update={"routes": routes})


def _route_request(
    profile: CapabilityProfile,
    path: str,
    *,
    stage_count: int,
    state: object,
    geometry: ImageGeometry,
    image_limit: int | None,
    native_diffusion: bool,
) -> RequestConstraints | None:
    """Complete request facts for this adapter path, constructed only at startup."""
    video_route = path in {"/v1/videos", "/v1/videos/sync", "/v1/realtime/video"}
    if video_route:
        if profile.support == "unsupported":
            return None
        # Checkpoint-dependent modes can still have known input facts, and
        # transports can establish exclusions without a model declaration.
        request = profile.request or RequestConstraints()
    elif profile.request is None or profile.support != "supported":
        return None
    else:
        request = profile.request
    inputs, parameters = dict(request.inputs), dict(request.parameters)
    mask = request.mask
    if path in {"/v1/images/generations", "/v1/images/edits"}:
        route_parameters = {
            "width": ParameterConstraint(support="supported", type="integer", minimum=1),
            "height": ParameterConstraint(support="supported", type="integer", minimum=1),
            "output_format": ParameterConstraint(support="supported", type="string", enum=("png", "jpeg", "webp")),
        }
        # A transport's generic schema cannot widen model-specific bounds.
        for name, constraint in route_parameters.items():
            parameters.setdefault(name, constraint)
        if path.endswith("/edits"):
            image = inputs.get("image")
            if image is not None and image.support == "supported" and image_limit is not None:
                maximum = min(image.max_count, image_limit) if image.max_count is not None else image_limit
                inputs["image"] = image.model_copy(update={"max_count": maximum})
            parameters["response_format"] = ParameterConstraint(support="supported", type="string", enum=("b64_json",))
            if stage_count > 1:
                # generate_diffusion_images only forwards ordinary reference
                # images. Native mask/separate-reference fields cannot survive.
                for role in ("mask_image", "reference_image"):
                    inputs[role] = InputConstraint(support="unsupported", max_count=0)
                if mask is not None:
                    mask = mask.model_copy(update={"support": "unsupported"})
            elif not native_diffusion:
                # The preserved architecture name describes the native owner;
                # the external adapter's loaded call signature lives in its worker.
                for role in ("mask_image", "reference_image"):
                    inputs[role] = InputConstraint()
                mask = None
            elif mask is not None and mask.support == "supported":
                mask = mask.model_copy(update={"geometry": geometry})
            parameters["stream"] = ParameterConstraint(
                support="supported", type="boolean", enum=(False, True) if stage_count > 1 else (False,)
            )
        else:
            for role in ("image", "mask_image", "reference_image"):
                inputs[role] = InputConstraint(support="unsupported", max_count=0)
        output_maximum = getattr(getattr(state, "args", None), "max_generated_image_size", None)
        request = request.model_copy(
            update={"max_output_pixels": output_maximum if type(output_maximum) is int and output_maximum > 0 else None}
        )
    elif video_route:
        route_parameters = {
            "width": ParameterConstraint(support="supported", type="integer", minimum=1),
            "height": ParameterConstraint(support="supported", type="integer", minimum=1),
            "num_frames": ParameterConstraint(support="supported", type="integer", minimum=1),
            "fps": ParameterConstraint(support="supported", type="number", minimum=1),
        }
        for name, constraint in route_parameters.items():
            parameters.setdefault(name, constraint)
        # These HTTP adapters do not map a last-image role to last_image in
        # multi_modal_data, even for native pipelines that support it offline.
        inputs["last_frame"] = InputConstraint(support="unsupported", max_count=0)
        edit_roles = ("source_video", "video_noise_mask", "audio_noise_mask")
        if path == "/v1/realtime/video":
            if "image" in inputs:
                image = inputs["image"]
                maximum = min(image.max_count, 1) if image.max_count is not None else 1
                inputs["image"] = image.model_copy(update={"max_count": maximum})
            for role in ("video", "audio", *edit_roles):
                inputs[role] = InputConstraint(support="unsupported", max_count=0)
        elif getattr(get_route_handler(state, path), "supports_latent_mask_editing", None) is False:
            # This is the same gate used by the HTTP input validator. Opting
            # in only preserves model facts; it cannot prove missing inputs.
            for role in edit_roles:
                inputs[role] = InputConstraint(support="unsupported", max_count=0)
    elif path == "/v1/audio/speech":
        parameters["response_format"] = ParameterConstraint(
            support="supported", type="string", enum=("wav", "pcm", "flac", "mp3", "opus")
        )
        adapter = getattr(getattr(state, "openai_serving_speech", None), "_adapter", None)
        project = getattr(adapter, "get_request_constraints", None)
        if callable(project):
            adapter_request = project()
            inputs.update(adapter_request.inputs)
            parameters.update(adapter_request.parameters)
            request = request.model_copy(update={"required_any_of": adapter_request.required_any_of})
    elif path == "/v1/audio/generate":
        # create_audio_generate forwards caption and optional negative prompt;
        # it has no reference-media, voice, or lyrics+instructions mapping.
        for role in ("audio", "reference_audio", "reference_text", "voice", "image", "video", "instructions"):
            inputs[role] = InputConstraint(support="unsupported", max_count=0)
    return request.model_copy(update={"inputs": inputs, "parameters": parameters, "mask": mask})


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
        image_stages = get_image_stage_configs(state, engine_client)
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
            if callable(observer) and get_route_handler(state, "/v1/duplex") is None
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
                exposure = handler_exposure(state, path)
                if path in {"/v1/models", "/v1/omni/capabilities"}:
                    exposure = "enabled"
                if path in {"/v1/images/generations", "/v1/images/edits"}:
                    exposure = "enabled" if image_generation_error(image_stages) is None else "disabled"
                    if getattr(state, "engine_client", None) is None:
                        exposure = "disabled"
                    if path.endswith("/edits") and image_edit_error(image_stages) is not None:
                        exposure = "disabled"
                    if requires_image_chat(image_stages) and get_route_handler(state, "/v1/chat/completions") is None:
                        exposure = "disabled"
                if path in {"/v1/videos", "/v1/videos/sync", "/v1/realtime/video"}:
                    handler = get_route_handler(state, path)
                    video_stages = getattr(handler, "_stage_configs", None)
                    if path != "/v1/realtime/video":
                        # The HTTP form parser supplies app stages when the
                        # handler has none. WebSocket handlers use their engine.
                        video_stages = video_stages or getattr(state, "stage_configs", None)
                    video_stages = get_video_stage_configs(video_stages, engine_client)
                    if video_generation_error(video_stages) is not None:
                        exposure = "disabled"
                if path == "/v1/duplex" and duplex_admission_error(state) is not None:
                    exposure = "disabled"
                if path == "/v1/realtime":
                    duplex = get_route_handler(state, "/v1/duplex")
                    realtime = get_route_handler(state, path)
                    ming3 = get_ming3_chat_handler(state)
                    if duplex is None:
                        exposure = "enabled" if realtime is not None or ming3 is not None else "disabled"
                    else:
                        exposure = "unknown"
                    # Query parameters choose duplex vs turn-based sessions.
                    # Their mixed policies cannot be represented by one bool.
                dependencies = route.dependant.dependencies
                if any(dependency.call in single_api_dependencies for dependency in dependencies):
                    worker_exposure = single_api_exposure(state)
                    if exposure != "disabled" and worker_exposure != "enabled":
                        exposure = worker_exposure
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
        from vllm_omni.diffusion.data import uses_diffusers_adapter
        from vllm_omni.entrypoints.openai.app_state import _get_diffusion_od_config_from_state
        from vllm_omni.entrypoints.openai.images.helpers import _max_edit_input_images_from_config

        diffusion_config = _get_diffusion_od_config_from_state(state, engine_client)
        image_limit = _max_edit_input_images_from_config(diffusion_config)
        native_diffusion = not uses_diffusers_adapter(diffusion_config)
        self._engine = engine
        geometry = _initialized_image_geometry(engine)
        self._profiles: dict[OperationId, list[CapabilityInputProfile]] = {}
        for operation_id in OPERATIONS:
            projected = []
            for profile in self._declaration.get(operation_id).profiles:
                references = []
                for key in _OPERATION_ROUTES[operation_id]:
                    route = routes[key]
                    exposure = route.exposure
                    if route.path == "/v1/realtime/video" and {"video", "audio"}.intersection(profile.required_inputs):
                        exposure = "disabled"
                    references.append(
                        CapabilityRouteReference(
                            transport=route.transport,
                            method=route.method,
                            path=route.path,
                            exposure=exposure,
                            request=_route_request(
                                profile,
                                route.path,
                                stage_count=len(image_stages) if route.path.startswith("/v1/images/") else len(stages),
                                state=state,
                                geometry=geometry,
                                image_limit=image_limit,
                                native_diffusion=native_diffusion,
                            ),
                        )
                    )
                projected.append(
                    CapabilityInputProfile(
                        id=profile.id,
                        support=profile.support,
                        required_inputs=list(profile.required_inputs),
                        outputs=list(profile.outputs),
                        routes=references,
                        reason_code=profile.reason_code,
                        constraints=dict(profile.constraints),
                        request=profile.request,
                    )
                )
            self._profiles[operation_id] = projected

    async def get_capabilities(self) -> OmniCapabilitiesResponse:
        now = datetime.now(timezone.utc)
        geometry = _initialized_image_geometry(self._engine)
        execution_state: ExecutionState = "unknown"
        if self._observer is not None:
            # Execution observation is optional. A failed observer must not
            # hide the initialized deployment facts; cancellation still propagates.
            try:
                observed = await self._observer()
                if observed in ("unpaused", "paused", "unknown"):
                    execution_state = observed
                else:
                    logger.warning_once("Invalid execution observation; reporting execution.state=unknown.")
            except Exception as exc:
                logger.warning_once(
                    "Execution observation failed (%s); reporting execution.state=unknown.", type(exc).__name__
                )
        execution = CapabilityExecution(state=execution_state, scope=self._scope, observed_at=now)
        operations = {}
        for operation_id in OPERATIONS:
            declaration = self._declaration.get(operation_id)
            routes = [self._routes[key] for key in _OPERATION_ROUTES[operation_id]]
            exposures = {route.exposure for route in routes}
            exposure: Exposure = (
                "enabled" if "enabled" in exposures else "unknown" if "unknown" in exposures else "disabled"
            )
            operations[operation_id] = CapabilityOperation(
                support=declaration.support,
                exposure=exposure,
                reason_code="endpoint_disabled"
                if exposure == "disabled" and declaration.support != "unsupported"
                else declaration.reason_code,
                profiles=[_with_geometry(profile, geometry) for profile in self._profiles[operation_id]],
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
