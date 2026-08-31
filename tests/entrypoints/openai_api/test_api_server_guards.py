# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""API-server surface guards for the OpenAI Omni entrypoint.

Purpose
-------
Freeze the *observable* server contract so helper moves (P0.2), route
extraction (P0.3), and later endpoint-family PRs cannot silently change:

* which routes exist,
* how upstream routes are overridden at assembly time,
* unavailable / missing-handler envelopes,
* shared engine-error JSON fields,
* app-state keys wired by ``omni_init_app_state``.

CPU-only (fakes / TestClient). No checked-in OpenAPI golden JSON.

Common wrong changes these tests are meant to catch
----------------------------------------------------
* Route accidentally removed, renamed, method-changed, or double-registered
  after a move (census / assembled-app checks).
* OpenAPI drops an Omni HTTP path while the router still has it (or the reverse).
* Assembly forgets to strip overridden upstream chat / models / profiler
  routes, or also deletes unrelated upstream routes.
* Profiler / Omni router not mounted; storage not started; Omni engine
  exception handlers not registered.
* Timestamp middleware stops stamping HTTP, or starts mutating WebSocket scopes.
* Missing-handler WS/HTTP responses change status, JSON shape, or wording
  (clients treat these as capability probes).
* Engine error payloads lose ``request_id`` / ``error_stage_id``.
* Pure-diffusion vs multi-stage init drops an ``app.state`` key, or leaves a
  handler as ``None`` while the route is still registered.

Not covered here (add elsewhere if needed)
------------------------------------------
* Deep per-endpoint success behavior (family UTs).
* Byte-identical helper bodies after a move (P0.2 AST integrity).
* Live GPU / speech byte-identity checks.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient
from starlette.datastructures import State
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.entrypoints.openai import api_server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# FastAPI/Starlette auto-adds HEAD on every GET route. Including HEAD here
# would invent ("HEAD", path) entries for every GET and fail the current census.
# TODO: extend this set and the expected census if @router.head / @router.options appear.
_HTTP_METHODS = {"GET", "POST", "DELETE", "PUT", "PATCH"}

# Exact Omni router census on main. Update intentionally when routes change.
_EXPECTED_ROUTER_ROUTES = {
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/chat/completions/batch"),
    ("POST", "/v1/audio/speech"),
    ("POST", "/v1/audio/speech/batch"),
    ("POST", "/v1/audio/generate"),
    ("GET", "/v1/audio/voices"),
    ("POST", "/v1/audio/voices"),
    ("DELETE", "/v1/audio/voices/{name}"),
    ("WEBSOCKET", "/v1/audio/speech/stream"),
    ("WEBSOCKET", "/v1/video/chat/stream"),
    ("WEBSOCKET", "/v1/realtime/video"),
    ("WEBSOCKET", "/v1/realtime"),
    ("WEBSOCKET", "/v1/realtime/robot/openpi"),
    ("WEBSOCKET", "/v1/duplex"),
    ("GET", "/health"),
    ("GET", "/v1/models"),
    ("POST", "/v1/images/generations"),
    ("POST", "/v1/images/edits"),
    ("POST", "/v1/videos"),
    ("POST", "/v1/videos/sync"),
    ("GET", "/v1/videos"),
    ("GET", "/v1/videos/{video_id}"),
    ("DELETE", "/v1/videos/{video_id}"),
    ("GET", "/v1/videos/{video_id}/content"),
    ("POST", "/v1/omni/sleep"),
    ("POST", "/v1/omni/wakeup"),
}
_EXPECTED_PROFILER_ROUTES = {
    ("POST", "/start_profile"),
    ("POST", "/stop_profile"),
}

# App-state keys that must remain available across refactor phases.
# Presence is not enough: "not wired" is written as ``state.X = None``.
_DIFFUSION_APP_STATE_KEYS = {
    "engine_client",
    "log_stats",
    "args",
    "sleeping_stages",
    "stage_configs",
    "vllm_config",
    "diffusion_engine",
    "openai_serving_models",
    "serving_tokenization",
    "openai_serving_chat",
    "openai_serving_chat_batch",
    "openai_serving_speech",
    "openai_serving_audio_generate",
    "openai_serving_video",
    "openai_streaming_video_output",
    "openai_serving_duplex",
    "openai_streaming_speech",
    "openai_streaming_video",
    "openai_serving_realtime_robot",
    "enable_server_load_tracking",
    "server_load_metrics",
}
_DIFFUSION_MUST_BE_NONE = {
    "vllm_config",
    "serving_tokenization",
    "openai_serving_duplex",
    "openai_streaming_speech",
    "openai_streaming_video",
}
_DIFFUSION_MUST_BE_WIRED = _DIFFUSION_APP_STATE_KEYS - _DIFFUSION_MUST_BE_NONE
_MULTISTAGE_APP_STATE_KEYS = {
    "engine_client",
    "log_stats",
    "args",
    "sleeping_stages",
    "stage_configs",
    "vllm_config",
    "openai_serving_models",
    "serving_tokenization",
    "online_renderer",
    "openai_serving_chat",
    "openai_serving_chat_batch",
    "openai_serving_speech",
    "openai_serving_audio_generate",
    "openai_streaming_speech",
    "openai_streaming_video",
    "openai_serving_duplex",
    "openai_serving_realtime",
    "openai_serving_video",
    "openai_serving_realtime_robot",
    "enable_server_load_tracking",
    "server_load_metrics",
}
_MULTISTAGE_MUST_BE_NONE = {
    "openai_serving_duplex",
    "openai_serving_realtime_robot",
}
_MULTISTAGE_MUST_BE_WIRED = _MULTISTAGE_APP_STATE_KEYS - _MULTISTAGE_MUST_BE_NONE


def _route_entries(routes) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for route in routes:
        if isinstance(route, APIRoute):
            for method in sorted((route.methods or set()) & _HTTP_METHODS):
                entries.append((method, route.path))
        elif isinstance(route, APIWebSocketRoute):
            entries.append(("WEBSOCKET", route.path))
    return entries


def _route_manifest(routes) -> set[tuple[str, str]]:
    return set(_route_entries(routes))


def _assert_unique_route_entries(routes) -> None:
    entries = _route_entries(routes)
    duplicates = sorted({item for item in entries if entries.count(item) > 1})
    assert not duplicates, f"duplicate route registrations: {duplicates}"


def _matching_http_routes(routes, method: str, path: str) -> list[APIRoute]:
    return [
        route
        for route in routes
        if isinstance(route, APIRoute) and route.path == path and method in (route.methods or set())
    ]


def _single_http_route(routes, method: str, path: str) -> APIRoute:
    matches = _matching_http_routes(routes, method, path)
    assert len(matches) == 1, f"expected exactly one {method} {path}, got {len(matches)}"
    return matches[0]


def _assert_app_state_snapshot(
    state,
    *,
    expected_keys: set[str],
    must_be_wired: set[str],
    must_be_none: set[str],
) -> None:
    assert must_be_wired.isdisjoint(must_be_none)
    assert must_be_wired | must_be_none <= expected_keys
    present = {key for key in expected_keys if hasattr(state, key)}
    assert present == expected_keys
    not_wired = sorted(key for key in must_be_wired if getattr(state, key) is None)
    assert not not_wired, f"app.state keys registered but not wired: {not_wired}"
    unexpectedly_set = sorted(key for key in must_be_none if getattr(state, key) is not None)
    assert not unexpectedly_set, f"app.state keys expected to stay None: {unexpectedly_set}"


def _minimal_args(**overrides) -> SimpleNamespace:
    args = SimpleNamespace(
        tool_parser_plugin=None,
        reasoning_parser_plugin=None,
        reasoning_parser=None,
        structured_outputs_config=SimpleNamespace(reasoning_parser=None),
        enable_ssl_refresh=False,
        host="127.0.0.1",
        port=8000,
        uvicorn_log_level="info",
        disable_uvicorn_access_log=True,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=0,
        ssl_ciphers=None,
        h11_max_incomplete_event_size=None,
        h11_max_header_count=None,
        log_config_file=None,
        disable_access_log_for_endpoints=None,
        disable_log_stats=True,
        enable_log_requests=False,
        max_log_len=None,
        served_model_name=None,
        model="demo-model",
        enable_server_load_tracking=False,
        log_error_stack=False,
        chat_template=None,
        chat_template_content_format="auto",
        trust_request_chat_template=False,
        tool_server=None,
        lora_modules=None,
        response_role="assistant",
        default_chat_template_kwargs=None,
        return_tokens_as_token_ids=False,
        enable_auto_tool_choice=False,
        exclude_tools_when_tool_choice_none=False,
        tool_call_parser=None,
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
        enable_log_outputs=False,
        enable_log_deltas=False,
        tokens_only=False,
        shutdown_timeout=0,
        stage_configs_path=None,
        deploy_config=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _request_for(app: FastAPI, *, method: str = "GET", path: str = "/") -> Request:
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],
            "app": app,
            "state": {},
        }
    )


class _FakeSocket:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeEngineClient:
    def __init__(self, *, stage_configs=None, endpoint_restrictions=None, vllm_config=None) -> None:
        self.stage_configs = stage_configs if stage_configs is not None else []
        self.endpoint_restrictions = endpoint_restrictions if endpoint_restrictions is not None else {}
        self.model_config = SimpleNamespace()
        self.input_processor = object()
        self.renderer = object()
        self.errored = False
        self.vllm_config = vllm_config

    async def get_vllm_config(self):
        return self.vllm_config

    async def get_supported_tasks(self) -> tuple[str, ...]:
        return ("generate",)

    async def get_tokenizer(self):
        return SimpleNamespace(chat_template="dummy")

    async def check_health(self) -> None:
        return None


def _marker(name: str):
    return SimpleNamespace(name=name)


def test_router_route_manifest_is_explicit_and_complete() -> None:
    """Lock the Omni + profiler router census.

    Fails if a route is added/removed/renamed, a method changes, or a
    WebSocket path disappears during helper/router moves. Also fails if the
    same public route is registered more than once, which a set manifest alone
    would collapse.
    """
    # TODO(P0.3): retarget this object-level census when routes move onto family
    # routers. The assembled-app census is the client-facing lock that must survive.
    _assert_unique_route_entries(api_server.router.routes)
    _assert_unique_route_entries(api_server.profiler_router.routes)
    assert _route_manifest(api_server.router.routes) == _EXPECTED_ROUTER_ROUTES
    assert _route_manifest(api_server.profiler_router.routes) == _EXPECTED_PROFILER_ROUTES


def test_router_openapi_paths_cover_http_manifest() -> None:
    """Lock that OpenAPI advertises every HTTP census route/method.

    Fails if a decorator still registers a route but OpenAPI loses the path
    (or method), e.g. after moving route bodies / response models.
    """
    app = FastAPI()
    app.include_router(api_server.router)
    app.include_router(api_server.profiler_router)
    paths = app.openapi().get("paths", {})

    http_routes = {
        (method, path) for method, path in _EXPECTED_ROUTER_ROUTES | _EXPECTED_PROFILER_ROUTES if method != "WEBSOCKET"
    }
    for method, path in sorted(http_routes):
        assert path in paths, f"missing OpenAPI path {path}"
        assert method.lower() in paths[path], f"missing OpenAPI method {method} {path}"


@pytest.mark.asyncio
async def test_api_server_assembly_replaces_upstream_routes_and_mounts_omni_router(monkeypatch) -> None:
    """Lock worker assembly: override, mount, storage, handlers, full census.

    Fails if assembly stops replacing upstream chat/batch/models/profiler,
    deletes unrelated upstream routes, forgets Omni/profiler mounts, skips
    storage start, or drops Omni ``EngineDeadError`` / ``EngineGenerateError``
    handlers.
    """
    captured: dict[str, object] = {}

    def fake_build_openai_app(args, supported_tasks):
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def upstream_chat():
            return {"owner": "upstream"}

        @app.post("/v1/chat/completions/batch")
        async def upstream_batch():
            return {"owner": "upstream"}

        @app.get("/v1/models")
        async def upstream_models():
            return {"owner": "upstream"}

        @app.post("/start_profile")
        async def upstream_start_profile():
            return {"owner": "upstream"}

        @app.post("/stop_profile")
        async def upstream_stop_profile():
            return {"owner": "upstream"}

        @app.get("/v1/upstream-only")
        async def upstream_only():
            return {"owner": "upstream"}

        captured["supported_tasks"] = supported_tasks
        return app

    @asynccontextmanager
    async def fake_build_async_omni(*_args, **_kwargs):
        yield _FakeEngineClient(
            stage_configs=[{"engine_args": {"profiler_config": {"profiler": "torch"}}}],
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(_api_process_rank=0)),
        )

    async def fake_omni_init_app_state(engine_client, state, args):
        state.engine_client = engine_client
        state.initialized_by_omni = True
        state.openai_serving_video = SimpleNamespace(
            shutdown=lambda: captured.__setitem__("video_shutdown", True),
        )
        state.openai_serving_speech = SimpleNamespace(
            shutdown=lambda: captured.__setitem__("speech_shutdown", True),
        )

    async def fake_storage_start():
        captured["storage_started"] = True

    async def fake_serve_http(app, **_kwargs):
        captured["served_app"] = app
        task = asyncio.create_task(asyncio.sleep(0))
        await asyncio.sleep(0)
        return task

    monkeypatch.setattr(api_server, "build_openai_app", fake_build_openai_app)
    monkeypatch.setattr(api_server, "build_async_omni", fake_build_async_omni)
    monkeypatch.setattr(api_server, "omni_init_app_state", fake_omni_init_app_state)
    monkeypatch.setattr(
        api_server,
        "shutdown_unsupported_routes",
        lambda app, restrictions: captured.setdefault("restrictions", restrictions),
    )
    monkeypatch.setattr(api_server, "STORAGE_MANAGER", SimpleNamespace(start=fake_storage_start))
    monkeypatch.setattr(api_server, "serve_http", fake_serve_http)

    sock = _FakeSocket()
    await api_server.omni_run_server_worker("127.0.0.1:8000", sock, _minimal_args())

    served_wrapper = captured["served_app"]
    # Timestamp middleware wraps the FastAPI app.
    assert hasattr(served_wrapper, "_inner")
    served_app = served_wrapper._inner
    routes = served_app.routes
    _assert_unique_route_entries(routes)
    manifest = _route_manifest(routes)

    assert captured["supported_tasks"] == ("generate",)
    assert captured["storage_started"] is True
    assert captured["restrictions"] == {}
    assert captured["video_shutdown"] is True
    assert captured["speech_shutdown"] is True
    assert sock.closed is True
    assert served_app.state.initialized_by_omni is True

    # Upstream override: unrelated route kept; Omni owns overridden pairs once.
    # Endpoint identity ensures the surviving route is the Omni handler, not
    # the fake upstream handler with the same method/path.
    assert ("GET", "/v1/upstream-only") in manifest
    assert _single_http_route(routes, "POST", "/v1/chat/completions").endpoint is api_server.create_chat_completion
    assert (
        _single_http_route(routes, "POST", "/v1/chat/completions/batch").endpoint
        is api_server.create_batch_chat_completion
    )
    assert _single_http_route(routes, "GET", "/v1/models").endpoint is api_server.show_available_models
    assert _single_http_route(routes, "POST", "/start_profile").endpoint is api_server.start_profile
    assert _single_http_route(routes, "POST", "/stop_profile").endpoint is api_server.stop_profile

    # Assembled app includes Omni router HTTP/WS surface + profiler.
    for item in _EXPECTED_ROUTER_ROUTES:
        assert item in manifest, f"assembled app missing {item}"
    assert ("POST", "/start_profile") in manifest
    assert ("POST", "/stop_profile") in manifest

    assert served_app.exception_handlers[EngineGenerateError]
    assert served_app.exception_handlers[EngineDeadError]

    # Light OpenAPI check on the assembled FastAPI app (not the ASGI wrapper).
    openapi_paths = served_app.openapi().get("paths", {})
    assert "/v1/chat/completions" in openapi_paths
    assert "/v1/models" in openapi_paths
    assert "/health" in openapi_paths


@pytest.mark.asyncio
async def test_timestamp_middleware_stamps_http_and_passes_websocket(monkeypatch) -> None:
    """Lock outermost timestamp middleware behavior.

    Fails if HTTP requests lose ``request_timestamp``, or WebSocket scopes
    start getting stamped / mutated incorrectly.
    """
    captured: dict[str, object] = {}

    def fake_build_openai_app(args, supported_tasks):
        return FastAPI()

    @asynccontextmanager
    async def fake_build_async_omni(*_args, **_kwargs):
        yield _FakeEngineClient()

    async def fake_omni_init_app_state(engine_client, state, args):
        state.engine_client = engine_client

    async def fake_serve_http(app, **_kwargs):
        captured["served_app"] = app
        task = asyncio.create_task(asyncio.sleep(0))
        await asyncio.sleep(0)
        return task

    monkeypatch.setattr(api_server, "build_openai_app", fake_build_openai_app)
    monkeypatch.setattr(api_server, "build_async_omni", fake_build_async_omni)
    monkeypatch.setattr(api_server, "omni_init_app_state", fake_omni_init_app_state)
    monkeypatch.setattr(api_server, "shutdown_unsupported_routes", lambda *_a, **_k: None)
    monkeypatch.setattr(api_server, "STORAGE_MANAGER", SimpleNamespace(start=lambda: asyncio.sleep(0)))
    monkeypatch.setattr(api_server, "serve_http", fake_serve_http)

    await api_server.omni_run_server_worker("127.0.0.1:8000", _FakeSocket(), _minimal_args())
    middleware = captured["served_app"]

    http_scope = {"type": "http", "state": {}}
    ws_scope = {"type": "websocket"}

    async def _receive():
        return {"type": "http.disconnect"}

    async def _send(_message):
        return None

    called = {"http": False, "websocket": False}

    async def inner(scope, receive, send):
        called[scope["type"]] = True

    middleware._inner = inner
    await middleware(http_scope, _receive, _send)
    await middleware(ws_scope, _receive, _send)

    assert called["http"] is True
    assert called["websocket"] is True
    assert "request_timestamp" in http_scope["state"]
    assert isinstance(http_scope["state"]["request_timestamp"], float)
    assert "request_timestamp" not in ws_scope.get("state", {})


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/audio/speech/stream", {"type": "error", "message": "Streaming speech is not available"}),
        ("/v1/video/chat/stream", {"type": "error", "message": "Streaming video chat is not available"}),
        ("/v1/realtime/video", {"type": "error", "message": "Streaming video generation is not available"}),
        ("/v1/realtime", {"type": "error", "error": "Realtime API is not available", "code": "unsupported"}),
        (
            "/v1/realtime/robot/openpi",
            {"type": "error", "error": "Robot policy not available", "code": "unsupported"},
        ),
        ("/v1/duplex", {"type": "error", "error": "Duplex API is not available", "code": "unsupported"}),
    ],
)
def test_websocket_routes_emit_stable_unavailable_frames_and_close(path: str, payload: dict[str, str]) -> None:
    """Lock missing-handler WebSocket capability probes.

    Fails if unavailable JSON changes (keys/wording/code) or the socket stays
    open after the error frame — common when WS wrappers move between modules.
    """
    app = FastAPI()
    app.include_router(api_server.router)

    with TestClient(app) as client:
        with client.websocket_connect(path) as websocket:
            assert websocket.receive_json() == payload
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_text()


def test_health_without_engine_returns_stable_unhealthy_response() -> None:
    """Lock ``/health`` with no engine initialized.

    Fails if status drifts from 503 or the unhealthy payload shape/reason
    string changes during health-route extraction.
    """
    app = FastAPI()
    response = asyncio.run(api_server.health(_request_for(app, path="/health")))

    assert response.status_code == 503
    assert json.loads(response.body) == {
        "status": "unhealthy",
        "reason": "No engine initialized",
    }


def test_models_without_handler_returns_empty_openai_list() -> None:
    """Lock ``/v1/models`` when ``openai_serving_models`` is unset.

    Fails if the empty listing becomes an error, or the OpenAI list envelope
    changes (``object`` / ``data``).
    """
    app = FastAPI()
    response = asyncio.run(api_server.show_available_models(_request_for(app, path="/v1/models")))

    assert response.status_code == 200
    assert json.loads(response.body) == {"object": "list", "data": []}


def test_images_generation_without_engine_preserves_service_unavailable_error() -> None:
    """Lock images generation when the multi-stage engine is missing.

    Fails if the 503 detail string changes — clients/tests use it as a
    capability signal when bootstrap failed.
    """
    app = FastAPI()
    raw_request = _request_for(app, method="POST", path="/v1/images/generations")
    request = api_server.ImageGenerationRequest(prompt="a cat", model="demo-model")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_server.generate_images(request, raw_request))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Multi-stage engine not initialized. Start server with a multi-stage omni model."


def test_images_generation_without_multistage_chat_handler_preserves_unavailable_error(monkeypatch) -> None:
    """Lock images generation when chat serving was not wired.

    Fails if multi-stage images stop requiring ``openai_serving_chat``, or the
    503 detail changes after images/chat ownership splits.
    """
    app = FastAPI()
    app.state.engine_client = SimpleNamespace(stage_configs=[object(), object()])
    app.state.stage_configs = app.state.engine_client.stage_configs
    app.state.openai_serving_models = SimpleNamespace(base_model_paths=[SimpleNamespace(name="demo-model")])
    app.state.openai_serving_chat = None
    app.state.args = SimpleNamespace(max_generated_image_size=None)

    raw_request = _request_for(app, method="POST", path="/v1/images/generations")
    request = api_server.ImageGenerationRequest(prompt="a cat", model="demo-model")
    monkeypatch.setattr(api_server, "get_stage_type", lambda _stage_cfg: "diffusion")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_server.generate_images(request, raw_request))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "openai_serving_chat is not initialized for multi-stage image generation."


def test_speech_without_handler_preserves_not_found_http_error() -> None:
    """Lock speech HTTP when ``openai_serving_speech`` is unset.

    Fails if missing speech becomes a different status/detail after TTS route
    wrappers move out of ``api_server``.
    """
    app = FastAPI()
    app.state.openai_serving_speech = None
    raw_request = _request_for(app, method="POST", path="/v1/audio/speech")
    request = api_server.OpenAICreateSpeechRequest(input="hello", model="demo-model", voice="alloy")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_server.create_speech(request, raw_request))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "The model does not support Speech API"


def test_engine_error_json_response_includes_request_and_stage_fields(monkeypatch) -> None:
    """Lock shared ``EngineGenerateError`` JSON fields used by Omni handlers.

    Fails if ``request_id`` / ``error_stage_id`` are dropped when error helpers
    move (P0.2) or exception-handler wiring changes.

    Exercises the envelope through the registered handler, not the private
    ``_create_engine_error_json_response`` helper, so relocating that helper
    in P0.2 is not a false red.
    """
    app = FastAPI()
    # TODO(P0.2): retarget this call if ``_register_omni_exception_handlers``
    # leaves api_server (red here is a rename, not a surface regression).
    api_server._register_omni_exception_handlers(app)
    app.state.engine_client = SimpleNamespace(errored=True, engine=SimpleNamespace(is_alive=lambda: False))
    app.state.server = object()
    app.state.args = SimpleNamespace(log_error_stack=False)

    req = _request_for(app, method="POST", path="/v1/chat/completions")
    req.state.request_metadata = SimpleNamespace(request_id="req-123")

    monkeypatch.setattr(api_server, "terminate_if_errored", lambda **_kwargs: None)

    exc = EngineGenerateError("boom")
    exc.error_stage_id = 2  # type: ignore[attr-defined]
    handler = app.exception_handlers[EngineGenerateError]
    response = asyncio.run(handler(req, exc))
    payload = json.loads(response.body)

    assert "error" in payload
    assert payload["error"]["request_id"] == "req-123"
    assert payload["error"]["error_stage_id"] == 2
    assert response.status_code >= 400


def test_engine_dead_error_handler_registered_returns_json(monkeypatch) -> None:
    """Lock Omni ``EngineDeadError`` handler registration + JSON envelope.

    Fails if registration is skipped or the handler stops returning OpenAI-style
    JSON with ``request_id``.
    """
    app = FastAPI()
    # TODO(P0.2): retarget this call if ``_register_omni_exception_handlers``
    # leaves api_server (red here is a rename, not a surface regression).
    api_server._register_omni_exception_handlers(app)
    app.state.engine_client = SimpleNamespace(errored=True, engine=SimpleNamespace(is_alive=lambda: False))
    app.state.server = object()
    app.state.args = SimpleNamespace(log_error_stack=False)

    monkeypatch.setattr(api_server, "terminate_if_errored", lambda **_kwargs: None)

    handler = app.exception_handlers[EngineDeadError]
    req = _request_for(app)
    req.state.request_metadata = SimpleNamespace(request_id="dead-1")
    response = asyncio.run(handler(req, EngineDeadError()))
    payload = json.loads(response.body)
    assert payload["error"]["request_id"] == "dead-1"
    assert response.status_code >= 400


@pytest.mark.asyncio
async def test_pure_diffusion_app_state_key_snapshot(monkeypatch) -> None:
    """Lock pure-diffusion ``app.state`` keys after init, including live vs None.

    Fails if bootstrap drops keys route owners still read (e.g. video/speech/
    models), wires a handler that must stay None, or leaves a required
    handler as None.
    """
    stage = SimpleNamespace(engine_args={})
    engine = _FakeEngineClient(stage_configs=[stage])

    def _for_diffusion_factory(label: str):
        @classmethod
        def _factory(cls, *args, **kwargs):
            return _marker(label)

        return _factory

    monkeypatch.setattr(api_server, "get_stage_type", lambda _cfg: "diffusion")
    monkeypatch.setattr(api_server.OmniOpenAIServingChat, "for_diffusion", _for_diffusion_factory("chat"))
    monkeypatch.setattr(api_server.OmniOpenAIServingChatBatch, "for_diffusion", _for_diffusion_factory("chat_batch"))
    monkeypatch.setattr(
        api_server.OmniOpenAIServingAudioGenerate,
        "for_diffusion",
        _for_diffusion_factory("audio_generate"),
    )
    monkeypatch.setattr(api_server.OmniOpenAIServingVideo, "for_diffusion", _for_diffusion_factory("video"))
    monkeypatch.setattr(api_server.OmniStreamingVideoOutputHandler, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(api_server.OmniOpenAIServingSpeech, "for_diffusion", _for_diffusion_factory("speech"))
    monkeypatch.setattr(
        api_server.ServingRealtimeRobotOpenPI,
        "create_policy_server",
        classmethod(lambda cls, *a, **k: _marker("openpi")),
    )

    state = State()
    await api_server.omni_init_app_state(
        engine,
        state,
        _minimal_args(),
    )

    _assert_app_state_snapshot(
        state,
        expected_keys=_DIFFUSION_APP_STATE_KEYS,
        must_be_wired=_DIFFUSION_MUST_BE_WIRED,
        must_be_none=_DIFFUSION_MUST_BE_NONE,
    )
    assert state.diffusion_engine is engine


@pytest.mark.asyncio
async def test_pure_diffusion_speech_forwards_media_access_args(monkeypatch) -> None:
    stage = SimpleNamespace(engine_args={})
    engine = _FakeEngineClient(stage_configs=[stage])
    speech_kwargs = {}

    def _for_diffusion_factory(label: str):
        @classmethod
        def _factory(cls, *args, **kwargs):
            return _marker(label)

        return _factory

    @classmethod
    def _speech_factory(cls, *args, **kwargs):
        speech_kwargs.update(kwargs)
        return _marker("speech")

    monkeypatch.setattr(api_server, "get_stage_type", lambda _cfg: "diffusion")
    monkeypatch.setattr(api_server.OmniOpenAIServingChat, "for_diffusion", _for_diffusion_factory("chat"))
    monkeypatch.setattr(api_server.OmniOpenAIServingChatBatch, "for_diffusion", _for_diffusion_factory("chat_batch"))
    monkeypatch.setattr(
        api_server.OmniOpenAIServingAudioGenerate,
        "for_diffusion",
        _for_diffusion_factory("audio_generate"),
    )
    monkeypatch.setattr(api_server.OmniOpenAIServingVideo, "for_diffusion", _for_diffusion_factory("video"))
    monkeypatch.setattr(api_server.OmniStreamingVideoOutputHandler, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(api_server.OmniOpenAIServingSpeech, "for_diffusion", _speech_factory)
    monkeypatch.setattr(
        api_server.ServingRealtimeRobotOpenPI,
        "create_policy_server",
        classmethod(lambda cls, *a, **k: _marker("openpi")),
    )

    state = State()
    await api_server.omni_init_app_state(
        engine,
        state,
        _minimal_args(
            allowed_local_media_path="/allowed/media",
            allowed_media_domains=["media.example.com"],
        ),
    )

    assert speech_kwargs["allowed_local_media_path"] == "/allowed/media"
    assert speech_kwargs["allowed_media_domains"] == ["media.example.com"]


@pytest.mark.asyncio
async def test_multistage_app_state_key_snapshot(monkeypatch) -> None:
    """Lock multi-stage ``app.state`` keys after init, including live vs None.

    Fails if chat/speech/video/realtime/tokenization keys disappear or are
    left as None after init refactors — classic “route still registered,
    handler never wired” failure mode.
    """
    engine = _FakeEngineClient(
        stage_configs=[object(), object()],
        vllm_config=SimpleNamespace(
            lora_config=None,
            model_config=SimpleNamespace(),
            parallel_config=SimpleNamespace(_api_process_rank=0),
        ),
    )

    class _FakeModels:
        def __init__(self, *args, **kwargs):
            self.base_model_paths = kwargs.get("base_model_paths") or []

        async def init_static_loras(self):
            return None

    class _FakeCtor:
        def __init__(self, *args, **kwargs):
            pass

        def warmup(self):
            return None

    class _FakeSpeech(_FakeCtor):
        async def warmup(self):
            return None

    monkeypatch.setattr(api_server, "load_chat_template", lambda *_a, **_k: None)
    monkeypatch.setattr(api_server, "process_lora_modules", lambda modules, _defaults: modules or [])
    monkeypatch.setattr(api_server, "OpenAIServingModels", _FakeModels)
    monkeypatch.setattr(api_server, "OnlineRenderer", _FakeCtor)
    monkeypatch.setattr(api_server, "OpenAIServingResponses", _FakeCtor)
    monkeypatch.setattr(api_server, "OmniOpenAIServingChat", _FakeCtor)
    monkeypatch.setattr(api_server, "OmniOpenAIServingChatBatch", _FakeCtor)
    monkeypatch.setattr(api_server, "OpenAIServingCompletion", _FakeCtor)
    monkeypatch.setattr(api_server, "ServingPooling", _FakeCtor)
    monkeypatch.setattr(api_server, "OpenAIServingEmbedding", _FakeCtor)
    monkeypatch.setattr(api_server, "ServingClassification", _FakeCtor)
    monkeypatch.setattr(api_server, "ServingScores", _FakeCtor)
    monkeypatch.setattr(api_server, "ServingTokenization", _FakeCtor)
    monkeypatch.setattr(api_server, "OpenAIServingTranscription", _FakeCtor)
    monkeypatch.setattr(api_server, "OpenAIServingTranslation", _FakeCtor)
    monkeypatch.setattr(api_server, "AnthropicServingMessages", _FakeCtor)
    monkeypatch.setattr(api_server, "ServingTokens", _FakeCtor)
    monkeypatch.setattr(api_server, "OmniOpenAIServingSpeech", _FakeSpeech)
    monkeypatch.setattr(api_server, "OmniOpenAIServingAudioGenerate", _FakeCtor)
    monkeypatch.setattr(api_server, "OmniStreamingSpeechHandler", _FakeCtor)
    monkeypatch.setattr(api_server, "create_streaming_video_handler", lambda **_k: _marker("streaming_video"))
    monkeypatch.setattr(api_server, "OpenAIServingRealtime", _FakeCtor)
    monkeypatch.setattr(api_server, "OmniOpenAIServingVideo", _FakeCtor)
    monkeypatch.setattr(api_server, "should_enable_duplex_endpoint", lambda *_a, **_k: False)

    state = State()
    await api_server.omni_init_app_state(engine, state, _minimal_args())

    _assert_app_state_snapshot(
        state,
        expected_keys=_MULTISTAGE_APP_STATE_KEYS,
        must_be_wired=_MULTISTAGE_MUST_BE_WIRED,
        must_be_none=_MULTISTAGE_MUST_BE_NONE,
    )
