# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Discovery uses real FastAPI routes and initialized engine/config fixtures.

The engine fixture skips device startup; framework modules, router dependencies,
response validation, and the frontend pause gate are the production objects.
No model weights or tokenizer downloads are needed.
"""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.config.endpoint_policy import EndpointRestriction, OmniServingCapability, shutdown_unsupported_routes
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig, StageConfig, StageType
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.engine.omni_engine_base import OmniEngineBase
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.models.serving import _DiffusionServingModels
from vllm_omni.entrypoints.openai.serving_capabilities import OmniServingCapabilities
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.serving_capabilities import OPERATIONS, ServingCapabilities

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _engine(declaration=None):
    """Retain the real frontend gate, with device/engine startup omitted."""
    client = object.__new__(AsyncOmni)
    client._pause_cond = asyncio.Condition()
    client._paused = False
    client._sleeping_tags = set()
    client.engine = object.__new__(OmniEngineBase)
    client.engine.stage_configs = [
        StageConfig(stage_id=7, model_stage="encode", stage_type=StageType.LLM).to_omegaconf(),
        StageConfig(
            stage_id=42,
            model_stage="dit",
            stage_type=StageType.DIFFUSION,
            input_sources=[7],
            final_output=True,
            final_output_type="video",
        ).to_omegaconf(),
    ]
    client.engine.serving_capabilities = (
        declaration or get_diffusion_model_metadata("Cosmos3OmniDiffusersPipeline").serving_capabilities
    )
    client.generate = AsyncMock(side_effect=AssertionError("discovery must not generate"))
    client.wake_up = AsyncMock(side_effect=AssertionError("discovery must not wake workers"))
    client.check_health = AsyncMock(side_effect=AssertionError("discovery must not probe workers"))
    return client


def _app(client, *, count=1, restricted=(), missing_chat=False, duplex=False):
    app = FastAPI()
    app.include_router(api_server.router)
    app.state.engine_client = client
    app.state.stage_configs = client.stage_configs
    app.state.api_server_count = count
    app.state.openai_serving_models = _DiffusionServingModels(
        [
            BaseModelPath(name="public-primary", model_path="/private/checkpoint"),
            BaseModelPath(name="public-alias", model_path="/private/checkpoint"),
        ]
    )
    # Handler bodies are not called by discovery; presence is the initialized
    # deployment fact under test. Real generation is covered by family tests.
    for name in (
        "openai_serving_chat",
        "openai_serving_speech",
        "openai_serving_audio_generate",
        "openai_serving_video",
        "openai_streaming_video_output",
    ):
        setattr(app.state, name, object())
    if missing_chat:
        app.state.openai_serving_chat = None
    app.state.openai_serving_duplex = object() if duplex else None
    shutdown_unsupported_routes(app, restricted)
    app.state.omni_serving_capabilities = OmniServingCapabilities(
        app, client, restrictions=restricted, single_api_dependencies=(api_server._require_single_api_video_store,)
    )
    return app


def _routes(body):
    return {(route["transport"], route["method"], route["path"]): route["exposure"] for route in body["routes"]}


def test_real_route_and_openapi_report_aliases_and_original_stage_ids():
    engine = _engine()
    app = _app(engine)
    with TestClient(app) as client:
        models = client.get("/v1/models").json()
        response = client.get("/v1/omni/capabilities")
        assert response.status_code == 200
        body = response.json()
        assert {model["id"] for model in body["models"]} == {model["id"] for model in models["data"]}
        assert body["object"] == "omni.capabilities" and body["schema_version"] == 1
        assert body["stages"] == [
            {"id": "7", "type": "llm", "depends_on": []},
            {"id": "42", "type": "diffusion", "depends_on": ["7"]},
        ]
        assert body["models"][0]["stage_ids"] == ["7", "42"]
        assert body["models"][0]["operations"] == body["models"][1]["operations"]
        assert set(body["models"][0]["operations"]) == set(OPERATIONS)
        assert "/private/" not in response.text
        schema = client.get("/openapi.json").json()
        operation = schema["paths"]["/v1/omni/capabilities"]["get"]
        assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/OmniCapabilitiesResponse"
        )
    engine.generate.assert_not_called()
    engine.wake_up.assert_not_called()
    engine.check_health.assert_not_called()


def test_restriction_and_absent_handler_are_separate_from_model_support():
    restrictions = (EndpointRestriction(OmniServingCapability.IMAGE_EDITS, "Disabled by deployment"),)
    engine = _engine(get_diffusion_model_metadata("QwenImageEditPipeline").serving_capabilities)
    app = _app(engine, restricted=restrictions, missing_chat=True)
    with TestClient(app) as client:
        body = client.get("/v1/omni/capabilities").json()
    operations = body["models"][0]["operations"]
    assert operations["image_edit"]["support"] == "supported"
    assert operations["image_edit"]["exposure"] == "disabled"
    assert operations["image_edit"]["reason_code"] == "endpoint_disabled"
    assert operations["chat"]["support"] == "supported"
    assert operations["chat"]["exposure"] == "disabled"
    assert _routes(body)[("http", "POST", "/v1/images/edits")] == "disabled"


@pytest.mark.parametrize("count", [None, 1, 2])
def test_process_local_video_store_and_execution_scope(count):
    app = _app(_engine(), count=count)
    with TestClient(app) as client:
        body = client.get("/v1/omni/capabilities").json()
    routes = _routes(body)
    expected = "unknown" if count is None else "enabled" if count == 1 else "disabled"
    assert routes[("http", "POST", "/v1/videos")] == expected
    assert routes[("http", "POST", "/v1/videos/sync")] == "enabled"
    operation = body["models"][0]["operations"]["video_generation"]
    assert operation["exposure"] == "enabled"
    assert operation["execution"]["scope"] == ("deployment" if count == 1 else "worker")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "count,policy,resume",
    [
        (None, None, None),
        (1, None, None),
        (0, None, None),
        (True, None, None),
        ("2", None, None),
        (2, None, None),
        (2, "least_sessions", True),
        (2, "least_sessions", False),
    ],
)
async def test_duplex_exposure_matches_the_existing_admission_guard(count, policy, resume):
    engine = _engine()
    engine.duplex_session_config = (
        DuplexSessionRuntimeConfig(replica_policy=policy, max_sessions_per_replica=1) if policy is not None else None
    )
    engine.duplex_capabilities = DuplexCapabilities(supports_session_resume=resume) if resume is not None else None
    app = _app(engine, count=count, duplex=True)
    websocket = WebSocket(
        {"type": "websocket", "app": app},
        receive=AsyncMock(return_value={"type": "websocket.connect"}),
        send=AsyncMock(),
    )
    rejected = await api_server._reject_multi_api_duplex(websocket)
    response = await app.state.omni_serving_capabilities.get_capabilities()
    route = next(route for route in response.routes if route.path == "/v1/duplex")
    assert route.exposure == ("disabled" if rejected else "enabled")


@pytest.mark.parametrize("count", [None, 1, 2])
def test_existing_video_and_image_guards_can_disable_declared_support(count):
    engine = _engine(get_diffusion_model_metadata("Wan22VACEPipeline").serving_capabilities)
    # The old metadata default can label a pipeline as image. Discovery reports
    # the resulting video guard; it does not rewrite that initialization policy.
    engine.engine.stage_configs = [
        StageConfig(
            0, "diffusion", stage_type=StageType.DIFFUSION, final_output=True, final_output_type="image"
        ).to_omegaconf()
    ]
    with TestClient(_app(engine, count=count)) as client:
        body = client.get("/v1/omni/capabilities").json()
    video = body["models"][0]["operations"]["video_generation"]
    assert video["support"] == "supported" and video["exposure"] == "disabled"
    assert all(
        _routes(body)[key] == "disabled"
        for key in (
            ("http", "POST", "/v1/videos"),
            ("http", "POST", "/v1/videos/sync"),
            ("websocket", "WEBSOCKET", "/v1/realtime/video"),
        )
    )
    engine.engine.stage_configs = [
        StageConfig(0, "thinker", final_output=True, final_output_type="text").to_omegaconf()
    ]
    with TestClient(_app(engine, count=count)) as client:
        body = client.get("/v1/omni/capabilities").json()
    assert body["models"][0]["operations"]["image_generation"]["exposure"] == "disabled"


def test_query_observes_pause_and_partial_sleep_without_waking_or_mutating():
    engine = _engine()
    app = _app(engine)
    with TestClient(app) as client:

        def state():
            body = client.get("/v1/omni/capabilities").json()
            return body["models"][0]["operations"]["image_generation"]["execution"]["state"]

        assert state() == "unpaused"
        engine._paused = True
        assert state() == "paused"
        engine._paused = False
        engine._sleeping_tags.add("weights")
        assert state() == "paused"
        assert engine._sleeping_tags == {"weights"}
        engine._sleeping_tags.clear()
        assert state() == "unpaused"
    engine.generate.assert_not_called()
    engine.wake_up.assert_not_called()
    engine.check_health.assert_not_called()


def test_duplex_execution_is_unknown_and_missing_declaration_stays_unknown():
    engine = _engine(ServingCapabilities())
    app = _app(engine, duplex=True)
    with TestClient(app) as client:
        body = client.get("/v1/omni/capabilities").json()
    for operation in body["models"][0]["operations"].values():
        assert operation["support"] == "unknown"
        assert operation["reason_code"] == "declaration_missing"
        assert operation["execution"]["state"] == "unknown"


def test_projection_does_not_reparse_configuration_on_queries():
    engine = _engine()
    app = _app(engine)
    # Startup has captured the topology and declarations. Replacing their
    # original containers cannot alter a published snapshot mid-request.
    engine.engine.stage_configs.clear()
    engine.engine.serving_capabilities = ServingCapabilities()
    with TestClient(app) as client:
        first = client.get("/v1/omni/capabilities").json()
        second = client.get("/v1/omni/capabilities").json()
    assert len(first["stages"]) == 2
    assert first["models"][0]["operations"]["image_generation"]["support"] == "supported"
    assert second["observed_at"] >= first["observed_at"]


def test_uninitialized_service_returns_503_instead_of_resolving_models():
    app = FastAPI()
    app.include_router(api_server.router)
    with TestClient(app) as client:
        assert client.get("/v1/omni/capabilities").status_code == 503


async def build_initialized_test_app(
    monkeypatch, tmp_path, architecture="QwenImagePipeline", api_workers=1, api_key=None, allowed_origins=None
):
    """Build real serving objects around a weight-free engine fixture."""
    metadata = get_diffusion_model_metadata(architecture)
    engine = _engine(metadata.serving_capabilities)
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    engine.config_path = None
    engine.input_processor = None
    engine.engine.stage_configs = [
        StageConfig(
            0,
            "diffusion",
            stage_type=StageType.DIFFUSION,
            final_output=True,
            final_output_type=metadata.final_output_type or "image",
        ).to_omegaconf()
    ]
    engine.engine.stage_vllm_configs = [None]
    engine.engine.stage_clients = []
    # No backend was started. This is the already-initialized device-config
    # boundary; model/config resolution itself is never invoked by this test.
    monkeypatch.setattr(engine, "get_diffusion_od_config", lambda: None)
    args = api_server.make_arg_parser(FlexibleArgumentParser()).parse_args([])
    args.model = "local-engine-fixture"
    args.served_model_name = ["public-primary", "public-alias"]
    if api_key is not None:
        args.api_key = [api_key]
    if allowed_origins is not None:
        args.allowed_origins = allowed_origins
    app = api_server.build_openai_app(args, ("generate",))
    for path, method in (
        ("/v1/chat/completions", "POST"),
        ("/v1/chat/completions/batch", "POST"),
        ("/v1/models", "GET"),
        ("/health", "GET"),
    ):
        api_server.remove_route_from_app(app, path, {method})
    app.include_router(api_server.router)
    app.state.api_server_count = api_workers
    await api_server.omni_init_app_state(engine, app.state, args)
    app.state.omni_serving_capabilities = OmniServingCapabilities(
        app, engine, single_api_dependencies=(api_server._require_single_api_video_store,)
    )
    return app, engine


@pytest.mark.asyncio
@pytest.mark.parametrize("architecture", ["QwenImagePipeline", "Cosmos3OmniDiffusersPipeline"])
async def test_real_upstream_app_and_omni_handler_initialization(monkeypatch, tmp_path, architecture):
    """Keep the real upstream builder, Omni init, handlers, and HTTP transport."""
    app, engine = await build_initialized_test_app(monkeypatch, tmp_path, architecture)
    try:
        assert isinstance(app.state.openai_serving_chat, OmniOpenAIServingChat)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.get("/v1/omni/capabilities")
            assert response.status_code == 200
            body = response.json()
            operation = body["models"][0]["operations"]["image_generation"]
            assert operation["support"] == "supported" and operation["exposure"] == "enabled"
            assert [model["id"] for model in body["models"]] == ["public-primary", "public-alias"]
    finally:
        app.state.openai_serving_video.shutdown()
        # The diffusion speech factory owns no TTS executor; its speaker
        # registry is isolated under tmp_path by SPEAKER_SAMPLES_DIR above.
    engine.generate.assert_not_called()


@pytest.mark.asyncio
async def test_real_app_authentication_and_cors_protect_discovery(monkeypatch, tmp_path):
    app, engine = await build_initialized_test_app(
        monkeypatch, tmp_path, api_key="fixture-secret", allowed_origins=["https://director.example"]
    )
    observer = AsyncMock(wraps=engine.get_execution_state)
    monkeypatch.setattr(app.state.omni_serving_capabilities, "_observer", observer)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            for headers in ({}, {"Authorization": "Bearer wrong"}):
                assert (await client.get("/v1/omni/capabilities", headers=headers)).status_code == 401
            observer.assert_not_called()
            preflight = await client.options(
                "/v1/omni/capabilities",
                headers={
                    "Origin": "https://director.example",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "Authorization",
                },
            )
            assert preflight.status_code == 200
            assert preflight.headers["access-control-allow-origin"] == "https://director.example"
            observer.assert_not_called()
            response = await client.get(
                "/v1/omni/capabilities",
                headers={"Authorization": "Bearer fixture-secret", "Origin": "https://director.example"},
            )
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == "https://director.example"
            observer.assert_awaited_once()
    finally:
        app.state.openai_serving_video.shutdown()
    engine.generate.assert_not_called()
    engine.wake_up.assert_not_called()
    engine.check_health.assert_not_called()
