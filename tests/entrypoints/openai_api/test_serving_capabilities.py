# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Discovery uses real FastAPI routes and initialized engine/config fixtures.

The engine fixture skips device startup; framework modules, router dependencies,
response validation, and the frontend pause gate are the production objects.
No model weights or tokenizer downloads are needed.
"""

import asyncio
import io
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import State
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


@pytest.mark.asyncio
@pytest.mark.parametrize("architecture,maximum", [("Flux2KleinPipeline", 4), ("ZImagePipeline", 1)])
async def test_image_request_contract_preserves_model_and_route_facts(monkeypatch, tmp_path, architecture, maximum):
    app, engine = await build_initialized_test_app(monkeypatch, tmp_path, architecture)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/omni/capabilities")
    assert response.status_code == 200
    profile = response.json()["models"][0]["operations"]["image_edit"]["profiles"][0]
    assert profile["request"]["inputs"]["image"]["max_count"] == maximum
    route = next(route for route in profile["routes"] if route["path"] == "/v1/images/edits")
    assert route["exposure"] == "enabled"
    assert route["request"]["inputs"]["image"]["min_count"] == 1
    assert route["request"]["parameters"]["width"]["minimum"] == 1
    if architecture == "Flux2KleinPipeline":
        assert route["request"]["mask"]["geometry"]["state"] == "unknown"
        assert route["request"]["mask"]["max_source_count"] == 1
        assert route["request"]["mask"]["source_resize"]["max_pixels"] == 1048576
        assert "maximum" not in route["request"]["parameters"]["width"]
    else:
        assert route["request"]["inputs"]["mask_image"]["support"] == "unsupported"
    engine.generate.assert_not_called()


def _engine(declaration=None):
    """Retain the real frontend gate, with device/engine startup omitted."""
    client = object.__new__(AsyncOmni)
    client._pause_cond = asyncio.Condition()
    client._paused = False
    client._sleeping_tags = set()
    client.engine = object.__new__(OmniEngineBase)
    client.engine.stage_clients = []
    client.engine._diffusion_od_config_view = SimpleNamespace()
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


@pytest.mark.parametrize("observation", [ConnectionError("observer unavailable"), None, "invalid", [], True])
def test_execution_observer_failure_preserves_discovery_and_recovers(observation):
    engine = _engine()
    app = _app(engine)
    service = app.state.omni_serving_capabilities
    with TestClient(app, raise_server_exceptions=False) as client:
        healthy = client.get("/v1/omni/capabilities").json()
        service._observer = AsyncMock(side_effect=[observation, observation, "paused"])
        for _ in range(2):
            response = client.get("/v1/omni/capabilities")
            assert response.status_code == 200
            body = response.json()
            assert body["routes"] == healthy["routes"]
            assert body["stages"] == healthy["stages"]
            assert body["schema_version"] == healthy["schema_version"] == 1
            for model, original in zip(body["models"], healthy["models"], strict=True):
                assert model["id"] == original["id"]
                assert model["stage_ids"] == original["stage_ids"]
                for name, operation in model["operations"].items():
                    execution = operation.pop("execution")
                    assert execution["state"] == "unknown"
                    assert execution["scope"] == "deployment"
                    assert execution["observed_at"] == body["observed_at"]
                    assert operation == {k: v for k, v in original["operations"][name].items() if k != "execution"}
        recovered = client.get("/v1/omni/capabilities")
        assert recovered.status_code == 200
        assert recovered.json()["models"][0]["operations"]["chat"]["execution"]["state"] == "paused"
    engine.generate.assert_not_called()
    engine.wake_up.assert_not_called()
    engine.check_health.assert_not_called()


@pytest.mark.asyncio
async def test_execution_observer_cancellation_propagates():
    engine = _engine()
    service = _app(engine).state.omni_serving_capabilities
    service._observer = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await service.get_capabilities()
    engine.generate.assert_not_called()
    engine.wake_up.assert_not_called()
    engine.check_health.assert_not_called()


def test_image_edit_route_requires_its_actual_diffusion_stage_guard():
    engine = _engine()
    engine.engine.stage_configs = [
        StageConfig(
            stage_id=0, model_stage="image", stage_type=StageType.LLM, final_output=True, final_output_type="image"
        ).to_omegaconf()
    ]
    with TestClient(_app(engine)) as client:
        routes = _routes(client.get("/v1/omni/capabilities").json())
        source = io.BytesIO()
        Image.new("RGB", (16, 16)).save(source, format="PNG")
        response = client.post("/v1/images/edits", data={"prompt": "edit"}, files={"image": source.getvalue()})
        assert response.status_code == 503
        assert response.json()["detail"] == "No diffusion stage found in multi-stage pipeline."
    engine.generate.assert_not_called()
    assert routes[("http", "POST", "/v1/images/generations")] == "enabled"
    assert routes[("http", "POST", "/v1/images/edits")] == "disabled"


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


@dataclass
class _EditConfigView:
    supports_multimodal_inputs: object = True
    max_multimodal_image_inputs: object = 3


@dataclass
class _ConfigEngine:
    od_config: object

    def get_diffusion_od_config(self):
        return self.od_config


@pytest.mark.parametrize(
    "config,maximum",
    [
        (None, None),
        (_EditConfigView(None, 3), None),
        (_EditConfigView(False, 3), 1),
        (_EditConfigView(True, 3), 3),
        (_EditConfigView(True, None), None),
        (_EditConfigView(True, True), None),
        (_EditConfigView(True, "3"), None),
        (_EditConfigView(True, 0), None),
    ],
)
def test_startup_config_reading_needs_no_request(monkeypatch, config, maximum):
    from vllm_omni.entrypoints.openai import app_state
    from vllm_omni.entrypoints.openai.images import helpers

    engine = _engine(get_diffusion_model_metadata("Flux2KleinPipeline").serving_capabilities)
    engine.engine.stage_configs = [
        StageConfig(0, "diffusion", stage_type=StageType.DIFFUSION, final_output=True, final_output_type="image")
    ]
    app = _app(engine)
    app.state.diffusion_engine = _ConfigEngine(config)
    assert app_state._get_diffusion_od_config_from_state(app.state, engine) is config
    assert helpers._max_edit_input_images_from_config(config) == maximum
    request = Request({"type": "http", "app": app})
    assert helpers._get_max_edit_input_images(request, engine) == maximum
    with monkeypatch.context() as patch:
        patch.setattr(Request, "__init__", lambda *a, **kw: pytest.fail("startup must not create a Request"))
        app.state.omni_serving_capabilities = OmniServingCapabilities(app, engine)
    # Later queries read captured configuration, never these accessors again.
    monkeypatch.setattr(
        app_state, "_get_diffusion_od_config_from_state", lambda *a: pytest.fail("query must not read config")
    )
    with TestClient(app) as client:
        body = client.get("/v1/omni/capabilities").json()
    profile = body["models"][0]["operations"]["image_edit"]["profiles"][0]
    route = next(r for r in profile["routes"] if r["path"] == "/v1/images/edits")
    # Klein declares four sources; an initialized handler can tighten the limit.
    assert route["request"]["inputs"]["image"]["max_count"] == (maximum if maximum is not None else 4)
    engine.generate.assert_not_called()
    engine.wake_up.assert_not_called()


def test_diffusion_config_source_priority():
    from vllm_omni.entrypoints.openai.app_state import _get_diffusion_od_config_from_state

    engine_config = _EditConfigView(True, 4)
    app_config = _EditConfigView(False, 1)
    engine = _ConfigEngine(engine_config)
    state = State()
    assert _get_diffusion_od_config_from_state(state, engine) is engine_config
    state.diffusion_engine = _ConfigEngine(app_config)
    assert _get_diffusion_od_config_from_state(state, engine) is app_config
    # A legacy engine exposes only the initialized attribute; a getter, when
    # present, takes priority even if it returns unknown.
    state.diffusion_engine = State({"od_config": app_config})
    assert _get_diffusion_od_config_from_state(state, engine) is app_config
    state.diffusion_engine.get_diffusion_od_config = lambda: None
    assert _get_diffusion_od_config_from_state(state, engine) is None


@pytest.mark.parametrize(
    "count,exposure,status",
    [(None, "unknown", 503), (True, "unknown", 503), (0, "unknown", 503), (1, "enabled", 200), (2, "disabled", 409)],
)
def test_video_store_exposure_matches_real_dependency(count, exposure, status):
    app = _app(_engine(), count=count)

    @app.post("/v1/extension", dependencies=[Depends(api_server._require_single_api_video_store)])
    async def extension():
        return {}

    app.state.omni_serving_capabilities = OmniServingCapabilities(
        app, app.state.engine_client, single_api_dependencies=(api_server._require_single_api_video_store,)
    )
    with TestClient(app) as client:
        routes = _routes(client.get("/v1/omni/capabilities").json())
        assert routes[("http", "POST", "/v1/videos")] == exposure
        # An unknown extension does not become known just because its worker
        # topology is valid.
        assert routes[("http", "POST", "/v1/extension")] == ("unknown" if count == 1 else exposure)
        response = client.post("/v1/extension")
        assert response.status_code == status
        if status == 503:
            assert response.json()["detail"] == "API worker topology is not initialized."
        elif status == 409:
            assert "Asynchronous video jobs is not supported" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,attribute,payload,message",
    [
        (
            "/v1/chat/completions",
            "openai_serving_chat",
            {"model": "public-primary", "messages": [{"role": "user", "content": "hello"}]},
            "Chat Completions API",
        ),
        ("/v1/audio/speech", "openai_serving_speech", {"input": "hello"}, "Speech API"),
        ("/v1/audio/generate", "openai_serving_audio_generate", {"input": "ocean"}, "Audio Generate API"),
    ],
)
async def test_missing_handlers_match_actual_http_rejection(monkeypatch, tmp_path, path, attribute, payload, message):
    app, engine = await build_initialized_test_app(monkeypatch, tmp_path)
    setattr(app.state, attribute, None)
    app.state.omni_serving_capabilities = OmniServingCapabilities(app, engine)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            routes = _routes((await client.get("/v1/omni/capabilities")).json())
            assert routes[("http", "POST", path)] == "disabled"
            response = await client.post(path, json=payload)
            assert response.status_code == 404, response.text
            assert message in response.text
    finally:
        app.state.openai_serving_video.shutdown()
    engine.generate.assert_not_called()


@pytest.mark.parametrize(
    "path,attribute,message",
    [
        ("/v1/audio/speech/stream", "openai_streaming_speech", "Streaming speech is not available"),
        ("/v1/video/chat/stream", "openai_streaming_video", "Streaming video chat is not available"),
        ("/v1/realtime/video", "openai_streaming_video_output", "Streaming video generation is not available"),
        ("/v1/duplex", "openai_serving_duplex", "Duplex API is not available"),
        ("/v1/realtime/robot/openpi", "openai_serving_realtime_robot", "Robot policy not available"),
    ],
)
def test_missing_handlers_match_actual_websocket_rejection(path, attribute, message):
    engine = _engine()
    app = _app(engine)
    setattr(app.state, attribute, None)
    app.state.omni_serving_capabilities = OmniServingCapabilities(app, engine)
    with TestClient(app) as client:
        routes = _routes(client.get("/v1/omni/capabilities").json())
        assert routes[("websocket", "WEBSOCKET", path)] == "disabled"
        with client.websocket_connect(path) as socket:
            error = socket.receive_json()
            assert error["type"] == "error"
            assert error.get("message", error.get("error")) == message
    engine.generate.assert_not_called()


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


async def build_video_capability_test_app(monkeypatch, tmp_path, architecture, task_type=None, multistage=False):
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.config.omni_config import VllmOmniConfig
    from vllm_omni.config.stage_config import DeployConfig
    from vllm_omni.diffusion.capability_resolvers import resolve_minimax_h3_capabilities
    from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
    from vllm_omni.model_executor.models.minimax_h3.pipeline import MINIMAX_H3_PIPELINE

    overrides = {"model": "local-checkpoint", "model_class_name": architecture, "task_type": task_type}
    if multistage:
        stages = VllmOmniConfig.from_pipeline_config(
            MINIMAX_H3_PIPELINE, user_deploy_config=DeployConfig(), cli_overrides=overrides
        ).stage_configs
    else:
        stages = StageConfigFactory.create_typed_default_diffusion("local-checkpoint", overrides).stage_configs
    declaration = get_diffusion_model_metadata(architecture).serving_capabilities
    if architecture in ("MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline"):
        declaration = resolve_minimax_h3_capabilities(declaration, stages)
    app, engine = await build_initialized_test_app(monkeypatch, tmp_path, architecture)
    # Supply resolved metadata at the initialized-engine boundary. The real
    # app, video handler, projector and HTTP serialization remain in use;
    # the encoder worker/tokenizer is outside these discovery tests.
    engine.engine.stage_configs = stages
    engine.engine.serving_capabilities = declaration
    app.state.stage_configs = stages
    app.state.openai_serving_video.shutdown()
    app.state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
        engine, model_name="public-primary", stage_configs=stages
    )
    monkeypatch.setattr(engine, "get_diffusion_od_config", lambda: SimpleNamespace(model_class_name=architecture))
    app.state.omni_serving_capabilities = OmniServingCapabilities(
        app, engine, single_api_dependencies=(api_server._require_single_api_video_store,)
    )
    return app, engine


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", [None, "auto", "t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("multistage", [False, True])
async def test_h3_http_discovery_retains_known_edit_fields(monkeypatch, tmp_path, task_type, multistage):
    app, engine = await build_video_capability_test_app(
        monkeypatch, tmp_path, "MiniMaxH3Pipeline", task_type, multistage
    )
    # Querying the startup projection must not revisit configuration or load a worker.
    monkeypatch.setattr(engine, "get_diffusion_od_config", lambda: pytest.fail("query re-read config"))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.get("/v1/omni/capabilities")
        assert response.status_code == 200
        body = response.json()
        assert body["schema_version"] == 1
        operation = body["models"][0]["operations"]["video_generation"]
        assert operation["support"] == "unknown"
        for profile in operation["profiles"]:
            if profile["id"] == "text_to_video" and task_type == "ref2va":
                assert profile["support"] == "unsupported"
                continue
            assert profile["support"] == "unknown"
            assert profile["reason_code"] == "worker_checkpoint_tasks_unavailable"
            for route in profile["routes"]:
                request = route["request"]
                assert request["inputs"]["last_frame"]["support"] == "unsupported"
                for role in ("source_video", "video_noise_mask", "audio_noise_mask"):
                    assert role not in profile["required_inputs"]
                    if route["path"] == "/v1/realtime/video":
                        assert request["inputs"][role]["support"] == "unsupported"
                    else:
                        assert request["inputs"][role] == {"support": "supported", "min_count": 0, "max_count": 1}
    finally:
        app.state.openai_serving_video.shutdown()
    engine.generate.assert_not_called()
    engine.wake_up.assert_not_called()
    engine.check_health.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "architecture,profile_id,first_frame",
    [
        ("WanImageToVideoPipeline", "image_to_video", "supported"),
        ("WanVACEPipeline", "image_to_video", "unsupported"),
        ("SeedVR2Pipeline", "video_to_video", None),
    ],
)
async def test_video_http_discovery_uses_real_handler_edit_gate(
    monkeypatch, tmp_path, architecture, profile_id, first_frame
):
    app, engine = await build_video_capability_test_app(monkeypatch, tmp_path, architecture)
    try:
        assert app.state.openai_serving_video.supports_latent_mask_editing is False
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.get("/v1/omni/capabilities")
        assert response.status_code == 200
        operation = response.json()["models"][0]["operations"]["video_generation"]
        profile = next(profile for profile in operation["profiles"] if profile["id"] == profile_id)
        assert profile["support"] == "supported"
        for route in profile["routes"]:
            inputs = route["request"]["inputs"]
            for role in ("source_video", "video_noise_mask", "audio_noise_mask"):
                assert inputs[role] == {"support": "unsupported", "max_count": 0}
            if first_frame is None:
                assert "first_frame" not in inputs
            else:
                assert inputs["first_frame"]["support"] == first_frame
    finally:
        app.state.openai_serving_video.shutdown()
    engine.generate.assert_not_called()


@pytest.mark.asyncio
async def test_pan2_video_routes_preserve_first_frame_limits(monkeypatch, tmp_path):
    app, engine = await build_initialized_test_app(monkeypatch, tmp_path, "PAN2ModularPipeline")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.get("/v1/omni/capabilities")
        assert response.status_code == 200
        operation = response.json()["models"][0]["operations"]["video_generation"]
        assert operation["support"] == "supported" and operation["exposure"] == "enabled"
        profiles = {profile["id"]: profile for profile in operation["profiles"]}
        assert profiles["video_to_video"]["support"] == "unsupported"
        for name in ("text_to_video", "image_to_video"):
            profile = profiles[name]
            assert profile["outputs"] == ["video"]
            assert profile["required_inputs"] == (["text"] if name == "text_to_video" else ["text", "image"])
            for path in ("/v1/videos", "/v1/videos/sync"):
                route = next(route for route in profile["routes"] if route["path"] == path)
                assert route["exposure"] == "enabled"
                inputs = route["request"]["inputs"]
                assert inputs["video"]["support"] == "unsupported"
                assert inputs["image"]["max_count"] == (0 if name == "text_to_video" else 1)
                if name == "image_to_video":
                    assert inputs["image"]["min_count"] == 1
    finally:
        app.state.openai_serving_video.shutdown()
    engine.generate.assert_not_called()


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
