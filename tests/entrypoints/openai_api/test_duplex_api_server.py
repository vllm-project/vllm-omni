# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for serving-mode selection, duplex API wiring and warmup."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import State
from starlette.websockets import WebSocketDisconnect

from vllm_omni.config import stage_config
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig, PipelineConfig, StagePipelineConfig
from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex_omni import DuplexOmni
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.utils.tracking_parser import TrackingNamespace

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ``app.state`` of a duplex-only server. Presence is not enough: a route whose
# handler is missing answers "not available", so "not wired" is written as None.
_DUPLEX_APP_STATE_KEYS = {
    "engine_client",
    "log_stats",
    "args",
    "sleeping_stages",
    "stage_configs",
    "vllm_config",
    "diffusion_engine",
    "openai_serving_models",
    "serving_tokenization",
    "serving_tokens",
    "online_renderer",
    "openai_serving_chat",
    "openai_serving_chat_batch",
    "openai_serving_completion",
    "openai_serving_responses",
    "openai_serving_embedding",
    "openai_serving_pooling",
    "openai_serving_classification",
    "openai_serving_scores",
    "openai_serving_transcription",
    "openai_serving_translation",
    "openai_serving_speech",
    "openai_serving_audio_generate",
    "openai_serving_video",
    "openai_streaming_speech",
    "openai_streaming_video",
    "openai_streaming_video_output",
    "openai_serving_realtime",
    "openai_serving_realtime_robot",
    "anthropic_serving_messages",
    "openai_serving_duplex",
    "enable_server_load_tracking",
    "server_load_metrics",
}
#: Every turn-based service, plus the Realtime route that is not the duplex one.
#: ``openai_serving_chat`` is absent: a duplex engine also serves chat, through
#: the ordinary turn-based service, because DuplexOmni extends AsyncOmni.
#: ``online_renderer`` is absent for the same reason -- the chat service needs it.
_DUPLEX_MUST_BE_NONE = _DUPLEX_APP_STATE_KEYS - {
    "engine_client",
    "log_stats",
    "args",
    "sleeping_stages",
    "stage_configs",
    "vllm_config",
    "openai_serving_models",
    "openai_serving_duplex",
    "openai_serving_chat",
    "online_renderer",
    "enable_server_load_tracking",
    "server_load_metrics",
}
_DUPLEX_MUST_BE_WIRED = {
    "engine_client",
    "args",
    "stage_configs",
    "vllm_config",
    "openai_serving_models",
    "openai_serving_duplex",
    "openai_serving_chat",
}


class _FakeDuplexOmni(DuplexOmni):
    """A ``DuplexOmni`` without an engine: only what the app-state wiring reads."""

    def __init__(self) -> None:
        self.model = "demo-duplex-model"
        self._stage_configs = [object(), object(), object()]
        self._vllm_config = SimpleNamespace(
            lora_config=None,
            model_config=SimpleNamespace(),
            parallel_config=SimpleNamespace(_api_process_rank=0),
        )
        self._duplex_session_config = DuplexSessionRuntimeConfig()
        self._capabilities = DuplexCapabilities(supports_chat_completions=True)

    async def get_vllm_config(self):
        return self._vllm_config

    async def get_supported_tasks(self) -> tuple[str, ...]:
        return ("generate",)

    async def get_tokenizer(self):
        return SimpleNamespace(chat_template="{{ messages }}")

    @property
    def stage_configs(self) -> list[object]:
        return self._stage_configs

    @property
    def duplex_session_config(self) -> DuplexSessionRuntimeConfig:
        return self._duplex_session_config

    @property
    def duplex_capabilities(self) -> DuplexCapabilities:
        return self._capabilities

    # Properties on AsyncOmni, so the fake overrides rather than assigns them.
    @property
    def model_config(self) -> SimpleNamespace:
        return SimpleNamespace()

    @property
    def renderer(self) -> SimpleNamespace:
        return SimpleNamespace()


class _FakeModels:
    def __init__(self, *args, **kwargs) -> None:
        self.base_model_paths = kwargs.get("base_model_paths") or []


def _minimal_args(**overrides) -> SimpleNamespace:
    args = SimpleNamespace(
        model="demo-duplex-model",
        served_model_name=None,
        disable_log_stats=True,
        enable_log_requests=False,
        max_log_len=None,
        enable_server_load_tracking=False,
        trust_remote_code=True,
        deploy_config=None,
        # What the chat service reads when a duplex model also serves chat.
        chat_template=None,
        chat_template_content_format="auto",
        trust_request_chat_template=False,
        default_chat_template_kwargs=None,
        response_role="assistant",
        return_tokens_as_token_ids=False,
        enable_auto_tool_choice=False,
        exclude_tools_when_tool_choice_none=False,
        tool_call_parser=None,
        structured_outputs_config=SimpleNamespace(reasoning_parser=None),
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
        enable_log_outputs=False,
        enable_log_deltas=False,
        log_error_stack=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# --------------------------------------------------------------------------- #
# Startup: which entrypoint class serves the model                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pipeline_config", "expected"),
    [
        (SimpleNamespace(duplex_plugin="pkg.mod.Plugin"), True),
        (SimpleNamespace(duplex_plugin=None), False),
        (SimpleNamespace(duplex_plugin=""), False),
        (SimpleNamespace(), False),
        (None, False),
    ],
)
def test_duplex_serving_requires_a_pipeline_plugin(monkeypatch, tmp_path, pipeline_config, expected: bool) -> None:
    seen: dict[str, object] = {}
    deploy_path = tmp_path / "deploy.yaml"
    deploy_path.write_text("session_mode: duplex\n")

    def fake_get_pipeline_config(*, model, trust_remote_code, deploy_config_path=None, **_kwargs):
        seen.update(model=model, trust_remote_code=trust_remote_code, deploy_config_path=deploy_config_path)
        return pipeline_config

    monkeypatch.setattr(StageConfigFactory, "get_pipeline_config", fake_get_pipeline_config)

    kwargs = {"trust_remote_code": True, "deploy_config": str(deploy_path)}
    assert api_server._should_serve_duplex("demo-duplex-model", kwargs) is expected
    assert seen == {
        "model": "demo-duplex-model",
        "trust_remote_code": True,
        "deploy_config_path": str(deploy_path),
    }


def test_duplex_model_probe_propagates_resolution_errors(monkeypatch) -> None:
    """A pipeline that cannot be resolved must fail startup, not fall back to turn-based."""

    def fake_get_pipeline_config(**_kwargs):
        raise ValueError("needs trust_remote_code=True")

    monkeypatch.setattr(StageConfigFactory, "get_pipeline_config", fake_get_pipeline_config)

    with pytest.raises(ValueError, match="trust_remote_code"):
        api_server._should_serve_duplex("demo-duplex-model", {})


@pytest.fixture
def duplex_pipeline():
    return PipelineConfig(
        model_type="demo",
        stages=(StagePipelineConfig(stage_id=0, model_stage="demo", final_output=True),),
        duplex_plugin="pkg.mod.Plugin",
    )


@pytest.mark.parametrize("session_mode", ["turn", "duplex"])
@pytest.mark.parametrize("config_source", ["explicit", "default", "basename", "inherited"])
@pytest.mark.asyncio
async def test_serving_mode_selects_engine_and_preserves_stage_config(
    monkeypatch, tmp_path, duplex_pipeline, session_mode, config_source
):
    base_path = tmp_path / "base.yaml"
    base_path.write_text("session_mode: duplex\nstages:\n  - stage_id: 0\n    max_num_seqs: 4\n")
    deploy_path = tmp_path / "deploy.yaml"
    deploy_path.write_text(f"base_config: base.yaml\nsession_mode: {session_mode}\n")
    if config_source == "inherited":
        deploy_path = tmp_path / "overlay.yaml"
        deploy_path.write_text("base_config: deploy.yaml\n")
    pipeline = replace(duplex_pipeline, default_deploy_config_name=deploy_path.name)
    monkeypatch.setattr(StageConfigFactory, "get_pipeline_config", lambda **_: pipeline)
    monkeypatch.setattr(stage_config, "_DEPLOY_DIR", tmp_path)
    kwargs = {}
    if config_source != "default":
        kwargs["deploy_config"] = deploy_path.name if config_source == "basename" else str(deploy_path)

    created = []

    class TurnEngine:
        def __init__(self, model, **engine_kwargs):
            self.model = model
            self.kwargs = engine_kwargs
            self.closed = False
            created.append(self)

        def shutdown(self):
            self.closed = True

    class DuplexEngine(TurnEngine):
        pass

    monkeypatch.setattr(api_server, "AsyncOmni", TurnEngine)
    monkeypatch.setattr(api_server, "DuplexOmni", DuplexEngine)
    args = TrackingNamespace(
        Namespace(model="demo-model", disable_log_stats=True, **kwargs), frozenset({"model", *kwargs})
    )
    async with api_server.build_async_omni_from_stage_config(args) as engine:
        assert type(engine) is (DuplexEngine if session_mode == "duplex" else TurnEngine)
        assert engine.model == "demo-model"
        assert engine.kwargs == {**kwargs, "log_stats": False}
        assert not engine.closed
    assert len(created) == 1
    assert engine.closed
    deploy = stage_config.load_deploy_config(deploy_path)
    assert deploy.session_mode == session_mode
    assert deploy.stages[0].max_num_seqs == 4


@pytest.mark.parametrize("contents", ["{}\n", "session_mode: null\n", "session_mode: typo\n"])
def test_duplex_serving_rejects_missing_or_invalid_mode(monkeypatch, tmp_path, duplex_pipeline, contents):
    deploy_path = tmp_path / "deploy.yaml"
    deploy_path.write_text(contents)
    monkeypatch.setattr(StageConfigFactory, "get_pipeline_config", lambda **_: duplex_pipeline)
    with pytest.raises(ValueError, match="session_mode: turn or duplex"):
        api_server._should_serve_duplex("demo-model", {"deploy_config": str(deploy_path)})


def test_duplex_serving_requires_a_deploy_config(monkeypatch, duplex_pipeline):
    monkeypatch.setattr(StageConfigFactory, "get_pipeline_config", lambda **_: duplex_pipeline)
    with pytest.raises(ValueError, match="requires a deploy config"):
        api_server._should_serve_duplex("demo-model", {})


def test_duplex_serving_propagates_deploy_load_errors(monkeypatch, tmp_path, duplex_pipeline):
    monkeypatch.setattr(StageConfigFactory, "get_pipeline_config", lambda **_: duplex_pipeline)
    with pytest.raises(FileNotFoundError):
        api_server._should_serve_duplex("demo-model", {"deploy_config": str(tmp_path / "missing.yaml")})


@pytest.mark.parametrize("deploy_name,expected", [("minicpmo_4_5.yaml", True), ("minicpmo_4_5_turn.yaml", False)])
def test_minicpmo_serving_profiles(monkeypatch, deploy_name, expected):
    from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

    monkeypatch.setattr(StageConfigFactory, "get_pipeline_config", lambda **_: MINICPMO_4_5_PIPELINE)
    assert api_server._should_serve_duplex("openbmb/MiniCPM-o-4_5", {"deploy_config": deploy_name}) is expected
    assert api_server._should_serve_duplex("openbmb/MiniCPM-o-4_5", {}) is True


# --------------------------------------------------------------------------- #
# app.state of a duplex-only server                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("log_error_stack", [False, True])
async def test_duplex_app_state_wires_only_the_session_surfaces(monkeypatch, mocker, log_error_stack) -> None:
    """Lock the duplex ``app.state``: the two session-backed surfaces are live, every turn-based service is None.

    Fails if a turn-based service is wired into a duplex server (its route
    would answer instead of reporting "not available"), or if a key the routes
    read disappears. ``openai_serving_chat`` is the one shared name: the route
    is served, but by the adapter that runs it on a duplex session, never by
    the turn-based chat service.
    """
    monkeypatch.setattr(api_server, "OpenAIServingModels", _FakeModels)
    renderer_ctor = mocker.patch.object(api_server, "OnlineRenderer")
    monkeypatch.setattr(api_server, "OmniOpenAIServingChat", lambda **kwargs: SimpleNamespace(**kwargs))
    engine = _FakeDuplexOmni()
    state = State()

    await api_server.omni_init_app_state(engine, state, _minimal_args(log_error_stack=log_error_stack))

    present = {key for key in _DUPLEX_APP_STATE_KEYS if hasattr(state, key)}
    assert present == _DUPLEX_APP_STATE_KEYS
    not_wired = sorted(key for key in _DUPLEX_MUST_BE_WIRED if getattr(state, key) is None)
    assert not not_wired, f"duplex app.state keys registered but not wired: {not_wired}"
    unexpectedly_set = sorted(key for key in _DUPLEX_MUST_BE_NONE if getattr(state, key) is not None)
    assert not unexpectedly_set, f"turn-based services wired into a duplex server: {unexpectedly_set}"
    assert isinstance(state.openai_serving_duplex, OmniDuplexSessionHandler)
    assert state.openai_serving_chat is not None
    renderer_ctor.assert_called_once()
    assert renderer_ctor.call_args.kwargs["log_error_stack"] is log_error_stack
    assert state.openai_serving_chat.online_renderer is state.online_renderer is renderer_ctor.return_value
    state.online_renderer.warmup.assert_called_once_with()
    assert state.engine_client is engine
    assert state.vllm_config is engine._vllm_config


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #


class _RecordingHandler:
    """Stands in for ``OmniDuplexSessionHandler`` on the app state."""

    def __init__(self) -> None:
        self.queries: list[dict[str, str]] = []

    async def handle_realtime_session(self, websocket) -> None:
        await websocket.accept()
        self.queries.append(dict(websocket.query_params))
        await websocket.send_json({"type": "session.created", "session": {"id": "duplex-test"}})
        await websocket.close()


def _duplex_app(handler: object) -> FastAPI:
    app = FastAPI()
    app.include_router(api_server.router)
    app.state.openai_serving_duplex = handler
    # A duplex server serves no turn-based Realtime route.
    app.state.openai_serving_realtime = None
    return app


@pytest.mark.parametrize("path", ["/v1/duplex", "/v1/realtime?duplex=1"])
def test_both_duplex_routes_reach_the_session_handler(path: str) -> None:
    """``/v1/duplex`` is an alias of ``/v1/realtime?duplex=1``: same handler, same protocol."""
    handler = _RecordingHandler()

    with TestClient(_duplex_app(handler)) as client:
        with client.websocket_connect(path) as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_text()

    assert len(handler.queries) == 1


@pytest.mark.parametrize("flag", ["1", "true", "on"])
def test_realtime_route_switches_to_duplex_only_for_the_documented_flag_values(flag: str) -> None:
    handler = _RecordingHandler()

    with TestClient(_duplex_app(handler)) as client:
        with client.websocket_connect(f"/v1/realtime?duplex={flag}") as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_text()

    assert handler.queries == [{"duplex": flag}]


@pytest.mark.parametrize("query", ["", "?model=openbmb%2FMiniCPM-o-4_5"])
def test_realtime_without_a_duplex_flag_reaches_the_session_handler(query: str) -> None:
    """A bare connection to a duplex deployment is a duplex session.

    ``?duplex=1`` is the spelling the server advertises, but it is not a
    requirement: a deployment that declares a duplex plugin answers plain
    ``/v1/realtime`` too, so a stock Realtime client needs no vendor query
    parameter.
    """
    handler = _RecordingHandler()

    with TestClient(_duplex_app(handler)) as client:
        with client.websocket_connect(f"/v1/realtime{query}") as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_text()

    assert len(handler.queries) == 1


@pytest.mark.parametrize("query", ["?duplex=0", "?duplex=false", "?duplex=off"])
def test_realtime_opted_out_of_duplex_falls_through_to_the_turn_based_route(query: str) -> None:
    """An explicit opt-out selects the legacy handler, which a duplex server does not mount."""
    handler = _RecordingHandler()

    with TestClient(_duplex_app(handler)) as client:
        with client.websocket_connect(f"/v1/realtime{query}") as websocket:
            assert websocket.receive_json() == {
                "type": "error",
                "error": "Realtime API is not available",
                "code": "unsupported",
            }
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_text()

    assert handler.queries == []


# --------------------------------------------------------------------------- #
# Startup warmup gate                                                         #
# --------------------------------------------------------------------------- #


def _warmup_websocket(warmup_done: asyncio.Event | None, query: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(duplex_warmup_done=warmup_done)),
        query_params=query,
    )


@pytest.mark.asyncio
async def test_warmup_gate_holds_clients_until_the_warmup_session_finishes() -> None:
    warmup_done = asyncio.Event()
    websocket = _warmup_websocket(warmup_done, {})

    waiting = asyncio.create_task(api_server._wait_for_duplex_warmup(websocket))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiting), timeout=0.05)

    warmup_done.set()
    await asyncio.wait_for(waiting, timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("warmup_done", "query"),
    [
        (None, {}),
        (asyncio.Event(), {"vllm_omni_warmup": "1"}),
    ],
    ids=["no-warmup-configured", "the-warmup-connection-itself"],
)
async def test_warmup_gate_lets_the_warmup_connection_and_plain_servers_through(warmup_done, query) -> None:
    websocket = _warmup_websocket(warmup_done, query)

    await asyncio.wait_for(api_server._wait_for_duplex_warmup(websocket), timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "supports_chat, supported_tasks", [(False, ("generate",)), (True, ())], ids=["no-capability", "no-generation"]
)
async def test_duplex_chat_initialization_respects_model_gates(
    monkeypatch, mocker, supports_chat, supported_tasks
) -> None:
    """Neither gate may construct or warm up a chat renderer."""
    monkeypatch.setattr(api_server, "OpenAIServingModels", _FakeModels)
    renderer_ctor = mocker.patch.object(api_server, "OnlineRenderer")
    chat_ctor = mocker.patch.object(api_server, "OmniOpenAIServingChat")
    engine = _FakeDuplexOmni()
    engine._capabilities = DuplexCapabilities(supports_chat_completions=supports_chat)
    mocker.patch.object(engine, "get_supported_tasks", return_value=supported_tasks)
    state = State()

    await api_server.omni_init_app_state(engine, state, _minimal_args())

    assert state.openai_serving_chat is None
    assert state.openai_serving_chat_batch is None
    renderer_ctor.assert_not_called()
    chat_ctor.assert_not_called()
    assert isinstance(state.openai_serving_duplex, OmniDuplexSessionHandler)


def test_qwen_plugin_preserves_custom_turn_deployment_default(monkeypatch, tmp_path):
    from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_PIPELINE

    monkeypatch.setattr(StageConfigFactory, "get_pipeline_config", lambda **kwargs: QWEN3_OMNI_PIPELINE)
    deploy = tmp_path / "qwen.yaml"
    deploy.write_text("stages: []\n")
    assert not api_server._should_serve_duplex("qwen", {"deploy_config": str(deploy)})
    deploy.write_text("session_mode: duplex\nstages: []\n")
    assert api_server._should_serve_duplex("qwen", {"deploy_config": str(deploy)})


@pytest.mark.parametrize("flag", ["1", "true", "on"])
def test_turn_deployment_rejects_explicit_duplex_without_falling_back_to_stt(flag: str) -> None:
    app = _duplex_app(None)
    # Even an available STT service must not silently accept a duplex request.
    app.state.openai_serving_realtime = object()
    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/realtime?duplex={flag}") as websocket:
            assert websocket.receive_json() == {
                "type": "error",
                "code": "unsupported",
                "error": "VAD realtime is not enabled",
            }
            with pytest.raises(WebSocketDisconnect) as exc:
                websocket.receive_text()
            assert exc.value.code == 1008


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "explicit_template, tokenizer_template, tokenizer_error, expected",
    [
        ("explicit", None, False, "explicit"),
        (None, "tokenizer", False, None),
        (None, None, False, "model-json"),
        (None, None, True, "model-json"),
    ],
)
async def test_shared_chat_template_resolution(
    mocker, explicit_template, tokenizer_template, tokenizer_error, expected
):
    engine = _FakeDuplexOmni()
    tokenizer = mocker.patch.object(
        engine,
        "get_tokenizer",
        return_value=SimpleNamespace(chat_template=tokenizer_template),
        side_effect=RuntimeError("tokenizer unavailable") if tokenizer_error else None,
    )
    mocker.patch.object(api_server, "load_chat_template", return_value=explicit_template)
    model_template = mocker.patch.object(api_server, "_load_model_chat_template_json", return_value="model-json")

    assert await api_server._resolve_chat_template(engine, _minimal_args()) == expected
    assert tokenizer.call_count == (0 if explicit_template is not None else 1)
    assert model_template.call_count == (1 if expected == "model-json" else 0)
