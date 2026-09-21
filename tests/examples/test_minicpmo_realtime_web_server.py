# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVER_PATH = Path(__file__).resolve().parents[2] / "examples/online_serving/minicpmo/realtime_web/server.py"
spec = importlib.util.spec_from_file_location(
    "minicpmo_realtime_web_server_test",
    SERVER_PATH,
)
assert spec is not None and spec.loader is not None
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)

APP_DIR = server.APP_DIR
STATIC_DIR = server.STATIC_DIR
_join_ws_url = server._join_ws_url
build_app = server.build_app

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_join_ws_url_preserves_realtime_query():
    assert (
        _join_ws_url(
            "ws://127.0.0.1:8099/",
            "/v1/realtime",
            "duplex=1&model=openbmb%2FMiniCPM-o-4_5",
        )
        == "ws://127.0.0.1:8099/v1/realtime?duplex=1&model=openbmb%2FMiniCPM-o-4_5"
    )


def test_build_app_serves_health_and_injected_client_config():
    app = build_app(
        ws_backend="ws://127.0.0.1:9001",
        model="local/MiniCPM-o-4_5",
    )
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.text == "ok"

    index = client.get("/")
    assert index.status_code == 200
    assert '"model": "local/MiniCPM-o-4_5"' in index.text
    assert '"realtimePath": "v1/realtime"' in index.text
    assert "__FULL_DUPLEX_CONFIG__" not in index.text


def test_build_app_injects_public_realtime_url():
    public_realtime_url = "wss://proxy.example.test/backend/v1/realtime"
    app = build_app(
        ws_backend="ws://127.0.0.1:9001",
        public_realtime_url=public_realtime_url,
    )
    client = TestClient(app)

    index = client.get("/")

    assert index.status_code == 200
    assert f'"realtimePath": "{public_realtime_url}"' in index.text


def test_build_app_versions_client_bundle_and_disables_index_cache():
    client = TestClient(build_app())
    expected_version_hash = hashlib.sha256()
    for asset_name in (
        "app.js",
        "pcm_worklet.js",
        "playback_worklet.js",
        *[str(p.relative_to(STATIC_DIR)) for p in sorted((STATIC_DIR / "profiles").glob("*.js"))],
    ):
        expected_version_hash.update((STATIC_DIR / asset_name).read_bytes())
    expected_version = expected_version_hash.hexdigest()[:12]

    assert (APP_DIR / "index.html").is_file()
    assert (STATIC_DIR / "app.js").is_file()
    assert not (APP_DIR.parent / "static").exists()
    assert not (STATIC_DIR / "index.html").exists()

    index = client.get("/")

    assert index.status_code == 200
    assert f'src="static/app.js?v={expected_version}"' in index.text
    assert index.headers["cache-control"] == "no-store"


def test_build_app_exposes_realtime_websocket_and_static_assets():
    app = build_app()
    paths = {route.path for route in app.routes}

    assert "/v1/realtime" in paths
    assert "/static" in paths

    client = TestClient(app)
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/pcm_worklet.js").status_code == 200
    assert client.get("/static/playback_worklet.js").status_code == 200


def test_qwen_config_does_not_inject_reference_voice():
    client = TestClient(build_app(profile="qwen3-turn", adapter="vad", model="Qwen/Qwen3-Omni-30B-A3B-Instruct"))
    page = client.get("/").text
    assert '"profile": "qwen3-turn"' in page
    assert '"adapter": "vad"' in page
    assert '"refAudio": null' in page
    for name in ("common", "minicpm_native", "qwen3_turn"):
        assert client.get(f"/static/profiles/{name}.js").status_code == 200
    with pytest.raises(ValueError, match="ref-audio"):
        build_app(profile="qwen3-turn", ref_audio="unused.wav")


def test_config_cannot_close_its_script_element():
    page = TestClient(build_app(model="</script><script>alert(1)</script>")).get("/").text
    assert "</script><script>alert(1)" not in page
    assert "\\u003c/script>" in page


def test_proxy_reports_unreachable_backend(monkeypatch):
    from examples.online_serving.realtime_web import server as shared_server

    class RefusedConnection:
        async def __aenter__(self):
            raise ConnectionRefusedError(111, "Connection refused")

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(shared_server.websockets, "connect", lambda *args, **kwargs: RefusedConnection())
    client = TestClient(build_app(ws_backend="ws://127.0.0.1:8091"))
    with client.websocket_connect("/v1/realtime?duplex=0") as ws:
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["code"] == "backend_unavailable"
        assert "ws://127.0.0.1:8091" in event["error"]
        assert "--port" in event["error"]
