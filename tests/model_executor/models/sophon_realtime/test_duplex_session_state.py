# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session state of the Sophon-Realtime duplex plugin: closed sessions are forgotten, ``clear()`` invalidates what
was decided about the dropped audio, and the loopback turn checker authenticates against ``--api-key``."""

import gc
import http.server
import json
import threading
import time

import numpy as np
import pybase64 as base64
import pytest

from vllm_omni.model_executor.models.sophon_realtime.duplex import env as sophon_env
from vllm_omni.model_executor.models.sophon_realtime.duplex import plugin as plugin_mod
from vllm_omni.model_executor.models.sophon_realtime.duplex import turn_check

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _rid(session_id: str, turn: int = 1) -> str:
    b64 = base64.urlsafe_b64encode(session_id.encode()).decode().rstrip("=")
    return f"duplex-s.{b64}.e.1.r.stage0-turn{turn}"


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setattr(plugin_mod, "start_warmup", lambda checker: None)  # no background thread polling a server
    return plugin_mod.SophonRealtimeDuplexPlugin(encode_audio=lambda *a, **k: b"")


def _session(plugin, session_id):
    state = plugin.create_session_state()
    buf = state.audio_buffer
    buf.session_id = session_id
    return state, buf


# ---------------------------------------------------------------- session close
def test_close_session_forgets_buffer_history_requests_and_speculative_state(plugin):
    state_a, a = _session(plugin, "sess-a")
    state_b, b = _session(plugin, "sess-b")
    a.history.append({"user": np.zeros(16000, np.float32), "a_audio": None, "a_text": "hi"})
    a.prev_user = np.zeros(1600, np.float32)
    a.spec_ids.add("spec-a")
    plugin._spec_pending["spec-a"] = "drop"
    plugin._spec_rid["spec-a2"] = _rid("sess-a", 2)
    plugin._keys["key-a"] = a
    plugin._keys["key-b"] = b
    plugin._reqs[_rid("sess-a")] = {"kind": "reply", "buf": a}
    plugin._reqs[_rid("sess-a", 2)] = {"kind": "check", "buf": None}
    plugin._reqs[_rid("sess-b")] = {"kind": "reply", "buf": b}
    plugin.data_plane.mark_terminal(_rid("sess-a"))
    plugin.data_plane.mark_terminal(_rid("sess-b"))

    plugin.data_plane.close_session("sess-a", active_request_id=None)

    assert id(a) not in plugin._bufs and id(b) in plugin._bufs
    assert a.history == [] and a.prev_user is None
    assert set(plugin._reqs) == {_rid("sess-b")}
    assert "spec-a" not in plugin._spec_pending and "spec-a2" not in plugin._spec_rid
    assert set(plugin._keys) == {"key-b"}
    assert not plugin.data_plane.is_terminal(_rid("sess-a")) and plugin.data_plane.is_terminal(_rid("sess-b"))


def test_buffer_is_released_with_its_session_state(plugin):
    state, buf = _session(plugin, "sess-a")
    del buf
    assert len(plugin._bufs) == 1
    del state
    gc.collect()
    assert len(plugin._bufs) == 0


def test_close_stream_forgets_terminal_request(plugin):
    plugin.data_plane.mark_terminal("r1")
    plugin.data_plane.close_stream("r1")
    assert not plugin.data_plane.is_terminal("r1")


# ---------------------------------------------------------------- clear() vs in-flight turn checks
class _SlowChecker:
    def __init__(self, p):
        self.p, self.go, self.done = p, threading.Event(), threading.Event()

    def __call__(self, audio, lang, said):
        self.go.wait(5)
        try:
            return self.p, 0.0
        finally:
            self.done.set()


def _run_check(buf, *, clear_first):
    checker = _SlowChecker(0.99)
    buf.turn_checker = checker
    buf.utt.extend(np.zeros(16000, "<f4").tobytes())
    buf.check_inflight = True
    buf._start_check(0.5, 1)
    if clear_first:
        buf.clear()
    checker.go.set()
    assert checker.done.wait(5)
    time.sleep(0.05)  # the check thread takes the lock right after the checker returns


def test_turn_check_arms_a_reply_without_clear(plugin):
    _, buf = _session(plugin, "s")
    _run_check(buf, clear_first=False)
    assert buf.pending_reply


def test_clear_discards_a_turn_check_started_before_it(plugin):
    _, buf = _session(plugin, "s")
    _run_check(buf, clear_first=True)
    assert not buf.pending_reply
    assert not buf.check_inflight


def test_clear_drops_the_speculative_reply(plugin):
    _, buf = _session(plugin, "s")
    dropped = []
    buf.owner_drop = dropped.append
    buf.spec_id = "spec-1"
    buf.pending_reply = True
    gen = buf.gen
    buf.clear()
    assert dropped == ["spec-1"] and buf.spec_id is None
    assert not buf.pending_reply and buf.gen == gen + 1


# ---------------------------------------------------------------- loopback auth
class _Server(http.server.ThreadingHTTPServer):
    key = "sk-test"
    seen: list


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _auth_ok(self):
        self.server.seen.append(self.headers.get("Authorization"))
        if self.headers.get("Authorization") != "Bearer " + self.server.key:
            self.send_response(401)
            self.end_headers()
            return False
        return True

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self._auth_ok():
            self._json({"data": [{"id": "sophon"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self._auth_ok():
            top = [{"token": "A", "logprob": 0.0}]
            self._json({"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]})


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.seen = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_loopback_turn_check_sends_the_api_key(server):
    srv, url = server
    checker = turn_check.LoopbackTurnChecker(base_url=url, api_key="sk-test")
    assert checker.wait_ready(timeout_s=5)
    p_finished, p_bc = checker(np.zeros(16000, np.float32), "en", "")
    assert 0.0 <= p_finished <= 1.0 and 0.0 <= p_bc <= 1.0
    assert srv.seen and all(h == "Bearer sk-test" for h in srv.seen)


def test_loopback_wait_ready_gives_up_on_a_refused_key(server):
    _, url = server
    checker = turn_check.LoopbackTurnChecker(base_url=url, api_key="wrong")
    t = time.monotonic()
    assert not checker.wait_ready(timeout_s=30)
    assert time.monotonic() - t < 5


def test_api_key_comes_from_the_serve_command_line(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setattr(sophon_env.sys, "argv", ["vllm", "serve", "m", "--api-key", "sk-cli"])
    assert sophon_env.api_key() == "sk-cli"
    assert turn_check.LoopbackTurnChecker(base_url="http://x").auth_headers() == {"Authorization": "Bearer sk-cli"}
    monkeypatch.setattr(sophon_env.sys, "argv", ["vllm", "serve", "m"])
    monkeypatch.setenv("VLLM_API_KEY", "sk-env")
    assert sophon_env.api_key() == "sk-env"
    monkeypatch.delenv("VLLM_API_KEY")
    assert turn_check.LoopbackTurnChecker(base_url="http://x").auth_headers() == {}


def test_realtime_warmup_connects_with_the_api_key(monkeypatch):
    import websockets

    calls = {}

    def connect(uri, *, max_size=None, additional_headers=None):
        calls.update(uri=uri, headers=additional_headers)
        return "conn"

    monkeypatch.setattr(websockets, "connect", connect)
    assert turn_check._ws_connect("ws://127.0.0.1:1/v1/realtime", {"Authorization": "Bearer k"}) == "conn"
    assert calls["headers"] == {"Authorization": "Bearer k"}
