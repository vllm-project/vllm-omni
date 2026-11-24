"""Dropped render connections may retry only the unchanged render request."""

from types import SimpleNamespace

import pytest
import requests

from vllm_omni_cosmos3_client.rendering.transport import RenderSession


def test_render_retry_reuses_identical_payload_and_clears_dead_session(monkeypatch):
    calls = []
    closed = []
    replies = iter([requests.ConnectionError("dropped"), requests.Timeout("timeout"), SimpleNamespace(status_code=200)])

    class Session:
        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs["data"]))
            value = next(replies)
            if isinstance(value, Exception):
                raise value
            return value

        def close(self):
            closed.append(True)

    monkeypatch.setenv("COSMOS3_RENDER_RETRY_INTERVAL", "0")
    monkeypatch.delenv("EVAL_OUT", raising=False)
    transport = RenderSession(session_factory=Session, sleep=lambda _: None)
    response = transport.post("http://renderer/render", data=b"same-scene-and-sequence")
    assert response.status_code == 200
    assert calls == [("POST", "http://renderer/render", b"same-scene-and-sequence")] * 3
    assert len(closed) == 2


def test_reconnect_deadline_is_explicit(monkeypatch):
    class Session:
        def request(self, *args, **kwargs):
            raise requests.ConnectionError("down")

        def close(self):
            pass

    clock = iter([0, 0, 2])
    monkeypatch.setenv("COSMOS3_RENDER_RECONNECT_TIMEOUT", "1")
    monkeypatch.setenv("COSMOS3_RENDER_RETRY_INTERVAL", "0")
    transport = RenderSession(session_factory=Session, clock=lambda: next(clock), sleep=lambda _: None)
    with pytest.raises(TimeoutError, match="unavailable"):
        transport.get("http://renderer/health")
