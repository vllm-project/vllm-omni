# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded-retry contract for the MiniMax-H3 accuracy asset download (#10661).

The accuracy job fetches its conditioning inputs from cdn.hailuoai.com, whose
DNS resolution is intermittent from CI runners. The download retries transient
failures and 5xx; these cases pin that the retry stays bounded and that a 4xx
or a changed asset still fails immediately.
"""

from __future__ import annotations

import hashlib
import http.server
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests

from tests.e2e.accuracy.minimax_h3.asset_download import download_reference_asset

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PAYLOAD = b"mini-max-h3-official-asset"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
# Long enough that honoring it would dominate any plausible test timeout.
RETRY_AFTER_SECONDS = "60"


class _ScriptedHandler(http.server.BaseHTTPRequestHandler):
    """Serve a scripted status sequence and record every request."""

    statuses: list[tuple[int, dict[str, str]]] = []
    hits: list[float] = []

    def do_GET(self) -> None:  # noqa: N802 - http.server dispatch name
        index = len(type(self).hits)
        type(self).hits.append(time.monotonic())
        status, headers = type(self).statuses[min(index, len(type(self).statuses) - 1)]
        body = PAYLOAD if status == 200 else b""
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@contextmanager
def _scripted_server(statuses: Sequence[tuple[int, dict[str, str]]]) -> Iterator[tuple[str, list[float]]]:
    handler = type("Handler", (_ScriptedHandler,), {"statuses": list(statuses), "hits": []})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}/asset.bin", handler.hits
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _download(url: str, output_dir: Path, sha256: str = PAYLOAD_SHA256) -> Path:
    return download_reference_asset(output_dir, filename="asset.bin", url=url, sha256=sha256, label="test")


def test_429_with_retry_after_fails_fast(tmp_path: Path) -> None:
    with _scripted_server([(429, {"Retry-After": RETRY_AFTER_SECONDS})]) as (url, hits):
        started = time.monotonic()
        with pytest.raises(requests.exceptions.HTTPError):
            _download(url, tmp_path)
        elapsed = time.monotonic() - started
    assert len(hits) == 1, f"429 was sent {len(hits)} times"
    assert elapsed < 10, f"429 with Retry-After held the job for {elapsed:.1f}s"


def test_413_with_retry_after_fails_fast(tmp_path: Path) -> None:
    """urllib3 also retries 413 when Retry-After is honored; it must not here."""
    with _scripted_server([(413, {"Retry-After": RETRY_AFTER_SECONDS})]) as (url, hits):
        started = time.monotonic()
        with pytest.raises(requests.exceptions.HTTPError):
            _download(url, tmp_path)
        elapsed = time.monotonic() - started
    assert len(hits) == 1, f"413 was sent {len(hits)} times"
    assert elapsed < 10, f"413 with Retry-After held the job for {elapsed:.1f}s"


def test_404_is_not_retried(tmp_path: Path) -> None:
    with _scripted_server([(404, {})]) as (url, hits):
        started = time.monotonic()
        with pytest.raises(requests.exceptions.HTTPError):
            _download(url, tmp_path)
        elapsed = time.monotonic() - started
    assert len(hits) == 1
    assert elapsed < 10


def test_503_with_retry_after_retries_without_honoring_the_header(tmp_path: Path) -> None:
    """503 stays retryable, but the 60 s header must not be added per attempt.

    The policy allows four retries, and its backoff without the header sums to
    0 + 2 + 4 + 8 s. Honoring Retry-After would add 300 s.
    """
    with _scripted_server([(503, {"Retry-After": RETRY_AFTER_SECONDS})]) as (url, hits):
        started = time.monotonic()
        with pytest.raises(requests.exceptions.RetryError):
            _download(url, tmp_path)
        elapsed = time.monotonic() - started
    assert len(hits) == 5, f"503 was sent {len(hits)} times"
    assert elapsed < 30, f"503 retries with Retry-After took {elapsed:.1f}s"
    assert not (tmp_path / "asset.bin").exists()


def test_500_then_success_writes_the_pinned_asset(tmp_path: Path) -> None:
    with _scripted_server([(500, {}), (500, {}), (200, {})]) as (url, hits):
        path = _download(url, tmp_path)
    assert len(hits) == 3
    assert path.read_bytes() == PAYLOAD


def test_content_gate_still_rejects_a_changed_asset(tmp_path: Path) -> None:
    with _scripted_server([(200, {})]) as (url, hits):
        with pytest.raises(AssertionError, match="reference changed"):
            _download(url, tmp_path, sha256="0" * 64)
    assert len(hits) == 1
    assert not (tmp_path / "asset.bin").exists()
