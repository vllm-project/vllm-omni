# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any

import pytest
import requests

from tests.helpers.endpoint_audit import (
    EndpointHealthError,
    HttpEndpoint,
    audit_server_endpoints,
    discover_http_endpoints,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Response:
    def __init__(self, status_code: int, body: Any = None):
        self.status_code = status_code
        self._body = body
        self.closed = False

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


class _Session:
    def __init__(
        self,
        openapi_document,
        *,
        unhealthy_after: tuple[str, str] | None = None,
        request_error_at: tuple[str, str] | None = None,
    ):
        self.headers: dict[str, str] = {}
        self.openapi_document = openapi_document
        self.unhealthy_after = unhealthy_after
        self.request_error_at = request_error_at
        self.last_endpoint: tuple[str, str] | None = None
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any):
        self.requests.append(("GET", url, kwargs))
        if url.endswith("/openapi.json"):
            return _Response(200, self.openapi_document)
        if url.endswith("/health"):
            unhealthy = self.unhealthy_after is not None and self.last_endpoint == self.unhealthy_after
            return _Response(503 if unhealthy else 200, {"status": "unhealthy" if unhealthy else "healthy"})
        raise AssertionError(f"Unexpected GET {url}")

    def request(self, method: str, url: str, **kwargs: Any):
        path = "/" + url.split("/", 3)[-1]
        self.last_endpoint = (method, path)
        self.requests.append((method, url, kwargs))
        if self.last_endpoint == self.request_error_at:
            raise requests.ConnectionError("connection dropped")
        return _Response(422)

    def close(self):
        pass


def _document():
    return {
        "openapi": "3.1.0",
        "paths": {
            "/health": {"get": {"responses": {"200": {}}}},
            "/v1/completions": {
                "post": {
                    "requestBody": {
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Completion"}}}
                    },
                    "responses": {"200": {}},
                }
            },
            "/v1/items/{item_id}": {
                "get": {
                    "parameters": [
                        {"name": "item_id", "in": "path", "required": True, "schema": {"type": "string"}},
                        {"name": "limit", "in": "query", "required": True, "schema": {"type": "integer"}},
                    ],
                    "responses": {"200": {}},
                }
            },
            "/v1/omni/sleep": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
                    "responses": {"200": {}},
                }
            },
        },
        "components": {
            "schemas": {
                "Completion": {
                    "type": "object",
                    "required": ["model", "prompt"],
                    "properties": {"model": {"type": "string"}, "prompt": {"type": "string"}},
                }
            }
        },
    }


def test_discover_http_endpoints_ignores_non_operations():
    document = _document()
    document["paths"]["/health"]["parameters"] = []

    assert discover_http_endpoints(document) == (
        HttpEndpoint("GET", "/health"),
        HttpEndpoint("POST", "/v1/completions"),
        HttpEndpoint("GET", "/v1/items/{item_id}"),
        HttpEndpoint("POST", "/v1/omni/sleep"),
    )


def test_audit_uses_semantic_payloads_checks_health_and_reports_skips():
    session = _Session(_document())

    report = audit_server_endpoints("http://server", model="demo-model", session=session)

    assert report.status_codes == {
        ("GET", "/health"): 422,
        ("POST", "/v1/completions"): 422,
        ("GET", "/v1/items/{item_id}"): 422,
    }
    assert [(item.endpoint.method, item.endpoint.path) for item in report.skipped] == [("POST", "/v1/omni/sleep")]

    completion_call = next(call for call in session.requests if call[1].endswith("/v1/completions"))
    assert completion_call[2]["json"] == {
        "model": "demo-model",
        "prompt": "Endpoint audit",
        "max_tokens": 1,
    }
    item_call = next(call for call in session.requests if "/v1/items/endpoint-audit" in call[1])
    assert item_call[2]["params"] == {"limit": 1}

    health_calls = [call for call in session.requests if call[1].endswith("/health")]
    assert len(health_calls) == 5  # Initial check, /health probe, then one check per probed endpoint.


def test_audit_stops_at_endpoint_that_makes_server_unhealthy():
    session = _Session(_document(), unhealthy_after=("POST", "/v1/completions"))

    with pytest.raises(EndpointHealthError, match=r"POST /v1/completions: HTTP 503") as exc_info:
        audit_server_endpoints("http://server", model="demo-model", session=session)

    assert exc_info.value.result.endpoint == HttpEndpoint("POST", "/v1/completions")
    assert exc_info.value.result.status_code == 422
    assert exc_info.value.result.health_status_code == 503


def test_request_overrides_replace_generated_request_fields():
    session = _Session(_document())

    audit_server_endpoints(
        "http://server",
        model="demo-model",
        session=session,
        request_overrides={
            ("post", "/v1/completions"): {
                "json": {"custom": True},
                "timeout": 3.0,
                "stream": False,
            }
        },
    )

    completion_call = next(call for call in session.requests if call[1].endswith("/v1/completions"))
    assert completion_call[2]["json"] == {"custom": True}
    assert completion_call[2]["timeout"] == 3.0
    assert completion_call[2]["stream"] is False


def test_request_errors_are_reported_when_server_remains_healthy():
    session = _Session(_document(), request_error_at=("POST", "/v1/completions"))

    report = audit_server_endpoints("http://server", model="demo-model", session=session)

    result = next(item for item in report.results if item.endpoint == HttpEndpoint("POST", "/v1/completions"))
    assert result.status_code is None
    assert result.request_error == "ConnectionError: connection dropped"
    assert result.server_healthy
