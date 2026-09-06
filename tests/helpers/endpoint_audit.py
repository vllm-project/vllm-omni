# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Utilities for auditing every HTTP endpoint exposed by a live server."""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import quote

import requests

_HTTP_METHODS = frozenset({"delete", "get", "patch", "post", "put"})
_PATH_PARAMETER = re.compile(r"{([^}]+)}")

# These routes change process or persistent server state. They are discovered and
# reported, but are not safe for an automated smoke pass against a shared server.
DEFAULT_SKIPPED_PATHS: Mapping[str, str] = {
    "/load_lora_adapter": "loads persistent server state",
    "/unload_lora_adapter": "unloads persistent server state",
    "/start_profile": "starts the server profiler",
    "/stop_profile": "stops the server profiler",
    "/sleep": "puts the engine to sleep",
    "/wake_up": "changes engine sleep state",
    "/v1/omni/sleep": "puts pipeline stages to sleep",
    "/v1/omni/wakeup": "changes pipeline stage sleep state",
}


class _HttpSession(Protocol):
    @property
    def headers(self) -> MutableMapping[str, str]: ...

    def get(self, url: str, **kwargs: Any) -> Any: ...

    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True, order=True)
class HttpEndpoint:
    """An HTTP operation from the server's OpenAPI document."""

    method: str
    path: str


@dataclass(frozen=True)
class EndpointProbeResult:
    """Status captured for one endpoint and its immediate health check."""

    endpoint: HttpEndpoint
    request_path: str
    status_code: int | None
    health_status_code: int | None
    request_error: str | None = None
    health_error: str | None = None

    @property
    def server_healthy(self) -> bool:
        return self.health_error is None and self.health_status_code == 200


@dataclass(frozen=True)
class SkippedEndpoint:
    endpoint: HttpEndpoint
    reason: str


@dataclass(frozen=True)
class EndpointAuditReport:
    """Results from a complete endpoint pass."""

    results: tuple[EndpointProbeResult, ...]
    skipped: tuple[SkippedEndpoint, ...]

    @property
    def status_codes(self) -> dict[tuple[str, str], int | None]:
        return {(result.endpoint.method, result.endpoint.path): result.status_code for result in self.results}


class EndpointHealthError(AssertionError):
    """Raised when the server fails the health check following a probe."""

    def __init__(self, result: EndpointProbeResult):
        self.result = result
        endpoint = result.endpoint
        health_detail = result.health_error or f"HTTP {result.health_status_code}"
        super().__init__(f"Server became unhealthy after {endpoint.method} {endpoint.path}: {health_detail}")


def discover_http_endpoints(openapi_document: Mapping[str, Any]) -> tuple[HttpEndpoint, ...]:
    """Return the HTTP operations advertised by an OpenAPI document."""

    paths = openapi_document.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("OpenAPI document does not contain a 'paths' object")

    endpoints = []
    for path, path_item in paths.items():
        if not isinstance(path, str) or not isinstance(path_item, Mapping):
            continue
        for method in path_item:
            normalized_method = str(method).lower()
            if normalized_method in _HTTP_METHODS:
                endpoints.append(HttpEndpoint(normalized_method.upper(), path))
    return tuple(sorted(endpoints, key=lambda endpoint: (endpoint.path, endpoint.method)))


def _resolve_schema(schema: Any, document: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(schema, Mapping):
        return {}
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema

    value: Any = document
    for component in ref[2:].split("/"):
        if not isinstance(value, Mapping):
            return {}
        value = value.get(component.replace("~1", "/").replace("~0", "~"))
    return value if isinstance(value, Mapping) else {}


def _schema_value(schema: Any, document: Mapping[str, Any], *, name: str = "") -> Any:
    schema = _resolve_schema(schema, document)
    for key in ("example", "default", "const"):
        if key in schema:
            return schema[key]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    for key in ("oneOf", "anyOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and variants:
            if key == "allOf":
                merged: dict[str, Any] = {}
                for variant in variants:
                    value = _schema_value(variant, document, name=name)
                    if isinstance(value, Mapping):
                        merged.update(value)
                return merged
            return _schema_value(variants[0], document, name=name)

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            return {}
        return {
            field: _schema_value(properties[field], document, name=field) for field in required if field in properties
        }
    if schema_type == "array":
        return [_schema_value(schema.get("items", {}), document, name=name)]
    if schema_type == "integer":
        return max(1, int(schema.get("minimum", 1)))
    if schema_type == "number":
        return max(1.0, float(schema.get("minimum", 1.0)))
    if schema_type == "boolean":
        return False
    if name == "model":
        return "endpoint-audit-model"
    return "endpoint-audit"


def _operation_for(document: Mapping[str, Any], endpoint: HttpEndpoint) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    path_item = document["paths"][endpoint.path]
    operation = path_item[endpoint.method.lower()]
    if not isinstance(path_item, Mapping) or not isinstance(operation, Mapping):
        raise ValueError(f"Invalid OpenAPI operation for {endpoint.method} {endpoint.path}")
    return path_item, operation


def _apply_model(value: Any, model: str) -> Any:
    if isinstance(value, Mapping):
        return {key: model if key == "model" else _apply_model(item, model) for key, item in value.items()}
    if isinstance(value, list):
        return [_apply_model(item, model) for item in value]
    return value


def _default_request_kwargs(
    document: Mapping[str, Any], endpoint: HttpEndpoint, model: str
) -> tuple[str, dict[str, Any]]:
    path_item, operation = _operation_for(document, endpoint)
    parameters = [
        *path_item.get("parameters", []),
        *operation.get("parameters", []),
    ]
    path_values: dict[str, str] = {}
    query: dict[str, Any] = {}
    for parameter in parameters:
        parameter = _resolve_schema(parameter, document)
        name = parameter.get("name")
        location = parameter.get("in")
        if not isinstance(name, str) or location not in {"path", "query"}:
            continue
        value = _schema_value(parameter.get("schema", {}), document, name=name)
        if name == "model":
            value = model
        if location == "path":
            path_values[name] = str(value)
        elif parameter.get("required"):
            query[name] = value

    request_path = _PATH_PARAMETER.sub(
        lambda match: quote(path_values.get(match.group(1), "endpoint-audit-missing"), safe=""),
        endpoint.path,
    )
    kwargs: dict[str, Any] = {}
    if query:
        kwargs["params"] = query

    request_body = operation.get("requestBody")
    if isinstance(request_body, Mapping):
        content = request_body.get("content", {})
        if isinstance(content, Mapping):
            if "application/json" in content:
                media = content["application/json"]
                schema = media.get("schema", {}) if isinstance(media, Mapping) else {}
                kwargs["json"] = _apply_model(_schema_value(schema, document), model)
            else:
                form_type = next(
                    (item for item in ("multipart/form-data", "application/x-www-form-urlencoded") if item in content),
                    None,
                )
                if form_type is not None:
                    media = content[form_type]
                    schema = media.get("schema", {}) if isinstance(media, Mapping) else {}
                    value = _apply_model(_schema_value(schema, document), model)
                    kwargs["data"] = value if isinstance(value, Mapping) else {}
    return request_path, kwargs


def _semantic_request_overrides(model: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Minimal requests that pass validation far enough to exercise handlers."""

    prompt = "Endpoint audit"
    return {
        ("POST", "/v1/completions"): {"json": {"model": model, "prompt": prompt, "max_tokens": 1}},
        ("POST", "/v1/chat/completions"): {
            "json": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "stream": False,
            }
        },
        ("POST", "/v1/embeddings"): {"json": {"model": model, "input": prompt}},
        ("POST", "/pooling"): {"json": {"model": model, "input": prompt}},
        ("POST", "/tokenize"): {"json": {"model": model, "prompt": prompt}},
        ("POST", "/detokenize"): {"json": {"model": model, "tokens": [1]}},
        ("POST", "/v1/responses"): {"json": {"model": model, "input": prompt, "max_output_tokens": 1}},
        ("POST", "/v1/audio/speech"): {
            "json": {"model": model, "input": prompt, "voice": "endpoint-audit", "response_format": "wav"}
        },
        ("POST", "/v1/audio/generate"): {"json": {"model": model, "input": prompt}},
        ("POST", "/v1/images/generations"): {"json": {"model": model, "prompt": prompt}},
        ("POST", "/v1/videos"): {"data": {"model": model, "prompt": prompt}},
    }


def _health_check(
    session: _HttpSession, base_url: str, health_path: str, timeout: float
) -> tuple[int | None, str | None]:
    try:
        response = session.get(
            f"{base_url.rstrip('/')}/{health_path.lstrip('/')}",
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"
    try:
        return response.status_code, None
    finally:
        response.close()


def audit_server_endpoints(
    base_url: str,
    *,
    model: str,
    api_key: str = "EMPTY",
    timeout: float = 120.0,
    health_path: str = "/health",
    skipped_paths: Mapping[str, str] = DEFAULT_SKIPPED_PATHS,
    request_overrides: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    session: _HttpSession | None = None,
) -> EndpointAuditReport:
    """Hit every advertised HTTP endpoint and verify health after each request.

    Status codes are intentionally recorded rather than asserted: endpoint support is
    model-specific, so 4xx responses can be the correct outcome. A failed post-request
    health check raises :class:`EndpointHealthError` immediately.
    """

    owned_session = session is None
    http_session = cast(_HttpSession, requests.Session()) if session is None else session
    http_session.headers.setdefault("Authorization", f"Bearer {api_key}")
    try:
        openapi_response = http_session.get(f"{base_url.rstrip('/')}/openapi.json", timeout=timeout)
        try:
            openapi_response.raise_for_status()
            document = openapi_response.json()
        finally:
            openapi_response.close()
        if not isinstance(document, Mapping):
            raise ValueError("OpenAPI endpoint returned a non-object document")

        initial_status, initial_error = _health_check(http_session, base_url, health_path, timeout)
        if initial_error is not None or initial_status != 200:
            initial_result = EndpointProbeResult(
                endpoint=HttpEndpoint("GET", health_path),
                request_path=health_path,
                status_code=None,
                health_status_code=initial_status,
                health_error=initial_error,
            )
            raise EndpointHealthError(initial_result)

        overrides = _semantic_request_overrides(model)
        if request_overrides:
            overrides.update(
                {(method.upper(), path): dict(value) for (method, path), value in request_overrides.items()}
            )

        results: list[EndpointProbeResult] = []
        skipped: list[SkippedEndpoint] = []
        for endpoint in discover_http_endpoints(document):
            if reason := skipped_paths.get(endpoint.path):
                skipped.append(SkippedEndpoint(endpoint, reason))
                continue

            request_path, kwargs = _default_request_kwargs(document, endpoint, model)
            kwargs.update(overrides.get((endpoint.method, endpoint.path), {}))
            request_options = {
                "timeout": timeout,
                "allow_redirects": False,
                "stream": True,
                **kwargs,
            }
            request_error = None
            status_code = None
            try:
                response = http_session.request(
                    endpoint.method,
                    f"{base_url.rstrip('/')}/{request_path.lstrip('/')}",
                    **request_options,
                )
                status_code = response.status_code
                response.close()
            except requests.RequestException as exc:
                request_error = f"{type(exc).__name__}: {exc}"

            health_status, health_error = _health_check(http_session, base_url, health_path, timeout)
            result = EndpointProbeResult(
                endpoint=endpoint,
                request_path=request_path,
                status_code=status_code,
                health_status_code=health_status,
                request_error=request_error,
                health_error=health_error,
            )
            results.append(result)
            if not result.server_healthy:
                raise EndpointHealthError(result)

        return EndpointAuditReport(tuple(results), tuple(skipped))
    finally:
        if owned_session:
            http_session.close()


__all__ = [
    "DEFAULT_SKIPPED_PATHS",
    "EndpointAuditReport",
    "EndpointHealthError",
    "EndpointProbeResult",
    "HttpEndpoint",
    "SkippedEndpoint",
    "audit_server_endpoints",
    "discover_http_endpoints",
]
