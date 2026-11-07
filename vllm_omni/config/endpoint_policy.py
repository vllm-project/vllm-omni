# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Endpoint restriction policy for omni pipelines."""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.routing import Route, WebSocketRoute
from vllm.entrypoints.serve.exception_handling.error_response import create_error_response

from vllm_omni.entrypoints.serve.utils.routes import remove_route_from_app


class RouteTarget(NamedTuple):
    """A server path & supported methods."""

    path: str
    methods: frozenset[str]
    # Indicates an endpoint should be rejected if we are only serving part of the pipeline
    requires_full_pipeline: bool = True


class OmniServingCapability(Enum):
    """Server routes that endpoint policies keep or shut down."""

    HEALTH = RouteTarget("/health", frozenset({"GET"}), requires_full_pipeline=False)
    METRICS = RouteTarget("/metrics", frozenset({"GET"}), requires_full_pipeline=False)
    COMPLETIONS = RouteTarget("/v1/completions", frozenset({"POST"}))
    CHAT_COMPLETIONS = RouteTarget("/v1/chat/completions", frozenset({"POST"}))
    CHAT_COMPLETIONS_BATCH = RouteTarget("/v1/chat/completions/batch", frozenset({"POST"}))
    RESPONSES = RouteTarget("/v1/responses", frozenset({"POST"}))
    MESSAGES = RouteTarget("/v1/messages", frozenset({"POST"}))
    INVOCATIONS = RouteTarget("/invocations", frozenset({"POST"}))
    GENERATE_TOKENS = RouteTarget("/inference/v1/generate", frozenset({"POST"}))
    IMAGE_EDITS = RouteTarget("/v1/images/edits", frozenset({"POST"}))
    RUN = RouteTarget("/v1/run", frozenset({"POST"}), requires_full_pipeline=False)

    @property
    def path(self) -> str:
        return self.value.path

    @property
    def methods(self) -> frozenset[str]:
        return self.value.methods

    @property
    def requires_full_pipeline(self) -> bool:
        return self.value.requires_full_pipeline


@dataclass(frozen=True)
class EndpointRestriction:
    capability: OmniServingCapability
    reason: str


def build_rejection_handler(reason: str):
    """Build a rejection handler for a given endpoint for the provided reason."""

    async def rejection_handler(_raw_request: Request):
        error = create_error_response(message=reason)
        return JSONResponse(
            content=error.model_dump(),
            status_code=error.error.code,
        )

    return rejection_handler


def shutdown_full_pipeline_routes(app: FastAPI, reason: str) -> None:
    """Reject HTTP routes that need the full pipeline and remove WebSocket routes.

    This is useful for putting an API server in front of a single stage in a multistage
    pipeline.
    """
    keep_paths = {capability.path for capability in OmniServingCapability if not capability.requires_full_pipeline}
    rejection_handler = build_rejection_handler(reason)
    routes = []
    for route in app.router.routes:
        if isinstance(route, Route) and route.path not in keep_paths:
            routes.append(APIRoute(route.path, rejection_handler, methods=route.methods))
        elif not (isinstance(route, WebSocketRoute) and route.path not in keep_paths):
            routes.append(route)
    app.router.routes = routes


def shutdown_unsupported_routes(
    app: FastAPI,
    endpoint_restrictions: tuple[EndpointRestriction, ...],
) -> None:
    """Given an initialized FastAPI server instance and a set of model specific endpoint
    restrictions, remove the restricted routes and patch a handler that returns 400.
    """
    for end_restrict in endpoint_restrictions:
        capability = end_restrict.capability
        # Remove the route from the app
        remove_route_from_app(app, capability.path, capability.methods)

        # Patch the bad request error with the model specific
        # reason for shutting down this endpoint
        rejection_handler = build_rejection_handler(end_restrict.reason)

        app.add_api_route(
            capability.path,
            rejection_handler,
            methods=list(capability.methods),
        )
