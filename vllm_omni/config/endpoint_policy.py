# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Endpoint restriction policy for omni pipelines."""

from collections.abc import Set
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.routing import Route
from vllm.entrypoints.serve.exception_handling.error_response import create_error_response


class RouteTarget(NamedTuple):
    """A server path & supported methods."""

    path: str
    methods: frozenset[str]


class OmniServingCapability(Enum):
    """Serving capabilities that pipelines can shut down."""

    COMPLETIONS = RouteTarget("/v1/completions", frozenset({"POST"}))

    @property
    def path(self) -> str:
        return self.value.path

    @property
    def methods(self) -> frozenset[str]:
        return self.value.methods


@dataclass(frozen=True)
class EndpointRestriction:
    capability: OmniServingCapability
    reason: str


def build_rejection_handler(reason: str):
    """Build a rejection handler for a given endpoint for the provided reason."""

    async def rejection_handler(raw_request: Request):
        error = create_error_response(message=reason)
        return JSONResponse(
            content=error.model_dump(),
            status_code=error.error.code,
        )

    return rejection_handler


def remove_route_from_app(
    app: FastAPI,
    path: str,
    methods: Set[str],
) -> None:
    """Remove routes matching a path and one of the given HTTP methods."""
    routes_to_remove = [
        route
        for route in app.routes
        if isinstance(route, Route) and route.path == path and route.methods is not None and route.methods & methods
    ]
    for route in routes_to_remove:
        app.routes.remove(route)


def shutdown_unsupported_routes(
    app: FastAPI,
    endpoint_restrictions: tuple[EndpointRestriction, ...],
):
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
