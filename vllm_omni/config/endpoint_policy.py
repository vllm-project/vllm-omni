# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Endpoint restriction policy for omni pipelines."""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.serve.utils.error_response import create_error_response


class RouteTarget(NamedTuple):
    """A server path & supported methods."""

    path: str
    methods: frozenset[str]


class OmniServingCapability(Enum):
    """Serving capabilities that pipelines can shut down."""

    CHAT_COMPLETIONS_BATCH = RouteTarget("/v1/chat/completions/batch", frozenset({"POST"}))
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


# Routes that are not supported for any model, but are supported in vLLM.
# This is only temporary to avoid 500s for batched chat completions.
UNSUPPORTED_ROUTES: tuple[EndpointRestriction, ...] = (
    EndpointRestriction(
        OmniServingCapability.CHAT_COMPLETIONS_BATCH,
        "Batched chat completions are not yet supported by vLLM Omni.",
    ),
)


def build_rejection_handler(reason: str):
    """Build a rejection handler for a given endpoint for the provided reason."""

    async def rejection_handler(raw_request: Request):
        error = create_error_response(message=reason)
        return JSONResponse(
            content=error.model_dump(),
            status_code=error.error.code,
        )

    return rejection_handler


def shutdown_unsupported_routes(
    app: FastAPI,
    endpoint_restrictions: tuple[EndpointRestriction, ...],
):
    """Given an initialized FastAPI server instance and a set of model specific endpoint
    restrictions, remove the restricted routes and patch a handler that returns 400.
    """
    from vllm_omni.entrypoints.openai.api_server import _remove_route_from_app

    # Generally these should not overlap since there is no point. If they do,
    # we use the reason message in UNSUPPORTED_ROUTES, for consistent error messages.
    restricted_endpoints = (*endpoint_restrictions, *UNSUPPORTED_ROUTES)

    for end_restrict in restricted_endpoints:
        capability = end_restrict.capability
        # Remove the route from the app
        _remove_route_from_app(app, capability.path, capability.methods)

        # Patch the bad request error with the model specific
        # reason for shutting down this endpoint
        rejection_handler = build_rejection_handler(end_restrict.reason)

        app.add_api_route(
            capability.path,
            rejection_handler,
            methods=list(capability.methods),
        )
