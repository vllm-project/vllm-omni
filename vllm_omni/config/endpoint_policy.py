# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Endpoint restriction policy for omni pipelines."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from vllm.entrypoints.serve.exception_handling.error_response import create_error_response

from vllm_omni.entrypoints.serve.utils.routes import remove_route_from_app


class RouteTarget(NamedTuple):
    """A server path & supported methods."""

    path: str
    methods: frozenset[str]


class OmniServingCapability(Enum):
    """Serving capabilities that pipelines can shut down."""

    COMPLETIONS = RouteTarget("/v1/completions", frozenset({"POST"}))
    IMAGE_EDITS = RouteTarget("/v1/images/edits", frozenset({"POST"}))
    TOKENIZE = RouteTarget("/tokenize", frozenset({"POST"}))
    DETOKENIZE = RouteTarget("/detokenize", frozenset({"POST"}))
    RESPONSES = RouteTarget("/v1/responses", frozenset({"POST"}))
    RESPONSES_RETRIEVE = RouteTarget("/v1/responses/{response_id}", frozenset({"GET"}))
    RESPONSES_CANCEL = RouteTarget("/v1/responses/{response_id}/cancel", frozenset({"POST"}))
    MESSAGES = RouteTarget("/v1/messages", frozenset({"POST"}))

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


# Upstream ``build_app`` mounts these routes in every serving mode, but a mode
# only initializes the handlers it can actually serve: pure diffusion and duplex
# wire no text-generation handler, and only the multi-stage state wires a
# tokenization handler. The upstream handlers read the missing ``app.state``
# attribute (or dereference a ``None`` one) and answer HTTP 500, so a deployment
# that never wires the handler reports the capability gap as a 400 instead.
UNWIRED_HANDLER_ROUTES: tuple[tuple[OmniServingCapability, str, str], ...] = (
    (
        OmniServingCapability.COMPLETIONS,
        "openai_serving_completion",
        "The Completions API is unavailable because this deployment does not initialize a completions handler.",
    ),
    (
        OmniServingCapability.RESPONSES,
        "openai_serving_responses",
        "The Responses API is unavailable because this deployment does not initialize a responses handler.",
    ),
    (
        OmniServingCapability.RESPONSES_RETRIEVE,
        "openai_serving_responses",
        "The Responses API is unavailable because this deployment does not initialize a responses handler.",
    ),
    (
        OmniServingCapability.RESPONSES_CANCEL,
        "openai_serving_responses",
        "The Responses API is unavailable because this deployment does not initialize a responses handler.",
    ),
    (
        OmniServingCapability.MESSAGES,
        "anthropic_serving_messages",
        "The Messages API is unavailable because this deployment does not initialize a messages handler.",
    ),
    (
        OmniServingCapability.TOKENIZE,
        "serving_tokenization",
        "Tokenization is unavailable because this deployment does not initialize a tokenization handler.",
    ),
    (
        OmniServingCapability.DETOKENIZE,
        "serving_tokenization",
        "Detokenization is unavailable because this deployment does not initialize a tokenization handler.",
    ),
)


def unwired_endpoint_restrictions(
    state: State,
    *,
    already_restricted: Iterable[OmniServingCapability] = (),
) -> tuple[EndpointRestriction, ...]:
    """Restrictions for routes whose handler this deployment never initializes.

    A route belongs here when its ``app.state`` handler is missing or ``None``.
    Capabilities the deployment already restricts are left out so the reason a
    pipeline declared for them is the one clients receive.
    """
    restricted = set(already_restricted)
    return tuple(
        EndpointRestriction(capability, reason)
        for capability, attribute, reason in UNWIRED_HANDLER_ROUTES
        if capability not in restricted and getattr(state, attribute, None) is None
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
