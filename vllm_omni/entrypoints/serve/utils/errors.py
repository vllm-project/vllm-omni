# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Server-level error helpers for vLLM-Omni entrypoints.

Use this module for FastAPI/app exception handling and engine-failure response
construction that is not owned by a single OpenAI endpoint family."""

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.launcher import terminate_if_errored

# vLLM < 0.28 keeps create_error_response under serve.utils; 0.28+ moved it
# under serve.exception_handling and re-exports it from the package root.
try:
    from vllm.entrypoints.serve import create_error_response
except ImportError:
    from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

logger = init_logger(__name__)


def _get_request_id_from_request(req: Request) -> str | None:
    return req.state.request_metadata.request_id if hasattr(req.state, "request_metadata") else None


def _build_engine_error_payload(
    exc: EngineDeadError | EngineGenerateError,
    *,
    request_id: str | None,
) -> tuple[dict[str, Any], int]:
    err = create_error_response(exc)
    payload = err.model_dump()
    error_body = payload.get("error", {})

    error_body["request_id"] = request_id
    error_body["error_stage_id"] = getattr(exc, "error_stage_id", None)

    return payload, err.error.code


def _create_engine_error_json_response(
    req: Request,
    exc: EngineDeadError | EngineGenerateError,
) -> JSONResponse:
    request_id = _get_request_id_from_request(req)
    error_stage_id = getattr(exc, "error_stage_id", None)
    engine = req.app.state.engine_client

    if isinstance(exc, EngineDeadError):
        # Log Omni-specific diagnostic information for dead engines.
        orchestrator_alive = engine.engine.is_alive() if hasattr(engine, "engine") else "N/A"
        logger.error(
            "EngineDeadError: orchestrator_alive=%s, errored=%s, request_id=%s, error_stage_id=%s",
            orchestrator_alive,
            engine.errored,
            request_id,
            error_stage_id,
        )

    terminate_if_errored(
        server=req.app.state.server,
        engine=engine,
    )

    payload, status_code = _build_engine_error_payload(exc, request_id=request_id)
    return JSONResponse(content=payload, status_code=status_code)


def _register_omni_exception_handlers(app) -> None:
    """Override upstream vLLM exception handlers with Omni-aware versions.

    The upstream ``engine_error_handler`` is designed for ``AsyncLLM`` (single
    EngineCore process).  Omni uses a multi-stage orchestrator with different
    health semantics, so we register our own handlers that:

    - Log multi-stage diagnostic info (orchestrator liveness, per-stage health)
      when an ``EngineDeadError`` is caught.
    - Call ``terminate_if_errored``
    - Return an OpenAI-compatible error JSON response.
    """

    async def omni_engine_error_handler(
        req: Request,
        exc: EngineDeadError | EngineGenerateError,
    ):
        request_id = _get_request_id_from_request(req)

        if req.app.state.args.log_error_stack:
            logger.exception("Engine Exception caught. Request id: %s", request_id)

        return _create_engine_error_json_response(req, exc)

    app.exception_handler(EngineGenerateError)(omni_engine_error_handler)
    app.exception_handler(EngineDeadError)(omni_engine_error_handler)
