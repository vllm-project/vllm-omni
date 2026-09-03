# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OpenAI endpoint error helpers.

This module owns helpers that are specific to OpenAI-compatible response
shapes. Server-level exception handling belongs in
``vllm_omni.entrypoints.serve.utils.errors``."""

from http import HTTPStatus

from fastapi import Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.instrumentator.basic import base


class InvalidInputReferenceError(ValueError):
    def __init__(self, message: str = "Invalid input reference.") -> None:
        super().__init__(message)


def _error_response_to_json_response(
    err: ErrorResponse,
    *,
    status_code: HTTPStatus | int | None = None,
    default_status_code: HTTPStatus | int = HTTPStatus.BAD_REQUEST,
) -> JSONResponse:
    resolved_status = int(
        status_code
        if status_code is not None
        else (err.error.code if err.error and err.error.code is not None else default_status_code)
    )
    payload = err.model_dump()
    if err.error:
        payload["error"]["code"] = resolved_status
    return JSONResponse(content=payload, status_code=resolved_status)


# TODO(#5227, P1.1): Relocate to audio/speech helpers when speech routes move out of api_server.py.
def _create_speech_error_json_response(
    raw_request: Request,
    message: str,
    *,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> JSONResponse:
    err = base(raw_request).create_error_response(
        message=message,
        err_type=err_type,
        status_code=status_code,
    )
    return _error_response_to_json_response(err, status_code=status_code)
