# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight transfer API router for vLLM-Omni.

Exposes HTTP endpoints for dynamic weight updates during RL training.
Mirrors the upstream vLLM RLHF API structure but works with AsyncOmni's
multi-stage orchestrator.
"""

import json
from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _get_engine_client(request: Request):
    """Extract the AsyncOmni engine client from app state."""
    return request.app.state.engine_client


@router.post("/init_weight_transfer_engine")
async def init_weight_transfer_engine(request: Request):
    """Initialize the weight transfer engine across all stages.

    Request body:
        {
            "init_info": {
                "backend": "nccl"|"ipc"|"sparse_nccl"|"sharded_rdt"
            }
        }
    """
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Invalid JSON format",
        ) from e

    init_info = body.get("init_info")
    if init_info is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Missing 'init_info' in request body",
        )

    try:
        engine = _get_engine_client(request)
        await engine.init_weight_transfer_engine(init_info)
        return JSONResponse(content={"message": "Weight transfer initialized"})
    except Exception as err:
        logger.exception("Failed to initialize weight transfer engine")
        return JSONResponse(
            content={"error": f"Failed to initialize: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.post("/start_weight_update")
async def start_weight_update(request: Request):
    """Start a new weight update session across all stages."""
    try:
        engine = _get_engine_client(request)
        await engine.start_weight_update()
        return JSONResponse(content={"message": "Weight update started"})
    except Exception as err:
        logger.exception("Failed to start weight update")
        return JSONResponse(
            content={"error": f"Failed to start weight update: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.post("/update_weights")
async def update_weights(request: Request):
    """Send weight update chunk to all stages.

    Request body:
        {
            "update_info": {
                "names": ["layer.0.weight", ...],
                "tensors": [tensor_data, ...]
            }
        }
    """
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Invalid JSON format",
        ) from e

    update_info = body.get("update_info")
    if update_info is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Missing 'update_info' in request body",
        )

    try:
        engine = _get_engine_client(request)
        await engine.update_weights(update_info)
        return JSONResponse(content={"message": "Weights updated"})
    except Exception as err:
        logger.exception("Failed to update weights")
        return JSONResponse(
            content={"error": f"Failed to update weights: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.post("/finish_weight_update")
async def finish_weight_update(request: Request):
    """Finish the current weight update session across all stages."""
    try:
        engine = _get_engine_client(request)
        await engine.finish_weight_update()
        return JSONResponse(content={"message": "Weight update finished"})
    except Exception as err:
        logger.exception("Failed to finish weight update")
        return JSONResponse(
            content={"error": f"Failed to finish weight update: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
