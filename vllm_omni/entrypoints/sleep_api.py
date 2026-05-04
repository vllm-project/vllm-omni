# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from argparse import Namespace
from typing import Any

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.routing import Route
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

SLEEP_ROUTE_PATHS = {"/sleep", "/wake_up", "/is_sleeping"}
sleep_router = APIRouter()


class SleepRequest(BaseModel):
    stage_ids: list[int] | None = None
    level: int | None = None
    mode: str | None = None


class WakeUpRequest(BaseModel):
    stage_ids: list[int] | None = None
    tags: list[str] | None = None


def _engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _serialize_acks(acks: Any) -> list[Any]:
    if acks is None:
        return []
    if not isinstance(acks, list):
        acks = [acks]
    return [dataclasses.asdict(ack) if dataclasses.is_dataclass(ack) else ack for ack in acks]


def _remove_route_from_app(app, path: str, methods: set[str] | None = None) -> None:
    routes_to_remove = []
    for route in app.routes:
        if isinstance(route, Route) and route.path == path:
            if methods is None or (hasattr(route, "methods") and route.methods & methods):
                routes_to_remove.append(route)

    for route in routes_to_remove:
        app.routes.remove(route)


@sleep_router.post("/sleep")
async def sleep(raw_request: Request, request: SleepRequest | None = Body(default=None)):
    level = int(raw_request.query_params.get("level", request.level if request and request.level is not None else "1"))
    mode = raw_request.query_params.get("mode", request.mode if request and request.mode is not None else "abort")

    kwargs: dict[str, Any] = {"level": level, "mode": mode}
    if request and request.stage_ids is not None:
        kwargs["stage_ids"] = request.stage_ids

    acks = await _engine_client(raw_request).sleep(**kwargs)
    return JSONResponse(content={"status": "SUCCESS", "acks": _serialize_acks(acks)})


@sleep_router.post("/wake_up")
async def wake_up(raw_request: Request, request: WakeUpRequest | None = Body(default=None)):
    tags = raw_request.query_params.getlist("tags") or (request.tags if request else None)
    if tags == []:
        tags = None
    logger.info("wake up the engine with tags: %s", tags)

    kwargs: dict[str, Any] = {"tags": tags}
    if request and request.stage_ids is not None:
        kwargs["stage_ids"] = request.stage_ids

    acks = await _engine_client(raw_request).wake_up(**kwargs)
    return JSONResponse(content={"status": "SUCCESS", "acks": _serialize_acks(acks)})


@sleep_router.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    is_sleeping = await _engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": is_sleeping})


def include_sleep_router_if_enabled(app, args: Namespace) -> None:
    if not getattr(args, "enable_sleep_mode", False):
        return

    for path in SLEEP_ROUTE_PATHS:
        _remove_route_from_app(app, path)
    app.include_router(sleep_router)
    logger.info("Sleep/wake HTTP endpoints enabled (/sleep, /wake_up, /is_sleeping)")
