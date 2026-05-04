# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from starlette.routing import Route
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

SLEEP_ROUTE_PATHS = {"/sleep", "/wake_up", "/is_sleeping"}
sleep_router = APIRouter()


def _engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _remove_route_from_app(app, path: str, methods: set[str] | None = None) -> None:
    routes_to_remove = []
    for route in app.routes:
        if isinstance(route, Route) and route.path == path:
            if methods is None or (hasattr(route, "methods") and route.methods & methods):
                routes_to_remove.append(route)

    for route in routes_to_remove:
        app.routes.remove(route)


@sleep_router.post("/sleep")
async def sleep(raw_request: Request):
    level = int(raw_request.query_params.get("level", "1"))
    mode = raw_request.query_params.get("mode", "abort")
    await _engine_client(raw_request).sleep(level=level, mode=mode)
    return Response(status_code=200)


@sleep_router.post("/wake_up")
async def wake_up(raw_request: Request):
    tags = raw_request.query_params.getlist("tags")
    if tags == []:
        tags = None
    logger.info("wake up the engine with tags: %s", tags)
    await _engine_client(raw_request).wake_up(tags=tags)
    return Response(status_code=200)


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
