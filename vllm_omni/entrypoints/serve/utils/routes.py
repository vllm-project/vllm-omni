# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Route-table helpers for Omni server assembly.

Owns FastAPI app **and** router mutation used during server construction.
``endpoint_policy`` imports app-level removal from here; it does not own
route-table helpers itself.
"""

from collections.abc import Set

from fastapi import APIRouter, FastAPI
from starlette.routing import Route


def remove_route_from_app(
    app: FastAPI,
    path: str,
    methods: Set[str],
) -> None:
    """Remove app routes matching a path and one of the given HTTP methods."""
    routes_to_remove = [
        route
        for route in app.routes
        if isinstance(route, Route) and route.path == path and route.methods is not None and route.methods & methods
    ]
    for route in routes_to_remove:
        app.routes.remove(route)


def _remove_route_from_router(
    router: APIRouter,
    path: str,
    methods: set[str] | None = None,
) -> None:
    """Remove routes from an ``APIRouter`` by path and optionally by methods."""
    methods_set = {method.upper() for method in methods} if methods else None
    for route in list(router.routes):
        if getattr(route, "path", None) != path:
            continue
        if methods_set is not None:
            route_methods = {method.upper() for method in (getattr(route, "methods", None) or set())}
            if not (route_methods & methods_set):
                continue
        router.routes.remove(route)
