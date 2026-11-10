# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Same-origin proxy route feeding the ``model`` dropdown.

The browser cannot fetch ``{url}/models`` itself: for remote deployments a
browser-side ``localhost`` resolves to the user's own machine, not to the host
running ComfyUI.
"""

import ipaddress
import socket
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

from .utils.logger import get_logger

logger = get_logger(__name__)

MODELS_ROUTE = "/vllm_omni/models"

# The dropdown is a convenience, not a request path: keep the proxy snappy so
# a dead server degrades to a free-text field instead of freezing the node.
_TIMEOUT = aiohttp.ClientTimeout(total=5.0)


def _is_private_ip(host: str) -> bool:
    """Return True when ``host`` resolves to a private/loopback/link-local address."""
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                return True
    except socket.gaierror:
        return True  # Treat unresolvable as unsafe
    return False


async def _validate_url(base_url: str) -> None:
    """Raise ValueError when ``base_url`` targets a non-public host (SSRF guard)."""
    parsed = urlparse(base_url)
    if not parsed.hostname:
        raise ValueError("missing host")
    # Block IP-literal localhost variants like http://[::1]/ or http://0.0.0.0/
    if parsed.hostname in {"localhost", "0.0.0.0"} or _is_private_ip(parsed.hostname):
        raise ValueError("URL must target a public host")


async def fetch_served_models(base_url: str) -> list[str]:
    """GET ``{base_url}/models`` and return the served model ids."""
    url = base_url.rstrip("/") + "/models"
    # codeql[py/server-side-request-forgery]
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session, session.get(url) as response:
        if not response.ok:
            raise RuntimeError(f"GET {url} -> {response.status}: {await response.text()}")
        data = await response.json(content_type=None)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise RuntimeError(f"malformed model list payload: {data}")
    return [str(model["id"]) for model in data["data"]]


async def models_proxy_response(base_url: str) -> tuple[int, dict]:
    """Validate ``base_url`` and return the ``(status, payload)`` for the proxy route."""
    base_url = base_url.strip()
    if not base_url.startswith(("http://", "https://")):
        return 400, {"error": "an http(s) 'url' query parameter is required"}
    try:
        await _validate_url(base_url)
    except ValueError:
        return 400, {"error": "URL must target a public host"}
    try:
        return 200, {"models": await fetch_served_models(base_url)}
    except Exception as e:  # any upstream failure degrades the dropdown to free text
        logger.warning("model list proxy failed for %s: %s", base_url, e)
        return 502, {"error": "upstream request failed"}


def register_routes(prompt_server) -> None:
    """Register the model-list proxy route on the ComfyUI prompt server."""

    @prompt_server.routes.get(MODELS_ROUTE)
    async def vllm_omni_models(request: web.Request) -> web.Response:
        status, payload = await models_proxy_response(request.query.get("url", ""))
        return web.json_response(payload, status=status)
