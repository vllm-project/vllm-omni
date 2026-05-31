from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


@dataclass
class BackendState:
    name: str
    base_url: str
    inflight: int = 0


class BaselineDispatcher:
    def __init__(self, backend_urls: list[str], request_timeout_s: float) -> None:
        if not backend_urls:
            raise ValueError("At least one --backend-url is required")
        self.backends = [
            BackendState(name=f"backend-{idx}", base_url=url.rstrip("/"))
            for idx, url in enumerate(backend_urls)
        ]
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
        self._rr_index = 0
        self._request_timeout_s = request_timeout_s

    async def startup(self) -> None:
        timeout = self._request_timeout_s if self._request_timeout_s > 0 else None
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _choose_backend(self) -> int:
        async with self._lock:
            num_backends = len(self.backends)
            start = self._rr_index
            candidate_indices = list(range(num_backends))
            candidate_indices.sort(
                key=lambda idx: (
                    self.backends[idx].inflight,
                    (idx - start) % num_backends,
                ))
            chosen = candidate_indices[0]
            self.backends[chosen].inflight += 1
            self._rr_index = (chosen + 1) % num_backends
            return chosen

    async def _finish_backend(self, backend_index: int) -> None:
        async with self._lock:
            self.backends[backend_index].inflight = max(
                self.backends[backend_index].inflight - 1, 0)

    async def dispatch_json(self, path: str, body: dict, incoming_headers: dict[str, str]) -> Response:
        backend_index = await self._choose_backend()
        backend = self.backends[backend_index]
        headers = {
            key: value for key, value in incoming_headers.items()
            if key.lower() not in {"host", "content-length"}
        }
        assert self._client is not None
        try:
            response = await self._client.post(f"{backend.base_url}{path}", json=body, headers=headers)
        finally:
            await self._finish_backend(backend_index)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    async def proxy_get(self, path: str) -> Response:
        backend = self.backends[0]
        assert self._client is not None
        response = await self._client.get(f"{backend.base_url}{path}")
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    async def health(self) -> JSONResponse:
        assert self._client is not None
        statuses: list[dict] = []
        overall_healthy = True
        for backend in self.backends:
            try:
                response = await self._client.get(f"{backend.base_url}/health")
                healthy = response.status_code == 200
                detail = response.text
            except Exception as exc:
                healthy = False
                detail = str(exc)
            overall_healthy = overall_healthy and healthy
            statuses.append({
                "backend": backend.name,
                "url": backend.base_url,
                "healthy": healthy,
                "detail": detail,
                "inflight": backend.inflight,
            })
        return JSONResponse(
            status_code=200 if overall_healthy else 503,
            content={"status": "healthy" if overall_healthy else "degraded", "backends": statuses},
        )

    @staticmethod
    def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
        blocked = {"content-length", "transfer-encoding", "connection", "content-encoding"}
        return {key: value for key, value in headers.items() if key.lower() not in blocked}


def build_app(dispatcher: BaselineDispatcher) -> FastAPI:
    app = FastAPI(title="baseline dispatcher")

    @app.on_event("startup")
    async def _startup() -> None:
        await dispatcher.startup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await dispatcher.shutdown()

    @app.get("/health")
    async def health() -> JSONResponse:
        return await dispatcher.health()

    @app.get("/v1/models")
    async def models() -> Response:
        return await dispatcher.proxy_get("/v1/models")

    @app.post("/v1/images/generations")
    async def image_generations(request: Request) -> Response:
        body = await request.json()
        return await dispatcher.dispatch_json("/v1/images/generations", body, dict(request.headers))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round-robin baseline dispatcher for diffusion servers")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--backend-url",
        dest="backend_urls",
        action="append",
        required=True,
        help="Backend base URL. Repeat for each backend, e.g. http://127.0.0.1:18091",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=0.0,
        help="Dispatcher-to-backend request timeout in seconds. Set to 0 to disable timeout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dispatcher = BaselineDispatcher(args.backend_urls, args.request_timeout_s)
    uvicorn.run(build_app(dispatcher), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
