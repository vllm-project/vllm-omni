#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

BASE_COST = 512 * 512 * 20


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float_dict(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in parsed.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def parse_batch_caps(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in parsed.items():
        try:
            cap = int(v)
        except Exception:
            continue
        out[str(k)] = max(1, cap)
    return out


@dataclass(frozen=True)
class RequestType:
    width: int
    height: int
    steps: int

    @property
    def key(self) -> str:
        return f"{self.width}x{self.height}_{self.steps}"

    @property
    def cost(self) -> float:
        return (self.width * self.height * self.steps) / BASE_COST


@dataclass
class ImageIngressPayload:
    model: str
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    seed: int | None
    n: int
    request_id: str


@dataclass
class ImageIngressResult:
    created: int
    images_b64: list[str]
    error: str | None = None


@dataclass
class _PendingRequest:
    payload: ImageIngressPayload
    req_type: RequestType
    enqueue_ts: float
    future: asyncio.Future[ImageIngressResult]


class _IngressBatchDrrAdapter:
    def __init__(
        self,
        *,
        execute_batch: Callable[[RequestType, list[ImageIngressPayload]], Awaitable[list[ImageIngressResult]]],
        batch_caps: dict[str, int],
    ) -> None:
        self._execute_batch = execute_batch
        self.batch_caps = {k: max(1, int(v)) for k, v in batch_caps.items()}
        self.max_wait_ms = max(0, _env_int("OMNI_INGRESS_DRR_MAX_WAIT_MS", 250))
        self.strict_batching = _env_bool("OMNI_INGRESS_DRR_STRICT_BATCHING", False)
        self.q_base = max(1, _env_int("OMNI_INGRESS_DRR_Q_BASE", 12))
        self.age_threshold_ms = max(1, _env_int("OMNI_INGRESS_DRR_AGE_THRESHOLD_MS", 400))
        self.age_bonus_factor = max(0.0, _env_float("OMNI_INGRESS_DRR_AGE_BONUS_FACTOR", 1.0))
        self.max_queues = max(0, _env_int("OMNI_INGRESS_DRR_MAX_QUEUES", 0))
        self.default_budget_mode = (
    os.environ.get(
        "OMNI_INGRESS_DRR_DEFAULT_BUDGET_MODE", "weight_inv_cost"
    )
    .strip()
    .lower()
)
        if self.default_budget_mode not in {"inv_cost", "weight_inv_cost"}:
            self.default_budget_mode = "weight_inv_cost"
        self.queue_budget_overrides = {
            str(k): max(1.0, float(v))
            for k, v in _parse_float_dict(os.environ.get("OMNI_INGRESS_DRR_QUEUE_BUDGET_OVERRIDES")).items()
        }
        self.request_weight_overrides = {
            str(k): max(0.0, float(v))
            for k, v in _parse_float_dict(os.environ.get("OMNI_INGRESS_DRR_REQUEST_WEIGHTS")).items()
        }

        self._queues: dict[str, deque[_PendingRequest]] = {}
        self._types: dict[str, RequestType] = {}
        self._deficits: dict[str, float] = {}
        self._order: list[str] = []
        self._rr_idx = 0
        self._arrival_counts: dict[str, int] = {}
        self._arrival_total = 0
        self._lock = asyncio.Lock()
        self._cv = asyncio.Condition(self._lock)
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._closed = False

    def _estimate_weight(self, key: str) -> float:
        override = self.request_weight_overrides.get(key)
        if override is not None and override > 0:
            return override
        if self._arrival_total <= 0:
            return 1.0
        count = self._arrival_counts.get(key, 0)
        if count <= 0:
            return 1.0 / max(1, len(self._types))
        return count / self._arrival_total

    def _ratio(self, key: str, req_type: RequestType) -> float:
        if self.default_budget_mode == "inv_cost":
            return 1.0 / max(req_type.cost, 1e-9)
        # weight_inv_cost: if explicit weights are not provided, fallback to observed arrival share.
        weight = self._estimate_weight(key)
        return weight / max(req_type.cost, 1e-9)

    def _auto_quantum(self, key: str) -> int:
        ratios: dict[str, float] = {}
        for k, t in self._types.items():
            ratio = self._ratio(k, t)
            if ratio > 0:
                ratios[k] = ratio
        if not ratios:
            req_type = self._types.get(key)
            if req_type is None:
                return 1
            return max(1, round(self.q_base / req_type.cost))
        min_ratio = min(ratios.values())
        if min_ratio <= 0:
            return 1
        cur = ratios.get(key, min_ratio)
        return max(1, round(cur / min_ratio))

    def _quantum(self, req_type: RequestType) -> int:
        key = req_type.key
        override = self.queue_budget_overrides.get(key)
        if override is not None:
            return max(1, round(override))
        return self._auto_quantum(key)

    async def start(self) -> None:
        async with self._lock:
            if self._running:
                return
            self._running = True
            self._closed = False
            self._loop_task = asyncio.create_task(self._run_loop(), name="image-ingress-batch-drr-loop")

    async def stop(self) -> None:
        async with self._lock:
            self._running = False
            self._closed = True
            self._cv.notify_all()
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None

    async def submit(self, payload: ImageIngressPayload) -> ImageIngressResult:
        req_type = RequestType(width=payload.width, height=payload.height, steps=payload.steps)
        key = req_type.key
        fut: asyncio.Future[ImageIngressResult] = asyncio.get_running_loop().create_future()
        async with self._lock:
            if self._closed:
                raise RuntimeError("Ingress adapter is closed")
            if key not in self._queues and self.max_queues > 0 and len(self._queues) >= self.max_queues:
                raise ValueError(
                    "Ingress queue limit reached; refusing to remap request type "
                    f"{key} (width={payload.width}, height={payload.height}, steps={payload.steps})"
                )
            if key not in self._queues:
                self._queues[key] = deque()
                self._types[key] = req_type
                self._deficits[key] = 0.0
                self._order.append(key)
                self._order.sort(key=lambda x: self._types[x].cost)
            self._queues[key].append(
                _PendingRequest(payload=payload, req_type=req_type, enqueue_ts=time.perf_counter(), future=fut)
            )
            self._arrival_counts[key] = self._arrival_counts.get(key, 0) + 1
            self._arrival_total += 1
            self._cv.notify()
        return await fut

    def _select_queue_locked(self) -> tuple[str | None, bool]:
        active = [k for k in self._order if self._queues.get(k)]
        if not active:
            return None, False
        for key in active:
            self._deficits[key] += self._quantum(self._types[key])
        n = len(active)
        for offset in range(n):
            key = active[(self._rr_idx + offset) % n]
            q = self._queues[key]
            req_type = self._types[key]
            oldest_wait_ms = (time.perf_counter() - q[0].enqueue_ts) * 1000.0
            bonus = self.age_bonus_factor * self._quantum(req_type) if oldest_wait_ms >= self.age_threshold_ms else 0.0
            if self._deficits[key] + bonus >= req_type.cost:
                self._rr_idx = (self._rr_idx + offset + 1) % max(1, n)
                return key, False
        oldest_key = min(active, key=lambda k: self._queues[k][0].enqueue_ts)
        return oldest_key, True

    async def _run_loop(self) -> None:
        while True:
            async with self._lock:
                while self._running and not any(self._queues.get(k) for k in self._order):
                    await self._cv.wait()
                if not self._running:
                    break
                key, force = self._select_queue_locked()
                if key is None:
                    await self._cv.wait()
                    continue
                queue = self._queues[key]
                req_type = self._types[key]
                cap = self.batch_caps.get(key, 1)
                oldest_wait_ms = (time.perf_counter() - queue[0].enqueue_ts) * 1000.0
                if len(queue) < cap:
                    if self.strict_batching:
                        await self._cv.wait()
                        continue
                    if oldest_wait_ms < self.max_wait_ms:
                        wait_s = max((self.max_wait_ms - oldest_wait_ms) / 1000.0, 0.001)
                        try:
                            await asyncio.wait_for(self._cv.wait(), timeout=wait_s)
                        except asyncio.TimeoutError:
                            pass
                        continue
                batch_size = min(cap, len(queue))
                batch = [queue.popleft() for _ in range(batch_size)]
                if not force:
                    self._deficits[key] = max(0.0, self._deficits[key] - req_type.cost * batch_size)
            await self._dispatch_batch(req_type, batch)

    async def _dispatch_batch(self, req_type: RequestType, batch: list[_PendingRequest]) -> None:
        if not batch:
            return
        payloads = [x.payload for x in batch]
        try:
            outputs = await self._execute_batch(req_type, payloads)
            if len(outputs) != len(batch):
                raise RuntimeError(f"batch result size mismatch: expected {len(batch)}, got {len(outputs)}")
            for node, output in zip(batch, outputs, strict=True):
                if not node.future.done():
                    node.future.set_result(output)
        except Exception as exc:
            for node in batch:
                if not node.future.done():
                    node.future.set_result(
                        ImageIngressResult(created=int(time.time()), images_b64=[], error=f"dispatch failed: {exc!r}")
                    )


class ImageIngressDispatcherService:
    """Single-file server ingress scheduler for /v1/images/generations."""

    def __init__(
        self,
        *,
        batch_execute_fn: Callable[[RequestType, list[ImageIngressPayload]], Awaitable[list[ImageIngressResult]]],
        batch_caps: dict[str, int],
    ) -> None:
        self._batch_execute_fn = batch_execute_fn
        self._adapter = _IngressBatchDrrAdapter(execute_batch=self._execute_batch, batch_caps=batch_caps)
        self._started = False
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            await self._adapter.start()
            self._started = True

    async def stop(self) -> None:
        async with self._start_lock:
            if not self._started:
                return
            await self._adapter.stop()
            self._started = False

    async def submit(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        steps: int,
        seed: int | None,
        n: int,
        request_id: str,
    ) -> ImageIngressResult:
        if not self._started:
            await self.start()
        payload = ImageIngressPayload(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed if seed is not None else random.randint(0, 2**32 - 1),
            n=n,
            request_id=request_id,
        )
        return await self._adapter.submit(payload)

    async def _execute_batch(
    self,
    req_type: RequestType,
    payloads: list[ImageIngressPayload],
) -> list[ImageIngressResult]:
        return await self._batch_execute_fn(req_type, payloads)
