# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Control-plane lifecycle for one colocated, rank-paired sharing group.

No model or serving imports: StageRuntime supplies the launch/RPC/termination
adapter. A stage-2 loader consumes Bootstrap from additional_config[CONFIG_KEY]
and installs a WorkerSharingSession before declaring its worker ready.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

import msgspec

from .contracts import Identity, SessionState

CONFIG_KEY = "weight_sharing_bootstrap"
CONTROL_METHOD = "weight_sharing_control"
RANK_RESULTS_KEY = "weight_sharing_ranks"
logger = logging.getLogger(__name__)


class DeviceBudget(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Bytes reserved per physical GPU; private budgets include runtime peaks.

    capacity_bytes is the operator's available budget, excluding other tenants.
    Both runtime peaks are reserved; this layer assumes no execution exclusion.
    """

    capacity_bytes: int
    shared_bytes: int
    provider_bytes: int
    consumer_bytes: int
    kv_cache_bytes: int
    reserve_bytes: int

    def validate(self) -> None:
        values = msgspec.structs.astuple(self)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("sharing memory budgets must be nonnegative integer bytes")
        if self.capacity_bytes == 0 or self.kv_cache_bytes == 0 or sum(values[1:]) > self.capacity_bytes:
            raise ValueError("sharing memory reservations exceed capacity or KV budget is empty")


@dataclass(frozen=True)
class SharingPlan:
    """Internal Python configuration; public YAML/request policy is separate."""

    provider_stage: int
    consumer_stage: int
    device_uuids: tuple[str, ...]
    budgets: tuple[DeviceBudget, ...]
    timeout: float = 60.0
    poll_interval: float = 1.0
    generation: int = 0

    def __post_init__(self) -> None:
        if self.provider_stage == self.consumer_stage or min(self.provider_stage, self.consumer_stage) < 0:
            raise ValueError("sharing requires distinct provider and consumer stages")
        if not self.device_uuids or len(set(self.device_uuids)) != len(self.device_uuids):
            raise ValueError("sharing requires unique physical GPU UUIDs")
        if any(not uuid.startswith("GPU-") for uuid in self.device_uuids):
            raise ValueError("sharing requires physical GPU UUIDs, not CUDA ordinals")
        if len(self.budgets) != len(self.device_uuids):
            raise ValueError("each rank needs a memory budget")
        if self.generation < 0 or any(not math.isfinite(v) or v <= 0 for v in (self.timeout, self.poll_interval)):
            raise ValueError("invalid sharing generation or deadline")
        for budget in self.budgets:
            budget.validate()
        if len({budget.kv_cache_bytes for budget in self.budgets}) != 1:
            raise ValueError("the AR cache configuration requires a uniform per-rank KV budget")


class Bootstrap(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    role: str
    identities: tuple[Identity, ...]
    budgets: tuple[DeviceBudget, ...]
    bundles: tuple[bytes, ...] = ()


class RankResult(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    identity: Identity
    state: SessionState
    shared_bytes: int
    private_bytes: int
    process_id: int
    payload: bytes = b""
    receipt: bytes = b""


class SharingHost(Protocol):
    async def launch(self, stage: int, bootstrap: Bootstrap) -> None: ...

    async def rpc(self, stage: int, operation: str, messages: tuple[bytes, ...]) -> object: ...

    async def quiesce(self, stage: int) -> None: ...

    def check_health(self, stage: int) -> None: ...

    async def terminate(self, stage: int) -> None:
        """Return only after all stage workers have exited; raise if unconfirmed."""
        ...


class SharingRuntime:
    """Owned by the orchestrator event loop; never executes CUDA operations."""

    # Prevent process-manager finalizers from destroying the provider after an
    # initialization exception when consumer exit could not be established.
    _retained: dict[str, SharingRuntime] = {}

    def __init__(self, plan: SharingPlan, host: SharingHost):
        self.plan = plan
        self.host = host
        group = uuid4().hex
        self.identities = tuple(
            Identity(
                group,
                plan.generation,
                uuid4().hex,
                str(plan.provider_stage),
                str(plan.consumer_stage),
                rank,
                len(plan.device_uuids),
                uuid,
            )
            for rank, uuid in enumerate(plan.device_uuids)
        )
        self.state = "NEW"
        self.error = ""
        self._started: list[int] = []
        self._terminated: set[int] = set()
        self._failure_reported = False

    @property
    def termination_complete(self) -> bool:
        return all(stage in self._terminated for stage in self._started)

    async def _rpc(self, stage: int, operation: str, messages: tuple[bytes, ...] = ()) -> list[RankResult]:
        raw = await asyncio.wait_for(self.host.rpc(stage, operation, messages), self.plan.timeout)
        results = msgspec.convert(raw, type=list[RankResult])
        if len(results) != len(self.identities):
            raise RuntimeError(f"stage {stage}: missing rank response to {operation}")
        results.sort(key=lambda result: result.identity.rank)
        for expected, result, budget in zip(self.identities, results, self.plan.budgets, strict=True):
            if result.identity != expected:
                raise RuntimeError(f"stage {stage}: stale, duplicate or mispaired rank response")
            if result.state == SessionState.POISONED:
                raise RuntimeError(f"stage {stage}, rank {expected.rank}: sharing session poisoned")
            private_limit = budget.consumer_bytes
            if stage == self.plan.provider_stage:
                private_limit = budget.provider_bytes + budget.kv_cache_bytes
            if not 0 <= result.shared_bytes <= budget.shared_bytes or not 0 <= result.private_bytes <= private_limit:
                raise RuntimeError(f"stage {stage}, rank {expected.rank}: sharing memory budget exceeded")
        return results

    @staticmethod
    def _require(results: list[RankResult], state: SessionState) -> None:
        if any(result.state != state for result in results):
            raise RuntimeError(f"sharing operation did not reach {state.value} on all ranks")

    async def start(self) -> None:
        if self.state != "NEW":
            raise RuntimeError("a sharing group cannot be started or replayed twice")
        self.state = "STARTING"
        p, c = self.plan.provider_stage, self.plan.consumer_stage
        try:
            self._started.append(p)  # Includes partially launched stages in rollback.
            await self.host.launch(p, Bootstrap("provider", self.identities, self.plan.budgets))
            exports = await self._rpc(p, "claim")
            self._require(exports, SessionState.CLAIMED)
            if any(not result.payload for result in exports):
                raise RuntimeError("provider returned an empty export bundle")
            self._started.append(c)
            await self.host.launch(
                c,
                Bootstrap("consumer", self.identities, self.plan.budgets, tuple(result.payload for result in exports)),
            )
            imports = await self._rpc(c, "status")
            self._require(imports, SessionState.MAPPED)
            if any(not result.receipt for result in imports):
                raise RuntimeError("consumer returned no import receipt")
            confirmed = await self._rpc(p, "confirm_import", tuple(result.receipt for result in imports))
            self._require(confirmed, SessionState.ATTACHED)
            self.state = "READY"
        except BaseException as exc:
            self.fail(f"startup: {exc}")
            await self.terminate()
            raise

    def fail(self, reason: str) -> None:
        if self.state != "CLOSED":
            self.state = "FAILED"
            self.error = reason

    async def monitor(self, stop: asyncio.Event | None = None) -> None:
        try:
            while self.state == "READY":
                if stop is None:
                    await asyncio.sleep(self.plan.poll_interval)
                else:
                    try:
                        await asyncio.wait_for(stop.wait(), self.plan.poll_interval)
                        return
                    except asyncio.TimeoutError:
                        pass
                if self.state != "READY":
                    return
                # Worker RPCs queue behind inference. A long DiT request must
                # not be mistaken for a dead peer because a status RPC timed out.
                # All session mutations go through the checked control path.
                self.host.check_health(self.plan.provider_stage)
                self.host.check_health(self.plan.consumer_stage)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail(f"health: {exc}")
            raise

    async def close(self) -> None:
        if self.state == "CLOSED":
            return
        p, c = self.plan.provider_stage, self.plan.consumer_stage
        try:
            if self.state == "READY":
                self.state = "CLOSING"
                for stage in (p, c):
                    await asyncio.wait_for(self.host.quiesce(stage), self.plan.timeout)
                deadline = asyncio.get_running_loop().time() + self.plan.timeout
                while True:
                    results = await self._rpc(c, "close")
                    if all(result.state == SessionState.CLOSED for result in results):
                        break
                    if any(result.state not in (SessionState.RELEASING, SessionState.CLOSED) for result in results):
                        raise RuntimeError("unexpected consumer close state")
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("consumer still holds shared views")
                    await asyncio.sleep(
                        min(self.plan.poll_interval, max(0, deadline - asyncio.get_running_loop().time()))
                    )
                if any(not result.receipt for result in results):
                    raise RuntimeError("consumer closed without a release receipt")
                self._require(
                    await self._rpc(p, "confirm_release", tuple(r.receipt for r in results)), SessionState.PEER_RELEASED
                )
                self._require(await self._rpc(p, "release"), SessionState.CLOSED)
        except BaseException as exc:
            self.fail(f"shutdown: {exc}")
            raise
        finally:
            await self.terminate()

    async def terminate(self) -> None:
        # Never destroy the provider while a possibly mapped consumer is alive.
        # If consumer death cannot be confirmed, retain the provider and fail.
        if self.state == "FAILED" and not self._failure_reported:
            self._failure_reported = True
            for stage in reversed(self._started):
                if stage not in self._terminated:
                    try:
                        await asyncio.wait_for(self.host.rpc(stage, "fail", ()), min(self.plan.timeout, 5.0))
                    except Exception:
                        logger.warning("Stage %s could not acknowledge failed sharing session", stage, exc_info=True)
        for stage in reversed(self._started):
            if stage not in self._terminated:
                try:
                    await self.host.terminate(stage)
                except BaseException as exc:
                    self.fail(f"stage {stage} exit unconfirmed: {exc}")
                    self._retained[self.identities[0].group_id] = self
                    logger.error("Retaining sharing group %s: %s", self.identities[0].group_id, self.error)
                    raise
                self._terminated.add(stage)
        self._retained.pop(self.identities[0].group_id, None)
        if self.state != "FAILED":
            self.state = "CLOSED"
