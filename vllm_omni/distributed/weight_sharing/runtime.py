# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Control-plane lifecycle for one colocated, rank-paired sharing group.

No model or serving imports: StageRuntime supplies the launch/RPC/termination
adapter. The model loader consumes Bootstrap from additional_config[CONFIG_KEY]
and installs a WorkerSharingSession before declaring its worker ready.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import uuid4

import msgspec

from .contracts import Identity, SessionState
from .session import SharedWeightGroupConfig

CONFIG_KEY = "weight_sharing_bootstrap"
CONTROL_METHOD = "weight_sharing_control"
RANK_RESULTS_KEY = "weight_sharing_ranks"
logger = logging.getLogger(__name__)


class DeviceBudget(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Resolved group limit and fixed KV sizes per GPU, not allocation requests.

    capacity_bytes = total memory * group ratio; reserve_bytes is the remainder.
    Private peaks are measured, never inferred from individual stage ratios.
    """

    capacity_bytes: int
    reserve_bytes: int
    kv_cache_bytes: int
    consumer_kv_cache_bytes: int

    def validate(self) -> None:
        values = msgspec.structs.astuple(self)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("sharing memory budgets must be nonnegative integer bytes")
        if min(self.capacity_bytes, self.kv_cache_bytes, self.consumer_kv_cache_bytes) == 0:
            raise ValueError("sharing capacity and both KV cache sizes must be positive")
        kv_bytes = self.kv_cache_bytes + self.consumer_kv_cache_bytes
        if kv_bytes > self.capacity_bytes:
            raise ValueError(
                f"fixed KV caches exceed group capacity "
                f"(KV={kv_bytes / 1024**3:.3f}, capacity={self.capacity_bytes / 1024**3:.3f} GiB)"
            )


@dataclass(frozen=True)
class SharingPlan:
    """Resolved deployment; identities and model agreement are frozen before launch."""

    provider_stage: int
    consumer_stage: int
    device_uuids: tuple[str, ...]
    budgets: tuple[DeviceBudget, ...]
    model_id: str
    checkpoint_revision: str
    layout: str
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
        if any(
            len({getattr(budget, field) for budget in self.budgets}) != 1
            for field in ("kv_cache_bytes", "consumer_kv_cache_bytes")
        ):
            raise ValueError("stage cache configurations require uniform per-rank KV budgets")
        if not all((self.model_id, self.checkpoint_revision, self.layout)):
            raise ValueError("sharing requires a frozen model, checkpoint revision and layout")


class Bootstrap(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    groups: tuple[SharedWeightGroupConfig, ...]
    budgets: tuple[DeviceBudget, ...]


class RankResult(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    identity: Identity
    state: SessionState
    shared_bytes: int
    private_bytes: int
    payload: bytes = b""
    receipt: bytes = b""
    free_bytes: int | None = None
    resident_bytes: int = 0  # Current private allocator reservation, included in private_bytes (peak).


class SharingHost(Protocol):
    async def launch(self, stage: int, bootstrap: Bootstrap) -> None: ...

    async def rpc(self, stage: int, operation: str, messages: tuple[bytes, ...]) -> object: ...

    async def quiesce(self, stage: int) -> None: ...

    def check_health(self, stage: int) -> None: ...

    async def terminate(self, stage: int) -> None:
        """Shut down the stage; consumer exit must be confirmed before returning."""
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
        for expected, result in zip(self.identities, results, strict=True):
            if result.identity != expected:
                raise RuntimeError(f"stage {stage}: stale, duplicate or mispaired rank response")
            if result.state == SessionState.POISONED:
                raise RuntimeError(f"stage {stage}, rank {expected.rank}: sharing session poisoned")
            if (
                result.shared_bytes < 0
                or not 0 <= result.resident_bytes <= result.private_bytes
                or (result.free_bytes is not None and result.free_bytes < 0)
            ):
                raise RuntimeError(f"stage {stage}, rank {expected.rank}: invalid memory measurements")
        return results

    def _check_memory(self, providers: list[RankResult], consumers: list[RankResult] | None = None) -> None:
        """Admit measured peaks together; before DiT loads only its KV lower bound is known."""
        gib = 1024**3
        for index, (provider, budget) in enumerate(zip(providers, self.plan.budgets, strict=True)):
            consumer = consumers[index] if consumers is not None else None
            if consumer is not None and consumer.shared_bytes != provider.shared_bytes:
                raise RuntimeError("provider and consumer disagree on shared storage sizes")
            ar_private = max(provider.private_bytes, budget.kv_cache_bytes)
            dit_private = max(consumer.private_bytes if consumer else 0, budget.consumer_kv_cache_bytes)
            required = provider.shared_bytes + ar_private + dit_private
            if required > budget.capacity_bytes:
                raise RuntimeError(
                    f"rank {index}: sharing startup memory budget exceeded "
                    f"(shared={provider.shared_bytes / gib:.3f}, AR private/KV={ar_private / gib:.3f}, "
                    f"DiT private/KV={dit_private / gib:.3f}, total={required / gib:.3f}, "
                    f"limit={budget.capacity_bytes / gib:.3f} GiB)"
                )
            # Preserve space for both measured peaks to recur simultaneously.
            # Device free memory also accounts for non-PyTorch allocations and other processes.
            headroom = ar_private - provider.resident_bytes + dit_private - (consumer.resident_bytes if consumer else 0)
            free_samples = [r.free_bytes for r in (provider, consumer) if r is not None and r.free_bytes is not None]
            free = min(free_samples) if free_samples else None
            if free is not None and free < budget.reserve_bytes + headroom:
                raise RuntimeError(
                    f"rank {index}: insufficient memory for sharing startup "
                    f"(free={free / gib:.3f}, required={(budget.reserve_bytes + headroom) / gib:.3f} GiB)"
                )

    @staticmethod
    def _require(results: list[RankResult], state: SessionState) -> None:
        if any(result.state != state for result in results):
            raise RuntimeError(f"sharing operation did not reach {state.value} on all ranks")

    def bootstrap(self, role: Literal["provider", "consumer"], bundles: tuple[bytes, ...] = ()) -> Bootstrap:
        if len(bundles) != (len(self.identities) if role == "consumer" else 0):
            raise ValueError("sharing bundles must cover every consumer rank")
        return Bootstrap(
            tuple(
                SharedWeightGroupConfig(
                    group_id=i.group_id,
                    generation=i.generation,
                    session_id=i.session_id,
                    provider_id=i.provider_id,
                    consumer_id=i.consumer_id,
                    role=role,
                    rank=i.rank,
                    tp_size=i.tp_size,
                    device_uuid=i.device_uuid,
                    model_id=self.plan.model_id,
                    checkpoint_revision=self.plan.checkpoint_revision,
                    layout=self.plan.layout,
                    payload=bundles[i.rank] if bundles else None,
                )
                for i in self.identities
            ),
            self.plan.budgets,
        )

    async def start(self) -> None:
        if self.state != "NEW":
            raise RuntimeError("a sharing group cannot be started or replayed twice")
        self.state = "STARTING"
        p, c = self.plan.provider_stage, self.plan.consumer_stage
        try:
            self._started.append(p)  # Includes partially launched stages in rollback.
            await self.host.launch(p, self.bootstrap("provider"))
            exports = await self._rpc(p, "claim")
            self._require(exports, SessionState.CLAIMED)
            self._check_memory(exports)
            if any(not result.payload for result in exports):
                raise RuntimeError("provider returned an empty export bundle")
            self._started.append(c)
            await self.host.launch(
                c,
                self.bootstrap("consumer", tuple(result.payload for result in exports)),
            )
            imports = await self._rpc(c, "status")
            self._require(imports, SessionState.MAPPED)
            self._check_memory(exports, imports)
            if any(not result.receipt for result in imports):
                raise RuntimeError("consumer returned no import receipt")
            confirmed = await self._rpc(p, "confirm_import", tuple(result.receipt for result in imports))
            self._require(confirmed, SessionState.ATTACHED)
            self._check_memory(confirmed, imports)
            self.state = "READY"
            logger.info(
                "Weight sharing startup budget check passed: stages %s -> %s, limit GiB/rank=%s, shared GiB/rank=%s, "
                "AR private/KV peak GiB=%s, DiT private/KV peak GiB=%s. "
                "Request-dependent runtime peaks may exceed startup measurements.",
                p,
                c,
                [round(budget.capacity_bytes / 1024**3, 3) for budget in self.plan.budgets],
                [round(rank.shared_bytes / 1024**3, 3) for rank in imports],
                [round(rank.private_bytes / 1024**3, 3) for rank in confirmed],
                [round(rank.private_bytes / 1024**3, 3) for rank in imports],
            )
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
