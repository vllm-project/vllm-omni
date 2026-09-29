# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MembershipController: distributed replica lifecycle management.

Extracted from Orchestrator to keep request-flow code free of distributed
concerns. Owns the OmniCoordClientForHub, watches for replica
disappearances, and handles register/unregister by building head-side
clients via an injected factory and mutating StagePool membership.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from vllm.logger import init_logger

from vllm_omni.distributed.omni_coordinator import (
    LoadBalancer,
)
from vllm_omni.distributed.omni_coordinator.messages import ReplicaStatus
from vllm_omni.distributed.omni_coordinator.omni_coord_client_for_hub import OmniCoordClientForHub
from vllm_omni.engine.messages import EngineQueueMessage, ErrorMessage
from vllm_omni.engine.stage_pool import StagePool

logger = init_logger(__name__)

RemoteReplicaFactory = Callable[[int, int], Any]


class MembershipController:
    """Manages dynamic replica attach/detach for distributed mode.

    Constructed by DistStageRuntime and passed to Orchestrator.
    The Orchestrator delegates register/unregister messages here.
    """

    WATCH_INTERVAL_S: float = 0.5

    def __init__(
        self,
        stage_pools: list[StagePool],
        coordinator_pub_address: str,
        load_balancer_factory: Callable[[], LoadBalancer],
        remote_replica_factory: RemoteReplicaFactory,
    ) -> None:
        self._stage_pools = stage_pools
        self._remote_replica_factory = remote_replica_factory
        self._membership_tasks: set[asyncio.Task[None]] = set()
        self._attaching_remote_replicas: set[tuple[int, int]] = set()
        self._attached_remote_replicas: set[tuple[int, int]] = set()
        self._remote_replica_keys_by_input_addr: dict[str, tuple[int, int]] = {}
        self._watch_generation = 0
        self._attach_start_generation: dict[tuple[int, int], int] = {}
        self._up_generation_by_input_addr: dict[tuple[int, str], int] = {}
        self._observed_up_attached_addrs: set[tuple[int, str]] = set()
        self._shutdown_event = asyncio.Event()
        self._watcher_task: asyncio.Task[None] | None = None
        self._output_queue: asyncio.Queue[EngineQueueMessage] | None = None
        self._cleanup_callback: Callable[[list[str]], Awaitable[None]] | None = None
        self._replica_removed_callback: Callable[[int, int], None] | None = None

        self._hub = OmniCoordClientForHub(coordinator_pub_address)
        factory = load_balancer_factory
        for pool in self._stage_pools:
            pool.attach_hub(self._hub)
            pool.attach_load_balancer(factory())

    def start(self) -> asyncio.Task[None]:
        """Start the replica watcher as a background task. Returns the task."""
        self._watcher_task = asyncio.create_task(self._watch_replica_list(), name="membership-watcher")
        return self._watcher_task

    async def handle_register(self, stage_id: int, replica_id: int) -> None:
        """Handle a register_remote_replica message (fire-and-forget)."""
        if self._shutdown_event.is_set():
            logger.info(
                "[MembershipController] ignoring register during shutdown stage=%d replica=%d",
                stage_id,
                replica_id,
            )
            return

        key = (stage_id, replica_id)
        if key in self._attached_remote_replicas:
            logger.info(
                "[MembershipController] ignoring register for attached replica stage=%d replica=%d",
                stage_id,
                replica_id,
            )
            return
        if key in self._attaching_remote_replicas:
            logger.info(
                "[MembershipController] ignoring register while attach is in flight stage=%d replica=%d",
                stage_id,
                replica_id,
            )
            return

        self._attaching_remote_replicas.add(key)
        self._attach_start_generation[key] = self._watch_generation
        self._spawn_task(
            self._register_once(stage_id, replica_id),
            label=f"register-s{stage_id}-r{replica_id}",
        )

    async def handle_unregister(
        self,
        stage_id: int,
        input_addr: str,
        output_queue: asyncio.Queue[EngineQueueMessage] | None = None,
        cleanup_callback: Callable[[list[str]], Awaitable[None]] | None = None,
    ) -> None:
        """Handle an unregister_remote_replica message."""
        pool = self._pool_for_stage_id(stage_id)
        if pool is None:
            return
        effective_output_queue = output_queue if output_queue is not None else self._output_queue
        effective_cleanup_callback = cleanup_callback if cleanup_callback is not None else self._cleanup_callback
        replica_id = pool.get_replica_id_by_addr(input_addr)
        client = pool.get_client_by_addr(input_addr)
        affected = pool.invalidate_addr(input_addr)
        self._detach_replica(stage_id, input_addr)
        if client is not None and replica_id is not None and self._replica_removed_callback is not None:
            self._replica_removed_callback(stage_id, replica_id)
        if affected and effective_cleanup_callback is not None:
            await effective_cleanup_callback(affected)
        if affected and effective_output_queue is not None:
            for req_id in affected:
                await effective_output_queue.put(
                    ErrorMessage(error="stage replica disappeared", request_id=req_id, stage_id=stage_id)
                )

    def shutdown(self) -> None:
        """Signal stop and close the hub."""
        self._shutdown_event.set()
        if self._hub is not None:
            self._hub.close()
            self._hub = None
        if self._watcher_task is not None and not self._watcher_task.done():
            self._watcher_task.cancel()

    async def drain_tasks(self, timeout: float = 10.0) -> None:
        """Wait for in-flight membership tasks to complete."""
        if self._membership_tasks:
            await asyncio.wait(self._membership_tasks, timeout=timeout)

    # ---- Internal ----

    def _spawn_task(self, coro: Awaitable[None], *, label: str) -> None:
        task = asyncio.create_task(coro, name=f"membership-{label}")
        self._membership_tasks.add(task)

        def _on_done(t: asyncio.Task[None]) -> None:
            self._membership_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error("[MembershipController] %s task crashed", label, exc_info=exc)

        task.add_done_callback(_on_done)

    async def _watch_replica_list(self) -> None:
        """Convert hub replica disappearances into unregister actions."""
        last_up: set[tuple[int, str]] = set()
        while not self._shutdown_event.is_set():
            try:
                snap = self._hub.get_replica_list()
                self._watch_generation += 1
                current = {(rep.stage_id, rep.input_addr) for rep in snap.replicas if rep.status == ReplicaStatus.UP}
                attached = {
                    (stage_id, input_addr)
                    for input_addr, (stage_id, _replica_id) in self._remote_replica_keys_by_input_addr.items()
                }
                self._observed_up_attached_addrs.update(attached & current)
                attaching_stages = {stage_id for stage_id, _replica_id in self._attaching_remote_replicas}
                for addr_key in current:
                    if addr_key[0] in attaching_stages:
                        self._up_generation_by_input_addr[addr_key] = self._watch_generation

                # Preserve watcher cleanup for untracked/static clients. A
                # controller-attached client is eligible only after this
                # attachment has been observed UP at least once.
                untracked_disappeared = (last_up - current) - attached
                confirmed_attached_disappeared = self._observed_up_attached_addrs - current
                for stage_id, addr in untracked_disappeared | confirmed_attached_disappeared:
                    self._spawn_task(
                        self.handle_unregister(stage_id, addr),
                        label=f"unregister-s{stage_id}",
                    )
                last_up = current
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[MembershipController] _watch_replica_list iteration failed")

            try:
                await asyncio.sleep(self.WATCH_INTERVAL_S)
            except asyncio.CancelledError:
                raise

    async def _register_once(self, stage_id: int, replica_id: int) -> None:
        key = (stage_id, replica_id)
        attach_start_generation = self._attach_start_generation[key]
        try:
            input_addr = await self._do_register(stage_id, replica_id)
            if input_addr is None:
                return
            self._attached_remote_replicas.add(key)
            self._remote_replica_keys_by_input_addr[input_addr] = key
            addr_key = (stage_id, input_addr)
            if self._up_generation_by_input_addr.get(addr_key, -1) > attach_start_generation:
                self._observed_up_attached_addrs.add(addr_key)
        finally:
            self._attaching_remote_replicas.discard(key)
            self._attach_start_generation.pop(key, None)
            if not any(attaching_stage_id == stage_id for attaching_stage_id, _ in self._attaching_remote_replicas):
                self._up_generation_by_input_addr = {
                    addr_key: generation
                    for addr_key, generation in self._up_generation_by_input_addr.items()
                    if addr_key[0] != stage_id
                }

    async def _do_register(self, stage_id: int, replica_id: int) -> str | None:
        pool = self._pool_for_stage_id(stage_id)
        if pool is None:
            logger.warning("[MembershipController] register: stage_id %d out of range", stage_id)
            return None
        client = await asyncio.to_thread(self._remote_replica_factory, stage_id, replica_id)
        if self._shutdown_event.is_set():
            try:
                client.shutdown()
            except Exception:
                logger.exception(
                    "[MembershipController] failed to shutdown late replica client stage=%d replica=%d",
                    stage_id,
                    replica_id,
                )
            return None
        try:
            input_addr = StagePool._client_input_addr(client)
            if input_addr is None:
                raise RuntimeError(
                    f"remote replica factory for stage {stage_id} produced a client without input address"
                )
            pool.add_client(input_addr, client, replica_id=replica_id)
        except Exception:
            try:
                client.shutdown()
            except Exception:
                logger.exception(
                    "[MembershipController] failed to shutdown rejected replica client stage=%d replica=%d",
                    stage_id,
                    replica_id,
                )
            raise
        logger.info(
            "[MembershipController] attached remote replica stage=%d replica=%d addr=%s",
            stage_id,
            replica_id,
            input_addr,
        )
        return input_addr

    def _detach_replica(self, stage_id: int, input_addr: str) -> None:
        key = self._remote_replica_keys_by_input_addr.pop(input_addr, None)
        if key is not None:
            self._attached_remote_replicas.discard(key)
            self._attaching_remote_replicas.discard(key)
            self._observed_up_attached_addrs.discard((stage_id, input_addr))
        pool = self._pool_for_stage_id(stage_id)
        if pool is None:
            return
        client = pool.remove_client(input_addr)
        if client is None:
            return
        try:
            client.shutdown()
        except Exception:
            logger.exception("[MembershipController] failed to shutdown client stage=%d addr=%s", stage_id, input_addr)
        logger.info("[MembershipController] detached replica stage=%d addr=%s", stage_id, input_addr)

    def install_unregister_handlers(
        self,
        *,
        output_queue: asyncio.Queue[EngineQueueMessage],
        cleanup_callback: Callable[[list[str]], Awaitable[None]],
        replica_removed_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Install shared cleanup sinks for watcher-driven unregister events."""
        self._output_queue = output_queue
        self._cleanup_callback = cleanup_callback
        self._replica_removed_callback = replica_removed_callback

    def _pool_for_stage_id(self, stage_id: int) -> StagePool | None:
        if not (0 <= stage_id < len(self._stage_pools)):
            return None
        return self._stage_pools[stage_id]
