"""Helpers for launching and handshaking omni engine cores."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import socket
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from multiprocessing import connection
from types import SimpleNamespace
from typing import Any

import msgspec
import zmq
from omegaconf import OmegaConf
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_ports_list, zmq_socket_ctx
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineProcManager,
    EngineZmqAddresses,
    get_engine_zmq_addresses,
    wait_for_engine_startup,
)
from vllm.v1.executor import Executor

from vllm_omni.distributed.omni_connectors.utils import initialization
from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.stage_init_utils import (
    acquire_device_locks,
    build_diffusion_config,
    initialize_diffusion_stage,
    release_device_locks,
)
from vllm_omni.entrypoints.utils import inject_omni_kv_config
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

StageRoute = tuple[int, int]

# Sentinel that signals "auto-assign me a replica_id" on the wire. Negative
# values are not valid replica ids, so any sub-zero value works equivalently.
AUTO_ASSIGN_REPLICA_ID = -1

# Callback signature for OmniMasterServer.on_register. Fires only for
# auto-assigned replicas (new, headless-launched). The arguments are
# (stage_id, replica_id, allocation).
OnRegisterCallback = Callable[[int, int, "StageAllocation"], None]

# Poll period (ms) used by the registration/handshake loop.
_POLL_PERIOD_MS = 5_000
# Default timeout (s) for a stage to send READY.
_DEFAULT_STARTUP_TIMEOUT_S = 300


def _serialize_stage_config(stage_config: Any) -> Any:
    """Convert a stage config to msgpack-friendly builtins."""
    if stage_config is None or isinstance(stage_config, (str, bytes, int, float, bool)):
        return stage_config

    if OmegaConf.is_config(stage_config):
        return _serialize_stage_config(OmegaConf.to_container(stage_config, resolve=True))

    if dataclasses.is_dataclass(stage_config):
        return _serialize_stage_config(dataclasses.asdict(stage_config))

    if isinstance(stage_config, dict):
        return {key: _serialize_stage_config(value) for key, value in stage_config.items() if not callable(value)}

    if isinstance(stage_config, (list, tuple, set)):
        return [_serialize_stage_config(item) for item in stage_config if not callable(item)]

    if hasattr(stage_config, "items"):
        return {key: _serialize_stage_config(value) for key, value in stage_config.items() if not callable(value)}

    if hasattr(stage_config, "__dict__"):
        return {
            key: _serialize_stage_config(value)
            for key, value in vars(stage_config).items()
            if not key.startswith("_") and not callable(value)
        }

    return stage_config


# ---------------------------------------------------------------------------
# Per-stage address allocation
# ---------------------------------------------------------------------------


@dataclass
class StageAllocation:
    """ZMQ addresses reserved for a single stage."""

    # Per-stage handshake socket (OmniMasterServer binds, engine connects)
    handshake_bind_address: str
    handshake_connect_address: str
    # Input channel: client binds ROUTER, engine connects DEALER
    input_bind_address: str
    input_connect_address: str
    # Output channel: client binds PULL, engine connects PUSH
    output_bind_address: str
    output_connect_address: str
    # The replica's routable IP (from registration). For LLM remote replicas
    # the ZMQ sockets are bound on the head, but the KV connector runs on
    # the replica host — this field preserves the replica's actual IP so the
    # orchestrator can advertise the correct KV sender endpoint.
    replica_host: str | None = None


@dataclass(frozen=True)
class StageCoordinatorAddresses:
    """Optional DP coordinator addresses registered for a stage."""

    coordinator_input: str | None = None
    coordinator_output: str | None = None
    frontend_stats_publish_address: str | None = None


@dataclass
class StageReplicaResources:
    """Resources created while launching one stage replica."""

    manager: Any | None = None
    coordinator: Any | None = None
    addresses: EngineZmqAddresses | None = None
    lock_fds: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OmniMasterServer
# ---------------------------------------------------------------------------


def _port_from_zmq_address(address: str | None) -> int | None:
    """Extract the TCP port from a ``tcp://host:port`` ZMQ address, else ``None``.

    Non-TCP transports (``ipc://``, ``inproc://``) and unparsable values return
    ``None`` so callers can skip seeding them into the port-dedup set.
    """
    if not address:
        return None
    tail = address.rsplit(":", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


class OmniMasterServer:
    """Registration server for single-stage engine startup."""

    def __init__(
        self,
        master_address: str,
        master_port: int,
        stage_ids: list[int],
        stage_replica_counts: dict[int, int] | None = None,
        *,
        coordinator_router_address: str | None = None,
        on_register: OnRegisterCallback | None = None,
        head_local_replicas: dict[int, list[int]] | None = None,
    ) -> None:
        self._address = master_address
        self._port = master_port
        self._stage_routes: dict[StageRoute, StageAllocation] = {}
        self._stage_configs: dict[StageRoute, Any] = {}
        self._stage_coordinator_addresses: dict[StageRoute, StageCoordinatorAddresses] = {}
        self._stage_config_events: dict[StageRoute, threading.Event] = {}
        # Ports already handed out by *this* server. ``get_open_ports_list`` only
        # guarantees uniqueness *within* a single call; a per-route call cannot
        # see ports drawn for other routes, so two stages/replicas can draw the
        # same ephemeral port and the second engine to ``bind()`` it dies with
        # ``zmq.error.ZMQError: Address already in use``. Multi-stage models
        # (e.g. Qwen3-Omni: thinker/talker/code2wav) allocate many routes and hit
        # this intermittently. We dedup every allocated port against this set,
        # seeded with the registration port (and coordinator ROUTER port) that
        # are already bound on the same host.
        self._allocated_ports: set[int] = set()
        if master_port:
            self._allocated_ports.add(int(master_port))
        # Coordinator ROUTER address echoed back in every registration reply
        # so OmniCoordClientForStage knows where to connect from inside the
        # engine subprocess.
        self._coordinator_router_address = coordinator_router_address
        coord_port = _port_from_zmq_address(coordinator_router_address)
        if coord_port is not None:
            self._allocated_ports.add(coord_port)
        # Fires only for *newly assigned* (auto-assigned) replicas, not for
        # head-side pre-allocated slots that already have head-side clients.
        self._on_register = on_register
        # Per-stage allocation lock + auto-assign cursor, so concurrent
        # registrations from multiple headless processes for the same stage
        # don't race on the routing table.
        self._alloc_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        stage_replica_counts = dict(stage_replica_counts or {})

        # Slots the *head* itself will fill via ``_launch_omni_core_engines``
        # / its own ``register_stage_with_omni_master`` call. Auto-assigning
        # headless registrations must skip these even when they appear
        # ``_stage_configs``-unfilled — otherwise a fast headless on the same
        # host can race the head's own registration and steal slot 0.
        self._head_local_slots: set[StageRoute] = set()
        for sid, rids in (head_local_replicas or {}).items():
            for rid in rids:
                self._head_local_slots.add((int(sid), int(rid)))

        for sid in stage_ids:
            replica_count = int(stage_replica_counts.get(sid, 1))
            # Allow 0 explicitly so non-self stages (head distributed mode)
            # can declare "no local replicas; remote ones will register
            # dynamically".
            if replica_count < 0:
                raise ValueError(f"stage_replica_counts[{sid}] must be >= 0, got {replica_count}")
            for replica_id in range(replica_count):
                self._allocate_route_locked(sid, replica_id)

        logger.info(
            "[OmniMasterServer] Pre-allocated addresses for stages %s (master=%s:%d)",
            list(stage_ids),
            master_address,
            master_port,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def address(self) -> str:
        """Return the registration address exposed to stage launchers."""
        return self._address

    @property
    def port(self) -> int:
        """Return the registration port exposed to stage launchers."""
        return self._port

    def get_allocation(self, stage_id: int, replica_id: int = 0) -> StageAllocation:
        """Return the full address allocation for *stage_id*."""
        return self._stage_routes[(stage_id, replica_id)]

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _allocate_route_locked(self, stage_id: int, replica_id: int) -> StageAllocation:
        """Allocate handshake/input/output ports for ``(stage_id, replica_id)``.

        Idempotent: if the route already exists, returns the existing
        allocation unchanged. Caller is responsible for holding
        ``self._alloc_lock`` when needed.
        """
        route = (stage_id, replica_id)
        existing = self._stage_routes.get(route)
        if existing is not None:
            return existing

        self._stage_config_events[route] = threading.Event()
        self._stage_coordinator_addresses[route] = StageCoordinatorAddresses()
        hs_port, inp_port, out_port = self._alloc_unique_ports(3)
        alloc = StageAllocation(
            handshake_bind_address=f"tcp://{self._address}:{hs_port}",
            handshake_connect_address=f"tcp://{self._address}:{hs_port}",
            input_bind_address=f"tcp://{self._address}:{inp_port}",
            input_connect_address=f"tcp://{self._address}:{inp_port}",
            output_bind_address=f"tcp://{self._address}:{out_port}",
            output_connect_address=f"tcp://{self._address}:{out_port}",
        )
        self._stage_routes[route] = alloc
        return alloc

    def _alloc_unique_ports(self, count: int) -> list[int]:
        """Return ``count`` open ports unique across every route this server owns.

        ``get_open_ports_list`` dedups only within its own call, so we redraw any
        port that collides with one already recorded in ``self._allocated_ports``
        (registration/coordinator ports plus every previously-allocated route
        port) and register the winners before returning. Callers already hold the
        relevant allocation context (``__init__`` is single-threaded;
        registration holds ``self._alloc_lock``).
        """
        picked: list[int] = []
        # Bounded retry budget so a pathological host (near port exhaustion)
        # fails loudly instead of spinning forever.
        for _ in range(64):
            for port in get_open_ports_list(count=count - len(picked)):
                if port in self._allocated_ports:
                    continue
                self._allocated_ports.add(port)
                picked.append(port)
                if len(picked) == count:
                    return picked
        raise RuntimeError(
            f"[OmniMasterServer] Could not allocate {count} unique open ports "
            f"(host={self._address}); {len(self._allocated_ports)} already in use."
        )

    def _next_free_replica_id(self, stage_id: int) -> int:
        """Return the next replica id to assign for an auto-assign registration.

        Strategy: prefer filling a pre-allocated-but-unfilled slot (one that
        ``__init__`` reserved in ``_stage_routes`` but no registration has
        completed yet) so the head's bootstrap path — which waits on
        ``_stage_config_events[(stage_id, replica_id)]`` for specific
        pre-allocated ids — unblocks. Only when every pre-allocated slot for
        this stage has been filled do we allocate a fresh id.

        Slots in ``_head_local_slots`` are reserved for the head's own
        ``_launch_omni_core_engines`` registration. Auto-assign must skip
        them even when ``_stage_configs`` shows them unfilled — otherwise a
        same-host headless that registers before the head's own
        ``register_stage_with_omni_master`` call would steal slot 0.

        Without this, a headless contributor using ``--omni-dp-size-local > 1``
        (auto-assign mode) would skip past pre-allocated slot 0 and pick ids
        beyond ``num_replicas``, deadlocking the head's
        ``connect_remote_engine_cores`` wait.
        """
        # Pre-allocated slots that haven't received a registration yet are
        # tracked by absence from ``_stage_configs``. Head-owned slots are
        # not auto-assignable.
        for sid, rid in sorted(self._stage_routes):
            if sid != stage_id:
                continue
            if (sid, rid) in self._head_local_slots:
                continue
            if (sid, rid) not in self._stage_configs:
                return rid
        # Every pre-allocated slot is filled (or head-owned); allocate a
        # fresh id past the existing routes.
        used = {rid for (sid, rid) in self._stage_routes if sid == stage_id}
        rid = 0
        while rid in used:
            rid += 1
        return rid

    def register_stage_config(
        self,
        stage_id: int,
        stage_config: Any,
        coordinator_addresses: StageCoordinatorAddresses | None = None,
        replica_id: int = 0,
    ) -> None:
        """Store the latest stage registration payload for *stage_id*."""
        key = (stage_id, replica_id)
        if key not in self._stage_routes:
            raise KeyError(key)
        self._stage_configs[key] = stage_config
        if coordinator_addresses is not None:
            self._stage_coordinator_addresses[key] = coordinator_addresses
        self._stage_config_events[key].set()

    def get_stage_config(self, stage_id: int, timeout_s: float | None = None, replica_id: int = 0) -> Any:
        """Return the stage config for *stage_id*, waiting if necessary."""
        key = (stage_id, replica_id)
        if key not in self._stage_routes:
            raise KeyError(key)

        if key in self._stage_configs:
            return self._stage_configs[key]

        if not self._stage_config_events[key].wait(timeout=timeout_s):
            raise TimeoutError(f"Timed out waiting for stage config for stage {stage_id} replica {replica_id}.")

        return self._stage_configs[key]

    def get_stage_coordinator_addresses(
        self,
        stage_id: int,
        timeout_s: float | None = None,
        replica_id: int = 0,
    ) -> StageCoordinatorAddresses:
        """Return the registered coordinator addresses for *stage_id*."""
        key = (stage_id, replica_id)
        if key not in self._stage_routes:
            raise KeyError(key)

        if not self._stage_config_events[key].is_set():
            if not self._stage_config_events[key].wait(timeout=timeout_s):
                raise TimeoutError(
                    f"Timed out waiting for stage registration for stage {stage_id} replica {replica_id}."
                )

        return self._stage_coordinator_addresses[key]

    def get_zmq_addresses(self, stage_id: int, replica_id: int = 0) -> EngineZmqAddresses:
        """Return EngineZmqAddresses using the *bind* (client) side addresses."""
        alloc = self.get_allocation(stage_id, replica_id)
        return EngineZmqAddresses(
            inputs=[alloc.input_bind_address],
            outputs=[alloc.output_bind_address],
        )

    def get_engine_zmq_addresses(self, stage_id: int, replica_id: int = 0) -> EngineZmqAddresses:
        """Return EngineZmqAddresses using the *connect* (engine) addresses."""
        alloc = self.get_allocation(stage_id, replica_id)
        return EngineZmqAddresses(
            inputs=[alloc.input_connect_address],
            outputs=[alloc.output_connect_address],
        )

    def get_replica_host(self, stage_id: int, replica_id: int = 0) -> str | None:
        """Return the replica's routable IP if it registered one."""
        alloc = self._stage_routes.get((stage_id, replica_id))
        if alloc is None:
            return None
        return alloc.replica_host

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background server thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="OmniMasterServer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[OmniMasterServer] Listening on tcp://%s:%d",
            self.address,
            self.port,
        )

    def stop(self) -> None:
        """Signal stop and join the background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ------------------------------------------------------------------
    # Internal server logic
    # ------------------------------------------------------------------

    def _run(self) -> None:
        ctx = zmq.Context()
        try:
            self._serve(ctx)
        except Exception:
            logger.exception("[OmniMasterServer] Server thread crashed")
        finally:
            ctx.term()

    def _serve(self, ctx: zmq.Context) -> None:  # type: ignore[type-arg]
        # Registration socket for the initial stage registration.
        # Per-stage handshake sockets are bound by the launch helpers.
        reg_socket: zmq.Socket = ctx.socket(zmq.ROUTER)  # type: ignore[attr-defined]
        reg_socket.bind(f"tcp://{self.address}:{self.port}")

        poller = zmq.Poller()
        poller.register(reg_socket, zmq.POLLIN)

        # The server runs until ``stop()`` is called so that headless replicas
        # spawned after the head finished its initial bring-up can still
        # register dynamically. ``pending`` is kept around purely for
        # debug-level logging of which pre-allocated slots have not yet
        # registered; once empty it does not terminate the loop.
        pending: set[StageRoute] = set(self._stage_routes.keys())

        while not self._stop_event.is_set():
            events: list[tuple[zmq.Socket, int]] = poller.poll(_POLL_PERIOD_MS)  # type: ignore[assignment]
            if not events:
                if pending:
                    logger.debug(
                        "[OmniMasterServer] Still waiting for registration from pre-allocated slots: %s",
                        pending,
                    )
                continue

            for sock, _ in events:
                if sock is reg_socket:
                    route = self._handle_registration(reg_socket)
                    if route is not None:
                        pending.discard(route)

        # Cleanup
        reg_socket.close(linger=0)
        logger.info("[OmniMasterServer] Server thread exiting.")

    def _handle_registration(self, reg_socket: zmq.Socket) -> StageRoute | None:  # type: ignore[type-arg]
        """Receive a stage registration and reply with the handshake address.

        Returns ``(stage_id, replica_id)`` on success or ``None`` on failure.
        """
        frames = reg_socket.recv_multipart()
        if len(frames) < 2:
            logger.warning(
                "[OmniMasterServer] Unexpected registration frame count: %d",
                len(frames),
            )
            return None
        identity = frames[0]
        msg_bytes = frames[-1]
        try:
            msg = msgspec.msgpack.decode(msg_bytes)
        except Exception as exc:
            logger.warning("[OmniMasterServer] Failed to decode registration message: %s", exc)
            return None

        stage_id_raw = msg.get("stage_id")
        if not isinstance(stage_id_raw, int) or stage_id_raw < 0:
            logger.warning(
                "[OmniMasterServer] Registration missing or invalid stage_id: %r",
                stage_id_raw,
            )
            return None
        stage_id: int = stage_id_raw

        incoming_replica_id = int(msg.get("replica_id", 0) or 0)
        was_auto_assigned = incoming_replica_id < 0

        # Distinguish two registration shapes:
        #   - Pre-allocated slots (concrete replica_id >= 0): the head built
        #     this slot during _initialize_stages. Just confirm it; do NOT
        #     fire ``on_register`` (the head already has a head-side client).
        #   - Auto-assigned slots (replica_id == AUTO_ASSIGN_REPLICA_ID):
        #     a *new* replica from a headless launcher. Allocate, then
        #     fire ``on_register`` so the orchestrator attaches.
        with self._alloc_lock:
            if was_auto_assigned:
                replica_id = self._next_free_replica_id(stage_id)
                # When auto-assign picks a slot the head pre-allocated (and
                # is therefore waiting on in ``connect_remote_engine_cores``),
                # the head's bootstrap path builds the head-side client. We
                # must NOT also fire ``on_register`` for it; otherwise the
                # orchestrator would build a duplicate client and overwrite
                # the bootstrap-built one in the pool, leaking it.
                preexisting_slot = (stage_id, replica_id) in self._stage_routes
                alloc = self._allocate_route_locked(stage_id, replica_id)
                if preexisting_slot:
                    was_auto_assigned = False
            else:
                replica_id = incoming_replica_id
                if (stage_id, replica_id) not in self._stage_routes:
                    # Tolerate explicit replica_ids that haven't been
                    # pre-allocated (e.g. headless that wants a specific id).
                    alloc = self._allocate_route_locked(stage_id, replica_id)
                    was_auto_assigned = True
                else:
                    alloc = self._stage_routes[(stage_id, replica_id)]

            # Cross-host remote replicas connect to head-owned sockets, but
            # still advertise their routable host for KV connector endpoints.
            new_bind_address = msg.get("replica_bind_address")
            if new_bind_address:
                alloc.replica_host = new_bind_address
                logger.info(
                    "[OmniMasterServer] Stage %d replica %d registered from host %s "
                    "(serving sockets remain head-owned)",
                    stage_id,
                    replica_id,
                    new_bind_address,
                )

            # Mark the slot as filled *inside* the lock. Without this,
            # concurrent auto-assign registrations from a second headless
            # could call ``_next_free_replica_id`` between the lock
            # releasing above and the ``register_stage_config`` call
            # below, observe the slot as unfilled, and hand the same
            # pre-allocated handshake/input/output addresses to two
            # different replicas — which then collide on
            # ``zmq_socket_ctx(handshake_address, ROUTER, bind=True)``.
            self.register_stage_config(
                stage_id,
                msg.get("stage_config"),
                coordinator_addresses=StageCoordinatorAddresses(
                    coordinator_input=msg.get("coordinator_input"),
                    coordinator_output=msg.get("coordinator_output"),
                    frontend_stats_publish_address=msg.get("frontend_stats_publish_address"),
                ),
                replica_id=replica_id,
            )

        # Fire on_register only for genuinely new (auto-assigned or newly
        # allocated) replicas, on the ROUTER thread. Callback is expected to
        # be cheap and non-blocking (e.g. enqueue onto an asyncio queue).
        if was_auto_assigned and self._on_register is not None:
            try:
                self._on_register(stage_id, replica_id, alloc)
            except Exception:
                logger.exception(
                    "[OmniMasterServer] on_register callback failed for stage=%d replica=%d",
                    stage_id,
                    replica_id,
                )

        response = msgspec.msgpack.encode(
            {
                "handshake_address": alloc.handshake_connect_address,
                "input_address": alloc.input_bind_address,
                "output_address": alloc.output_bind_address,
                "replica_id": replica_id,
                "coordinator_router_address": self._coordinator_router_address,
            }
        )
        # ROUTER-DEALER: reply is [identity, payload] (no empty delimiter).
        reg_socket.send_multipart([identity, response])
        logger.info(
            "[OmniMasterServer] Stage %d replica %d registered (auto=%s); handshake=%s",
            stage_id,
            replica_id,
            was_auto_assigned,
            alloc.handshake_connect_address,
        )
        return (stage_id, replica_id)


@dataclass(frozen=True)
class StageRegistrationResponse:
    """Reply payload returned by :class:`OmniMasterServer` after a successful registration."""

    handshake_address: str
    input_address: str
    output_address: str
    replica_id: int
    coordinator_router_address: str | None


def _detect_local_bind_address(master_address: str, master_port: int) -> str:
    """Return the local IP the kernel would use to reach the master.

    Uses a connected UDP socket as a routing-table probe: ``connect()`` on
    SOCK_DGRAM sends no packets but forces a route lookup, after which
    ``getsockname()[0]`` exposes the source IP that an outbound packet to
    ``(master_address, master_port)`` would carry. For a remote master this
    returns the NIC IP that's reachable from the master, which is exactly the
    address the headless's per-stage ZMQ sockets must bind on.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((master_address, master_port))
        return s.getsockname()[0]
    finally:
        s.close()


def _engine_count(parallel_config: Any) -> int:
    data_parallel_size_local = getattr(parallel_config, "data_parallel_size_local", None)
    if data_parallel_size_local is not None and data_parallel_size_local > 0:
        return int(data_parallel_size_local)
    return max(1, int(parallel_config.data_parallel_size))


def _single_diffusion_parallel_config(*, local_client: bool) -> Any:
    return SimpleNamespace(
        data_parallel_size_local=1 if local_client else 0,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
    )


def register_stage_with_omni_master(
    *,
    omni_master_address: str,
    omni_master_port: int,
    omni_stage_id: int,
    omni_stage_config: Any = None,
    coordinator: DPCoordinator | None = None,
    replica_id: int | None = 0,
    replica_bind_address: str | None = None,
) -> StageRegistrationResponse:
    """Register a stage with the omni master server.

    Pass ``replica_id=None`` to request auto-assignment of a free replica
    id by the master (used by headless launchers).
    """

    if replica_id is None:
        wire_replica_id = AUTO_ASSIGN_REPLICA_ID
    else:
        wire_replica_id = int(replica_id)

    reg_ctx = zmq.Context()
    try:
        reg_sock: zmq.Socket = reg_ctx.socket(zmq.DEALER)  # type: ignore[attr-defined]
        try:
            reg_sock.connect(f"tcp://{omni_master_address}:{omni_master_port}")
            payload: dict[str, Any] = {
                "stage_id": omni_stage_id,
                "replica_id": wire_replica_id,
                "stage_config": _serialize_stage_config(omni_stage_config),
            }
            if coordinator is not None:
                coordinator_input, coordinator_output = coordinator.get_engine_socket_addresses()
                payload["coordinator_input"] = coordinator_input
                payload["coordinator_output"] = coordinator_output
                payload["frontend_stats_publish_address"] = coordinator.get_stats_publish_address()

            # Advertise this host for KV connector routing. Serving/control
            # sockets stay head-owned; remote workers only connect to them.
            if replica_bind_address is None:
                replica_bind_address = _detect_local_bind_address(omni_master_address, omni_master_port)
            payload["replica_bind_address"] = replica_bind_address

            reg_sock.send(msgspec.msgpack.encode(payload))
            timeout_ms = _DEFAULT_STARTUP_TIMEOUT_S * 1_000
            if not reg_sock.poll(timeout=timeout_ms):
                raise RuntimeError(
                    f"Timed out waiting for registration "
                    f"response from OmniMasterServer "
                    f"({omni_master_address}:{omni_master_port}) "
                    f"for stage {omni_stage_id}."
                )
            response_bytes = reg_sock.recv()
            response_msg = msgspec.msgpack.decode(response_bytes)
            handshake_address: str = response_msg["handshake_address"]
            input_address: str = response_msg["input_address"]
            output_address: str = response_msg["output_address"]
            assigned_replica_id: int = int(response_msg.get("replica_id", wire_replica_id))
            coord_router_addr: str | None = response_msg.get("coordinator_router_address")
            logger.info(
                "Stage %d replica %d registered; handshake_address=%s",
                omni_stage_id,
                assigned_replica_id,
                handshake_address,
            )
        finally:
            reg_sock.close(linger=0)
    finally:
        reg_ctx.term()

    return StageRegistrationResponse(
        handshake_address=handshake_address,
        input_address=input_address,
        output_address=output_address,
        replica_id=assigned_replica_id,
        coordinator_router_address=coord_router_addr,
    )


@contextlib.contextmanager
def connect_remote_engine_cores(
    vllm_config: VllmConfig,
    omni_master_server: OmniMasterServer,
    stage_id: int,
    replica_id: int = 0,
) -> Iterator[StageReplicaResources]:
    """Wait for remote engine cores to connect through the omni handshake."""
    addresses = omni_master_server.get_zmq_addresses(stage_id, replica_id=replica_id)
    parallel_config = vllm_config.parallel_config
    remote_engine_count = _engine_count(parallel_config)
    start_index = parallel_config.data_parallel_rank if parallel_config.data_parallel_rank is not None else 0
    coordinator = None

    registered_coordinator_addresses = omni_master_server.get_stage_coordinator_addresses(
        stage_id,
        replica_id=replica_id,
    )
    addresses.coordinator_input = registered_coordinator_addresses.coordinator_input
    addresses.coordinator_output = registered_coordinator_addresses.coordinator_output
    addresses.frontend_stats_publish_address = registered_coordinator_addresses.frontend_stats_publish_address

    engines_to_handshake = [CoreEngine(index=start_index + i, local=False) for i in range(remote_engine_count)]

    logger.info(
        "Waiting for %d remote engine(s) for stage %d replica %d",
        remote_engine_count,
        stage_id,
        replica_id,
    )

    handshake_bind_address = omni_master_server.get_allocation(stage_id, replica_id=replica_id).handshake_bind_address

    with zmq_socket_ctx(handshake_bind_address, zmq.ROUTER, bind=True) as handshake_socket:
        yield StageReplicaResources(
            coordinator=coordinator,
            addresses=addresses,
        )
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            vllm_config.parallel_config,
            False,  # coordinated_dp
            vllm_config.cache_config,
            None,  # proc_manager (remote — no local procs)
            None,  # coord_process
        )


@contextlib.contextmanager
def connect_remote_diffusion_proc(
    omni_master_server: OmniMasterServer,
    stage_id: int,
    replica_id: int = 0,
) -> Iterator[StageReplicaResources]:
    """Wait for a remote headless diffusion proc to connect to head-owned sockets."""
    addresses = omni_master_server.get_zmq_addresses(stage_id, replica_id=replica_id)
    handshake_bind_address = omni_master_server.get_allocation(
        stage_id,
        replica_id=replica_id,
    ).handshake_bind_address

    logger.info("Waiting for remote diffusion proc for stage %d replica %d", stage_id, replica_id)
    with zmq_socket_ctx(handshake_bind_address, zmq.ROUTER, bind=True) as handshake_socket:
        yield StageReplicaResources(addresses=addresses)
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            [CoreEngine(index=0, local=False)],
            _single_diffusion_parallel_config(local_client=False),
            False,
            None,
            None,
            None,
        )


@contextlib.contextmanager
def scoped_spawn_device_env(
    stage_visible_devices: str | None,
    spawn_device_lock: threading.Lock | None,
) -> Iterator[None]:
    """Briefly scope device visibility while spawning a stage subprocess."""
    if stage_visible_devices is None or spawn_device_lock is None:
        yield
        return

    device_control_env = current_omni_platform.device_control_env_var
    with spawn_device_lock:
        previous_visible_devices = os.environ.get(device_control_env)
        try:
            current_omni_platform.set_device_control_env_var(stage_visible_devices)
            yield
        finally:
            if previous_visible_devices is None:
                current_omni_platform.unset_device_control_env_var()
            else:
                current_omni_platform.set_device_control_env_var(previous_visible_devices)


@contextlib.contextmanager
def _launch_omni_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    omni_master_server: OmniMasterServer,
    stage_id: int,
    stage_config: Any = None,
    replica_id: int = 0,
    *,
    omni_coordinator_address: str | None = None,
    stage_visible_devices: str | None = None,
    spawn_device_lock: threading.Lock | None = None,
) -> Iterator[tuple[CoreEngineProcManager, DPCoordinator | None, EngineZmqAddresses]]:
    """Launch local engine cores using the omni registration flow.

    When ``omni_coordinator_address`` is provided, the spawned engine
    subprocesses use :class:`StageEngineCoreProcManager` and each
    instantiates an :class:`OmniCoordClientForStage` after the handshake
    completes so the head's :class:`OmniCoordinator` knows about them.
    """
    addresses = omni_master_server.get_zmq_addresses(stage_id, replica_id=replica_id)
    parallel_config = vllm_config.parallel_config
    local_engine_count = _engine_count(parallel_config)
    dp_rank = parallel_config.data_parallel_rank if parallel_config.data_parallel_rank is not None else 0
    local_start_index = 0
    start_index = dp_rank

    # Run the DP Coordinator process with rank 0 when in online DP mode.
    # The coordinator is needed for:
    # 1. Internal/hybrid LB: collecting and publishing queue stats
    # 2. MoE models: wave coordination in addition to stats
    run_coordinator = vllm_config.needs_dp_coordinator and dp_rank == 0

    if run_coordinator:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )

        addresses.coordinator_input, addresses.coordinator_output = coordinator.get_engine_socket_addresses()
        addresses.frontend_stats_publish_address = coordinator.get_stats_publish_address()

        logger.info(
            "[omni] Started DP Coordinator process for stage %d replica %d (PID: %d)",
            stage_id,
            replica_id,
            coordinator.proc.pid,
        )
    else:
        coordinator = None

    logger.info(
        "Starting %d local engine(s) for stage %d replica %d (dp_rank=%d)",
        local_engine_count,
        stage_id,
        replica_id,
        dp_rank,
    )

    # Register the stage once and reuse the returned per-stage handshake
    # address for all local engine-core processes.
    registration = register_stage_with_omni_master(
        omni_master_address=omni_master_server.address,
        omni_master_port=omni_master_server.port,
        omni_stage_id=stage_id,
        omni_stage_config=stage_config,
        coordinator=coordinator,
        replica_id=replica_id,
    )
    handshake_address = registration.handshake_address

    # One CoreEngine entry per local engine so wait_for_engine_startup can
    # track the HELLO/READY handshake for each of them.
    engines_to_handshake = [CoreEngine(index=start_index + i, local=True) for i in range(local_engine_count)]

    # Bind the pre-allocated handshake socket for this stage.
    handshake_bind_address = omni_master_server.get_allocation(stage_id, replica_id=replica_id).handshake_bind_address

    with zmq_socket_ctx(handshake_bind_address, zmq.ROUTER, bind=True) as handshake_socket:
        if omni_coordinator_address is not None:
            # Use the omni subclass so each spawned subprocess instantiates
            # an OmniCoordClientForStage and heartbeats to the coordinator.
            from vllm_omni.engine.stage_engine_core_proc_manager import StageEngineCoreProcManager

            with scoped_spawn_device_env(stage_visible_devices, spawn_device_lock):
                local_engine_manager: CoreEngineProcManager = StageEngineCoreProcManager(
                    local_engine_count=local_engine_count,
                    start_index=start_index,
                    local_start_index=local_start_index,
                    vllm_config=vllm_config,
                    local_client=True,
                    handshake_address=handshake_address,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    omni_stage_id=stage_id,
                    omni_coordinator_address=omni_coordinator_address,
                    omni_replica_base_id=replica_id,
                )
        else:
            with scoped_spawn_device_env(stage_visible_devices, spawn_device_lock):
                local_engine_manager = CoreEngineProcManager(
                    local_engine_count=local_engine_count,
                    start_index=start_index,
                    local_start_index=local_start_index,
                    vllm_config=vllm_config,
                    local_client=True,
                    handshake_address=handshake_address,
                    executor_class=executor_class,
                    log_stats=log_stats,
                )

        yield local_engine_manager, coordinator, addresses
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            parallel_config.data_parallel_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )


@contextlib.contextmanager
def launch_stage_replica(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    stage_id: int,
    *,
    replica_id: int = 0,
    stage_config: Any = None,
    omni_master_server: OmniMasterServer | None = None,
    omni_coordinator_address: str | None = None,
    stage_visible_devices: str | None = None,
    spawn_device_lock: threading.Lock | None = None,
) -> Iterator[StageReplicaResources]:
    """Launch a local LLM stage replica.

    This is the common entry point for colocated and distributed local LLM
    replicas. Distributed launches delegate to ``_launch_omni_core_engines`` so
    registration/address ownership stays centralized in ``OmniMasterServer``.
    Colocated launches use an IPC handshake without master registration but
    keep the same returned resource bundle.
    """
    if omni_master_server is not None:
        with _launch_omni_core_engines(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            omni_master_server=omni_master_server,
            stage_id=stage_id,
            stage_config=stage_config,
            replica_id=replica_id,
            omni_coordinator_address=omni_coordinator_address,
            stage_visible_devices=stage_visible_devices,
            spawn_device_lock=spawn_device_lock,
        ) as resources:
            engine_manager, coordinator, addresses = resources
            yield StageReplicaResources(
                manager=engine_manager,
                coordinator=coordinator,
                addresses=addresses,
            )
        return

    from vllm.utils.network_utils import get_open_zmq_ipc_path

    from vllm_omni.engine.stage_engine_core_proc_manager import StageEngineCoreProcManager

    addresses = get_engine_zmq_addresses(vllm_config)
    handshake_address = get_open_zmq_ipc_path()
    engines_to_handshake = [CoreEngine(index=0, local=True)]
    with scoped_spawn_device_env(stage_visible_devices, spawn_device_lock):
        engine_manager = StageEngineCoreProcManager(
            local_engine_count=1,
            start_index=0,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=True,
            handshake_address=handshake_address,
            executor_class=executor_class,
            log_stats=log_stats,
            omni_stage_id=stage_id,
            omni_coordinator_address=omni_coordinator_address,
            omni_replica_base_id=replica_id,
        )

    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        yield StageReplicaResources(
            manager=engine_manager,
            addresses=addresses,
        )
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            vllm_config.parallel_config,
            False,  # coordinated_dp
            vllm_config.cache_config,
            engine_manager,
            None,  # coordinator_proc
        )


def launch_headless_llm_replica(
    *,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    omni_master_address: str,
    omni_master_port: int,
    stage_id: int,
    stage_config: Any,
    coordinator: DPCoordinator | None = None,
    replica_bind_address: str | None = None,
) -> CoreEngineProcManager:
    """Register and launch one headless LLM replica.

    Headless LLM workers are connectors: the head binds the handshake/input/
    output sockets, while this process connects to the addresses allocated by
    ``OmniMasterServer``. Keep that registration/manager wiring in the startup
    layer so the CLI only handles argument/config normalization.
    """
    from vllm_omni.engine.stage_engine_core_proc_manager import StageEngineCoreProcManager

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in headless mode")
    dp_rank = parallel_config.data_parallel_rank if parallel_config.data_parallel_rank is not None else 0

    response = register_stage_with_omni_master(
        omni_master_address=omni_master_address,
        omni_master_port=omni_master_port,
        omni_stage_id=stage_id,
        omni_stage_config=stage_config,
        coordinator=coordinator,
        replica_id=None,
        replica_bind_address=replica_bind_address,
    )

    manager = StageEngineCoreProcManager(
        local_engine_count=local_engine_count,
        start_index=dp_rank,
        local_start_index=0,
        vllm_config=vllm_config,
        local_client=False,
        handshake_address=response.handshake_address,
        executor_class=executor_class,
        log_stats=log_stats,
        omni_stage_id=stage_id,
        omni_coordinator_address=response.coordinator_router_address,
        omni_replica_base_id=response.replica_id,
    )
    logger.info(
        "[Headless] Stage %d replica id=%d up (coord=%s)",
        stage_id,
        response.replica_id,
        response.coordinator_router_address,
    )
    return manager


def launch_headless_llm_replicas(
    *,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    omni_master_address: str,
    omni_master_port: int,
    stage_id: int,
    stage_config: Any,
    omni_dp_size_local: int,
    per_replica_devices: list[str | None],
    replica_bind_address: str | None = None,
) -> None:
    """Launch, monitor, and clean up all local headless LLM replicas."""
    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in headless mode")

    dp_rank = parallel_config.data_parallel_rank if parallel_config.data_parallel_rank is not None else 0
    coordinator = None
    if vllm_config.needs_dp_coordinator and dp_rank == 0:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )
        logger.info(
            "[Headless] Started DP Coordinator process for stage %d (PID: %d)",
            stage_id,
            coordinator.proc.pid,
        )

    logger.info(
        "[Headless] Launching %d omni replica(s) (vLLM dp_size_local=%d each) for stage %d "
        "via OmniMasterServer at %s:%d",
        omni_dp_size_local,
        local_engine_count,
        stage_id,
        omni_master_address,
        omni_master_port,
    )

    try:

        def _launch_one(rep_idx: int) -> Any:
            return launch_headless_llm_replica(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                omni_master_address=omni_master_address,
                omni_master_port=omni_master_port,
                stage_id=stage_id,
                stage_config=stage_config,
                coordinator=coordinator,
                replica_bind_address=replica_bind_address,
            )

        launch_headless_replica_group(
            stage_id=stage_id,
            omni_dp_size_local=omni_dp_size_local,
            per_replica_devices=per_replica_devices,
            launch_one=_launch_one,
        )
    finally:
        if coordinator is not None:
            coordinator.shutdown()


def launch_headless_diffusion_replica(
    *,
    model: str,
    od_config: Any,
    stage_config: Any,
    stage_id: int,
    omni_master_address: str,
    omni_master_port: int,
    replica_bind_address: str | None = None,
) -> Any:
    """Register and launch one headless diffusion replica.

    Headless diffusion follows the LLM remote-attach model: the head binds
    handshake/input/output sockets, while this backend process only connects.
    """
    response = register_stage_with_omni_master(
        omni_master_address=omni_master_address,
        omni_master_port=omni_master_port,
        omni_stage_id=stage_id,
        omni_stage_config=stage_config,
        replica_id=None,
        replica_bind_address=replica_bind_address,
    )
    from vllm_omni.diffusion import stage_diffusion_proc

    manager = stage_diffusion_proc.StageDiffusionProcManager.launch_headless(
        model=model,
        od_config=od_config,
        handshake_address=response.handshake_address,
        addresses=EngineZmqAddresses(
            inputs=[response.input_address],
            outputs=[response.output_address],
        ),
        omni_coordinator_address=response.coordinator_router_address,
        omni_stage_id=stage_id,
        omni_replica_id=response.replica_id,
    )
    logger.info(
        "[Headless] Diffusion replica id=%d for stage %d is up (coord=%s)",
        response.replica_id,
        stage_id,
        response.coordinator_router_address,
    )
    return manager


@contextlib.contextmanager
def replica_device_env(stage_id: int, devices: str | None):
    """Temporarily scope device visibility for one replica launch."""
    device_control_env = current_omni_platform.device_control_env_var
    previous_visible_devices = os.environ.get(device_control_env)
    try:
        if devices is not None:
            stage_init_utils.setup_stage_devices(stage_id, {"devices": devices})
        yield
    finally:
        if previous_visible_devices is None:
            current_omni_platform.unset_device_control_env_var()
        else:
            current_omni_platform.set_device_control_env_var(previous_visible_devices)


def get_headless_replica_devices(
    stage_cfg: Any,
    stage_id: int,
    omni_dp_size_local: int,
) -> list[str | None]:
    """Return per-replica device slices for a headless stage."""
    runtime_cfg = getattr(stage_cfg, "runtime", None)
    devices_str: str | None = None
    if runtime_cfg is not None:
        devices_str = (
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "devices", None)
        )
    if not devices_str:
        return [None] * omni_dp_size_local

    devices_per_replica = stage_init_utils.get_stage_devices_per_replica(stage_cfg)
    per_replica_devices = stage_init_utils.split_devices_for_replicas(
        devices_str, omni_dp_size_local, devices_per_replica, stage_id
    )
    logger.info(
        "[Headless] Stage %d: %d local replicas, devices_per_replica=%d, per-replica devices: %s",
        stage_id,
        omni_dp_size_local,
        devices_per_replica,
        per_replica_devices,
    )
    return per_replica_devices


def wait_for_manager_liveness(engine_managers: list[Any]) -> None:
    """Block until one or more engine managers report process exit."""
    if len(engine_managers) == 1:
        engine_managers[0].monitor_engine_liveness()
        return

    def _monitor_target(mgr: Any) -> None:
        try:
            mgr.monitor_engine_liveness()
        except Exception:
            logger.exception("[Headless] monitor_engine_liveness raised")

    monitor_threads: list[threading.Thread] = []
    for mgr in engine_managers:
        t = threading.Thread(
            target=_monitor_target,
            args=(mgr,),
            name=f"omni-replica-monitor-{id(mgr):x}",
        )
        t.start()
        monitor_threads.append(t)
    for t in monitor_threads:
        t.join()


def wait_for_diffusion_manager_liveness(managers: list[Any]) -> None:
    """Block until one diffusion manager exits and surface non-zero exits."""
    sentinel_to_proc = {manager.proc.sentinel: manager.proc for manager in managers}
    died = connection.wait(list(sentinel_to_proc.keys()))
    first = sentinel_to_proc[died[0]]
    logger.info(
        "[Headless] Diffusion replica %s exited (code=%s).",
        first.name,
        first.exitcode,
    )
    if first.exitcode not in (None, 0):
        raise RuntimeError(f"Diffusion replica {first.name!r} exited with code {first.exitcode}")


def launch_headless_replica_group(
    *,
    stage_id: int,
    omni_dp_size_local: int,
    per_replica_devices: list[str | None],
    launch_one: Callable[[int], Any],
    wait_for_replicas: Callable[[list[Any]], None] = wait_for_manager_liveness,
) -> None:
    """Launch, monitor, and clean up a group of local headless replicas."""
    managers: list[Any] = []
    try:
        for rep_idx in range(omni_dp_size_local):
            with replica_device_env(stage_id, per_replica_devices[rep_idx]):
                managers.append(launch_one(rep_idx))
        wait_for_replicas(managers)
    finally:
        logger.info("[Headless] Shutting down stage %d (%d manager(s)).", stage_id, len(managers))
        for manager in managers:
            try:
                manager.shutdown()
            except Exception:
                logger.exception("[Headless] manager shutdown failed")


def launch_headless_diffusion_replicas(
    *,
    model: str,
    stage_cfg: Any,
    stage_configs: list[Any],
    stage_id: int,
    omni_master_address: str,
    omni_master_port: int,
    omni_dp_size_local: int,
    per_replica_devices: list[str | None],
    config_path: str,
    replica_bind_address: str | None = None,
) -> None:
    """Prepare diffusion config, launch replicas, monitor, and clean up."""
    omni_transfer_config = stage_init_utils.load_omni_transfer_config_for_model(model, config_path)
    omni_conn_cfg, omni_from, omni_to = initialization.resolve_omni_kv_config_for_stage(
        omni_transfer_config,
        stage_id,
    )

    metadata = stage_init_utils.extract_stage_metadata(stage_cfg)
    if omni_conn_cfg:
        inject_omni_kv_config(stage_cfg, omni_conn_cfg, omni_from, omni_to)
    # Headless single-stage launch must still infer cross-stage TP topology
    # from the loaded deploy config so heterogeneous KV routing keys match the
    # head process (e.g. from_tp=2, to_tp=1).
    stage_init_utils.inject_kv_stage_info(stage_cfg, stage_id, stage_configs)
    od_config = stage_init_utils.build_diffusion_config(model, stage_cfg, metadata)

    logger.info(
        "[Headless] Launching %d diffusion replica(s) for stage %d via OmniMasterServer at %s:%d",
        omni_dp_size_local,
        stage_id,
        omni_master_address,
        omni_master_port,
    )

    def _launch_one(rep_idx: int) -> Any:
        # Keep torch.distributed ports away from the ZMQ ephemeral range that
        # OmniMasterServer pre-allocates for sibling headless replicas.
        if omni_dp_size_local > 1:
            od_config.master_port = od_config.settle_port(
                61000 + rep_idx * 100,
                port_inc=37,
            )
        return launch_headless_diffusion_replica(
            model=model,
            od_config=od_config,
            stage_config=stage_cfg,
            stage_id=stage_id,
            omni_master_address=omni_master_address,
            omni_master_port=omni_master_port,
            replica_bind_address=replica_bind_address,
        )

    launch_headless_replica_group(
        stage_id=stage_id,
        omni_dp_size_local=omni_dp_size_local,
        per_replica_devices=per_replica_devices,
        launch_one=_launch_one,
        wait_for_replicas=wait_for_diffusion_manager_liveness,
    )


def launch_diffusion_stage_replica(
    *,
    model: str,
    stage_config: Any,
    metadata: Any,
    stage_init_timeout: int,
    batch_size: int,
    use_inline: bool,
    replica_id: int = 0,
    omni_master_server: OmniMasterServer | None = None,
    omni_coordinator_address: str | None = None,
) -> tuple[Any, StageReplicaResources]:
    """Launch a local diffusion stage replica.

    Colocated mode delegates to ``initialize_diffusion_stage``. Distributed
    local mode registers with ``OmniMasterServer`` and spawns a
    ``StageDiffusionProc`` that heartbeats to ``OmniCoordinator``.
    """
    if omni_master_server is None:
        client = initialize_diffusion_stage(
            metadata.stage_id,
            model,
            stage_config,
            metadata,
            stage_init_timeout=stage_init_timeout,
            batch_size=batch_size,
            use_inline=use_inline,
        )
        return client, StageReplicaResources()

    from vllm_omni.diffusion import stage_diffusion_proc
    from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

    od_config = build_diffusion_config(model, stage_config, metadata)
    parallel_config = getattr(od_config, "parallel_config", None)
    world_size = getattr(parallel_config, "world_size", 1)
    try:
        world_size = max(1, int(world_size))
    except (TypeError, ValueError):
        world_size = 1
    lock_fds = acquire_device_locks(
        metadata.stage_id,
        {"tensor_parallel_size": world_size},
        stage_init_timeout,
    )
    proc_manager = None
    try:
        registration = register_stage_with_omni_master(
            omni_master_address=omni_master_server.address,
            omni_master_port=omni_master_server.port,
            omni_stage_id=metadata.stage_id,
            omni_stage_config=stage_config,
            replica_id=replica_id,
        )
        proc_manager = stage_diffusion_proc.StageDiffusionProcManager(
            model=model,
            od_config=od_config,
            stage_init_timeout=stage_init_timeout,
            handshake_address=registration.handshake_address,
            addresses=EngineZmqAddresses(
                inputs=[registration.input_address],
                outputs=[registration.output_address],
            ),
            omni_coordinator_address=omni_coordinator_address,
            omni_stage_id=metadata.stage_id,
            omni_replica_id=replica_id,
        )
        client = StageDiffusionClient.from_addresses(
            metadata,
            request_address=proc_manager.addresses.inputs[0],
            response_address=proc_manager.addresses.outputs[0],
            proc_manager=proc_manager,
            batch_size=batch_size,
        )
        return client, StageReplicaResources(
            manager=proc_manager,
            addresses=proc_manager.addresses,
            lock_fds=lock_fds,
        )
    except Exception:
        if proc_manager is not None:
            proc_manager.shutdown()
        if lock_fds:
            release_device_locks(lock_fds)
        raise
