"""Helpers for launching and handshaking omni engine cores."""

from __future__ import annotations

import contextlib
import dataclasses
import socket
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import msgspec
import zmq
from omegaconf import OmegaConf
from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_ports_list, zmq_socket_ctx
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.utils import (
    STARTUP_POLL_PERIOD_MS,
    CoreEngine,
    CoreEngineProcManager,
    CoreEngineState,
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    wait_for_engine_startup,
)
from vllm.v1.executor import Executor

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


@dataclass(frozen=True)
class StageCoordinatorAddresses:
    """Optional DP coordinator addresses registered for a stage."""

    coordinator_input: str | None = None
    coordinator_output: str | None = None
    frontend_stats_publish_address: str | None = None


# ---------------------------------------------------------------------------
# OmniMasterServer
# ---------------------------------------------------------------------------


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
        # Coordinator ROUTER address echoed back in every registration reply
        # so OmniCoordClientForStage knows where to connect from inside the
        # engine subprocess.
        self._coordinator_router_address = coordinator_router_address
        # Fires only for *newly assigned* (auto-assigned) replicas, not for
        # head-side pre-allocated slots that already have head-side clients.
        self._on_register = on_register
        # Per-stage allocation lock + auto-assign cursor, so concurrent
        # registrations from multiple headless processes for the same stage
        # don't race on the routing table.
        self._alloc_lock = threading.Lock()
        self._stage_ids_known: set[int] = set(int(sid) for sid in stage_ids)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        stage_replica_counts = dict(stage_replica_counts or {})

        # Slots the *head* itself will fill via ``launch_omni_core_engines``
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

    @property
    def coordinator_router_address(self) -> str | None:
        """Return the OmniCoordinator ROUTER address echoed to replicas."""
        return self._coordinator_router_address

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
        hs_port, inp_port, out_port = get_open_ports_list(count=3)
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

    def _next_free_replica_id(self, stage_id: int) -> int:
        """Return the next replica id to assign for an auto-assign registration.

        Strategy: prefer filling a pre-allocated-but-unfilled slot (one that
        ``__init__`` reserved in ``_stage_routes`` but no registration has
        completed yet) so the head's bootstrap path — which waits on
        ``_stage_config_events[(stage_id, replica_id)]`` for specific
        pre-allocated ids — unblocks. Only when every pre-allocated slot for
        this stage has been filled do we allocate a fresh id.

        Slots in ``_head_local_slots`` are reserved for the head's own
        ``launch_omni_core_engines`` registration. Auto-assign must skip
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

    def get_client_addresses(self, stage_id: int, replica_id: int = 0) -> dict[str, str]:
        """Return the addresses the client-side sockets should *bind* to."""
        alloc = self.get_allocation(stage_id, replica_id)
        return {
            "input_address": alloc.input_bind_address,
            "output_address": alloc.output_bind_address,
        }

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

            # Cross-host override: when the registering replica advertised
            # its own bind address + ports, rewrite the StageAllocation so
            # each socket is rooted on the host that actually binds it
            # (the master's pre-allocated ports are unreachable from a
            # remote replica's host).
            #
            # Diffusion and LLM stages have different binder ownership:
            #
            #   Diffusion remote replica (StageDiffusionProc):
            #     - handshake: replica binds  -> rewrite to replica IP
            #     - input    : replica binds  -> rewrite to replica IP
            #     - output   : replica binds  -> rewrite to replica IP
            #
            #   LLM remote replica (CoreClient on head):
            #     - handshake: head binds (``connect_remote_engine_cores``)
            #                 -> keep on master IP, worker TCP-connects
            #     - input    : head binds (``CoreClient`` ROUTER)
            #                 -> keep on master IP, worker TCP-connects
            #     - output   : head binds (``CoreClient`` PULL — default
            #                 bind=True for PULL in ``make_zmq_socket``)
            #                 -> keep on master IP, worker TCP-connects
            #
            # The registrant indicates which case via the boolean
            # ``replica_binds_sockets`` payload flag. It defaults to
            # True (the diffusion / single-host case) so older callers
            # still get the previous full-rewrite semantics. For LLM
            # remote replicas, the master keeps every address on its
            # own host and the remote worker establishes 3 outbound
            # TCP connections to the master.
            new_bind_address = msg.get("replica_bind_address")
            if new_bind_address:
                replica_binds_sockets = bool(msg.get("replica_binds_sockets", True))
                if replica_binds_sockets:
                    hs_port = int(msg["replica_handshake_port"])
                    inp_port = int(msg["replica_input_port"])
                    out_port = int(msg["replica_output_port"])
                    hs_bind_addr = f"tcp://{new_bind_address}:{hs_port}"
                    inp_bind_addr = f"tcp://{new_bind_address}:{inp_port}"
                    out_bind_addr = f"tcp://{new_bind_address}:{out_port}"
                    alloc = StageAllocation(
                        handshake_bind_address=hs_bind_addr,
                        handshake_connect_address=hs_bind_addr,
                        input_bind_address=inp_bind_addr,
                        input_connect_address=inp_bind_addr,
                        output_bind_address=out_bind_addr,
                        output_connect_address=out_bind_addr,
                    )
                    self._stage_routes[(stage_id, replica_id)] = alloc
                logger.info(
                    "[OmniMasterServer] Stage %d replica %d cross-host bind (sockets bound on %s; replica_ip=%s)",
                    stage_id,
                    replica_id,
                    "replica" if replica_binds_sockets else "master",
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
    ``(master_address, master_port)`` would carry. For a co-located master
    this returns the loopback or eth0 IP (same effect as the legacy
    ``self._address`` behaviour); for a remote master it returns the
    NIC IP that's actually reachable from the master — which is exactly
    the address the headless's per-stage ZMQ sockets must bind on.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((master_address, master_port))
        return s.getsockname()[0]
    finally:
        s.close()


def register_stage_with_omni_master(
    *,
    omni_master_address: str,
    omni_master_port: int,
    omni_stage_id: int,
    omni_stage_config: Any = None,
    coordinator: DPCoordinator | None = None,
    return_addresses: bool = False,
    replica_id: int | None = 0,
    return_full_response: bool = False,
    replica_bind_address: str | None = None,
    replica_binds_sockets: bool = True,
) -> str | tuple[str, str, str] | StageRegistrationResponse:
    """Register a stage with the omni master server.

    Returns the per-stage handshake address by default. When
    ``return_addresses`` is true, also returns the stage input/output
    addresses allocated by the master. When ``return_full_response`` is
    true, returns the full :class:`StageRegistrationResponse` including the
    assigned ``replica_id`` and the OmniCoordinator ROUTER address (if
    published by the master).

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

            # Always advertise THIS host's local bind address + 3 locally
            # free ports so the master can root the per-stage socket
            # allocation on the replica's own interface. For a co-located
            # replica the detected IP matches the master's address and
            # the override is a no-op semantically; for a cross-host
            # replica it's what makes the headless's ROUTER bind succeed
            # (otherwise the master would hand back ``tcp://<master_ip>:port``
            # and ``zmq.bind`` would EADDRNOTAVAIL on the remote host).
            if replica_bind_address is None:
                replica_bind_address = _detect_local_bind_address(omni_master_address, omni_master_port)
            hs_port, inp_port, out_port = get_open_ports_list(count=3)
            payload["replica_bind_address"] = replica_bind_address
            payload["replica_handshake_port"] = hs_port
            payload["replica_input_port"] = inp_port
            payload["replica_output_port"] = out_port
            # ``False`` only for LLM headless replicas: the head's
            # ``connect_remote_engine_cores`` is the binder for the
            # handshake ROUTER, and ``CoreClient.__init__`` binds the
            # input ROUTER and the output PULL (``make_zmq_socket``
            # defaults bind=True for PULL). The master must keep all
            # three addresses on the master's host so the head can
            # ``bind`` them; the remote LLM worker TCP-connects across
            # hosts on all three.
            payload["replica_binds_sockets"] = bool(replica_binds_sockets)

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

    if return_full_response:
        return StageRegistrationResponse(
            handshake_address=handshake_address,
            input_address=input_address,
            output_address=output_address,
            replica_id=assigned_replica_id,
            coordinator_router_address=coord_router_addr,
        )
    if return_addresses:
        return handshake_address, input_address, output_address
    return handshake_address


def _wait_for_omni_engine_startup(
    handshake_socket: zmq.Socket,
    engine_addresses: EngineZmqAddresses,
    engines: list[CoreEngine],
    cache_config: CacheConfig,
) -> None:
    """Wait for omni-managed engines to finish the HELLO/READY handshake."""
    conn_pending = len(engines)
    start_pending = 0

    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)

    while conn_pending or start_pending:
        events = poller.poll(STARTUP_POLL_PERIOD_MS)
        if not events:
            logger.debug(
                "[omni] Waiting for %d engine(s) to connect, %d to start.",
                conn_pending,
                start_pending,
            )
            continue

        eng_identity, msg_bytes = handshake_socket.recv_multipart()
        eng_index = int.from_bytes(eng_identity, "little")
        engine = next((e for e in engines if e.identity == eng_identity), None)
        if engine is None:
            raise RuntimeError(f"[omni] Handshake message from unexpected engine rank: {eng_index}")

        msg = msgspec.msgpack.decode(msg_bytes)
        status: str = msg["status"]

        if status == "HELLO" and engine.state == CoreEngineState.NEW:
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(addresses=engine_addresses, parallel_config={})
            )
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)
            conn_pending -= 1
            start_pending += 1
            engine.state = CoreEngineState.CONNECTED
            logger.debug("[omni] HELLO from engine %d", eng_index)

        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            # Upstream vllm >=0.19 dropped `num_gpu_blocks` from the READY
            # handshake payload (the field is now communicated out-of-band via
            # stats/health topics); tolerate both legacy and new message
            # shapes so the omni handshake keeps working across rebases.
            reported_blocks = msg.get("num_gpu_blocks")
            if reported_blocks is not None:
                cache_config.num_gpu_blocks = (cache_config.num_gpu_blocks or 0) + int(reported_blocks)
            if engine_addresses.frontend_stats_publish_address is None:
                engine_addresses.frontend_stats_publish_address = msg.get("dp_stats_address")
            start_pending -= 1
            engine.state = CoreEngineState.READY
            logger.debug(
                "[omni] READY from engine %d (num_gpu_blocks=%s)",
                eng_index,
                "unknown" if reported_blocks is None else reported_blocks,
            )

        else:
            raise RuntimeError(f"[omni] Unexpected status '{status}' from engine {eng_index} in state {engine.state}.")


@contextlib.contextmanager
def connect_remote_engine_cores(
    vllm_config: VllmConfig,
    omni_master_server: OmniMasterServer,
    stage_id: int,
    replica_id: int = 0,
) -> Iterator[tuple[None, DPCoordinator | None, EngineZmqAddresses, None]]:
    """Wait for remote engine cores to connect through the omni handshake."""
    addresses = omni_master_server.get_zmq_addresses(stage_id, replica_id=replica_id)
    parallel_config = vllm_config.parallel_config
    # Mirror the engine-count logic from launch_omni_core_engines.
    remote_engine_count = (
        parallel_config.data_parallel_size_local
        if parallel_config.data_parallel_size_local is not None and parallel_config.data_parallel_size_local > 0
        else max(1, parallel_config.data_parallel_size)
    )
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
        yield None, coordinator, addresses, None

        _wait_for_omni_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            vllm_config.cache_config,
        )


@contextlib.contextmanager
def launch_omni_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    omni_master_server: OmniMasterServer,
    stage_id: int,
    stage_config: Any = None,
    replica_id: int = 0,
    *,
    omni_coordinator_address: str | None = None,
) -> Iterator[tuple[CoreEngineProcManager, DPCoordinator | None, EngineZmqAddresses]]:
    """Launch local engine cores using the omni registration flow.

    When ``omni_coordinator_address`` is provided, the spawned engine
    subprocesses use :class:`OmniCoreEngineProcManager` and each
    instantiates an :class:`OmniCoordClientForStage` after the handshake
    completes so the head's :class:`OmniCoordinator` knows about them.
    """
    addresses = omni_master_server.get_zmq_addresses(stage_id, replica_id=replica_id)
    parallel_config = vllm_config.parallel_config
    # Determine the number of local engines and their ranks.
    local_engine_count = (
        parallel_config.data_parallel_size_local
        if parallel_config.data_parallel_size_local is not None and parallel_config.data_parallel_size_local > 0
        else max(1, parallel_config.data_parallel_size)
    )
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
    handshake_address = register_stage_with_omni_master(
        omni_master_address=omni_master_server.address,
        omni_master_port=omni_master_server.port,
        omni_stage_id=stage_id,
        omni_stage_config=stage_config,
        coordinator=coordinator,
        replica_id=replica_id,
    )

    # One CoreEngine entry per local engine so wait_for_engine_startup can
    # track the HELLO/READY handshake for each of them.
    engines_to_handshake = [CoreEngine(index=start_index + i, local=True) for i in range(local_engine_count)]

    # Bind the pre-allocated handshake socket for this stage.
    handshake_bind_address = omni_master_server.get_allocation(stage_id, replica_id=replica_id).handshake_bind_address

    with zmq_socket_ctx(handshake_bind_address, zmq.ROUTER, bind=True) as handshake_socket:
        if omni_coordinator_address is not None:
            # Use the omni subclass so each spawned subprocess instantiates
            # an OmniCoordClientForStage and heartbeats to the coordinator.
            from vllm_omni.engine.omni_core_engine_proc_manager import OmniCoreEngineProcManager

            local_engine_manager: CoreEngineProcManager = OmniCoreEngineProcManager(
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

        # Wait for all local engine-core processes to complete the
        # standard HELLO/READY handshake — mirrors launch_core_engines.
        coordinated_dp = parallel_config.data_parallel_size > 1 and vllm_config.model_config.is_moe
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            coordinated_dp,
            vllm_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )
