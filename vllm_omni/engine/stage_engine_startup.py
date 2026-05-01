"""Helpers for launching and handshaking omni engine cores."""

from __future__ import annotations

import contextlib
import dataclasses
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import msgspec
import zmq
from omegaconf import OmegaConf
from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port, zmq_socket_ctx
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
    ) -> None:
        self._address = master_address
        self._port = master_port
        self._allocations: dict[int, StageAllocation] = {}
        self._stage_configs: dict[int, Any] = {}
        self._stage_coordinator_addresses: dict[int, StageCoordinatorAddresses] = {}
        self._stage_config_events: dict[int, threading.Event] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        for sid in stage_ids:
            self._stage_config_events[sid] = threading.Event()
            self._stage_coordinator_addresses[sid] = StageCoordinatorAddresses()
            hs_port = get_open_port()
            inp_port = get_open_port()
            out_port = get_open_port()
            self._allocations[sid] = StageAllocation(
                handshake_bind_address=f"tcp://{master_address}:{hs_port}",
                handshake_connect_address=f"tcp://{master_address}:{hs_port}",
                input_bind_address=f"tcp://{master_address}:{inp_port}",
                input_connect_address=f"tcp://{master_address}:{inp_port}",
                output_bind_address=f"tcp://{master_address}:{out_port}",
                output_connect_address=f"tcp://{master_address}:{out_port}",
            )

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

    def get_allocation(self, stage_id: int) -> StageAllocation:
        """Return the full address allocation for *stage_id*."""
        return self._allocations[stage_id]

    def register_stage_config(
        self,
        stage_id: int,
        stage_config: Any,
        coordinator_addresses: StageCoordinatorAddresses | None = None,
    ) -> None:
        """Store the latest stage registration payload for *stage_id*."""
        if stage_id not in self._allocations:
            raise KeyError(stage_id)
        self._stage_configs[stage_id] = stage_config
        if coordinator_addresses is not None:
            self._stage_coordinator_addresses[stage_id] = coordinator_addresses
        self._stage_config_events[stage_id].set()

    def get_stage_config(self, stage_id: int, timeout_s: float | None = None) -> Any:
        """Return the stage config for *stage_id*, waiting if necessary."""
        if stage_id not in self._allocations:
            raise KeyError(stage_id)

        if stage_id in self._stage_configs:
            return self._stage_configs[stage_id]

        if not self._stage_config_events[stage_id].wait(timeout=timeout_s):
            raise TimeoutError(f"Timed out waiting for stage config for stage {stage_id}.")

        return self._stage_configs[stage_id]

    def get_stage_coordinator_addresses(
        self,
        stage_id: int,
        timeout_s: float | None = None,
    ) -> StageCoordinatorAddresses:
        """Return the registered coordinator addresses for *stage_id*."""
        if stage_id not in self._allocations:
            raise KeyError(stage_id)

        if not self._stage_config_events[stage_id].is_set():
            if not self._stage_config_events[stage_id].wait(timeout=timeout_s):
                raise TimeoutError(f"Timed out waiting for stage registration for stage {stage_id}.")

        return self._stage_coordinator_addresses[stage_id]

    def get_client_addresses(self, stage_id: int) -> dict[str, str]:
        """Return the addresses the client-side sockets should *bind* to."""
        alloc = self._allocations[stage_id]
        return {
            "input_address": alloc.input_bind_address,
            "output_address": alloc.output_bind_address,
        }

    def get_zmq_addresses(self, stage_id: int) -> EngineZmqAddresses:
        """Return EngineZmqAddresses using the *bind* (client) side addresses."""
        alloc = self._allocations[stage_id]
        return EngineZmqAddresses(
            inputs=[alloc.input_bind_address],
            outputs=[alloc.output_bind_address],
        )

    def get_engine_zmq_addresses(self, stage_id: int) -> EngineZmqAddresses:
        """Return EngineZmqAddresses using the *connect* (engine) addresses."""
        alloc = self._allocations[stage_id]
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

        pending: set[int] = set(self._allocations.keys())

        while pending and not self._stop_event.is_set():
            events: list[tuple[zmq.Socket, int]] = poller.poll(_POLL_PERIOD_MS)  # type: ignore[assignment]
            if not events:
                logger.debug("[OmniMasterServer] Still waiting for registration from stages: %s", pending)
                continue

            for sock, _ in events:
                if sock is reg_socket:
                    sid = self._handle_registration(reg_socket)
                    if sid is not None:
                        pending.discard(sid)

        # Cleanup
        reg_socket.close(linger=0)
        logger.info("[OmniMasterServer] All stages registered; server thread exiting.")

    def _handle_registration(self, reg_socket: zmq.Socket) -> int | None:  # type: ignore[type-arg]
        """Receive a stage registration and reply with the handshake address.

        Returns the registered stage_id on success, or None on failure.
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

        stage_id: int | None = msg.get("stage_id")
        if stage_id not in self._allocations:
            logger.warning(
                "[OmniMasterServer] Received registration for unknown stage_id=%s",
                stage_id,
            )
            return None

        self.register_stage_config(
            stage_id,
            msg.get("stage_config"),
            coordinator_addresses=StageCoordinatorAddresses(
                coordinator_input=msg.get("coordinator_input"),
                coordinator_output=msg.get("coordinator_output"),
                frontend_stats_publish_address=msg.get("frontend_stats_publish_address"),
            ),
        )

        alloc = self._allocations[stage_id]
        response = msgspec.msgpack.encode(
            {
                "handshake_address": alloc.handshake_connect_address,
                "input_address": alloc.input_bind_address,
                "output_address": alloc.output_bind_address,
            }
        )
        # ROUTER-DEALER: reply is [identity, payload] (no empty delimiter).
        reg_socket.send_multipart([identity, response])
        logger.info(
            "[OmniMasterServer] Stage %d registered; assigned handshake=%s",
            stage_id,
            alloc.handshake_connect_address,
        )
        return stage_id


def register_stage_with_omni_master(
    *,
    omni_master_address: str,
    omni_master_port: int,
    omni_stage_id: int,
    omni_stage_config: Any = None,
    coordinator: DPCoordinator | None = None,
    return_addresses: bool = False,
) -> str | tuple[str, str, str]:
    """Register a stage with the omni master server.

    Returns the per-stage handshake address by default. When
    ``return_addresses`` is true, also returns the stage input/output
    addresses allocated by the master.
    """

    reg_ctx = zmq.Context()
    try:
        reg_sock: zmq.Socket = reg_ctx.socket(zmq.DEALER)  # type: ignore[attr-defined]
        try:
            reg_sock.connect(f"tcp://{omni_master_address}:{omni_master_port}")
            payload = {
                "stage_id": omni_stage_id,
                "stage_config": _serialize_stage_config(omni_stage_config),
            }
            if coordinator is not None:
                coordinator_input, coordinator_output = coordinator.get_engine_socket_addresses()
                payload["coordinator_input"] = coordinator_input
                payload["coordinator_output"] = coordinator_output
                payload["frontend_stats_publish_address"] = coordinator.get_stats_publish_address()

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
            response = msgspec.msgpack.decode(response_bytes)
            handshake_address: str = response["handshake_address"]
            input_address: str = response["input_address"]
            output_address: str = response["output_address"]
            logger.info(
                "Stage %d registered; handshake_address=%s",
                omni_stage_id,
                handshake_address,
            )
        finally:
            reg_sock.close(linger=0)
    finally:
        reg_ctx.term()

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
) -> Iterator[tuple[None, DPCoordinator | None, EngineZmqAddresses, None]]:
    """Wait for remote engine cores to connect through the omni handshake."""
    addresses = omni_master_server.get_zmq_addresses(stage_id)
    parallel_config = vllm_config.parallel_config
    # Mirror the engine-count logic from launch_omni_core_engines.
    remote_engine_count = (
        parallel_config.data_parallel_size_local
        if parallel_config.data_parallel_size_local is not None and parallel_config.data_parallel_size_local > 0
        else max(1, parallel_config.data_parallel_size)
    )
    start_index = parallel_config.data_parallel_rank if parallel_config.data_parallel_rank is not None else 0
    coordinator = None

    registered_coordinator_addresses = omni_master_server.get_stage_coordinator_addresses(stage_id)
    addresses.coordinator_input = registered_coordinator_addresses.coordinator_input
    addresses.coordinator_output = registered_coordinator_addresses.coordinator_output
    addresses.frontend_stats_publish_address = registered_coordinator_addresses.frontend_stats_publish_address

    engines_to_handshake = [CoreEngine(index=start_index + i, local=False) for i in range(remote_engine_count)]

    logger.info(
        "Waiting for %d remote engine(s) for stage %d",
        remote_engine_count,
        stage_id,
    )

    handshake_bind_address = omni_master_server.get_allocation(stage_id).handshake_bind_address

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
) -> Iterator[tuple[CoreEngineProcManager, DPCoordinator | None, EngineZmqAddresses]]:
    """Launch local engine cores using the omni registration flow."""
    addresses = omni_master_server.get_zmq_addresses(stage_id)
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
            "[omni] Started DP Coordinator process for stage %d (PID: %d)",
            stage_id,
            coordinator.proc.pid,
        )
    else:
        coordinator = None

    logger.info(
        "Starting %d local engine(s) for stage %d (dp_rank=%d)",
        local_engine_count,
        stage_id,
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
    )

    # One CoreEngine entry per local engine so wait_for_engine_startup can
    # track the HELLO/READY handshake for each of them.
    engines_to_handshake = [CoreEngine(index=start_index + i, local=True) for i in range(local_engine_count)]

    # Bind the pre-allocated handshake socket for this stage.
    handshake_bind_address = omni_master_server.get_allocation(stage_id).handshake_bind_address

    with zmq_socket_ctx(handshake_bind_address, zmq.ROUTER, bind=True) as handshake_socket:
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
