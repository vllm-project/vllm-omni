# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import zmq

from .messages import ReplicaEvent, ReplicaStatus

logger = logging.getLogger(__name__)


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_pci_bus_id(value: object) -> str | None:
    bus_id = _as_text(value).strip().lower()
    if not bus_id:
        return None
    # NVML commonly emits 00000000:3b:00.0 while Linux sysfs uses
    # 0000:3b:00.0. Keeping the last 12 characters normalizes both forms.
    return bus_id[-12:]


def _read_pci_locality(bus_id: str) -> str | None:
    """Return a Linux NUMA or PCI-root locality label for ``bus_id``."""
    device_path = Path("/sys/bus/pci/devices") / bus_id
    try:
        numa_node = int((device_path / "numa_node").read_text().strip())
        if numa_node >= 0:
            return f"numa:{numa_node}"
    except (OSError, ValueError):
        pass

    try:
        resolved = device_path.resolve(strict=True)
    except OSError:
        return None
    for parent in resolved.parents:
        if parent.name.startswith("pci") and ":" in parent.name:
            return f"pci:{parent.name[3:]}"
    return None


def _visible_nvml_handles(pynvml: Any, all_handles: list[Any]) -> list[Any]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return all_handles
    if not visible_devices.strip():
        return []

    selected: list[Any] = []
    for raw_device in visible_devices.split(","):
        device = raw_device.strip()
        if not device:
            continue
        try:
            if device.isdecimal():
                selected.append(pynvml.nvmlDeviceGetHandleByIndex(int(device)))
            else:
                selected.append(pynvml.nvmlDeviceGetHandleByUUID(device))
        except Exception:
            logger.debug(
                "Unable to resolve CUDA_VISIBLE_DEVICES entry %s with NVML",
                device,
                exc_info=True,
            )
            return []
    return selected


def _detect_cuda_topology_domain(hostname: str, *, pynvml: Any | None = None) -> str | None:
    """Detect the NVLink island, NUMA node, or PCI root of visible GPUs."""
    if pynvml is None:
        try:
            from vllm.utils.import_utils import import_pynvml

            pynvml = import_pynvml()
        except Exception:
            return None

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        all_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        selected_handles = _visible_nvml_handles(pynvml, all_handles)
        if not selected_handles:
            return None

        handle_by_bus: dict[str, Any] = {}
        selected_bus_ids: list[str] = []
        for handle in all_handles:
            bus_id = _normalize_pci_bus_id(pynvml.nvmlDeviceGetPciInfo(handle).busId)
            if bus_id is not None:
                handle_by_bus[bus_id] = handle
        for handle in selected_handles:
            bus_id = _normalize_pci_bus_id(pynvml.nvmlDeviceGetPciInfo(handle).busId)
            if bus_id is None:
                return None
            selected_bus_ids.append(bus_id)

        # Build NVLink connected components across all GPUs so independently
        # launched replicas in the same island derive the same stable label.
        parent = {bus_id: bus_id for bus_id in handle_by_bus}

        def find(bus_id: str) -> str:
            while parent[bus_id] != bus_id:
                parent[bus_id] = parent[parent[bus_id]]
                bus_id = parent[bus_id]
            return bus_id

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        max_links = int(getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 18))
        for local_bus, handle in handle_by_bus.items():
            for link in range(max_links):
                try:
                    if not pynvml.nvmlDeviceGetNvLinkState(handle, link):
                        continue
                    remote_info = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle, link)
                    remote_bus = _normalize_pci_bus_id(remote_info.busId)
                except Exception:
                    continue
                if remote_bus in handle_by_bus:
                    union(local_bus, remote_bus)

        selected_roots = {find(bus_id) for bus_id in selected_bus_ids}
        if len(selected_roots) == 1:
            root = next(iter(selected_roots))
            nvlink_island = sorted(bus_id for bus_id in handle_by_bus if find(bus_id) == root)
            if len(nvlink_island) > 1:
                return f"{hostname}/nvlink:{','.join(nvlink_island)}"

        locality = {_read_pci_locality(bus_id) for bus_id in selected_bus_ids}
        if len(locality) == 1 and None not in locality:
            return f"{hostname}/{next(iter(locality))}"

        # Conservative last resort: use the exact GPU set. This never treats
        # unrelated GPUs as local merely because they share a hostname.
        return f"{hostname}/gpu:{','.join(sorted(set(selected_bus_ids)))}"
    except Exception:
        logger.debug("Automatic CUDA topology detection failed", exc_info=True)
        return None
    finally:
        if initialized:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()


class OmniCoordClientForStage:
    """Client used by stage replicas to send events to OmniCoordinator.

    This client maintains a DEALER socket connected to OmniCoordinator's
    ROUTER endpoint and sends JSON-encoded events describing replica status.
    """

    def __init__(
        self,
        coord_zmq_addr: str,
        input_addr: str,
        output_addr: str,
        stage_id: int,
        topology_domain: str | None = None,
    ) -> None:
        """Initialize client and send initial registration / status-up event."""
        self._coord_zmq_addr = coord_zmq_addr
        self._input_addr = input_addr
        self._output_addr = output_addr
        self._stage_id = stage_id
        self._topology_domain = topology_domain

        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        try:
            self._socket.connect(self._coord_zmq_addr)
        except zmq.ZMQError as e:
            self._socket.close()
            raise RuntimeError(f"Failed to connect to coordinator at {self._coord_zmq_addr}: {e}") from e

        self._status = ReplicaStatus.UP
        self._queue_length = 0
        self._closed = False
        self._closing = False
        self._heartbeat_interval = 5.0
        self._stop_event = threading.Event()
        self._send_lock = threading.RLock()
        # Optional hook invoked from the heartbeat thread before each
        # heartbeat send. Stages set this to refresh ``queue_length`` (or any
        # other field) just-in-time. Exceptions raised by the hook are
        # suppressed and logged.
        self._on_heartbeat: Callable[[], None] | None = None

        self._send_event("update")

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _reconnect(self, max_retries: int = 3, retry_interval: float = 5.0) -> bool:
        """Best-effort reconnect with up to ``max_retries`` attempts.

        Each attempt closes the current socket/context, sleeps ``retry_interval``
        seconds, then creates a new DEALER socket and reconnects to the coordinator.
        Returns True on success, False if all attempts fail.
        """
        if max_retries <= 0:
            return False

        for attempt in range(1, max_retries + 1):
            with self._send_lock:
                if self._stop_event.is_set() or self._closed:
                    return False
                try:
                    self._socket.close(0)
                except zmq.ZMQError:
                    pass
                try:
                    self._ctx.term()
                except zmq.ZMQError:
                    pass

                try:
                    self._ctx = zmq.Context()
                    self._socket = self._ctx.socket(zmq.DEALER)
                    self._socket.connect(self._coord_zmq_addr)
                    return True
                except zmq.ZMQError as e:
                    logger.error(
                        "Stage client reconnect failed (attempt=%d/%d, coord=%s)",
                        attempt,
                        max_retries,
                        self._coord_zmq_addr,
                        exc_info=e,
                    )

            if retry_interval > 0:
                time.sleep(retry_interval)
        return False

    def _send_event(self, event_type: str) -> None:
        """Send a ReplicaEvent to OmniCoordinator.

        Wire format: input_addr, output_addr, stage_id, status, queue_length, event_type.
        For "update": includes status and queue_length from replica state.
        For "heartbeat": includes the latest queue_length (refreshed by the
        optional ``_on_heartbeat`` hook) so the coordinator can propagate
        live load to load balancers between explicit ``update`` events.

        On send failure (ZMQError / RuntimeError), attempts to reconnect up
        to 3 times (5s sleep each) and retries the send once after a
        successful reconnect. Raises if reconnect or the retry send fails.
        """
        with self._send_lock:
            if self._closed:
                raise RuntimeError("Client already closed")

            event = ReplicaEvent(
                input_addr=self._input_addr,
                output_addr=self._output_addr,
                stage_id=self._stage_id,
                event_type=event_type,
                status=self._status,
                queue_length=self._queue_length,
                topology_domain=self._topology_domain,
            )
            data = json.dumps(asdict(event)).encode("utf-8")

            try:
                self._socket.send(data, flags=zmq.NOBLOCK)
                return
            except zmq.Again:
                logger.debug("Send buffer full, dropping event")
                return
            except (RuntimeError, zmq.ZMQError) as e:
                # First send failed; try reconnecting a few times.
                if not self._reconnect(max_retries=3):
                    logger.error("Failed to send event and reconnect to coordinator", exc_info=e)
                    raise

                # Reconnected successfully; try sending once more.
                try:
                    self._socket.send(data, flags=zmq.NOBLOCK)
                except zmq.Again:
                    logger.debug("Send buffer full after reconnect, dropping event")
                except (RuntimeError, zmq.ZMQError) as e2:
                    logger.error("Failed to send event after successful reconnect", exc_info=e2)
                    raise

    def update_info(
        self,
        status: ReplicaStatus | None = None,
        queue_length: int | None = None,
    ) -> None:
        """Update replica information and notify OmniCoordinator.

        At least one of ``status`` or ``queue_length`` must be provided.
        """
        if status is None and queue_length is None:
            raise ValueError("At least one of status or queue_length must be provided")

        with self._send_lock:
            if self._closed or self._closing:
                raise RuntimeError("Client is closing or already closed")

            if status is not None:
                self._status = status
            if queue_length is not None:
                self._queue_length = queue_length

            self._send_event("update")

    def _heartbeat_loop(self) -> None:
        """Periodically send heartbeat events while the client is alive."""
        while not self._stop_event.wait(timeout=self._heartbeat_interval):
            if self._closed:
                break

            # Invoke the optional pre-heartbeat hook so callers (e.g. the
            # engine subprocess) can refresh ``queue_length`` from live state
            # before the heartbeat is sent. Exceptions are swallowed so a
            # buggy hook never breaks the heartbeat loop.
            hook = self._on_heartbeat
            if hook is not None:
                with contextlib.suppress(Exception):
                    hook()

            try:
                self._send_event("heartbeat")
            except (RuntimeError, zmq.ZMQError) as e:
                if self._closed or self._stop_event.is_set():
                    break
                logger.warning("Heartbeat send failed; will retry on next interval", exc_info=e)
                continue

    def close(self) -> None:
        """Send a final down event and close the underlying socket."""
        if self._closed:
            raise RuntimeError("Client already closed")

        # Stop heartbeat thread first to avoid concurrent sends during shutdown.
        self._stop_event.set()
        if hasattr(self, "_heartbeat_thread"):
            self._heartbeat_thread.join(timeout=1.0)

        with self._send_lock:
            if self._closed:
                raise RuntimeError("Client already closed")

            self._closing = True

            # Mark status as DOWN and send one last update.
            self._status = ReplicaStatus.DOWN
            try:
                self._send_event("update")
            except (RuntimeError, zmq.ZMQError):
                pass  # Socket may already be broken, proceed with close

            # Close DEALER socket and terminate this client's context.
            self._socket.close(0)
            try:
                self._ctx.term()
            except zmq.ZMQError:
                pass
            self._closed = True


def detect_topology_domain() -> str | None:
    """Best-effort detection of the local NVLink/NUMA/IB topology domain.

    Returns the value of the ``VLLM_OMNI_TOPOLOGY_DOMAIN`` environment variable
    if set. Otherwise, CUDA replicas are grouped by NVLink connected component,
    then NUMA node or PCI root complex. The exact visible GPU set is used as a
    conservative last resort, including when NVML is unavailable. Replicas
    that share a domain can communicate over fast local links, while
    cross-domain transfers traverse InfiniBand or TCP.

    Returns ``None`` if even the hostname cannot be determined, in which case
    the topology-aware balancer falls back to ``least-queue-length``.
    """
    env_domain = os.environ.get("VLLM_OMNI_TOPOLOGY_DOMAIN")
    if env_domain:
        return env_domain
    try:
        hostname = socket.gethostname()
    except OSError:
        return None
    detected_domain = _detect_cuda_topology_domain(hostname)
    if detected_domain is not None:
        return detected_domain

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return None
    device_ids = sorted({device.strip() for device in visible_devices.split(",") if device.strip()})
    if not device_ids or device_ids == ["-1"]:
        return None
    return f"{hostname}/cuda-visible:{','.join(device_ids)}"


def create_stage_coord_client(
    *,
    coord_zmq_addr: str,
    input_addr: str,
    output_addr: str,
    stage_id: int,
    queue_length_getter: Callable[[], int] | None = None,
    topology_domain: str | None = None,
) -> OmniCoordClientForStage:
    """Create a stage coordinator client with an optional heartbeat hook."""
    client = OmniCoordClientForStage(
        coord_zmq_addr=coord_zmq_addr,
        input_addr=input_addr,
        output_addr=output_addr,
        stage_id=stage_id,
        topology_domain=topology_domain,
    )
    if queue_length_getter is not None:

        def _refresh_queue_length() -> None:
            try:
                client._queue_length = max(int(queue_length_getter()), 0)
            except Exception:
                pass

        client._on_heartbeat = _refresh_queue_length
    return client
