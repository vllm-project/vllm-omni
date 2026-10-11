# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from time import time
from types import SimpleNamespace

import pytest

from vllm_omni.distributed.omni_coordinator import (
    LeastQueueLengthBalancer,
    RandomBalancer,
    ReplicaInfo,
    ReplicaStatus,
    RoundRobinBalancer,
    Task,
    TopologyAwareBalancer,
    detect_topology_domain,
)
from vllm_omni.distributed.omni_coordinator import omni_coord_client_for_stage as stage_client_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_load_balancer_select_returns_valid_index():
    """Verify RandomBalancer.select() returns a valid index for replicas."""
    # Task structure mirrors async_omni; RandomBalancer ignores task contents.
    task: dict = {
        "request_id": "test",
        "engine_inputs": None,
        "sampling_params": None,
    }

    now = time()
    replicas = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=1,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=ReplicaStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = RandomBalancer()

    index = balancer.select(task, replicas)

    assert isinstance(index, int)
    assert 0 <= index < len(replicas)


def test_round_robin_balancer_cycles_replicas():
    now = time()
    replicas = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=1,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = RoundRobinBalancer()
    results = [balancer.select({}, replicas) for _ in range(5)]

    # Default start_index=0 => 0,1,2,0,1
    assert results == [0, 1, 2, 0, 1]


def test_round_robin_balancer_empty_replicas_raises():
    with pytest.raises(ValueError, match="replicas must not be empty"):
        RoundRobinBalancer().select({}, [])


def test_round_robin_balancer_after_large_index_and_shorter_list():
    """Large start_index % len(replicas) then counter wraps with shorter list."""
    now = time()
    two = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]
    balancer = RoundRobinBalancer(start_index=7)
    assert balancer.select({}, two) == 1  # 7 % 2
    assert balancer.select({}, two) == 0  # next index wrapped to 0


def test_least_queue_length_balancer_picks_min_queue():
    now = time()
    replicas = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=ReplicaStatus.UP,
            queue_length=5,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = LeastQueueLengthBalancer()
    index = balancer.select({}, replicas)
    assert index == 1


def test_least_queue_length_balancer_empty_replicas_raises():
    with pytest.raises(ValueError, match="replicas must not be empty"):
        LeastQueueLengthBalancer().select({}, [])


def test_least_queue_length_balancer_equal_queues_uses_choice(mocker):
    now = time()
    replicas = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=3,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=3,
            last_heartbeat=now,
            registered_at=now,
        ),
        ReplicaInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=ReplicaStatus.UP,
            queue_length=3,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]
    balancer = LeastQueueLengthBalancer()
    mocker.patch(
        "vllm_omni.distributed.omni_coordinator.load_balancer.random.choice",
        return_value=2,
    )
    assert balancer.select({}, replicas) == 2


def test_least_queue_length_balancer_negative_queue_raises():
    now = time()
    replicas = [
        ReplicaInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=ReplicaStatus.UP,
            queue_length=-1,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]
    with pytest.raises(ValueError, match="queue_length must be non-negative"):
        LeastQueueLengthBalancer().select({}, replicas)


# ============================================================================
# TopologyAwareBalancer tests
# ============================================================================


def _make_replica(queue_length: int, topology_domain: str | None, now: float) -> ReplicaInfo:
    return ReplicaInfo(
        input_addr=f"tcp://host:{10000 + queue_length}",
        output_addr=f"tcp://host:{10000 + queue_length}-out",
        stage_id=0,
        status=ReplicaStatus.UP,
        queue_length=queue_length,
        last_heartbeat=now,
        registered_at=now,
        topology_domain=topology_domain,
    )


def test_topology_aware_balancer_prefers_same_domain():
    """When a topology_domain hint is present, only same-domain replicas are
    considered, and the one with the smallest queue length is picked."""
    now = time()
    replicas = [
        _make_replica(queue_length=5, topology_domain="A", now=now),
        _make_replica(queue_length=0, topology_domain="B", now=now),  # smallest overall, wrong domain
        _make_replica(queue_length=2, topology_domain="A", now=now),
    ]

    balancer = TopologyAwareBalancer()
    task: Task = {"topology_domain": "A"}

    index = balancer.select(task, replicas)

    # Only domain-A replicas are candidates; among them queue 2 < queue 5.
    assert replicas[index].topology_domain == "A"
    assert replicas[index].queue_length == 2


def test_topology_aware_balancer_falls_back_when_no_same_domain():
    """If no replica shares the requested domain, fall back to least-queue
    over all replicas."""
    now = time()
    replicas = [
        _make_replica(queue_length=5, topology_domain="A", now=now),
        _make_replica(queue_length=0, topology_domain="B", now=now),
        _make_replica(queue_length=2, topology_domain="A", now=now),
    ]

    balancer = TopologyAwareBalancer()
    task: Task = {"topology_domain": "C"}  # no replica in domain C

    index = balancer.select(task, replicas)

    assert index == 1  # replica with queue_length=0
    assert replicas[index].queue_length == 0


def test_topology_aware_balancer_no_hint_uses_least_queue():
    """Without a topology_domain hint, behaviour matches LeastQueueLength."""
    now = time()
    replicas = [
        _make_replica(queue_length=5, topology_domain="A", now=now),
        _make_replica(queue_length=0, topology_domain="B", now=now),
        _make_replica(queue_length=2, topology_domain=None, now=now),
    ]

    balancer = TopologyAwareBalancer()

    index = balancer.select({}, replicas)

    assert index == 1
    assert replicas[index].queue_length == 0


def test_topology_aware_balancer_empty_replicas_raises():
    with pytest.raises(ValueError, match="replicas must not be empty"):
        TopologyAwareBalancer().select({"topology_domain": "A"}, [])


def test_topology_aware_balancer_negative_queue_raises():
    now = time()
    replicas = [
        _make_replica(queue_length=-1, topology_domain="A", now=now),
    ]
    with pytest.raises(ValueError, match="queue_length must be non-negative"):
        TopologyAwareBalancer().select({"topology_domain": "A"}, replicas)


def test_topology_aware_balancer_rejects_negative_queue_outside_preferred_domain():
    now = time()
    replicas = [
        _make_replica(queue_length=0, topology_domain="A", now=now),
        _make_replica(queue_length=-1, topology_domain="B", now=now),
    ]
    with pytest.raises(ValueError, match="queue_length must be non-negative"):
        TopologyAwareBalancer().select({"topology_domain": "A"}, replicas)


def test_topology_aware_balancer_tie_breaks_randomly(mocker):
    now = time()
    replicas = [
        _make_replica(queue_length=3, topology_domain="A", now=now),
        _make_replica(queue_length=3, topology_domain="A", now=now),
        _make_replica(queue_length=3, topology_domain="A", now=now),
    ]
    balancer = TopologyAwareBalancer()
    mocker.patch(
        "vllm_omni.distributed.omni_coordinator.load_balancer.random.choice",
        return_value=2,
    )
    assert balancer.select({"topology_domain": "A"}, replicas) == 2


def test_topology_aware_balancer_handles_none_domain_replicas():
    """Replicas with topology_domain=None are never matched by a domain hint
    and only participate in the fallback path."""
    now = time()
    replicas = [
        _make_replica(queue_length=0, topology_domain=None, now=now),
        _make_replica(queue_length=1, topology_domain="A", now=now),
    ]

    balancer = TopologyAwareBalancer()
    task: Task = {"topology_domain": "A"}

    index = balancer.select(task, replicas)

    assert index == 1  # the domain-A replica, even though its queue is larger


def test_detect_topology_domain_prefers_env_var(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_TOPOLOGY_DOMAIN", "nvlink-group-1")

    assert detect_topology_domain() == "nvlink-group-1"


def test_detect_topology_domain_returns_none_without_reliable_locality(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_TOPOLOGY_DOMAIN", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(stage_client_module, "_detect_cuda_topology_domain", lambda _hostname: None)

    assert detect_topology_domain() is None


def test_detect_topology_domain_falls_back_to_visible_device_set(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_TOPOLOGY_DOMAIN", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3, 1")
    monkeypatch.setattr(stage_client_module, "_detect_cuda_topology_domain", lambda _hostname: None)
    monkeypatch.setattr(stage_client_module.socket, "gethostname", lambda: "worker-a")

    assert detect_topology_domain() == "worker-a/cuda-visible:1,3"


class _FakeNvml:
    NVML_NVLINK_MAX_LINKS = 2

    def __init__(self) -> None:
        self.buses = [
            "00000000:01:00.0",
            "00000000:02:00.0",
            "00000000:81:00.0",
            "00000000:82:00.0",
        ]
        self.links = {(0, 0): 1, (1, 0): 0, (2, 0): 3, (3, 0): 2}

    def nvmlInit(self) -> None:
        pass

    def nvmlShutdown(self) -> None:
        pass

    def nvmlDeviceGetCount(self) -> int:
        return len(self.buses)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetPciInfo(self, handle: int):
        return SimpleNamespace(busId=self.buses[handle].encode())

    def nvmlDeviceGetNvLinkState(self, handle: int, link: int) -> bool:
        return (handle, link) in self.links

    def nvmlDeviceGetNvLinkRemotePciInfo(self, handle: int, link: int):
        remote_handle = self.links[(handle, link)]
        return SimpleNamespace(busId=self.buses[remote_handle].encode())


def test_detect_cuda_topology_domain_distinguishes_nvlink_islands(monkeypatch):
    fake_nvml = _FakeNvml()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    first_domain = stage_client_module._detect_cuda_topology_domain("worker-a", pynvml=fake_nvml)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    second_domain = stage_client_module._detect_cuda_topology_domain("worker-a", pynvml=fake_nvml)

    assert first_domain == "worker-a/nvlink:0000:01:00.0,0000:02:00.0"
    assert second_domain == "worker-a/nvlink:0000:81:00.0,0000:82:00.0"


def test_detect_cuda_topology_domain_uses_numa_without_nvlink(monkeypatch):
    fake_nvml = _FakeNvml()
    fake_nvml.links.clear()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(stage_client_module, "_read_pci_locality", lambda _bus_id: "numa:3")

    domain = stage_client_module._detect_cuda_topology_domain("worker-a", pynvml=fake_nvml)

    assert domain == "worker-a/numa:3"


@pytest.mark.parametrize("visible_devices", ["", "-1"])
def test_detect_topology_domain_respects_no_visible_devices(monkeypatch, visible_devices):
    monkeypatch.delenv("VLLM_OMNI_TOPOLOGY_DOMAIN", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)

    assert stage_client_module._detect_cuda_topology_domain("worker-a", pynvml=_FakeNvml()) is None
    monkeypatch.setattr(stage_client_module, "_detect_cuda_topology_domain", lambda _hostname: None)
    assert detect_topology_domain() is None
