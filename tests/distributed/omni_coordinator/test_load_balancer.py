# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from time import time

import pytest

from vllm_omni.distributed.omni_coordinator import (
    InstanceInfo,
    LeastQueueLengthBalancer,
    RandomBalancer,
    RoundRobinBalancer,
    StageStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_load_balancer_select_returns_valid_index():
    """Verify RandomBalancer.select() returns a valid index for instances."""
    # Task structure mirrors async_omni; RandomBalancer ignores task contents.
    task: dict = {
        "request_id": "test",
        "engine_inputs": None,
        "sampling_params": None,
    }

    now = time()
    instances = [
        InstanceInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=1,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=StageStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = RandomBalancer()

    index = balancer.select(task, instances)

    assert isinstance(index, int)
    assert 0 <= index < len(instances)


def test_round_robin_balancer_cycles_instances():
    now = time()
    instances = [
        InstanceInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=1,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=StageStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = RoundRobinBalancer()
    results = [balancer.select({}, instances) for _ in range(5)]

    # Default start_index=0 => 0,1,2,0,1
    assert results == [0, 1, 2, 0, 1]


def test_round_robin_balancer_empty_instances_raises():
    with pytest.raises(ValueError, match="instances must not be empty"):
        RoundRobinBalancer().select({}, [])


def test_round_robin_balancer_after_large_index_and_shorter_list():
    """Large start_index % len(instances) then counter wraps with shorter list."""
    now = time()
    two = [
        InstanceInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=StageStatus.UP,
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
    instances = [
        InstanceInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=StageStatus.UP,
            queue_length=5,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = LeastQueueLengthBalancer()
    index = balancer.select({}, instances)
    assert index == 1


def test_least_queue_length_balancer_empty_instances_raises():
    with pytest.raises(ValueError, match="instances must not be empty"):
        LeastQueueLengthBalancer().select({}, [])


def test_least_queue_length_balancer_equal_queues_uses_choice(mocker):
    now = time()
    instances = [
        InstanceInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=3,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=3,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=StageStatus.UP,
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
    assert balancer.select({}, instances) == 2


def test_least_queue_length_balancer_negative_queue_raises():
    now = time()
    instances = [
        InstanceInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=-1,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]
    with pytest.raises(ValueError, match="queue_length must be non-negative"):
        LeastQueueLengthBalancer().select({}, instances)
