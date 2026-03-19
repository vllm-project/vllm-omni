# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from time import time

from vllm_omni.distributed.omni_coordinator import (
    InstanceInfo,
    RandomBalancer,
    StageStatus,
)


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
