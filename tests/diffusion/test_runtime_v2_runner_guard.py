# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _config(*, tp=1, sp=1, cfg=1, num_gpus=None, ring=1):
    world_size = tp * sp * cfg
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            sequence_parallel_size=sp,
            cfg_parallel_size=cfg,
            ring_degree=ring,
            world_size=world_size,
        ),
        num_gpus=world_size if num_gpus is None else num_gpus,
    )


@pytest.mark.parametrize("tp,sp,cfg", [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)])
def test_single_group_topology_matches_parallel_world(tp, sp, cfg):
    topology = RuntimeV2Runner._build_topology(_config(tp=tp, sp=sp, cfg=cfg))
    group = topology.groups[0]
    assert len(topology.groups) == 1
    assert len(group.ranks) == len(topology.workers) == tp * sp * cfg
    assert (group.parallel_spec.tp, group.parallel_spec.sp, group.parallel_spec.cfg) == (
        tp,
        sp,
        cfg,
    )


@pytest.mark.parametrize("num_gpus", [2, 8])
def test_single_group_rejects_mismatched_world_size(num_gpus):
    with pytest.raises(ValueError, match="single execution group"):
        RuntimeV2Runner._build_topology(_config(tp=2, sp=2, num_gpus=num_gpus))


def test_single_group_rejects_ring_parallelism():
    with pytest.raises(NotImplementedError, match="ring sequence parallelism"):
        RuntimeV2Runner._build_topology(_config(sp=2, ring=2))
