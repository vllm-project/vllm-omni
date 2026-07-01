# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.distributed.parallel_state as parallel_state


class _FakeGroup:
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        parallel_mode: str,
        **kwargs,
    ) -> None:
        self.group_ranks = group_ranks
        self.parallel_mode = parallel_mode
        self.device_group = object()
        self.ulysses_group = kwargs.get("ulysses_group")
        self.ring_group = kwargs.get("ring_group")
        self.local_group = next(group for group in group_ranks if local_rank in group)
        self.world_size = len(self.local_group)
        self.rank_in_group = self.local_group.index(local_rank)

    def destroy(self) -> None:
        pass


@pytest.mark.cpu
@pytest.mark.core_model
def test_moe_ep_maps_diffusion_sp_cfg_dp_to_vllm_groups(monkeypatch):
    """MoE+EP rank layout should map SP->PCP, CFG*DP->DP, and TP*SP*CFG*DP->EP."""
    local_rank = 0
    world_size = 32
    created_groups: list[_FakeGroup] = []

    def fake_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        parallel_mode=None,
        group_name=None,
        **kwargs,
    ):
        del backend, group_name
        group = _FakeGroup(
            [list(ranks) for ranks in group_ranks],
            local_rank,
            parallel_mode or "",
            **kwargs,
        )
        created_groups.append(group)
        return group

    fake_world_group = SimpleNamespace(
        rank_in_group=local_rank,
        local_rank=local_rank,
        device_group=object(),
    )
    fake_forward_context = SimpleNamespace(omni_diffusion_config=SimpleNamespace(is_moe=True))
    monkeypatch.setattr(parallel_state.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_backend", lambda *_args, **_kwargs: "gloo")
    monkeypatch.setattr(parallel_state.torch.distributed, "new_group", lambda ranks: tuple(ranks))
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: fake_world_group)
    monkeypatch.setattr(parallel_state, "get_forward_context", lambda: fake_forward_context)
    monkeypatch.setattr(parallel_state, "init_model_parallel_group", fake_init_model_parallel_group)
    monkeypatch.setattr(parallel_state, "init_dit_group", lambda *_args, **_kwargs: None)

    for name in ("_DP", "_CFG", "_SP", "_PP", "_FS", "_EXPERT_PARALLEL_GROUP_RANKS"):
        monkeypatch.setattr(parallel_state, name, None)
    for name in ("_TP", "_PCP", "_DP", "_EP", "_PP"):
        monkeypatch.setattr(parallel_state.vllm_parallel_state, name, None, raising=False)

    parallel_state.initialize_model_parallel(
        tensor_parallel_size=2,
        sequence_parallel_size=2,
        ulysses_degree=2,
        ring_degree=1,
        pipeline_parallel_size=2,
        cfg_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
        backend="gloo",
    )

    assert parallel_state.vllm_parallel_state._PCP is parallel_state._SP
    assert parallel_state.vllm_parallel_state._PCP.world_size == 2
    assert parallel_state._DP.world_size == 2
    assert parallel_state.vllm_parallel_state._DP is not parallel_state._DP
    assert parallel_state.vllm_parallel_state._DP.world_size == 4
    assert parallel_state.vllm_parallel_state._EP.world_size == 16
    assert parallel_state.vllm_parallel_state._TP.world_size == 2
    assert parallel_state._PP.world_size == 2

    assert parallel_state.vllm_parallel_state._PCP.local_group == [0, 2]
    assert parallel_state.vllm_parallel_state._DP.local_group == [0, 8, 16, 24]
    assert parallel_state.vllm_parallel_state._EP.local_group == [
        0,
        1,
        2,
        3,
        8,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
        24,
        25,
        26,
        27,
    ]
    assert parallel_state.get_expert_parallel_group_ranks() == [
        [
            0,
            1,
            2,
            3,
            8,
            9,
            10,
            11,
            16,
            17,
            18,
            19,
            24,
            25,
            26,
            27,
        ],
        [
            4,
            5,
            6,
            7,
            12,
            13,
            14,
            15,
            20,
            21,
            22,
            23,
            28,
            29,
            30,
            31,
        ],
    ]

    ep_groups = [group.local_group for group in created_groups if group.parallel_mode == "expert"]
    assert ep_groups == [parallel_state.vllm_parallel_state._EP.local_group]


@pytest.mark.cpu
@pytest.mark.core_model
def test_cfg_parallel_keeps_diffusion_dp_without_ep(monkeypatch):
    """vLLM DP should keep diffusion DP when expert parallelism is not enabled."""
    local_rank = 0
    world_size = 8

    def fake_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        parallel_mode=None,
        group_name=None,
        **kwargs,
    ):
        del backend, group_name
        return _FakeGroup(
            [list(ranks) for ranks in group_ranks],
            local_rank,
            parallel_mode or "",
            **kwargs,
        )

    fake_world_group = SimpleNamespace(
        rank_in_group=local_rank,
        local_rank=local_rank,
        device_group=object(),
    )
    monkeypatch.setattr(parallel_state.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_backend", lambda *_args, **_kwargs: "gloo")
    monkeypatch.setattr(parallel_state.torch.distributed, "new_group", lambda ranks: tuple(ranks))
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: fake_world_group)
    monkeypatch.setattr(parallel_state, "init_model_parallel_group", fake_init_model_parallel_group)
    monkeypatch.setattr(parallel_state, "init_dit_group", lambda *_args, **_kwargs: None)

    for name in ("_DP", "_CFG", "_SP", "_PP", "_FS", "_EXPERT_PARALLEL_GROUP_RANKS"):
        monkeypatch.setattr(parallel_state, name, None)
    for name in ("_TP", "_PCP", "_DP", "_EP", "_PP"):
        monkeypatch.setattr(parallel_state.vllm_parallel_state, name, None, raising=False)

    parallel_state.initialize_model_parallel(
        tensor_parallel_size=2,
        sequence_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        pipeline_parallel_size=1,
        cfg_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=False,
        backend="gloo",
    )

    assert parallel_state._DP.world_size == 2
    assert parallel_state.vllm_parallel_state._DP is parallel_state._DP
    assert parallel_state.vllm_parallel_state._DP.world_size == 2
    assert parallel_state.vllm_parallel_state._DP.local_group == [0, 4]
    assert parallel_state.vllm_parallel_state._PCP is None
    assert parallel_state.vllm_parallel_state._EP is None
    assert parallel_state._EXPERT_PARALLEL_GROUP_RANKS is None


@pytest.mark.cpu
@pytest.mark.core_model
def test_non_moe_ep_fails_before_vllm_ep_remap(monkeypatch):
    """Non-MoE diffusion configs should not create vLLM PCP/DP/EP remap state."""
    local_rank = 0
    world_size = 8

    def fake_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        parallel_mode=None,
        group_name=None,
        **kwargs,
    ):
        del backend, group_name
        return _FakeGroup(
            [list(ranks) for ranks in group_ranks],
            local_rank,
            parallel_mode or "",
            **kwargs,
        )

    fake_world_group = SimpleNamespace(
        rank_in_group=local_rank,
        local_rank=local_rank,
        device_group=object(),
    )
    fake_forward_context = SimpleNamespace(omni_diffusion_config=SimpleNamespace(is_moe=False))
    monkeypatch.setattr(parallel_state.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_backend", lambda *_args, **_kwargs: "gloo")
    monkeypatch.setattr(parallel_state.torch.distributed, "new_group", lambda ranks: tuple(ranks))
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: fake_world_group)
    monkeypatch.setattr(parallel_state, "get_forward_context", lambda: fake_forward_context)
    monkeypatch.setattr(parallel_state, "init_model_parallel_group", fake_init_model_parallel_group)
    monkeypatch.setattr(parallel_state, "init_dit_group", lambda *_args, **_kwargs: None)

    for name in ("_DP", "_CFG", "_SP", "_PP", "_FS", "_EXPERT_PARALLEL_GROUP_RANKS"):
        monkeypatch.setattr(parallel_state, name, None)
    for name in ("_TP", "_PCP", "_DP", "_EP", "_PP"):
        monkeypatch.setattr(parallel_state.vllm_parallel_state, name, None, raising=False)

    with pytest.raises(RuntimeError, match="Expert parallelism enabled for a non-MoE model"):
        parallel_state.initialize_model_parallel(
            tensor_parallel_size=2,
            sequence_parallel_size=2,
            ulysses_degree=2,
            ring_degree=1,
            pipeline_parallel_size=1,
            cfg_parallel_size=2,
            data_parallel_size=1,
            enable_expert_parallel=True,
            backend="gloo",
        )

    assert parallel_state.vllm_parallel_state._PCP is None
    assert parallel_state.vllm_parallel_state._EP is None
    assert parallel_state._EXPERT_PARALLEL_GROUP_RANKS is None
