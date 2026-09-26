# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
        self.device_communicator = kwargs.get("device_communicator")
        reduce_scatter = kwargs.get("reduce_scatter")
        if reduce_scatter is not None:
            self.reduce_scatter = reduce_scatter
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

    def fake_init_vllm_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        group_name,
        use_all2all=False,
    ):
        del backend
        group = _FakeGroup(
            [list(ranks) for ranks in group_ranks],
            local_rank,
            f"vllm_{group_name}",
            device_communicator=object(),
            reduce_scatter=lambda tensor, **kwargs: tensor,
        )
        group.use_all2all = use_all2all
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
    monkeypatch.setattr(parallel_state, "init_vllm_model_parallel_group", fake_init_vllm_model_parallel_group)

    for name in ("_DP", "_CFG", "_SP", "_PP", "_EXPERT_PARALLEL_GROUP_RANKS"):
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

    assert parallel_state.vllm_parallel_state._PCP is not parallel_state._SP
    assert parallel_state.vllm_parallel_state._PCP.world_size == 2
    assert parallel_state._DP.world_size == 2
    assert parallel_state.vllm_parallel_state._DP is not parallel_state._DP
    assert parallel_state.vllm_parallel_state._DP.world_size == 4
    assert parallel_state.vllm_parallel_state._EP.world_size == 16
    assert parallel_state.vllm_parallel_state._TP.world_size == 2
    assert parallel_state._PP.world_size == 2

    assert parallel_state.vllm_parallel_state._PCP.device_communicator is not None
    assert parallel_state.vllm_parallel_state._DP.device_communicator is not None
    assert parallel_state.vllm_parallel_state._EP.device_communicator is not None
    # vLLM 0.27's MoE oracle asserts the EP communicator has an all2all
    # manager, which vLLM only builds for groups created with use_all2all.
    assert parallel_state.vllm_parallel_state._EP.use_all2all is True
    assert hasattr(parallel_state.vllm_parallel_state._PCP, "reduce_scatter")
    assert hasattr(parallel_state.vllm_parallel_state._DP, "reduce_scatter")
    assert hasattr(parallel_state.vllm_parallel_state._EP, "reduce_scatter")

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

    vllm_group_names = [group.parallel_mode for group in created_groups if group.parallel_mode.startswith("vllm_")]
    assert vllm_group_names == ["vllm_pcp", "vllm_tp", "vllm_dp", "vllm_ep"]
    ep_groups = [group.local_group for group in created_groups if group.parallel_mode == "vllm_ep"]
    assert ep_groups == [parallel_state.vllm_parallel_state._EP.local_group]


@pytest.mark.cpu
@pytest.mark.core_model
@pytest.mark.parametrize(
    ("fully_shard_degree", "error_message"),
    [(0, "fully_shard_degree must be positive"), (3, "must be divisible by fully_shard_degree")],
)
def test_invalid_hsdp_shard_size_fails_before_group_creation(monkeypatch, fully_shard_degree: int, error_message: str):
    """Reject invalid HSDP shard sizes before allocating any process groups."""
    world_size = 4
    created_groups: list[object] = []

    def fail_if_group_created(*args, **kwargs):
        del args, kwargs
        created_groups.append(object())
        raise AssertionError("a process group was created before HSDP validation")

    fake_world_group = SimpleNamespace(
        rank_in_group=0,
        local_rank=0,
        device_group=object(),
    )
    monkeypatch.setattr(parallel_state.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: fake_world_group)
    monkeypatch.setattr(parallel_state, "init_model_parallel_group", fail_if_group_created)

    for name in ("_DP", "_CFG", "_SP", "_PP", "_FS", "_HSDP_REPLICATE", "_EXPERT_PARALLEL_GROUP_RANKS"):
        monkeypatch.setattr(parallel_state, name, None)
    for name in ("_TP", "_PCP", "_DP", "_EP", "_PP"):
        monkeypatch.setattr(parallel_state.vllm_parallel_state, name, None, raising=False)

    with pytest.raises(ValueError, match=error_message):
        parallel_state.initialize_model_parallel(
            fully_shard_degree=fully_shard_degree,
            use_hsdp=True,
            backend="gloo",
        )

    assert created_groups == []


@pytest.mark.cpu
@pytest.mark.core_model
def test_cfg_parallel_keeps_diffusion_dp_without_ep(monkeypatch):
    """Without EP, keep diffusion DP and retain rank metadata for MC2."""
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

    for name in ("_DP", "_CFG", "_SP", "_PP", "_EXPERT_PARALLEL_GROUP_RANKS"):
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
    assert parallel_state._EXPERT_PARALLEL_GROUP_RANKS == [
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]


@pytest.mark.cpu
@pytest.mark.core_model
def test_destroy_model_parallel_clears_vllm_pipeline_group(monkeypatch):
    """A destroyed diffusion PP group must not block the next initialization."""

    class _DestroyableGroup:
        def __init__(self) -> None:
            self.destroy_calls = 0

        def destroy(self) -> None:
            self.destroy_calls += 1

    pipeline_group = _DestroyableGroup()
    for name in ("_DP", "_CFG", "_SP"):
        monkeypatch.setattr(parallel_state, name, None)
    monkeypatch.setattr(parallel_state, "_PP", pipeline_group)
    monkeypatch.setattr(parallel_state, "_EXPERT_PARALLEL_GROUP_RANKS", None)
    for name in ("_DP", "_PCP", "_TP", "_EP"):
        monkeypatch.setattr(parallel_state.vllm_parallel_state, name, None, raising=False)
    monkeypatch.setattr(parallel_state.vllm_parallel_state, "_PP", pipeline_group, raising=False)

    parallel_state.destroy_model_parallel()

    assert pipeline_group.destroy_calls == 1
    assert parallel_state.vllm_parallel_state._PP is None


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

    for name in ("_DP", "_CFG", "_SP", "_PP", "_EXPERT_PARALLEL_GROUP_RANKS"):
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


@pytest.mark.cpu
@pytest.mark.core_model
def test_init_vllm_group_forwards_use_all2all_only_when_supported(monkeypatch):
    """use_all2all reaches vLLM only on versions whose API accepts it.

    vLLM 0.27 builds the EP all2all manager only for groups created with
    use_all2all=True (its MoE oracle then asserts the manager exists —
    build 2953, HunyuanImage3-DIT accuracy). Older vLLM has no such kwarg,
    so forwarding it unconditionally would TypeError at init.
    """
    received = {}

    def modern(group_ranks, local_rank, backend, group_name=None, use_device_communicator=True, use_all2all=False):
        received["modern"] = use_all2all
        return object()

    monkeypatch.setattr(parallel_state.vllm_parallel_state, "init_model_parallel_group", modern)
    parallel_state.init_vllm_model_parallel_group([[0]], 0, "gloo", "ep", use_all2all=True)
    assert received["modern"] is True

    parallel_state.init_vllm_model_parallel_group([[0]], 0, "gloo", "tp")
    assert received["modern"] is False

    def legacy(group_ranks, local_rank, backend, group_name=None, use_device_communicator=True):
        received["legacy"] = True
        return object()

    monkeypatch.setattr(parallel_state.vllm_parallel_state, "init_model_parallel_group", legacy)
    parallel_state.init_vllm_model_parallel_group([[0]], 0, "gloo", "ep", use_all2all=True)
    assert received["legacy"] is True
