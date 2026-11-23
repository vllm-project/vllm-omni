# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.platforms.interface import DeviceCapability

import vllm_omni.diffusion.models.magi2.layers.cuda.fused_moe_kernels as kernels

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 16,
    "num_warps": 16,
    "num_stages": 1,
}


class _LaunchProbe:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def _inputs(*, gate_up: bool):
    route_weights = torch.ones((2, 6), dtype=torch.float32)
    if gate_up:
        A = torch.zeros((2, 4), dtype=torch.bfloat16)
        B = torch.zeros((2, 8, 4), dtype=torch.bfloat16)
        C = torch.empty((12, 4), dtype=torch.bfloat16)
        top_k = 6
    else:
        A = torch.zeros((12, 4), dtype=torch.bfloat16)
        B = torch.zeros((2, 4, 4), dtype=torch.bfloat16)
        C = torch.empty((2, 6, 4), dtype=torch.bfloat16)
        top_k = 1
    sorted_ids = torch.zeros(128, dtype=torch.int32)
    expert_ids = torch.zeros(1, dtype=torch.int32)
    padded = torch.tensor([128], dtype=torch.int32)
    return A, B, C, route_weights, sorted_ids, expert_ids, padded, top_k


@pytest.mark.parametrize(
    ("gate_up", "fuse_swiglu", "mul_weight"),
    [(True, True, False), (False, False, True)],
)
def test_bf16_launch_contract(monkeypatch, gate_up, fuse_swiglu, mul_weight):
    probe = _LaunchProbe()
    monkeypatch.setattr(kernels, "_fused_moe_bf16_kernel", probe)
    args = _inputs(gate_up=gate_up)
    kernels.invoke_fused_moe_bf16(*args[:-1], top_k=args[-1], config=CONFIG, fuse_swiglu=fuse_swiglu)
    assert len(probe.calls) == 1
    _, launch_args, launch_kwargs = probe.calls[0]
    assert launch_args[0].dtype == torch.bfloat16
    assert launch_kwargs["compute_type"] == kernels.tl.bfloat16
    assert launch_kwargs["FUSE_SWIGLU"] is fuse_swiglu
    assert launch_kwargs["MUL_ROUTED_WEIGHT"] is mul_weight
    assert launch_kwargs["SWIGLU_ALPHA"] == (1.702 if fuse_swiglu else 0.0)
    assert launch_kwargs["SWIGLU_LIMIT"] == (7.0 if fuse_swiglu else 0.0)


def test_rejects_non_bf16_activation(monkeypatch):
    args = list(_inputs(gate_up=True))
    args[0] = args[0].float()
    with pytest.raises(ValueError, match="requires BF16"):
        kernels.invoke_fused_moe_bf16(*args[:-1], top_k=args[-1], config=CONFIG, fuse_swiglu=True)


@pytest.mark.parametrize(
    "change",
    [
        {"BLOCK_SIZE_N": 32},
        {"num_warps": 3},
        {"num_stages": 5},
        {"unexpected": 1},
    ],
)
def test_rejects_invalid_launch_config(monkeypatch, change):
    config = dict(CONFIG)
    config.update(change)
    args = _inputs(gate_up=True)
    with pytest.raises(ValueError, match="invalid MAGI-2 BF16 MoE launch config"):
        kernels.invoke_fused_moe_bf16(*args[:-1], top_k=args[-1], config=config, fuse_swiglu=True)


def test_rejects_incompatible_swiglu_output():
    args = list(_inputs(gate_up=True))
    args[2] = torch.empty((12, 8), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="output"):
        kernels.invoke_fused_moe_bf16(*args[:-1], top_k=args[-1], config=CONFIG, fuse_swiglu=True)


@pytest.mark.parametrize("gate_up", [True, False])
def test_rejects_wrong_routing_top_k(gate_up):
    args = _inputs(gate_up=gate_up)
    with pytest.raises(ValueError, match="routing layout"):
        kernels.invoke_fused_moe_bf16(*args[:-1], top_k=2, config=CONFIG, fuse_swiglu=gate_up)


@pytest.mark.parametrize(
    ("argument", "replacement", "message"),
    [
        (4, torch.zeros(128, dtype=torch.float32), "sorted_token_ids"),
        (5, torch.zeros(1, dtype=torch.float32), "expert_ids"),
        (6, torch.tensor([128.0]), "num_tokens_post_padded"),
        (6, torch.tensor([128, 256]), "num_tokens_post_padded"),
        (4, torch.zeros(127, dtype=torch.int32), "block aligned"),
        (5, torch.empty(0, dtype=torch.int32), "one expert per padded token block"),
    ],
)
def test_rejects_invalid_route_metadata(argument, replacement, message):
    args = list(_inputs(gate_up=True))
    args[argument] = replacement
    with pytest.raises(ValueError, match=message):
        kernels.invoke_fused_moe_bf16(*args[:-1], top_k=args[-1], config=CONFIG, fuse_swiglu=True)


@pytest.fixture
def expert_launch(monkeypatch):
    class ResourceError(Exception):
        pass

    # vLLM's CPU-only Triton placeholder has no runtime error module.
    monkeypatch.setattr(
        kernels,
        "triton",
        SimpleNamespace(runtime=SimpleNamespace(errors=SimpleNamespace(OutOfResources=ResourceError))),
    )
    monkeypatch.setattr(kernels, "_RESOLVED_BLOCK_CONFIG", {})
    monkeypatch.setattr(
        kernels.current_omni_platform, "get_device_capability", lambda device_id: DeviceCapability(10, 0)
    )
    probe = _LaunchProbe()
    monkeypatch.setattr(kernels, "_mh_moe_kernel", probe)
    return probe, ResourceError


def _expert_inputs(tokens=65, d_head=256, d_expert=1280, dtype=torch.bfloat16):
    return (
        torch.zeros(tokens, 1, d_head, dtype=dtype),
        torch.arange(tokens, dtype=torch.int32),
        torch.ones(tokens),
        torch.tensor([0, tokens]),
        torch.zeros(1, d_head, d_expert, dtype=dtype),
        torch.zeros(1, d_head, d_expert, dtype=dtype),
        torch.zeros(1, d_expert, d_head, dtype=dtype),
    )


@pytest.mark.parametrize("deterministic", [False, True])
def test_expert_tile_fallback_rebuilds_metadata_and_caches(monkeypatch, expert_launch, deterministic):
    probe, resource_error = expert_launch

    class OversizedTile:
        def __getitem__(self, grid):
            record = probe[grid]

            def launch(*args, **kwargs):
                record(*args, **kwargs)
                if args[29] == 128:
                    raise resource_error("tensor memory requires 528 columns, limit 512")
                args[4].fill_(1)

            return launch

    monkeypatch.setattr(kernels, "_mh_moe_kernel", OversizedTile())
    for tokens in (65, 129):
        output = kernels.triton_mh_moe_forward(*_expert_inputs(tokens), deterministic=deterministic)
        torch.testing.assert_close(output, torch.ones_like(output))

    assert [args[29] for _, args, _ in probe.calls] == [128, 64, 64]
    assert [grid for grid, _, _ in probe.calls] == [(2,), (3,), (4,)]
    assert [args[8].tolist() for _, args, _ in probe.calls] == [[0, 1], [0, 2], [0, 3]]
    key = (torch.device("cpu"), torch.bfloat16, 256, 1280, deterministic)
    assert kernels._RESOLVED_BLOCK_CONFIG[key] == (64, 64, 32, 2, 8)


@pytest.mark.parametrize(
    ("d_head", "d_expert", "dtype", "deterministic"),
    [(256, 1280, torch.bfloat16, True), (64, 96, torch.bfloat16, False), (256, 1280, torch.float16, False)],
)
def test_expert_tile_cache_is_scoped(expert_launch, d_head, d_expert, dtype, deterministic):
    probe, _ = expert_launch
    kernels._RESOLVED_BLOCK_CONFIG[(torch.device("cpu"), torch.bfloat16, 256, 1280, False)] = (64, 64, 32, 2, 8)
    kernels.triton_mh_moe_forward(*_expert_inputs(1, d_head, d_expert, dtype), deterministic=deterministic)
    assert probe.calls[0][1][29] == 128


def test_expert_tile_cache_is_device_specific(expert_launch):
    probe, _ = expert_launch
    kernels._RESOLVED_BLOCK_CONFIG[(torch.device("cuda", 1), torch.bfloat16, 256, 1280, False)] = (64, 64, 32, 2, 8)
    kernels.triton_mh_moe_forward(*_expert_inputs(1))
    assert probe.calls[0][1][29] == 128


@pytest.mark.parametrize("cached", [False, True])
def test_expert_tile_exhaustion_propagates(monkeypatch, expert_launch, cached):
    probe, resource_error = expert_launch
    cache_key = (torch.device("cpu"), torch.bfloat16, 256, 1280, False)
    if cached:
        kernels._RESOLVED_BLOCK_CONFIG[cache_key] = (64, 64, 32, 2, 8)

    class UnsupportedTiles:
        def __getitem__(self, grid):
            record = probe[grid]

            def launch(*args, **kwargs):
                record(*args, **kwargs)
                raise resource_error("insufficient resources")

            return launch

    monkeypatch.setattr(kernels, "_mh_moe_kernel", UnsupportedTiles())
    with pytest.raises(resource_error, match="insufficient resources"):
        kernels.triton_mh_moe_forward(*_expert_inputs(1))
    assert [kwargs["num_warps"] for _, _, kwargs in probe.calls] == ([8, 4] if cached else [8, 8, 4])
    assert cache_key not in kernels._RESOLVED_BLOCK_CONFIG


def test_expert_tile_rejects_cached_tile_and_retries(monkeypatch, expert_launch):
    probe, resource_error = expert_launch
    cache_key = (torch.device("cpu"), torch.bfloat16, 256, 1280, False)
    kernels._RESOLVED_BLOCK_CONFIG[cache_key] = (128, 64, 32, 2, 8)

    class RejectCachedThenSucceed:
        def __getitem__(self, grid):
            record = probe[grid]

            def launch(*args, **kwargs):
                record(*args, **kwargs)
                if args[29] == 128:
                    raise resource_error("tensor memory requires 528 columns, limit 512")
                args[4].fill_(1)

            return launch

    monkeypatch.setattr(kernels, "_mh_moe_kernel", RejectCachedThenSucceed())
    output = kernels.triton_mh_moe_forward(*_expert_inputs(1))
    torch.testing.assert_close(output, torch.ones_like(output))
    repeated = kernels.triton_mh_moe_forward(*_expert_inputs(1))
    torch.testing.assert_close(repeated, torch.ones_like(repeated))

    assert [args[29] for _, args, _ in probe.calls] == [128, 64, 64]
    assert kernels._RESOLVED_BLOCK_CONFIG[cache_key] == (64, 64, 32, 2, 8)


def test_expert_tile_propagates_other_errors(monkeypatch, expert_launch):
    class BrokenKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise RuntimeError("compiler bug")

            return launch

    monkeypatch.setattr(kernels, "_mh_moe_kernel", BrokenKernel())
    with pytest.raises(RuntimeError, match="compiler bug"):
        kernels.triton_mh_moe_forward(*_expert_inputs(1))
    assert not kernels._RESOLVED_BLOCK_CONFIG


@pytest.mark.parametrize("major", [None, 8, 9, 10])
def test_expert_tile_candidates_use_requested_device(monkeypatch, major):
    def capability(device_id):
        assert device_id == 3
        return None if major is None else DeviceCapability(major, 0)

    monkeypatch.setattr(kernels.current_omni_platform, "get_device_capability", capability)
    candidates = kernels._block_config_candidates(3)
    assert candidates[-1] == (64, 64, 32, 2, 4)
    assert len(candidates) == (3 if major == 10 else 1)
