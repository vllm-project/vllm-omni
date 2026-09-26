# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

import vllm_omni.diffusion.models.magi2.fused_moe_kernels as kernels

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
