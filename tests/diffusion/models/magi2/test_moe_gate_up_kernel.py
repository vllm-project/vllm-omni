# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gate/up GEMM with a register SwiGLU7 epilogue, against the shipped kernel.

CPU kernel cases run under the Triton interpreter (``TRITON_INTERPRET=1``; on
hosts without a GPU also ``CUDA_VISIBLE_DEVICES=`` so vLLM keeps Triton
enabled). The interpreter's ``tl.dot`` does not decode BF16 operands, so those
cases launch the kernels directly on FP32 tensors and compare with a
tolerance. Device cases go through ``invoke_fused_moe_bf16`` and require
bit-identical output.
"""

import os

import pytest
import torch

import vllm_omni.diffusion.models.magi2.fused_moe_kernels as kernels
import vllm_omni.diffusion.models.magi2.mh_moe as moe
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

requires_triton = pytest.mark.skipif(not kernels.HAS_TRITON, reason="Triton is unavailable")
requires_interpreter = pytest.mark.skipif(
    not kernels.HAS_TRITON or os.environ.get("TRITON_INTERPRET") != "1",
    reason="CPU kernel launches need TRITON_INTERPRET=1",
)

SHIPPED_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 16,
    "num_warps": 16,
    "num_stages": 1,
}


def _config(block_m=128, block_n=128, block_k=64, num_warps=16, num_stages=1):
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 16,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _config_id(config):
    return "bm{BLOCK_SIZE_M}-bn{BLOCK_SIZE_N}-bk{BLOCK_SIZE_K}-w{num_warps}-s{num_stages}".format(**config)


class _LaunchProbe:
    def __init__(self):
        self.calls = []
        self.handle = object()

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return self.handle

        return launch


def _gate_up_problem(num_tokens, top_k, num_experts, hidden, intermediate, *, dtype, device="cpu", seed=0):
    """Skewed routes: expert 0 takes every token, expert ``4 % num_experts`` every fifth.

    The other routes go to expert 1 and the remaining experts stay empty. The
    weight scale drives a few percent of gate/up values past the SwiGLU7 clamps.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, hidden, generator=generator).to(dtype)
    scale = 3.0 / hidden**0.5
    w13 = (torch.randn(num_experts, 2 * intermediate, hidden, generator=generator) * scale).to(dtype)
    route_ids = torch.randint(0, 2, (num_tokens, top_k), generator=generator)
    route_ids[:, 0] = 0
    if top_k > 1:
        route_ids[::5, 1] = 4 % num_experts
    weights = torch.rand(num_tokens, top_k, generator=generator)
    return x.to(device), w13.to(device), route_ids.to(device), weights.to(device)


def _nan_padded(tensor, pad=16):
    """View ``tensor`` inside NaN-padded rows: an unmasked K-tail read poisons the output."""
    storage = torch.full((*tensor.shape[:-1], tensor.shape[-1] + pad), float("nan"), dtype=tensor.dtype)
    storage[..., : tensor.shape[-1]] = tensor
    return storage[..., : tensor.shape[-1]]


def _launch(variant, a, b, c, weights, metadata, *, top_k, config, fuse_swiglu=True):
    """Launch a kernel: BF16 through its launcher, FP32 directly with the launcher's arguments."""
    if a.dtype == torch.bfloat16:
        args = (a, b, c, weights, *metadata)
        if variant == "register":
            return kernels.invoke_fused_moe_bf16(
                *args, top_k=top_k, config=config, fuse_swiglu=True, split_gate_up=True
            )
        return kernels.invoke_fused_moe_bf16(*args, top_k=top_k, config=config, fuse_swiglu=fuse_swiglu)
    sorted_ids = metadata[0]
    grid = (
        kernels.triton.cdiv(sorted_ids.numel(), config["BLOCK_SIZE_M"])
        * kernels.triton.cdiv(b.shape[1], config["BLOCK_SIZE_N"]),
    )
    shape_and_strides = (
        b.shape[1],
        b.shape[2],
        sorted_ids.numel(),
        weights.numel(),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(-2),
        c.stride(-1),
    )
    even_ks = b.shape[2] % config["BLOCK_SIZE_K"] == 0
    if variant == "register":
        return kernels._fused_moe_bf16_gate_up_kernel[grid](
            a,
            b,
            c,
            *metadata,
            *shape_and_strides,
            top_k=top_k,
            compute_type=kernels.tl.float32,
            even_Ks=even_ks,
            SWIGLU_ALPHA=1.702,
            SWIGLU_LIMIT=7.0,
            **config,
        )
    return kernels._fused_moe_bf16_kernel[grid](
        a,
        b,
        c,
        weights,
        *metadata,
        *shape_and_strides,
        MUL_ROUTED_WEIGHT=not fuse_swiglu,
        top_k=top_k,
        compute_type=kernels.tl.float32,
        even_Ks=even_ks,
        FUSE_SWIGLU=fuse_swiglu,
        SWIGLU_ALPHA=1.702 if fuse_swiglu else 0.0,
        SWIGLU_LIMIT=7.0 if fuse_swiglu else 0.0,
        **config,
    )


def _gate_up(variant, x, w13, weights, metadata, config):
    out = torch.full((weights.numel(), w13.shape[1] // 2), float("nan"), dtype=x.dtype, device=x.device)
    _launch(variant, x, w13, out, weights, metadata, top_k=weights.shape[1], config=config)
    return out


def _down(hidden, w_down, weights, metadata, config):
    out = torch.full((*weights.shape, w_down.shape[2]), float("nan"), dtype=hidden.dtype, device=hidden.device)
    _launch(
        "shipped", hidden, w_down.transpose(1, 2), out, weights, metadata, top_k=1, config=config, fuse_swiglu=False
    )
    return out


def _torch_gate_up(x, w13, route_ids):
    top_k = route_ids.shape[1]
    gemm = torch.einsum("rk,rnk->rn", x.float().repeat_interleave(top_k, 0), w13[route_ids.reshape(-1)].float())
    gate = gemm[:, 0::2].clamp(max=7.0)
    up = gemm[:, 1::2].clamp(-7.0, 7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(x.dtype)


def _launch_inputs(*, block_m=128, intermediate=4, hidden=4):
    """Two tokens with top-6 routes over two experts; metadata for one block."""
    x = torch.zeros((2, hidden), dtype=torch.bfloat16)
    w13 = torch.zeros((2, 2 * intermediate, hidden), dtype=torch.bfloat16)
    out = torch.empty((12, intermediate), dtype=torch.bfloat16)
    weights = torch.ones((2, 6), dtype=torch.float32)
    sorted_ids, expert_ids, num_padded = moe._allocate_bf16_route_buffers(12, 1, block_m, torch.device("cpu"))
    return [x, w13, out, weights, sorted_ids, expert_ids, num_padded]


# Launch contract and validation.


@pytest.mark.cpu
@requires_triton
@pytest.mark.parametrize("block_m", [64, 128])
@pytest.mark.parametrize(("hidden", "even_ks"), [(256, True), (80, False)])
def test_gate_up_launch_contract(monkeypatch, block_m, hidden, even_ks):
    probe, shipped = _LaunchProbe(), _LaunchProbe()
    monkeypatch.setattr(kernels, "_fused_moe_bf16_gate_up_kernel", probe)
    monkeypatch.setattr(kernels, "_fused_moe_bf16_kernel", shipped)
    args = _launch_inputs(block_m=block_m, intermediate=96, hidden=hidden)
    config = _config(block_m=block_m, block_n=256)

    handle = kernels.invoke_fused_moe_bf16(*args, top_k=6, config=config, fuse_swiglu=True, split_gate_up=True)

    assert handle is probe.handle
    assert len(probe.calls) == 1 and not shipped.calls
    (grid, launch_args, launch_kwargs) = probe.calls[0]
    x, w13, out, weights, sorted_ids, expert_ids, num_padded = args
    assert grid(config) == (sorted_ids.numel() // block_m * kernels.triton.cdiv(192, 256),)
    assert all(a is b for a, b in zip(launch_args[:6], (x, w13, out, sorted_ids, expert_ids, num_padded)))
    assert launch_args[6:10] == (192, hidden, sorted_ids.numel(), 12)
    assert launch_args[10:] == (hidden, 1, 192 * hidden, 1, hidden, 96, 1)
    assert launch_kwargs == {
        "top_k": 6,
        "compute_type": kernels.tl.bfloat16,
        "even_Ks": even_ks,
        "SWIGLU_ALPHA": 1.702,
        "SWIGLU_LIMIT": 7.0,
        **config,
    }


@pytest.mark.cpu
@requires_triton
@pytest.mark.parametrize("fuse_swiglu", [True, False])
def test_shipped_launcher_returns_launch_result_and_accepts_block_m_64(monkeypatch, fuse_swiglu):
    probe = _LaunchProbe()
    monkeypatch.setattr(kernels, "_fused_moe_bf16_kernel", probe)
    x, weight, out, weights, sorted_ids, expert_ids, num_padded = _launch_inputs(block_m=64)
    if not fuse_swiglu:
        # Down GEMM: one activation row per route, written as [tokens, top_k, hidden].
        x = torch.zeros((12, 4), dtype=torch.bfloat16)
        weight = torch.zeros((2, 4, 4), dtype=torch.bfloat16)
        out = torch.empty((2, 6, 4), dtype=torch.bfloat16)
    handle = kernels.invoke_fused_moe_bf16(
        x,
        weight,
        out,
        weights,
        sorted_ids,
        expert_ids,
        num_padded,
        top_k=6 if fuse_swiglu else 1,
        config=_config(block_m=64),
        fuse_swiglu=fuse_swiglu,
    )
    assert handle is probe.handle
    assert probe.calls[0][2]["BLOCK_SIZE_M"] == 64


@pytest.mark.cpu
@requires_triton
@pytest.mark.parametrize(
    "config",
    [
        SHIPPED_CONFIG,
        _config(block_m=64, block_n=512, block_k=128, num_warps=4, num_stages=3),
        _config(block_m=128, block_n=128, block_k=32, num_warps=8, num_stages=2),
    ],
)
def test_gate_up_accepts_supported_configs(monkeypatch, config):
    probe = _LaunchProbe()
    monkeypatch.setattr(kernels, "_fused_moe_bf16_gate_up_kernel", probe)
    args = _launch_inputs(block_m=config["BLOCK_SIZE_M"])
    kernels.invoke_fused_moe_bf16(*args, top_k=6, config=config, fuse_swiglu=True, split_gate_up=True)
    assert len(probe.calls) == 1


@pytest.mark.cpu
@pytest.mark.parametrize(
    "change",
    [
        {"BLOCK_SIZE_M": 32},
        {"BLOCK_SIZE_N": 64},
        {"BLOCK_SIZE_K": 16},
        {"GROUP_SIZE_M": 8},
        {"num_warps": 32},
        {"num_stages": 4},
        {"num_stages": 1.0},
        {"unexpected": 1},
    ],
)
def test_gate_up_rejects_unsupported_config(change):
    config = dict(SHIPPED_CONFIG)
    config.update(change)
    with pytest.raises(ValueError, match="invalid MAGI-2 BF16 MoE launch config"):
        kernels.invoke_fused_moe_bf16(*_launch_inputs(), top_k=6, config=config, fuse_swiglu=True, split_gate_up=True)


@pytest.mark.cpu
def test_gate_up_rejects_missing_config_key():
    config = dict(SHIPPED_CONFIG)
    del config["num_stages"]
    with pytest.raises(ValueError, match="invalid MAGI-2 BF16 MoE launch config"):
        kernels.invoke_fused_moe_bf16(*_launch_inputs(), top_k=6, config=config, fuse_swiglu=True, split_gate_up=True)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("index", "replacement", "top_k", "message"),
    [
        (0, torch.zeros((2, 4), dtype=torch.float16), 6, "requires BF16"),
        (1, torch.zeros((2, 7, 4), dtype=torch.bfloat16), 6, "even number of interleaved gate/up rows"),
        (2, torch.empty((12, 8), dtype=torch.bfloat16), 6, "output"),
        (2, torch.empty((4, 12), dtype=torch.bfloat16).t(), 6, "output"),
        (3, torch.ones((2, 6), dtype=torch.int64), 6, "routing weights"),
        (4, torch.zeros(128, dtype=torch.float32), 6, "sorted_token_ids"),
        (6, torch.tensor([128, 256]), 6, "num_tokens_post_padded"),
        (None, None, 1, "routing layout"),
    ],
)
def test_gate_up_rejects_invalid_arguments(index, replacement, top_k, message):
    args = _launch_inputs()
    if index is not None:
        args[index] = replacement
    with pytest.raises(ValueError, match=message):
        kernels.invoke_fused_moe_bf16(*args, top_k=top_k, config=SHIPPED_CONFIG, fuse_swiglu=True, split_gate_up=True)


@pytest.mark.cpu
@requires_triton
def test_split_gate_up_requires_fuse_swiglu(monkeypatch):
    probes = _LaunchProbe(), _LaunchProbe()
    monkeypatch.setattr(kernels, "_fused_moe_bf16_gate_up_kernel", probes[0])
    monkeypatch.setattr(kernels, "_fused_moe_bf16_kernel", probes[1])
    # A valid down-GEMM launch, so only split_gate_up can reject it.
    down_args = [
        torch.zeros((12, 4), dtype=torch.bfloat16),
        torch.zeros((2, 4, 4), dtype=torch.bfloat16),
        torch.empty((2, 6, 4), dtype=torch.bfloat16),
        *_launch_inputs()[3:],
    ]
    with pytest.raises(ValueError, match="split_gate_up requires fuse_swiglu=True"):
        kernels.invoke_fused_moe_bf16(*down_args, top_k=1, config=SHIPPED_CONFIG, fuse_swiglu=False, split_gate_up=True)
    assert not probes[0].calls and not probes[1].calls
    kernels.invoke_fused_moe_bf16(*down_args, top_k=1, config=SHIPPED_CONFIG, fuse_swiglu=False)
    assert len(probes[1].calls) == 1


@pytest.mark.cpu
@requires_triton
@pytest.mark.parametrize("fuse_swiglu", [True, False])
@pytest.mark.parametrize("split_kwargs", [{}, {"split_gate_up": False}], ids=["default", "explicit"])
def test_unsplit_launch_uses_shipped_kernel(monkeypatch, fuse_swiglu, split_kwargs):
    gate_up, shipped = _LaunchProbe(), _LaunchProbe()
    monkeypatch.setattr(kernels, "_fused_moe_bf16_gate_up_kernel", gate_up)
    monkeypatch.setattr(kernels, "_fused_moe_bf16_kernel", shipped)
    args = _launch_inputs(intermediate=96, hidden=80)
    top_k = 6
    if not fuse_swiglu:
        # Down GEMM: one activation row per route, written as [tokens, top_k, hidden].
        args[:3] = (
            torch.zeros((12, 96), dtype=torch.bfloat16),
            torch.zeros((2, 80, 96), dtype=torch.bfloat16),
            torch.empty((2, 6, 80), dtype=torch.bfloat16),
        )
        top_k = 1
    # Legal for the shipped kernel only, not for the split gate/up kernel.
    config = _config(block_n=64, block_k=32, num_warps=32, num_stages=4)

    handle = kernels.invoke_fused_moe_bf16(*args, top_k=top_k, config=config, fuse_swiglu=fuse_swiglu, **split_kwargs)

    assert handle is shipped.handle
    assert not gate_up.calls and len(shipped.calls) == 1
    (grid, launch_args, launch_kwargs) = shipped.calls[0]
    a, b, c, weights, sorted_ids, expert_ids, num_padded = args
    assert grid(config) == (sorted_ids.numel() // 128 * kernels.triton.cdiv(b.shape[1], 64),)
    assert all(x is y for x, y in zip(launch_args[:7], (a, b, c, weights, sorted_ids, expert_ids, num_padded)))
    assert launch_args[7:] == (
        b.shape[1],
        b.shape[2],
        sorted_ids.numel(),
        12,
        a.shape[1],
        1,
        b.shape[1] * b.shape[2],
        1,
        b.shape[2],
        c.shape[-1],
        1,
    )
    assert launch_kwargs == {
        "MUL_ROUTED_WEIGHT": not fuse_swiglu,
        "top_k": top_k,
        "compute_type": kernels.tl.bfloat16,
        "even_Ks": not fuse_swiglu,  # K 80 leaves a tail for BLOCK_SIZE_K 32, K 96 does not.
        "FUSE_SWIGLU": fuse_swiglu,
        "SWIGLU_ALPHA": 1.702 if fuse_swiglu else 0.0,
        "SWIGLU_LIMIT": 7.0 if fuse_swiglu else 0.0,
        **config,
    }


@pytest.mark.cpu
@pytest.mark.parametrize(("metadata_block", "launch_block"), [(64, 128), (128, 64)])
@pytest.mark.parametrize("variant", ["register", "shipped"])
def test_launchers_reject_route_metadata_for_another_block_size(variant, metadata_block, launch_block):
    """64-row metadata read with 128-row tiles would mix experts in one tile."""
    args = _launch_inputs(block_m=metadata_block)
    config = _config(block_m=launch_block)
    with pytest.raises(ValueError, match="one expert per padded token block|block aligned"):
        if variant == "register":
            kernels.invoke_fused_moe_bf16(*args, top_k=6, config=config, fuse_swiglu=True, split_gate_up=True)
        else:
            kernels.invoke_fused_moe_bf16(*args, top_k=6, config=config, fuse_swiglu=True)


# BLOCK_SIZE_M=64 route metadata.


@pytest.mark.cpu
@pytest.mark.parametrize("preallocated", [False, True])
def test_align_bf16_routes_block_m_64(preallocated):
    num_experts = 9
    # Expert 2 spans three 64-row blocks, expert 7 has a single route and
    # experts 0, 1, 3, 4, 6 and 8 stay empty.
    route_ids = torch.tensor([2] * 150 + [5] * 64 + [7] + [5] * 3).reshape(-1, 2)
    route_ids = route_ids[torch.randperm(route_ids.shape[0], generator=torch.Generator().manual_seed(3))]
    route_count = route_ids.numel()
    buffers = moe._allocate_bf16_route_buffers(route_count, num_experts, 64, route_ids.device)
    capacity = buffers[0].numel()
    assert capacity % 64 == 0 and capacity >= route_count + num_experts * 63
    assert buffers[1].numel() == capacity // 64 and buffers[2].numel() == 1

    sorted_ids, expert_ids, num_padded = moe._align_bf16_routes(
        route_ids, num_experts, 64, buffers if preallocated else None
    )
    if preallocated:
        assert all(a is b for a, b in zip((sorted_ids, expert_ids, num_padded), buffers))
    assert sorted_ids.numel() == capacity and expert_ids.numel() == capacity // 64
    counts = torch.bincount(route_ids.reshape(-1), minlength=num_experts)
    live_blocks = int(((counts + 63) // 64).sum())
    assert live_blocks == 3 + 2 + 1
    assert num_padded.tolist() == [live_blocks * 64]
    assert expert_ids[:live_blocks].tolist() == [2, 2, 2, 5, 5, 7]
    flat_ids = route_ids.reshape(-1)
    live = sorted_ids[: live_blocks * 64].view(live_blocks, 64)
    for block, expert in enumerate(expert_ids[:live_blocks].tolist()):
        routes = live[block][live[block] < route_count].long()
        assert torch.all(flat_ids[routes] == expert)
    valid = live[live < route_count]
    assert sorted(valid.tolist()) == list(range(route_count))
    assert torch.all(live[live >= route_count] == route_count)
    assert torch.all(sorted_ids[live_blocks * 64 :] == route_count)


# Kernel parity under the Triton interpreter (FP32 storage).

_PROBLEMS = {
    # tokens, top_k, experts, hidden (K), intermediate (N / 2)
    "k_tail_n_tail": (37, 3, 6, 80, 96),
    "even_k_exact_n": (50, 2, 5, 128, 64),
    "n_narrower_than_tile": (3, 6, 8, 64, 32),
}
_CPU_CONFIGS = [
    _config(block_m=128, block_n=128, block_k=64),
    _config(block_m=64, block_n=128, block_k=64),
    _config(block_m=64, block_n=256, block_k=32, num_warps=8, num_stages=2),
    _config(block_m=128, block_n=512, block_k=128, num_warps=4, num_stages=3),
    _config(block_m=64, block_n=512, block_k=32),
]
_FP32_TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}


def _assert_same_writes_and_close(actual, expected):
    assert torch.equal(actual.isnan(), expected.isnan())
    torch.testing.assert_close(actual, expected, equal_nan=True, **_FP32_TOLERANCE)


@pytest.mark.cpu
@requires_interpreter
@pytest.mark.parametrize("problem", list(_PROBLEMS))
@pytest.mark.parametrize("config", _CPU_CONFIGS, ids=_config_id)
def test_gate_up_kernel_matches_shipped_kernel_on_interpreter(problem, config):
    num_tokens, top_k, num_experts, hidden, intermediate = _PROBLEMS[problem]
    x, w13, route_ids, weights = _gate_up_problem(
        num_tokens, top_k, num_experts, hidden, intermediate, dtype=torch.float32
    )
    x, w13 = _nan_padded(x), _nan_padded(w13)
    shipped_metadata = moe._align_bf16_routes(route_ids, num_experts, 128)
    metadata = moe._align_bf16_routes(route_ids, num_experts, config["BLOCK_SIZE_M"])
    shipped = _gate_up("shipped", x, w13, weights, shipped_metadata, SHIPPED_CONFIG)
    actual = _gate_up("register", x, w13, weights, metadata, config)

    assert not shipped.isnan().any() and not actual.isnan().any()
    _assert_same_writes_and_close(actual, shipped)
    expected = _torch_gate_up(x, w13, route_ids)
    torch.testing.assert_close(actual, expected, **_FP32_TOLERANCE)
    gate_up = torch.einsum("rk,rnk->rn", x.repeat_interleave(top_k, 0), w13[route_ids.reshape(-1)])
    assert (gate_up.abs() > 7).any(), "the problem must exercise the SwiGLU7 clamps"


def _padded_block_problem(block_m, *, dtype, device="cpu"):
    """Live blocks for experts 0, -1 and 1, then a dead block past the padded count."""
    x, w13, _, weights = _gate_up_problem(10, 2, 3, 64, 96, dtype=dtype, device=device)
    route_count = weights.numel()
    sorted_ids = torch.full((4 * block_m,), route_count, dtype=torch.int32)
    sorted_ids[:7] = torch.arange(7)
    sorted_ids[block_m : block_m + 6] = torch.arange(7, 13)
    sorted_ids[2 * block_m : 2 * block_m + 7] = torch.arange(13, 20)
    sorted_ids[3 * block_m] = 0
    # Expert 99 does not exist: reading it would fault or write garbage.
    expert_ids = torch.tensor([0, -1, 1, 99], dtype=torch.int32)
    num_padded = torch.tensor([3 * block_m], dtype=torch.int32)
    metadata = (sorted_ids.to(device), expert_ids.to(device), num_padded.to(device))
    skipped = torch.zeros(route_count, dtype=torch.bool, device=device)
    skipped[7:13] = True
    route_experts = torch.tensor([0] * 13 + [1] * 7, device=device).view(-1, 2)
    return x, w13, weights, metadata, skipped, route_experts


@pytest.mark.cpu
@requires_interpreter
@pytest.mark.parametrize("block_m", [64, 128])
def test_gate_up_kernel_skips_unowned_and_dead_blocks_on_interpreter(block_m):
    x, w13, weights, metadata, skipped, route_experts = _padded_block_problem(block_m, dtype=torch.float32)
    x, w13 = _nan_padded(x), _nan_padded(w13)
    shipped = _gate_up("shipped", x, w13, weights, metadata, _config(block_m=block_m))
    actual = _gate_up("register", x, w13, weights, metadata, _config(block_m=block_m, block_n=256, block_k=32))

    assert torch.all(actual[skipped].isnan()) and not actual[~skipped].isnan().any()
    _assert_same_writes_and_close(actual, shipped)
    expected = _torch_gate_up(x, w13, route_experts)
    torch.testing.assert_close(actual[~skipped], expected[~skipped], **_FP32_TOLERANCE)


@pytest.mark.cpu
@requires_interpreter
def test_shipped_kernel_runs_both_gemms_at_block_m_64_on_interpreter():
    num_tokens, top_k, num_experts, hidden, intermediate = _PROBLEMS["k_tail_n_tail"]
    x, w13, route_ids, weights = _gate_up_problem(
        num_tokens, top_k, num_experts, hidden, intermediate, dtype=torch.float32
    )
    w_down = torch.randn(num_experts, intermediate, hidden, generator=torch.Generator().manual_seed(1)) * 0.1
    results = {}
    for block_m in (64, 128):
        metadata = moe._align_bf16_routes(route_ids, num_experts, block_m)
        gate_up = _gate_up("shipped", x, w13, weights, metadata, _config(block_m=block_m))
        results[block_m] = (gate_up, _down(gate_up, w_down, weights, metadata, _config(block_m=block_m)))

    for result_64, result_128 in zip(results[64], results[128]):
        assert not result_64.isnan().any()
        _assert_same_writes_and_close(result_64, result_128)
    gate_up, routed = results[64]
    torch.testing.assert_close(gate_up, _torch_gate_up(x, w13, route_ids), **_FP32_TOLERANCE)
    expected = torch.einsum("rn,rnk->rk", gate_up, w_down[route_ids.reshape(-1)]) * weights.reshape(-1, 1)
    torch.testing.assert_close(routed.view(-1, hidden), expected, **_FP32_TOLERANCE)


# Device ship gate: BF16 output bits must match the shipped kernel exactly.


def _gpu_device(device_type: str) -> torch.device:
    if device_type == "musa":
        if getattr(torch.version, "musa", None) is None or not hasattr(torch, "musa") or not torch.musa.is_available():
            pytest.skip("MUSA runtime and device required")
    elif not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() == 0:
        pytest.skip("CUDA runtime and device required")
    return torch.device(device_type)


_DEVICE_PARAMS = [
    pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.gpu]),
    pytest.param("musa", marks=[pytest.mark.musa, pytest.mark.gpu]),
]
_DEVICE_CONFIGS = [
    _config(block_m=128, block_n=128, block_k=64),
    _config(block_m=64, block_n=128, block_k=64),
    _config(block_m=64, block_n=128, block_k=64, num_warps=8),
    _config(block_m=128, block_n=256, block_k=64),
    _config(block_m=64, block_n=256, block_k=32, num_warps=8, num_stages=2),
    _config(block_m=128, block_n=128, block_k=128),
]


_DEVICE_PROBLEMS = {
    # MAGI-2 head width (256) and expert width (1280) with skewed top-6 routing.
    "magi2": (257, 6, 8, 256, 1280),
    # Masked K loads (even_Ks=False) and an N tail narrower than some tiles.
    "k_tail_n_tail": _PROBLEMS["k_tail_n_tail"],
}


def _device_problem(device):
    return _gate_up_problem(*_DEVICE_PROBLEMS["magi2"], dtype=torch.bfloat16, device=device)


def _bits(tensor):
    # Raw BF16 bits: NaN payloads, infinities and signed zeros all count.
    return tensor.view(torch.int16)


@pytest.mark.parametrize("device_type", _DEVICE_PARAMS)
@pytest.mark.parametrize("problem", list(_DEVICE_PROBLEMS))
@pytest.mark.parametrize("config", _DEVICE_CONFIGS, ids=_config_id)
def test_gate_up_kernel_is_bit_identical_to_shipped_kernel_on_device(device_type, problem, config):
    device = _gpu_device(device_type)
    num_tokens, top_k, num_experts, hidden, intermediate = _DEVICE_PROBLEMS[problem]
    x, w13, route_ids, weights = _gate_up_problem(
        num_tokens, top_k, num_experts, hidden, intermediate, dtype=torch.bfloat16, device=device
    )
    shipped_metadata = moe._align_bf16_routes(route_ids, num_experts, 128)
    shipped = _gate_up("shipped", x, w13, weights, shipped_metadata, SHIPPED_CONFIG)
    metadata = moe._align_bf16_routes(route_ids, num_experts, config["BLOCK_SIZE_M"])
    actual = _gate_up("register", x, w13, weights, metadata, config)
    assert not shipped.isnan().any() and not actual.isnan().any()
    assert torch.equal(_bits(actual), _bits(shipped))


@pytest.mark.parametrize("device_type", _DEVICE_PARAMS)
@pytest.mark.parametrize("block_m", [64, 128])
def test_gate_up_kernel_skips_unowned_and_dead_blocks_on_device(device_type, block_m):
    device = _gpu_device(device_type)
    x, w13, weights, metadata, skipped, _ = _padded_block_problem(block_m, dtype=torch.bfloat16, device=device)
    shipped = _gate_up("shipped", x, w13, weights, metadata, _config(block_m=block_m))
    actual = _gate_up("register", x, w13, weights, metadata, _config(block_m=block_m))
    assert torch.all(actual[skipped].isnan()) and not actual[~skipped].isnan().any()
    assert torch.equal(actual.isnan(), shipped.isnan())
    assert torch.equal(_bits(actual), _bits(shipped))


@pytest.mark.parametrize("device_type", _DEVICE_PARAMS)
def test_shipped_kernel_block_m_64_is_bit_identical_on_device(device_type):
    device = _gpu_device(device_type)
    x, w13, route_ids, weights = _device_problem(device)
    w_down = (torch.randn(8, 1280, 256, generator=torch.Generator().manual_seed(1)) / 36).to(torch.bfloat16)
    w_down = w_down.to(device)
    results = {}
    for block_m in (64, 128):
        metadata = moe._align_bf16_routes(route_ids, 8, block_m)
        gate_up = _gate_up("shipped", x, w13, weights, metadata, _config(block_m=block_m))
        results[block_m] = (gate_up, _down(gate_up, w_down, weights, metadata, _config(block_m=block_m)))
    for result_64, result_128 in zip(results[64], results[128]):
        assert not result_128.isnan().any()
        assert torch.equal(_bits(result_64), _bits(result_128))
