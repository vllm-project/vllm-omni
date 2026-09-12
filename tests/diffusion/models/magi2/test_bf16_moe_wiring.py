# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_omni.diffusion.models.magi2.mh_moe as moe
from vllm_omni.diffusion.models.magi2 import fused_moe_kernels
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def _moe_inputs(
    num_tokens: int,
    num_heads: int,
    *,
    hidden_size: int = 7,
    intermediate_size: int = 9,
    num_experts: int = 4,
    top_k: int = 2,
    seed: int = 419,
):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, num_heads, hidden_size, generator=generator).to(torch.bfloat16)
    shape = (num_heads * num_experts, hidden_size, intermediate_size)
    gate = (torch.randn(shape, generator=generator) * 0.3).to(torch.bfloat16)
    up = (torch.randn(shape, generator=generator) * 0.7 - 0.5).to(torch.bfloat16)
    down = torch.randn(num_heads * num_experts, intermediate_size, hidden_size, generator=generator)
    down = (down * 0.2).to(torch.bfloat16)
    # Repeat head-local IDs so different heads must use different expert banks.
    # Experts 1 and 3 have no routes, exercising empty experts and padding.
    selected = torch.tensor([0, 2]) if top_k == 2 else torch.arange(top_k)
    indices = selected.expand(num_heads, num_tokens, top_k).contiguous()
    probabilities = torch.rand(num_heads, num_tokens, top_k, generator=generator)
    probabilities = (probabilities / probabilities.sum(-1, keepdim=True)).contiguous()
    return x, probabilities, indices, gate, up, down


def _reference_forward(x, probabilities, indices, gate, up, down):
    """FP32 fused-expert math with BF16 activation and weighted-route stores."""
    num_tokens, num_heads, _ = x.shape
    experts_per_head = gate.shape[0] // num_heads
    output = torch.zeros_like(x)
    for head in range(num_heads):
        for token in range(num_tokens):
            routes = []
            for choice in range(indices.shape[-1]):
                expert = head * experts_per_head + int(indices[head, token, choice])
                gate_value = (x[token, head].float() @ gate[expert].float()).clamp(max=7)
                up_value = (x[token, head].float() @ up[expert].float()).clamp(-7, 7)
                hidden = (gate_value * torch.sigmoid(1.702 * gate_value) * (up_value + 1)).to(x.dtype)
                down_value = hidden.float() @ down[expert].float()
                routes.append((down_value * probabilities[head, token, choice]).to(x.dtype))
            output[token, head] = torch.stack(routes).sum(0)
    return output


def _assert_call_args_identity(call_args, *expected):
    assert len(call_args.args) == len(expected)
    assert all(actual is wanted for actual, wanted in zip(call_args.args, expected))


@pytest.mark.cpu
def test_reference_keeps_fp32_gate_up_until_activation():
    x = torch.ones(1, 1, 2, dtype=torch.bfloat16)
    probabilities = torch.ones(1, 1, 1)
    indices = torch.zeros(1, 1, 1, dtype=torch.long)
    weight = torch.tensor([[[1.0], [2**-8]]], dtype=torch.bfloat16)
    down = torch.ones(1, 1, 2, dtype=torch.bfloat16)
    actual = _reference_forward(x, probabilities, indices, weight, weight, down)

    projection = torch.tensor(1.0 + 2**-8)
    expected = (projection * torch.sigmoid(1.702 * projection) * (projection + 1)).to(x.dtype)
    rounded = projection.to(x.dtype).float()
    early_rounded = (rounded * torch.sigmoid(1.702 * rounded) * (rounded + 1)).to(x.dtype)
    torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=0)
    assert not torch.equal(actual, early_rounded.expand_as(actual))


def _torch_invoke(a, b, c, weights, sorted_ids, expert_ids, num_padded, *, top_k, config, fuse_swiglu):
    """Execute the launcher's route contract with Torch instead of Triton."""
    route_count = weights.numel()
    output = c.view(route_count, c.shape[-1])
    block_size = config["BLOCK_SIZE_M"]
    assert int(num_padded.item()) == sorted_ids.numel()
    assert expert_ids.numel() * block_size == sorted_ids.numel()
    written = []
    for block, expert in enumerate(expert_ids.tolist()):
        assert 0 <= expert < b.shape[0]
        routes = sorted_ids[block * block_size : (block + 1) * block_size].long()
        routes = routes[routes < route_count]
        written.extend(routes.tolist())
        gemm = a[routes // top_k].float() @ b[expert].float().T
        if fuse_swiglu:
            gate = gemm[:, 0::2].clamp(max=7)
            up = gemm[:, 1::2].clamp(-7, 7)
            result = gate * torch.sigmoid(1.702 * gate) * (up + 1)
        else:
            result = gemm * weights.reshape(-1)[routes, None]
        output[routes] = result.to(c.dtype)
    assert sorted(written) == list(range(route_count))


@pytest.mark.cpu
@pytest.mark.parametrize("route_ids", [[], [2, 0, 2, 0, 2], [3] * 9 + [0, 3, 0]])
def test_align_bf16_routes_preserves_routes_and_pads_expert_blocks(route_ids):
    ids = torch.tensor(route_ids, dtype=torch.int64)
    sorted_ids, expert_ids, num_padded = moe._align_bf16_routes(ids, num_experts=5, block_size=4)
    counts = torch.bincount(ids, minlength=5)
    expected_blocks = ((counts + 3) // 4).sum().item()
    assert num_padded.tolist() == [expected_blocks * 4]
    assert expert_ids.numel() == expected_blocks
    assert sorted_ids.dtype == expert_ids.dtype == num_padded.dtype == torch.int32
    assert sorted_ids.numel() == num_padded.item()
    valid_ids = sorted_ids[sorted_ids < ids.numel()].long()
    assert sorted(valid_ids.tolist()) == list(range(ids.numel()))
    for block, expert in enumerate(expert_ids.tolist()):
        block_routes = sorted_ids[block * 4 : (block + 1) * 4]
        valid = block_routes[block_routes < ids.numel()].long()
        assert torch.all(ids[valid] == expert)
        assert torch.all(block_routes[block_routes >= ids.numel()] == ids.numel())


@pytest.mark.cpu
@pytest.mark.parametrize(("num_tokens", "num_heads"), [(0, 3), (1, 1), (5, 2), (7, 3)])
def test_bf16_forward_head_offsets_and_interleaved_weights(monkeypatch, num_tokens, num_heads):
    inputs = _moe_inputs(num_tokens, num_heads)
    calls = []

    def invoke(*args, **kwargs):
        calls.append(kwargs["fuse_swiglu"])
        return _torch_invoke(*args, **kwargs)

    monkeypatch.setattr(moe, "invoke_fused_moe_bf16", invoke)
    actual = moe._bf16_fused_moe_forward(*inputs)
    expected = _reference_forward(*inputs)
    assert actual.shape == inputs[0].shape
    assert actual.dtype == torch.bfloat16
    if num_tokens:
        assert calls == [True, False]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("flag", "device_type", "dtype", "deterministic", "expected_path"),
    [
        (None, "cuda", torch.bfloat16, None, "triton"),
        ("0", "cuda", torch.bfloat16, None, "triton"),
        ("0", "cuda", torch.bfloat16, "0", "triton"),
        ("1", "cpu", torch.bfloat16, None, "torch"),
        ("enabled", "cuda", torch.bfloat16, None, "triton"),
        ("1", "cuda", torch.float16, None, "triton"),
        ("1", "cuda", torch.float32, None, "triton"),
        ("1", "cuda", torch.bfloat16, None, "bf16"),
        ("1", "cuda", torch.bfloat16, "0", "bf16"),
        ("1", "cuda", torch.bfloat16, "1", "triton"),
    ],
)
def test_local_forward_dispatch_contract(
    monkeypatch,
    flag,
    device_type,
    dtype,
    deterministic,
    expected_path,
):
    config = moe.Magi2MultiHeadMoEConfig(
        hidden_size=4,
        num_heads=1,
        num_experts=2,
        top_k=1,
        expert_intermediate_size=4,
        params_dtype=torch.bfloat16,
    )
    layer = moe.Magi2MultiHeadMoE(config, ep_group=moe.Magi2ParallelGroup(None, 1, 0))
    if device_type == "cpu":
        x_heads = torch.randn(1, 1, 4, dtype=dtype)
    else:
        # Accelerator metadata exercises dispatch without allocating on a GPU.
        x_heads = SimpleNamespace(
            device=SimpleNamespace(type=device_type),
            dtype=dtype,
            is_cuda=True,
        )
    probabilities = torch.tensor([[[1.0]]])
    indices = torch.tensor([[[0]]])
    route = Mock(return_value=(probabilities, indices))
    gather_ids, sorted_probs, offsets = object(), object(), object()
    sort = Mock(return_value=(gather_ids, sorted_probs, offsets))
    fast = Mock(return_value="bf16-output")
    torch_kernel = Mock(return_value="torch-output")
    triton_kernel = Mock(return_value="triton-output")
    monkeypatch.setattr(layer, "_route", route)
    monkeypatch.setattr(moe, "global_sort_routes", sort)
    monkeypatch.setattr(moe, "_bf16_fused_moe_forward", fast)
    monkeypatch.setattr(moe, "torch_mh_moe_forward", torch_kernel)
    monkeypatch.setattr(moe, "triton_mh_moe_forward", triton_kernel)
    if flag is None:
        monkeypatch.delenv("MAGI2_USE_BF16_MOE_KERNEL", raising=False)
    else:
        monkeypatch.setenv("MAGI2_USE_BF16_MOE_KERNEL", flag)
    if deterministic is None:
        monkeypatch.delenv("MAGI2_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("MAGI2_DETERMINISTIC", deterministic)

    actual = layer._local_forward(x_heads)

    route.assert_called_once()
    _assert_call_args_identity(route.call_args, x_heads)
    if expected_path == "bf16":
        assert actual == "bf16-output"
        fast.assert_called_once()
        _assert_call_args_identity(
            fast.call_args, x_heads, probabilities, indices, layer.W_gate, layer.W_up, layer.W_down
        )
        assert fast.call_args.kwargs == {}
        sort.assert_not_called()
        torch_kernel.assert_not_called()
        triton_kernel.assert_not_called()
    else:
        sort.assert_called_once()
        _assert_call_args_identity(sort.call_args, probabilities, indices, layer.num_experts)
        fast.assert_not_called()
        if expected_path == "torch":
            assert actual == "torch-output"
            torch_kernel.assert_called_once()
            _assert_call_args_identity(
                torch_kernel.call_args,
                x_heads,
                gather_ids,
                sorted_probs,
                offsets,
                layer.W_gate,
                layer.W_up,
                layer.W_down,
            )
            triton_kernel.assert_not_called()
        else:
            assert actual == "triton-output"
            triton_kernel.assert_called_once()
            _assert_call_args_identity(
                triton_kernel.call_args,
                x_heads,
                gather_ids,
                sorted_probs,
                offsets,
                layer.W_gate,
                layer.W_up,
                layer.W_down,
            )
            assert triton_kernel.call_args.kwargs == {
                "deterministic": deterministic == "1",
            }
            torch_kernel.assert_not_called()


@pytest.mark.cpu
def test_local_forward_fast_path_propagates_kernel_failure(monkeypatch):
    config = moe.Magi2MultiHeadMoEConfig(
        hidden_size=4,
        num_heads=1,
        num_experts=2,
        top_k=1,
        expert_intermediate_size=4,
        params_dtype=torch.bfloat16,
    )
    layer = moe.Magi2MultiHeadMoE(config, ep_group=moe.Magi2ParallelGroup(None, 1, 0))
    x_heads = SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        dtype=torch.bfloat16,
        is_cuda=True,
    )
    probabilities = torch.tensor([[[1.0]]])
    indices = torch.tensor([[[0]]])
    monkeypatch.setattr(layer, "_route", Mock(return_value=(probabilities, indices)))
    sort = Mock()
    torch_kernel = Mock()
    triton_kernel = Mock()
    error = RuntimeError("BF16 kernel failed")
    fast = Mock(side_effect=error)
    monkeypatch.setattr(moe, "global_sort_routes", sort)
    monkeypatch.setattr(moe, "_bf16_fused_moe_forward", fast)
    monkeypatch.setattr(moe, "torch_mh_moe_forward", torch_kernel)
    monkeypatch.setattr(moe, "triton_mh_moe_forward", triton_kernel)
    monkeypatch.setenv("MAGI2_USE_BF16_MOE_KERNEL", "1")
    monkeypatch.delenv("MAGI2_DETERMINISTIC", raising=False)

    with pytest.raises(RuntimeError, match="BF16 kernel failed") as exc_info:
        layer._local_forward(x_heads)

    assert exc_info.value is error
    fast.assert_called_once()
    sort.assert_not_called()
    torch_kernel.assert_not_called()
    triton_kernel.assert_not_called()


def _gpu_device(device_type: str) -> torch.device:
    if device_type == "musa":
        if getattr(torch.version, "musa", None) is None or not hasattr(torch, "musa") or not torch.musa.is_available():
            pytest.skip("MUSA runtime and device required")
    elif not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() == 0:
        pytest.skip("CUDA runtime and device required")
    return torch.device(device_type)


@pytest.mark.parametrize(
    "device_type",
    [
        pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.gpu]),
        pytest.param("musa", marks=[pytest.mark.musa, pytest.mark.gpu]),
    ],
)
@pytest.mark.parametrize(
    ("num_tokens", "num_heads", "hidden_size", "intermediate_size", "top_k", "num_experts"),
    [
        (1, 1, 64, 80, 2, 4),
        (17, 3, 64, 80, 2, 4),
        (137, 2, 64, 80, 2, 4),
        (2, 3, 256, 1280, 6, 8),
        (129, 1, 256, 1280, 6, 8),
    ],
)
def test_bf16_forward_real_kernel_parity(
    device_type, num_tokens, num_heads, hidden_size, intermediate_size, top_k, num_experts
):
    device = _gpu_device(device_type)
    inputs = _moe_inputs(
        num_tokens,
        num_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        num_experts=num_experts,
    )
    expected = _reference_forward(*inputs)
    actual = moe._bf16_fused_moe_forward(*(tensor.to(device) for tensor in inputs)).cpu()
    assert torch.isfinite(actual).all()
    # BF16 activation/output rounding and cancellation across routes amplify
    # small arithmetic differences. Bound both individual and aggregate error;
    # retain the tighter absolute tolerance for the small masked-tile cases.
    atol = 5e-1 if intermediate_size == 1280 else 4e-2
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=atol)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-6)
    assert relative_l2 < 1e-3


@pytest.mark.parametrize(
    "device_type",
    [pytest.param("cuda", marks=pytest.mark.cuda), pytest.param("musa", marks=pytest.mark.musa)],
)
@pytest.mark.parametrize("seed", [419, 420])
@pytest.mark.parametrize(
    ("num_tokens", "num_heads", "hidden_size", "intermediate_size", "top_k", "num_experts"),
    [
        (1, 1, 64, 96, 2, 4),
        (17, 3, 64, 96, 2, 4),
        (137, 2, 64, 96, 2, 4),
        (2, 3, 256, 1280, 6, 8),
        (129, 1, 256, 1280, 6, 8),
    ],
)
def test_bf16_forward_matches_reference_routes_and_fp32_reduction(
    monkeypatch, device_type, seed, num_tokens, num_heads, hidden_size, intermediate_size, top_k, num_experts
):
    """Compare expert math and FP32 reduction, not BF16 scatter accumulation."""
    device = _gpu_device(device_type)
    inputs = _moe_inputs(
        num_tokens,
        num_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        num_experts=num_experts,
        seed=seed,
    )
    x, probabilities, indices, gate, up, down = (tensor.to(device) for tensor in inputs)
    gather_ids, sorted_probs, offsets = moe.global_sort_routes(probabilities, indices, num_experts)
    reference_routes, new_routes = [], []
    original_scatter = fused_moe_kernels._deterministic_scatter
    original_invoke = moe.invoke_fused_moe_bf16

    def capture_reference(sorted_output, *args):
        reference_routes.append(sorted_output.detach().cpu())
        return original_scatter(sorted_output, *args)

    def capture_new(*args, **kwargs):
        result = original_invoke(*args, **kwargs)
        if not kwargs.get("fuse_swiglu", False):
            new_routes.append(args[2].detach().cpu().reshape(-1, hidden_size))
        return result

    monkeypatch.setattr(fused_moe_kernels, "_deterministic_scatter", capture_reference)
    monkeypatch.setattr(moe, "invoke_fused_moe_bf16", capture_new)
    # Capture before BF16 scatter; this path intentionally reduces in FP32.
    moe.triton_mh_moe_forward(x, gather_ids, sorted_probs, offsets, gate, up, down, deterministic=True)
    actual = moe._bf16_fused_moe_forward(x, probabilities, indices, gate, up, down).cpu()
    assert len(reference_routes) == len(new_routes) == 1
    flat_experts = (inputs[2] + torch.arange(num_heads).view(-1, 1, 1) * num_experts).reshape(-1)
    order = flat_experts.argsort(stable=True)
    new_sorted = new_routes[0].index_select(0, order)
    scatter_ids = gather_ids.cpu().long() * num_heads + flat_experts[order] // num_experts
    expected_fp32 = torch.zeros(num_tokens * num_heads, hidden_size, dtype=torch.float32)
    expected_fp32.index_add_(0, scatter_ids, reference_routes[0].float())
    expected = expected_fp32.to(x.dtype).view(num_tokens, num_heads, hidden_size)
    atol = 5e-1 if intermediate_size == 1280 else 4e-2
    for result, reference in ((new_sorted, reference_routes[0]), (actual, expected)):
        assert torch.isfinite(result).all() and torch.isfinite(reference).all()
        torch.testing.assert_close(result, reference, rtol=2e-2, atol=atol)
        relative_l2 = (result.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-6)
        assert relative_l2 < 1e-3


@pytest.mark.parametrize(
    "device_type",
    [
        pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.gpu]),
        pytest.param("musa", marks=[pytest.mark.musa, pytest.mark.gpu]),
    ],
)
def test_bf16_reference_atomic_add_capability(device_type):
    """The non-deterministic BF16 reference path must compile on each GPU.

    This is intentionally separate from candidate parity: it exercises the
    ``deterministic=False`` branch and therefore the ``tl.atomic_add`` used by
    the original MAGI-2 reference path.  An unsupported backend must fail the
    test with its compiler error instead of being silently treated as a skip.
    """

    device = _gpu_device(device_type)
    inputs = _moe_inputs(
        num_tokens=2,
        num_heads=1,
        hidden_size=64,
        intermediate_size=96,
        top_k=2,
        num_experts=4,
        seed=419,
    )
    x, probabilities, indices, gate, up, down = (tensor.to(device) for tensor in inputs)
    gather_ids, sorted_probs, offsets = moe.global_sort_routes(probabilities, indices, 4)
    actual = moe.triton_mh_moe_forward(x, gather_ids, sorted_probs, offsets, gate, up, down, deterministic=False)
    deterministic = moe.triton_mh_moe_forward(x, gather_ids, sorted_probs, offsets, gate, up, down, deterministic=True)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, deterministic, rtol=2e-2, atol=4e-2)
