# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.ring import fused_merge, ring_utils
from vllm_omni.diffusion.attention.backends.ring.fused_merge import try_fused_ring_merge
from vllm_omni.diffusion.attention.backends.ring.ring_utils import update_out_and_lse

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA"
)


def _reference(out, lse, block_out, block_lse):
    """Compute the merged distribution independently in double precision."""
    lse, block_lse = lse.double(), block_lse.double()
    merged_lse = torch.logaddexp(lse, block_lse)
    merged_out = out.double() * (lse - merged_lse).exp() + block_out.double() * (block_lse - merged_lse).exp()
    return merged_out.float(), merged_lse.float()


def _assert_close(actual, expected):
    for tensor, reference in zip(actual, expected):
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, reference, rtol=2e-6, atol=2e-6)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@pytest.mark.parametrize(
    "dtype,layout,shape",
    [
        (torch.bfloat16, "bhs", (1, 17, 3, 128)),
        (torch.float16, "bsh", (2, 7, 4, 64)),
        (torch.float32, "bhs_padded", (2, 9, 3, 80)),
        (torch.bfloat16, "bsh_padded", (1, 17, 3, 64)),
        (torch.float16, "bhs1", (1, 4, 4, 128)),
        (torch.float32, "strided", (2, 7, 3, 80)),
    ],
)
@torch.inference_mode()
def test_ring_merge_layouts_and_input_preservation(dtype, layout, shape):
    torch.manual_seed(42)
    batch, seq_len, heads, dim = shape
    out = torch.randn(shape, device="cuda")
    block_out = torch.randn(shape, device="cuda", dtype=dtype)
    lse = torch.randn(batch, heads, seq_len, device="cuda").transpose(1, 2).unsqueeze(-1)
    canonical = torch.randn(batch, seq_len, heads, device="cuda")
    lse_layout = "bhs" if layout.startswith("bhs") else "bsh"
    block_lse = canonical.transpose(1, 2).contiguous() if lse_layout == "bhs" else canonical
    if layout.endswith("padded"):
        pad_shape = (batch, heads, 3) if lse_layout == "bhs" else (batch, 3, heads)
        # Padding must never contribute, even when it contains NaNs.
        block_lse = torch.cat(
            [block_lse, torch.full(pad_shape, torch.nan, device="cuda")], dim=2 if lse_layout == "bhs" else 1
        )
    if layout == "bhs1":
        block_lse = block_lse.unsqueeze(-1)
    if layout == "strided":
        out = torch.randn(batch, heads, seq_len, dim, device="cuda").transpose(1, 2)
        block_out = torch.randn(batch, seq_len, heads, dim * 2, device="cuda", dtype=dtype)[..., ::2]

    inputs = (out, lse, block_out, block_lse)
    snapshots = [tensor.clone() for tensor in inputs]
    expected = _reference(out, lse, block_out, canonical.unsqueeze(-1))
    actual = update_out_and_lse(*inputs, lse_layout=lse_layout, use_fused_merge=True)
    _assert_close(actual, expected)
    for tensor, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0, equal_nan=True)

    # Ensure eligible inputs exercise the kernel, rather than passing via eager fallback.
    normalized = block_lse.squeeze(-1) if block_lse.ndim == 4 else block_lse
    normalized = normalized[:, :, :seq_len].transpose(1, 2) if lse_layout == "bhs" else normalized[:, :seq_len]
    fused = try_fused_ring_merge(out, lse, block_out, normalized.unsqueeze(-1))
    assert fused is not None
    _assert_close(fused, expected)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_ring_merge_eight_hops(dtype):
    torch.manual_seed(13)
    blocks = torch.randn(8, 2, 19, 3, 80, device="cuda", dtype=dtype)
    block_lses = torch.randn(8, 2, 19, 3, 1, device="cuda") * 8
    out, lse = None, None
    for block_out, block_lse in zip(blocks, block_lses):
        out, lse = update_out_and_lse(out, lse, block_out, block_lse, lse_layout="bsh", use_fused_merge=True)

    expected_lse = block_lses.double().logsumexp(dim=0)
    expected_out = (blocks.double() * (block_lses.double() - expected_lse).exp()).sum(dim=0)
    _assert_close((out, lse), (expected_out.float(), expected_lse.float()))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@torch.inference_mode()
def test_ring_merge_extreme_lse():
    lse = torch.tensor([-1000, 1000, -1000, 1000, -torch.inf, 0, -torch.inf, torch.inf], device="cuda")
    block_lse = torch.tensor([1000, -1000, -1000, 1000, 0, -torch.inf, -torch.inf, torch.inf], device="cuda")
    lse, block_lse = lse.view(1, -1, 1, 1), block_lse.view(1, -1, 1, 1)
    out = torch.arange(8, device="cuda", dtype=torch.float32).view(1, 8, 1, 1).expand(-1, -1, -1, 64)
    block_out = -out
    actual = try_fused_ring_merge(out, lse, block_out, block_lse)
    assert actual is not None
    _assert_close(
        tuple(t[:, :4] for t in actual), _reference(out[:, :4], lse[:, :4], block_out[:, :4], block_lse[:, :4])
    )
    # Preserve the existing eager behavior for nonfinite inputs, including NaNs.
    expected = (out - torch.sigmoid(block_lse - lse) * (out - block_out), lse - F.logsigmoid(lse - block_lse))
    for tensor, reference in zip(actual, expected):
        torch.testing.assert_close(tensor[:, 4:], reference[:, 4:], rtol=0, atol=0, equal_nan=True)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@torch.inference_mode()
def test_ring_merge_slice_updates_only_selected_rows():
    torch.manual_seed(14)
    out = torch.randn(2, 9, 3, 64, device="cuda")
    lse = torch.randn(2, 9, 3, 1, device="cuda")
    block_out = torch.randn(2, 4, 3, 64, device="cuda", dtype=torch.bfloat16)
    block_lse = torch.randn(2, 4, 3, 1, device="cuda")
    selected = (slice(None), slice(1, 9, 2))
    expected_out, expected_lse = out.clone(), lse.clone()
    expected_out[selected], expected_lse[selected] = _reference(out[selected], lse[selected], block_out, block_lse)
    actual = update_out_and_lse(out, lse, block_out, block_lse, slice_=selected, lse_layout="bsh", use_fused_merge=True)
    assert actual[0] is out and actual[1] is lse
    _assert_close(actual, (expected_out, expected_lse))
    torch.testing.assert_close(out[:, ::2], expected_out[:, ::2], rtol=0, atol=0)
    torch.testing.assert_close(lse[:, ::2], expected_lse[:, ::2], rtol=0, atol=0)


@pytest.mark.cpu
@torch.inference_mode()
def test_ring_merge_cpu_fallback():
    out, block_out = torch.randn(2, 5, 3, 8), torch.randn(2, 5, 3, 8)
    lse, block_lse = torch.randn(2, 5, 3, 1), torch.randn(2, 5, 3, 1)
    assert try_fused_ring_merge(out, lse, block_out, block_lse) is None
    _assert_close(
        update_out_and_lse(out, lse, block_out, block_lse, lse_layout="bsh", use_fused_merge=True),
        _reference(out, lse, block_out, block_lse),
    )


@pytest.mark.cpu
@torch.inference_mode()
def test_ring_merge_shared_call_does_not_opt_in(monkeypatch):
    def unexpected_fusion(*_args):
        pytest.fail("Shared Ring callers must keep their existing merge by default")

    monkeypatch.setattr(ring_utils, "try_fused_ring_merge", unexpected_fusion)
    out, block_out = torch.randn(1, 5, 2, 8), torch.randn(1, 5, 2, 8)
    lse, block_lse = torch.randn(1, 5, 2, 1), torch.randn(1, 5, 2, 1)
    actual = update_out_and_lse(out, lse, block_out, block_lse, lse_layout="bsh")
    _assert_close(actual, _reference(out, lse, block_out, block_lse))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
def test_ring_merge_autograd_falls_back():
    torch.manual_seed(15)
    inputs = [torch.randn(shape, device="cuda", requires_grad=True) for shape in [(1, 5, 3, 8), (1, 5, 3, 1)] * 2]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    assert try_fused_ring_merge(*inputs) is None
    actual = update_out_and_lse(*inputs, lse_layout="bsh", use_fused_merge=True)
    out, lse, block_out, block_lse = reference_inputs
    expected = (out - torch.sigmoid(block_lse - lse) * (out - block_out), lse - F.logsigmoid(lse - block_lse))
    sum(t.square().sum() for t in actual).backward()
    sum(t.square().sum() for t in expected).backward()
    for tensor, reference in zip(inputs, reference_inputs):
        torch.testing.assert_close(tensor.grad, reference.grad, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@torch.inference_mode()
def test_ring_merge_cuda_graph_reads_current_inputs():
    out, block_out = torch.randn(2, 17, 3, 64, device="cuda"), torch.randn(2, 17, 3, 64, device="cuda")
    lse, block_lse = torch.randn(2, 17, 3, 1, device="cuda"), torch.randn(2, 17, 3, 1, device="cuda")
    assert try_fused_ring_merge(out, lse, block_out, block_lse) is not None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = try_fused_ring_merge(out, lse, block_out, block_lse)
    assert actual is not None
    for _ in range(2):
        block_out.normal_()
        block_lse.normal_()
        graph.replay()
        _assert_close(actual, _reference(out, lse, block_out, block_lse))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@torch.inference_mode()
def test_ring_merge_fullgraph_compile_keeps_torch_fallback(monkeypatch):
    # Tracing must use the Torch expression, not enter the eager Triton kernel.
    monkeypatch.setattr(fused_merge, "_ring_merge_kernel", None)
    compiled = torch.compile(ring_utils._update_out_and_lse, backend="inductor", fullgraph=True)
    torch.manual_seed(16)
    out = torch.empty(2, 19, 3, 80, device="cuda")
    block_out = torch.empty_like(out, dtype=torch.bfloat16)
    lse = torch.empty(2, 3, 19, device="cuda").transpose(1, 2).unsqueeze(-1)
    block_lse = torch.full((2, 3, 24), torch.nan, device="cuda")
    for _ in range(2):
        out.normal_()
        block_out.normal_()
        lse.normal_()
        block_lse[:, :, :19].normal_()
        actual = compiled(out, lse, block_out, block_lse, "bhs", use_fused_merge=True)
        expected = _reference(out, lse, block_out, block_lse[:, :, :19].transpose(1, 2).unsqueeze(-1))
        _assert_close(actual, expected)


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@requires_cuda
@torch.inference_mode()
def test_ring_merge_other_current_device_falls_back(monkeypatch):
    # Fail before a potentially invalid launch if the device guard regresses.
    monkeypatch.setattr(fused_merge, "_ring_merge_kernel", None)
    torch.manual_seed(17)
    with torch.accelerator.device_index(0):
        device = torch.device("cuda:1")
        out = torch.randn(2, 7, 3, 64, device=device)
        block_out = torch.randn_like(out, dtype=torch.bfloat16)
        lse = torch.randn(2, 7, 3, 1, device=device)
        block_lse = torch.randn_like(lse)

        assert torch.accelerator.current_device_index() == 0
        assert try_fused_ring_merge(out, lse, block_out, block_lse) is None
        actual = update_out_and_lse(out, lse, block_out, block_lse, lse_layout="bsh", use_fused_merge=True)
        _assert_close(actual, _reference(out, lse, block_out, block_lse))
        assert all(tensor.device == device for tensor in actual)
        assert torch.accelerator.current_device_index() == 0
