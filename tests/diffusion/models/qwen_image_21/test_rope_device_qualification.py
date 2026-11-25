# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from diffusers.models.normalization import RMSNorm

from tests.helpers.device import DEVICE_TYPE
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import exact_bf16_pointwise as pointwise
from vllm_omni.diffusion.layers import exact_bf16_qk_norm_rope as fused_qk
from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_npu_mixed_prompt_rotary_matches_single_requests():
    rope = transformer.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    prepare = transformer.QwenImage21SequencePrepare(torch.nn.Identity(), rope)
    layout = [(1, 2, 2)]
    image_slots = torch.tensor([[False] * 6 + [True]] * 2, device="npu")
    text_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1] * 6], device="npu")
    device = text_mask.device
    mask = transformer._qwen_image21_rope_text_mask(text_mask, device, sp_active=False)
    _, _, _, target_freqs = prepare(
        torch.zeros(2, 4, 64, device=device), torch.zeros(2, 6, 64, device=device), layout, image_slots, False, mask
    )

    x = torch.randn(2, 4, 3, 128, dtype=torch.bfloat16, device=device)
    actual = transformer._apply_qwen_image21_rotary_emb_npu(x, target_freqs)
    for i, length in enumerate((4, 6)):
        image_mask = torch.tensor([False] * length + [True] * 4, device=device)
        single_freqs = rope(layout, image_mask, device)[-4:]
        expected = transformer._apply_qwen_image21_rotary_emb_npu(x[i : i + 1], single_freqs)
        torch.testing.assert_close(actual[i : i + 1], expected, rtol=0, atol=0)

    with pytest.raises(ValueError, match="without sequence parallelism"):
        transformer._qwen_image21_rope_text_mask(text_mask, device, sp_active=True)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "version,hip,device,expected",
    [
        ("2.13.0+cu130", None, "cuda:1", False),
        ("2.12.0", None, "cuda:1", None),
        ("2.13.0", "7.0", "cuda:1", None),
        ("2.13.0", None, "cpu", None),
    ],
)
def test_exact_rope_stack_and_input_device(monkeypatch, version, hip, device, expected):
    backend = Mock()
    backend.get_device_capability.return_value = (10, 3)
    monkeypatch.setattr(torch, "get_device_module", Mock(return_value=backend))
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.version, "hip", hip)
    assert pointwise.exact_rope_fuse_real_sine(torch.device(device)) is expected
    if expected is not None:
        backend.get_device_capability.assert_called_once_with(torch.device(device))
    else:
        backend.get_device_capability.assert_not_called()


@pytest.mark.cpu
@pytest.mark.parametrize(
    "device_name,version,cuda_version,expected",
    [
        ("NVIDIA A800-SXM4-80GB", "2.14.0+cu130", "13.0", True),
        ("NVIDIA A800-SXM4-80GB", "2.14.0+cu128", "12.8", None),
        ("NVIDIA A800-SXM4-80GB", "2.14.1+cu130", "13.0", None),
        ("NVIDIA A100-SXM4-80GB", "2.14.0+cu130", "13.0", None),
    ],
)
def test_exact_rope_a800_qualification(monkeypatch, device_name, version, cuda_version, expected):
    device = torch.device("cuda:1")
    backend = Mock()
    backend.get_device_capability.return_value = (8, 0)
    backend.get_device_name.return_value = device_name
    monkeypatch.setattr(torch, "get_device_module", Mock(return_value=backend))
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch.version, "hip", None)

    assert pointwise.exact_rope_fuse_real_sine(device) is expected
    if expected is not None:
        backend.get_device_capability.assert_called_once_with(device)
        backend.get_device_name.assert_called_once_with(device)


@pytest.mark.cpu
@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (10, 0), (12, 0)])
@pytest.mark.parametrize("path", ["rotary", "qk_rotary"])
@torch.no_grad()
def test_unqualified_device_declines_before_allocation(monkeypatch, capability, path):
    device = torch.device("cuda", 1)

    def tensor(shape, dtype=torch.bfloat16):
        value = Mock(device=device, dtype=dtype, shape=shape, ndim=len(shape))
        value.stride.return_value = 1
        value.numel.return_value = 256
        value.is_contiguous.return_value = True
        return value

    q = tensor((1, 2, 1, 128))
    weight = tensor((128,))
    freqs = tensor((2, 64), torch.complex64)
    backend = Mock()
    backend.get_device_capability.return_value = capability
    monkeypatch.setattr(torch, "get_device_module", Mock(return_value=backend))
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(pointwise, "HAS_TRITON", True)
    monkeypatch.setattr(fused_qk, "HAS_TRITON", True)
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("unqualified kernel allocation")))

    if path == "rotary":
        assert pointwise.rotary(q, freqs) is None
    else:
        assert fused_qk.qk_rotary(q, q, weight, weight, freqs, 1e-6) is None
    backend.get_device_capability.assert_called_once_with(device)


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
@pytest.mark.parametrize("fuse_real_sine,expected_bits", [(False, 0xBF34), (True, 0xBF35)])
@torch.no_grad()
def test_exact_rope_imaginary_rounding_boundary(monkeypatch, fuse_real_sine, expected_bits):
    if not pointwise.HAS_TRITON:
        pytest.skip("Triton required")
    monkeypatch.setattr(pointwise, "exact_rope_fuse_real_sine", lambda device: fuse_real_sine)
    monkeypatch.setattr(fused_qk, "exact_rope_fuse_real_sine", lambda device: fuse_real_sine)

    q = torch.ones(1, 1, 1, 128, device=DEVICE_TYPE, dtype=torch.bfloat16)
    weight = torch.ones_like(q).flatten()
    weight[0] = -0.71484375
    weight[1] = -1.375
    freqs = torch.ones(1, 64, device=DEVICE_TYPE, dtype=torch.complex64)
    freqs[0, 0] = complex(0.8144440650939941, -0.5802420973777771)

    # The two FMA orders land on opposite sides of a BF16 rounding boundary.
    rotated = pointwise.rotary(weight.view_as(q), freqs)
    fused = fused_qk.qk_rotary(q, q, weight, weight, freqs, 1e-6)
    assert rotated is not None
    assert fused is not None
    for output in (rotated, *fused):
        assert (output.view(torch.int16)[0, 0, 0, 1].item() & 0xFFFF) == expected_bits


@pytest.mark.cpu
@torch.no_grad()
def test_validated_pointwise_declines_kernels_while_compiling(monkeypatch):
    forbidden = Mock()
    forbidden.__getitem__ = Mock(side_effect=AssertionError("user Triton kernel launched while compiling"))
    monkeypatch.setattr(pointwise, "_residual_kernel", forbidden, raising=False)
    monkeypatch.setattr(pointwise, "_silu_mul_kernel", forbidden, raising=False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    x, y, gate = (torch.randn(4, 8, dtype=torch.bfloat16) for _ in range(3))

    torch.testing.assert_close(pointwise.residual(x, y, gate, _validated=True), x + gate * y, rtol=0, atol=0)
    torch.testing.assert_close(
        pointwise.silu_mul(gate, y, _validated=True), torch.nn.functional.silu(gate) * y, rtol=0, atol=0
    )


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards=1)
@torch.no_grad()
def test_validated_pointwise_compiles_with_inductor():
    """Eager callers may validate once and pass the flag into regionally compiled blocks."""
    torch.manual_seed(0)
    x, y, gate = (torch.randn(2, 1024, device=DEVICE_TYPE, dtype=torch.bfloat16) for _ in range(3))

    def block(x, y, gate):
        return pointwise.residual(x, pointwise.silu_mul(gate, y, _validated=True), gate, _validated=True)

    def reference(x, y, gate):
        return x + gate * (torch.nn.functional.silu(gate) * y)

    compiled = torch.compile(block, backend="inductor", fullgraph=True)
    expected = torch.compile(reference, backend="inductor", fullgraph=True)(x, y, gate)
    torch.testing.assert_close(compiled(x, y, gate), expected, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@torch.no_grad()
def test_unqualified_attention_uses_reference_norm_and_rope(monkeypatch):
    sequence = 17
    torch.manual_seed(sequence)
    projection = torch.randn(2, sequence, 3, 32, 128, device=DEVICE_TYPE, dtype=torch.bfloat16)
    freqs = torch.polar(torch.ones(sequence, 64, device=DEVICE_TYPE), torch.randn(sequence, 64, device=DEVICE_TYPE))
    attention = object.__new__(transformer.QwenImage21Attention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = attention.num_kv_heads = 32
    attention.head_dim = 128
    attention.norm_q = RMSNorm(128, eps=1e-6).to("cuda", torch.bfloat16)
    attention.norm_k = RMSNorm(128, eps=1e-6).to("cuda", torch.bfloat16)
    for norm in (attention.norm_q, attention.norm_k):
        norm.weight.copy_(torch.randn_like(norm.weight) * 0.1 + 1)

    def reference(x):
        pairs = torch.view_as_complex(x.float().reshape(2, sequence, 32, 64, 2))
        return torch.view_as_real(pairs * freqs.unsqueeze(1)).flatten(3).to(x.dtype)

    expected_q = reference(attention.norm_q(projection[:, :, 0]))
    expected_k = reference(attention.norm_k(projection[:, :, 1]))
    attention.to_qkv = lambda _: (projection.flatten(2), None)
    observed = []

    def check_attention(q, k, v, metadata):
        torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
        observed.append(True)
        return v

    attention.attn = check_attention
    attention.to_out = torch.nn.Identity()
    # Exercise the real fallback chain on CUDA while simulating an SM90 input device.
    monkeypatch.setattr(torch.get_device_module("cuda"), "get_device_capability", lambda device: (9, 0))
    forbidden = Mock()
    forbidden.__getitem__ = Mock(side_effect=AssertionError("unqualified RoPE kernel launch"))
    monkeypatch.setattr(pointwise, "_rope_kernel", forbidden)
    monkeypatch.setattr(fused_qk, "_full", forbidden)

    raw = projection[:, :, 0]
    torch.testing.assert_close(transformer._apply_qwen_image21_rotary_emb(raw, freqs), reference(raw), rtol=0, atol=0)
    result = attention(torch.empty(2, sequence, 4096, device=DEVICE_TYPE, dtype=torch.bfloat16), freqs)
    assert observed == [True]
    torch.testing.assert_close(result, projection[:, :, 2].flatten(2), rtol=0, atol=0)
