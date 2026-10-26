# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the optional H3 Omni adapter, including real INT8 routing.

Only device kernels are numerical doubles; no NPU accuracy or speed is claimed.
"""

import sys
from contextlib import contextmanager
from copy import deepcopy
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.minimax_h3 import vae_ops
from vllm_omni.quantization import int8_config
from vllm_omni.quantization.int8_config import DiffusionInt8Config, NPUInt8OnlineLinearMethod

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

HAS_VAE_INT8 = find_spec("vllm_omni.diffusion.models.minimax_h3.vae_int8") is not None
if HAS_VAE_INT8:
    from vllm_omni.diffusion.models.minimax_h3 import vae_int8


@pytest.fixture(autouse=True)
def cpu_platform(monkeypatch):
    """CPU contracts also run on hosts where an accelerator is installed.

    CustomOp selects its bound implementation at construction. NPU routing
    tests override only is_npu after all competing backends are disabled here.
    """
    for backend in ("is_rocm", "is_cuda", "is_npu", "is_xpu", "is_musa"):
        monkeypatch.setattr(vae_ops.current_omni_platform, backend, lambda: False)


def all_to_all_4D(value, scatter_dim, gather_dim, *, group):
    raise AssertionError("A CPU test must provide the spatial communication boundary")


def get_parallel_state():
    raise AssertionError("A CPU test must provide the spatial process group")


def _vit_norm_input(_module, value):
    return value.float()


def _reference_rope(value, cos, sin):
    """H3's full-coefficient NeoX formula, retaining each batch's positions."""
    if cos.ndim == 3:
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    width = cos.shape[-1]
    prefix, suffix = value[..., :width], value[..., width:]
    first, second = prefix.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((prefix * cos.to(value.dtype) + rotated * sin.to(value.dtype), suffix), dim=-1)


class Attention(nn.Module):
    """Small reference for the remote H3 Attention's public contract."""

    def __init__(self):
        super().__init__()
        self.dim_head = 8
        self.spatial_parallel = 0
        self.to_qkv = nn.Linear(16, 48)
        self.to_out = nn.Linear(16, 16)
        self.norm_q = nn.RMSNorm(8, eps=1e-5, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(8, eps=1e-5, elementwise_affine=False)

    def perform_attention(self, query, key, value, pack_info):
        result = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        return result.transpose(1, 2)

    def forward(self, hidden_states, rotary_pos_emb=None, pack_info=None):
        batch, sequence, _ = hidden_states.shape
        query, key, value = self.to_qkv(hidden_states).view(batch, sequence, 2, 24).chunk(3, dim=-1)
        if self.spatial_parallel:
            group = get_parallel_state()["sp_process_group"]
            query = all_to_all_4D(query, 2, 1, group=group)
            key = all_to_all_4D(key, 2, 1, group=group)
            value = all_to_all_4D(value, 2, 1, group=group)
        if self.norm_q is not None:
            query = self.norm_q(_vit_norm_input(self.norm_q, query)).to(query.dtype)
        if self.norm_k is not None:
            key = self.norm_k(_vit_norm_input(self.norm_k, key)).to(key.dtype)
        if rotary_pos_emb is not None:
            query = _reference_rope(query, *rotary_pos_emb)
            key = _reference_rope(key, *rotary_pos_emb)
        result = self.perform_attention(query, key, value, {} if pack_info is None else pack_info)
        if self.spatial_parallel:
            result = all_to_all_4D(result, 1, 2, group=group)
        return self.to_out(result.reshape(batch, sequence, 16))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(16, 32)
        self.w2 = nn.Linear(16, 16)

    def forward(self, value):
        gate, value = self.w1(value).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(16, eps=1e-5)
        self.norm2 = nn.RMSNorm(16, eps=1e-5)
        self.attn = Attention()
        self.ff = FeedForward()

    def forward(self, value, rotary_pos_emb=None, pack_info=None):
        value = value + self.attn(self.norm1(value.float()).to(value.dtype), rotary_pos_emb, pack_info)
        return value + self.ff(self.norm2(value.float()).to(value.dtype))


def _model(block_count=1):
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.transformer_blocks = nn.ModuleList([Block() for _ in range(block_count)])
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
def test_affine_norm_retains_checkpoint_and_module_properties(dtype, training):
    owner = nn.Module()
    owner.norm = nn.RMSNorm(8, eps=2e-5, dtype=dtype).train(training)
    owner.norm.weight.requires_grad_(not training)
    with torch.no_grad():
        owner.norm.weight.copy_(torch.linspace(0.25, 1.5, 8).to(dtype))
    expected_state = deepcopy(owner.state_dict())
    expected_device = owner.norm.weight.device
    assert vae_ops._replace_rms_norm(owner, "norm")
    assert isinstance(owner.norm, RMSNorm)
    assert owner.norm.variance_epsilon == 2e-5
    assert owner.norm.training == training
    assert owner.norm.weight.requires_grad == (not training)
    assert owner.norm.weight.dtype == dtype
    assert owner.norm.weight.device == expected_device
    assert owner.state_dict().keys() == expected_state.keys()
    torch.testing.assert_close(owner.norm.weight, expected_state["norm.weight"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "norm",
    [nn.LayerNorm(8), nn.RMSNorm(8, eps=None), nn.RMSNorm(8, eps=None, elementwise_affine=False)],
)
def test_unsupported_norm_semantics_retain_original_object(norm):
    owner = nn.Module()
    owner.norm = norm
    assert not vae_ops._replace_rms_norm(owner, "norm")
    assert owner.norm is norm


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("eps", [1e-5, 3e-4])
def test_non_affine_norm_preserves_semantics_and_checkpoint(dtype, training, eps):
    owner = nn.Module()
    owner.norm = nn.RMSNorm(8, eps=eps, elementwise_affine=False).train(training)
    reference = deepcopy(owner.norm)
    state = deepcopy(owner.state_dict())
    assert vae_ops._replace_rms_norm(owner, "norm")
    norm = owner.norm
    assert isinstance(norm, nn.RMSNorm)
    assert norm.weight is None and not norm.elementwise_affine
    assert norm.normalized_shape == (8,) and norm.eps == eps
    assert norm.training == training
    assert not list(norm.parameters())
    assert owner.state_dict().keys() == state.keys()
    owner.load_state_dict(state, strict=True)
    gamma = dict(norm.named_buffers())["_unit_weight"]
    assert gamma.shape == (8,) and gamma.dtype == torch.float32
    assert not gamma.requires_grad
    torch.testing.assert_close(gamma, torch.ones(8), rtol=0, atol=0)
    assert not vae_ops._replace_rms_norm(owner, "norm")
    assert owner.norm is norm
    # Per-head Q/K chunks are strided; normalization must stay on the last axis.
    value = torch.randn(2, 3, 2, 24, dtype=dtype).chunk(3, dim=-1)[1]
    assert not value.is_contiguous()
    torch.testing.assert_close(norm(value), reference(value), rtol=0, atol=0)
    owner.to(dtype=dtype)
    assert norm._unit_weight.dtype == dtype
    assert not owner.state_dict()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_non_affine_attention_preserves_fp32_norm_and_qk_output_dtype(dtype):
    attn = Attention().to(dtype=dtype)
    reference = deepcopy(attn)
    model = nn.Module()
    model.decoder = nn.Module()
    block = Block()
    block.attn = attn
    model.decoder.transformer_blocks = nn.ModuleList([block])
    vae_ops.patch_minimax_h3_video_vae(model)
    inputs, outputs = [], []
    for norm in (attn.norm_q, attn.norm_k):
        norm.register_forward_pre_hook(lambda _module, args: inputs.append(args[0].dtype))
        norm.register_forward_hook(lambda _module, _args, result: outputs.append(result.dtype))
    qk_dtypes = []

    def record_qk_dtype(q, k, v, pack):
        qk_dtypes.append((q.dtype, k.dtype))
        return q

    attn.perform_attention = record_qk_dtype
    reference.perform_attention = lambda q, k, v, pack: q
    value = torch.randn(1, 3, 16, dtype=dtype)
    torch.testing.assert_close(attn(value), reference(value), rtol=0, atol=0)
    assert inputs == outputs == [torch.float32, torch.float32]
    assert qk_dtypes == [(dtype, dtype)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,coefficient_batch,rank", [(8, 2, 4), (6, 2, 4), (6, 1, 3)])
def test_native_rope_preserves_batch_and_suffix(dtype, width, coefficient_batch, rank):
    torch.manual_seed(5985)
    value = torch.randn(2, 3, 2, 8, dtype=dtype)
    cos = torch.randn(coefficient_batch, 3, 1, width)
    sin = torch.randn_like(cos)
    if rank == 3:
        cos, sin = cos.squeeze(2), sin.squeeze(2)
    actual = vae_ops._apply_h3_omni_rope(value, cos, sin)
    expected = _reference_rope(value, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == dtype
    assert torch.equal(actual[..., width:], value[..., width:])


@pytest.mark.parametrize(
    "value_shape,cos_shape,sin_shape",
    [
        ((2, 3, 8), (2, 3, 1, 6), (2, 3, 1, 6)),
        ((2, 3, 2, 8), (2, 3, 1, 6), (2, 3, 1, 8)),
        ((2, 3, 2, 8), (3, 6), (3, 6)),
        ((2, 3, 2, 8), (2, 3, 2, 6), (2, 3, 2, 6)),
        ((2, 3, 2, 8), (3, 3, 1, 6), (3, 3, 1, 6)),
        ((2, 3, 2, 8), (2, 2, 1, 6), (2, 2, 1, 6)),
        ((2, 3, 2, 8), (2, 3, 1, 0), (2, 3, 1, 0)),
        ((2, 3, 2, 8), (2, 3, 1, 3), (2, 3, 1, 3)),
        ((2, 3, 2, 8), (2, 3, 1, 10), (2, 3, 1, 10)),
    ],
)
def test_rope_rejects_incompatible_layouts(value_shape, cos_shape, sin_shape):
    with pytest.raises(ValueError):
        vae_ops._apply_h3_omni_rope(torch.zeros(value_shape), torch.zeros(cos_shape), torch.zeros(sin_shape))


def test_rope_rejects_coefficient_device_mismatch():
    with pytest.raises(ValueError):
        vae_ops._apply_h3_omni_rope(
            torch.zeros(2, 3, 2, 8), torch.empty(2, 3, 1, 6, device="meta"), torch.zeros(2, 3, 1, 6)
        )


_NPU_DEVICE = SimpleNamespace(type="npu", index=0)


class _NPUTensor(torch.Tensor):
    """CPU storage with an NPU device tag, used only to exercise routing."""

    @property
    def device(self):
        return _NPU_DEVICE


@pytest.mark.parametrize("buffer_dtype", [torch.float32, torch.float16])
def test_non_affine_norm_dispatches_to_npu_and_reuses_registered_gamma(monkeypatch, buffer_dtype):
    monkeypatch.setattr(vae_ops.current_omni_platform, "is_npu", lambda: True)
    owner = nn.Module()
    owner.norm = nn.RMSNorm(64, eps=3e-5, elementwise_affine=False)
    assert vae_ops._replace_rms_norm(owner, "norm")
    norm = owner.norm
    # Only the device tag and kernel are doubles; run the real adapter forward.
    norm._unit_weight = norm._unit_weight.to(dtype=buffer_dtype).as_subclass(_NPUTensor)
    gamma = norm._unit_weight
    value = torch.randn(1, 7, 32, 192).chunk(3, dim=-1)[1].as_subclass(_NPUTensor)
    expected = F.rms_norm(value.as_subclass(torch.Tensor), (64,), None, 3e-5)
    calls = []

    def npu_rms_norm(actual_value, *, gamma, epsilon):
        calls.append((actual_value, gamma, epsilon))
        assert actual_value.is_contiguous()
        assert actual_value.shape == value.shape
        assert gamma.dtype == actual_value.dtype == torch.float32
        torch.testing.assert_close(actual_value.as_subclass(torch.Tensor), value.as_subclass(torch.Tensor))
        torch.testing.assert_close(gamma.as_subclass(torch.Tensor), torch.ones(64))
        return F.rms_norm(actual_value.as_subclass(torch.Tensor), (64,), None, epsilon), None

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rms_norm=npu_rms_norm))
    for _ in range(2):
        torch.testing.assert_close(norm(value), expected, rtol=0, atol=0)
    assert len(calls) == 2
    assert all(call[2] == 3e-5 for call in calls)
    if buffer_dtype == torch.float32:
        assert all(call[1] is gamma for call in calls)
    assert norm._unit_weight is gamma
    assert gamma.dtype == buffer_dtype
    assert not owner.state_dict()


@pytest.mark.parametrize(
    "npu_platform,npu_tensor,dtype,sequence",
    [
        (True, False, torch.float32, 3),
        (False, True, torch.float32, 3),
        (True, True, torch.float64, 3),
        (True, True, torch.float32, 0),
    ],
)
def test_non_affine_norm_uses_native_outside_npu_scope(monkeypatch, npu_platform, npu_tensor, dtype, sequence):
    monkeypatch.setattr(vae_ops.current_omni_platform, "is_npu", lambda: npu_platform)
    owner = nn.Module()
    owner.norm = nn.RMSNorm(8, eps=1e-5, elementwise_affine=False)
    reference = deepcopy(owner.norm)
    assert vae_ops._replace_rms_norm(owner, "norm")
    value = torch.randn(1, sequence, 2, 8, dtype=dtype)
    expected = reference(value)
    if npu_tensor:
        value = value.as_subclass(_NPUTensor)

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("NPU RMSNorm must not run outside its supported scope")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rms_norm=unexpected_kernel))
    torch.testing.assert_close(owner.norm(value).as_subclass(torch.Tensor), expected, rtol=0, atol=0)


def test_selected_non_affine_npu_kernel_failure_propagates(monkeypatch):
    monkeypatch.setattr(vae_ops.current_omni_platform, "is_npu", lambda: True)
    owner = nn.Module()
    owner.norm = nn.RMSNorm(8, eps=1e-5, elementwise_affine=False)
    assert vae_ops._replace_rms_norm(owner, "norm")
    owner.norm._unit_weight = owner.norm._unit_weight.as_subclass(_NPUTensor)

    def failing_kernel(*args, **kwargs):
        raise RuntimeError("NPU RMSNorm execution failed")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rms_norm=failing_kernel))
    with pytest.raises(RuntimeError, match="NPU RMSNorm execution failed"):
        owner.norm(torch.ones(1, 3, 2, 8).as_subclass(_NPUTensor))


def test_replaced_rms_norm_dispatches_to_real_npu_forward(monkeypatch):
    monkeypatch.setattr(vae_ops.current_omni_platform, "is_npu", lambda: True)
    owner = nn.Module()
    owner.norm = nn.RMSNorm(8, eps=3e-5)
    with torch.no_grad():
        owner.norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
    assert vae_ops._replace_rms_norm(owner, "norm")
    value = torch.randn(2, 3, 8).as_subclass(_NPUTensor)
    expected = torch.full_like(value, 17)
    calls = []

    def npu_rms_norm(actual_value, *, gamma, epsilon):
        calls.append((actual_value, gamma, epsilon))
        return expected, None

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rms_norm=npu_rms_norm))
    actual = owner.norm(value)
    assert actual is expected
    assert len(calls) == 1
    assert calls[0][0] is value
    assert calls[0][1] is owner.norm.weight
    assert calls[0][2] == 3e-5


@pytest.mark.parametrize(
    "npu_tensor,npu_platform,batch,sequence,dtype,expected_fused",
    [
        (True, True, 1, 3, torch.float32, True),
        (True, True, 1, 3, torch.float16, True),
        (True, True, 1, 3, torch.bfloat16, True),
        (False, True, 1, 3, torch.float32, False),
        (True, False, 1, 3, torch.float32, False),
        (True, True, 2, 3, torch.float32, True),
        (True, True, 4, 3, torch.float32, True),
        (True, True, 1, 0, torch.float32, False),
        (True, True, 1, 3, torch.float64, False),
    ],
)
def test_npu_rope_routing_without_mindie(monkeypatch, npu_tensor, npu_platform, batch, sequence, dtype, expected_fused):
    monkeypatch.setattr(vae_ops.current_omni_platform, "is_npu", lambda: npu_platform)
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    value = torch.randn(batch, sequence, 2, 8, dtype=dtype)
    cos = torch.randn(batch, sequence, 1, 6, dtype=dtype)
    sin = torch.randn_like(cos)
    if npu_tensor:
        value, cos, sin = (item.as_subclass(_NPUTensor) for item in (value, cos, sin))
    calls = []

    def fused(prefix, actual_cos, actual_sin):
        calls.append((prefix, actual_cos, actual_sin))
        return torch.full_like(prefix, 17)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=fused))
    actual = vae_ops._apply_h3_omni_rope(value, cos, sin)
    assert bool(calls) == expected_fused
    if expected_fused:
        assert calls[0][0].shape == (batch, sequence, 2, 6)
        assert calls[0][1].shape == (batch, sequence, 1, 6)
        assert torch.equal(actual[..., :6], torch.full_like(value[..., :6], 17))
    else:
        torch.testing.assert_close(actual, _reference_rope(value, cos, sin), rtol=0, atol=0)
    assert torch.equal(actual[..., 6:], value[..., 6:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,coefficient_batch", [(2, 1), (2, 2), (4, 1), (4, 4)])
@pytest.mark.parametrize("rank", [3, 4])
def test_npu_rope_preserves_batch_coefficients_and_partial_width(monkeypatch, dtype, batch, coefficient_batch, rank):
    monkeypatch.setattr(vae_ops.current_omni_platform, "is_npu", lambda: True)
    torch.manual_seed(5985)
    # Match H3's per-head QKV packing and partial rotary prefix strides.
    value = torch.randn(batch, 5, 2, 3 * 64, dtype=dtype).chunk(3, dim=-1)[0]
    angles = torch.randn(coefficient_batch, 5, 1, 48)
    cos, sin = angles.cos(), angles.sin()
    if coefficient_batch > 1:
        assert not torch.equal(cos[0], cos[1])
    if rank == 3:
        cos, sin = cos.squeeze(2), sin.squeeze(2)
    value, cos, sin = (item.as_subclass(_NPUTensor) for item in (value, cos, sin))
    assert not value.is_contiguous()
    calls = []

    def fused(prefix, actual_cos, actual_sin):
        calls.append(prefix)
        assert prefix.shape == (batch, 5, 2, 48)
        assert actual_cos.shape == actual_sin.shape == (coefficient_batch, 5, 1, 48)
        assert all(item.is_contiguous() and item.dtype == dtype for item in (prefix, actual_cos, actual_sin))
        expected_cos = cos.unsqueeze(2) if rank == 3 else cos
        expected_sin = sin.unsqueeze(2) if rank == 3 else sin
        torch.testing.assert_close(actual_cos, expected_cos.to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(actual_sin, expected_sin.to(dtype), rtol=0, atol=0)
        return _reference_rope(prefix, actual_cos, actual_sin)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=fused))
    actual = vae_ops._apply_h3_omni_rope(value, cos, sin)
    assert len(calls) == 1
    torch.testing.assert_close(actual, _reference_rope(value, cos, sin), rtol=0, atol=0)
    assert torch.equal(actual[..., 48:], value[..., 48:])


def test_selected_npu_rope_failure_propagates(monkeypatch):
    monkeypatch.setattr(vae_ops.current_omni_platform, "is_npu", lambda: True)
    value = torch.ones(1, 3, 2, 8).as_subclass(_NPUTensor)
    cos = torch.ones(1, 3, 1, 6).as_subclass(_NPUTensor)

    def failing_kernel(*args, **kwargs):
        raise RuntimeError("NPU RoPE execution failed")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=failing_kernel))
    with pytest.raises(RuntimeError, match="NPU RoPE execution failed"):
        vae_ops._apply_h3_omni_rope(value, cos, cos)


def test_attention_retains_per_head_qkv_packing_and_pack_info(monkeypatch):
    model = _model()
    attn = model.decoder.transformer_blocks[0].attn
    attn.norm_q = attn.norm_k = None
    with torch.no_grad():
        attn.to_qkv.weight.zero_()
        attn.to_qkv.bias.copy_(torch.arange(48))
        attn.to_out.weight.copy_(torch.eye(16))
        attn.to_out.bias.zero_()
    pack_info = {"opaque": object()}
    received = []

    def attention(query, key, value, actual_pack_info):
        received.append(actual_pack_info)
        return query + 100 * key + 10000 * value

    monkeypatch.setattr(attn, "perform_attention", attention)
    vae_ops.patch_minimax_h3_video_vae(model)
    actual = attn(torch.zeros(2, 3, 16), pack_info=pack_info)
    packed = torch.arange(48).reshape(2, 24)
    query, key, value = packed.chunk(3, dim=-1)
    expected = (query + 100 * key + 10000 * value).reshape(16).expand(2, 3, 16).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert received[0] is pack_info
    attn(torch.zeros(2, 3, 16))
    assert received[1] == {}


def test_spatial_attention_preserves_communication_and_operation_order(monkeypatch):
    model = _model()
    attn = model.decoder.transformer_blocks[0].attn
    attn.spatial_parallel = 1
    events, group = [], object()

    class Norm(nn.Module):
        def __init__(self, label):
            super().__init__()
            self.label = label

        def forward(self, value):
            events.append(self.label)
            assert value.dtype == torch.float32
            return value

    def all_to_all(value, scatter_dim, gather_dim, *, group):
        events.append((scatter_dim, gather_dim, group))
        return value

    def attention(query, key, value, pack_info):
        events.append("attention")
        return query

    def rope(value, cos, sin):
        events.append("rope")
        return value

    monkeypatch.setitem(Attention.forward.__globals__, "all_to_all_4D", all_to_all)
    monkeypatch.setitem(Attention.forward.__globals__, "get_parallel_state", lambda: {"sp_process_group": group})
    attn.norm_q, attn.norm_k = Norm("norm_q"), Norm("norm_k")
    monkeypatch.setattr(attn, "perform_attention", attention)
    monkeypatch.setattr(vae_ops, "_apply_h3_omni_rope", rope)
    vae_ops.patch_minimax_h3_video_vae(model)
    attn.to_qkv.register_forward_hook(lambda *_args: events.append("qkv"))
    attn.to_out.register_forward_hook(lambda *_args: events.append("out"))
    # Helpers are captured on this instance, independent of subsequent global edits.
    monkeypatch.setitem(Attention.forward.__globals__, "get_parallel_state", lambda: {})
    coefficients = torch.zeros(1, 3, 1, 6)
    attn(torch.ones(1, 3, 16), (coefficients, coefficients))
    assert events == [
        "qkv",
        (2, 1, group),
        (2, 1, group),
        (2, 1, group),
        "norm_q",
        "norm_k",
        "rope",
        "rope",
        "attention",
        (1, 2, group),
        "out",
    ]


@pytest.mark.parametrize("with_rope", [False, True])
def test_full_block_matches_remote_reference_and_patch_is_idempotent(with_rope):
    torch.manual_seed(29)
    reference = _model(2)
    actual = deepcopy(reference)
    value = torch.randn(2, 3, 16)
    coefficients = (torch.randn(2, 3, 1, 6), torch.randn(2, 3, 1, 6)) if with_rope else None
    keys = set(actual.state_dict())
    vae_ops.patch_minimax_h3_video_vae(actual)
    norm = actual.decoder.transformer_blocks[0].norm1
    q_norm = actual.decoder.transformer_blocks[0].attn.norm_q
    forward = actual.decoder.transformer_blocks[0].attn.forward
    vae_ops.patch_minimax_h3_video_vae(actual)
    assert actual.decoder.transformer_blocks[0].norm1 is norm
    assert actual.decoder.transformer_blocks[0].attn.norm_q is q_norm
    assert actual.decoder.transformer_blocks[0].attn.forward is forward
    assert set(actual.state_dict()) == keys
    for expected_block, actual_block in zip(reference.decoder.transformer_blocks, actual.decoder.transformer_blocks):
        torch.testing.assert_close(actual_block(value, coefficients), expected_block(value, coefficients))


@pytest.mark.parametrize("owner", ["model", "decoder", "block", "attn", "norm1", "norm_q"])
def test_preflight_rejects_compiled_modules_before_any_mutation(owner):
    model = _model(2)
    last = model.decoder.transformer_blocks[-1]
    owners = {
        "model": model,
        "decoder": model.decoder,
        "block": last,
        "attn": last.attn,
        "norm1": last.norm1,
        "norm_q": last.attn.norm_q,
    }
    owners[owner]._compiled_forward = lambda: None
    first = model.decoder.transformer_blocks[0]
    old_norm, old_forward = first.norm1, first.attn.forward.__func__
    with pytest.raises((ValueError, RuntimeError)):
        vae_ops.patch_minimax_h3_video_vae(model)
    assert first.norm1 is old_norm
    assert first.attn.forward.__func__ is old_forward
    assert not getattr(first.attn, "_minimax_h3_omni_ops_patched", False)


@pytest.mark.parametrize(
    "broken",
    [
        "attention",
        "norm",
        "head_dim",
        "norm_shape",
        "non_affine_shape",
        "all_to_all_4D",
        "get_parallel_state",
        "_vit_norm_input",
    ],
)
def test_preflight_checks_all_blocks_and_helpers_before_mutation(monkeypatch, broken):
    model = _model(2)
    first, last = model.decoder.transformer_blocks
    if broken == "attention":
        last.attn.perform_attention = None
    elif broken == "norm":
        del last.norm2
    elif broken == "head_dim":
        last.attn.dim_head = 0
    elif broken == "norm_shape":
        last.norm2 = nn.RMSNorm((2, 8), eps=1e-5)
    elif broken == "non_affine_shape":
        last.attn.norm_q = nn.RMSNorm((2, 8), eps=1e-5, elementwise_affine=False)
    else:
        monkeypatch.setitem(Attention.forward.__globals__, broken, None)
    old_norm, old_forward = first.norm1, first.attn.forward.__func__
    with pytest.raises((ValueError, RuntimeError)):
        vae_ops.patch_minimax_h3_video_vae(model)
    assert first.norm1 is old_norm
    assert first.attn.forward.__func__ is old_forward
    assert not getattr(first.attn, "_minimax_h3_omni_ops_patched", False)


@pytest.mark.parametrize("blocks", [None, [], nn.ModuleList()])
def test_preflight_requires_nonempty_module_list(blocks):
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.transformer_blocks = blocks
    with pytest.raises((ValueError, RuntimeError)):
        vae_ops.patch_minimax_h3_video_vae(model)


@pytest.fixture
def cpu_int8_kernels(monkeypatch):
    """Retain the installer, wrapper, and NPUInt8OnlineLinearMethod."""
    state = SimpleNamespace(matmuls=0)

    def dynamic_quant(value):
        scale = value.float().abs().amax(-1).clamp_min(1e-12) / 127
        quantized = (value.float() / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
        return quantized, scale

    def quant_matmul(value, weight, weight_scale, *, bias, pertoken_scale, output_dtype):
        assert bias is None
        state.matmuls += 1
        result = (value.float() * pertoken_scale[:, None]) @ (weight.float() * weight_scale[None, :])
        return result.to(output_dtype)

    @contextmanager
    def autocast(device_type, enabled=True, **kwargs):
        assert device_type == "npu" and not enabled
        yield

    original_is_autocast_enabled = torch.is_autocast_enabled
    monkeypatch.setattr(vae_int8, "_require_npu_device", lambda device: None)
    monkeypatch.setattr(
        int8_config, "torch_npu", SimpleNamespace(npu_dynamic_quant=dynamic_quant, npu_quant_matmul=quant_matmul)
    )
    monkeypatch.setattr(torch, "autocast", autocast)
    monkeypatch.setattr(
        torch,
        "is_autocast_enabled",
        lambda device_type="cuda": False if device_type == "npu" else original_is_autocast_enabled(device_type),
    )
    return state


@pytest.mark.skipif(not HAS_VAE_INT8, reason="Requires VAE INT8")
@pytest.mark.parametrize("ignored", [False, True])
def test_combo_uses_real_int8_install_and_all_four_linears(cpu_int8_kernels, ignored):
    torch.manual_seed(17)
    int8_only = _model()
    combo = deepcopy(int8_only)
    paths = (("attn", "to_qkv"), ("attn", "to_out"), ("ff", "w1"), ("ff", "w2"))
    config = DiffusionInt8Config(
        is_checkpoint_int8_serialized=False,
        activation_scheme="dynamic",
        ignored_layers=["video_vae.decoder.transformer_blocks.0.ff.w2"] if ignored else [],
    )
    vae_ops.patch_minimax_h3_video_vae(combo)
    for model in (int8_only, combo):
        report = vae_int8.install_h3_video_vae_int8(
            model.decoder,
            config,
            quant_device=torch.device("cpu"),
            storage_device=torch.device("cpu"),
            compute_dtype=torch.float16,
        )
        assert report.quantized_count == 4 - int(ignored)
    reference_block = int8_only.decoder.transformer_blocks[0]
    combo_block = combo.decoder.transformer_blocks[0]
    calls, installed = [], []
    for parent, leaf in paths:
        reference = getattr(getattr(reference_block, parent), leaf)
        actual = getattr(getattr(combo_block, parent), leaf)
        installed.append(actual)
        if isinstance(actual, vae_int8.H3VAEInt8Linear):
            assert type(actual._int8_method) is NPUInt8OnlineLinearMethod
            assert torch.equal(actual.weight, reference.weight)
            assert torch.equal(actual.weight_scale, reference.weight_scale)
        actual.register_forward_hook(lambda _module, _args, _output, path=(parent, leaf): calls.append(path))
    value = torch.randn(2, 3, 16)
    coefficients = (torch.randn(2, 3, 1, 6), torch.randn(2, 3, 1, 6))
    with torch.inference_mode():
        expected = reference_block(value, coefficients)
        before = cpu_int8_kernels.matmuls
        actual = combo_block(value, coefficients)
    assert cpu_int8_kernels.matmuls - before == 4 - int(ignored)
    assert calls == list(paths)
    # Both paths share fixed quantized weights; this measures only the adapter's change.
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    vae_ops.patch_minimax_h3_video_vae(combo)
    assert all(
        getattr(getattr(combo_block, parent), leaf) is module for (parent, leaf), module in zip(paths, installed)
    )
