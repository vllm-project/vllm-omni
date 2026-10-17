# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Quantization wiring for the DreamZero DiT.

Every vLLM parallel linear gets `quant_config` and a `prefix` equal to its
module path; small plain-`nn.Linear` projections stay bf16.

CPU-only: the parallel linears, the patch conv and the attention layer are
replaced by stand-ins.
"""

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

DIM = 24
NUM_HEADS = 2


class _FakeLinear(nn.Module):
    """Stand-in recording `quant_config` and `prefix`."""

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        *args,
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
        hidden_size: int | None = None,
        head_size: int | None = None,
        total_num_heads: int | None = None,
        total_num_kv_heads: int | None = None,
        **kwargs,
    ):
        del args, kwargs
        super().__init__()
        self.prefix = prefix
        self.quant_config = quant_config
        self.input_size = hidden_size if hidden_size is not None else input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(1))
        else:
            self.register_parameter("bias", None)
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads if total_num_kv_heads is not None else total_num_heads


class _FakeConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(1, 2, 2), **kwargs):
        del kwargs
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = tuple(kernel_size)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.enable_linear = True


class _FakeAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        del args, kwargs
        super().__init__()


@pytest.fixture
def dz(monkeypatch):
    from vllm_omni.diffusion.models.dreamzero import causal_wan_model as m

    for name in ("ColumnParallelLinear", "RowParallelLinear", "QKVParallelLinear"):
        monkeypatch.setattr(m, name, _FakeLinear)
    monkeypatch.setattr(m, "Conv3dLayer", _FakeConv3d)
    monkeypatch.setattr(m, "Attention", _FakeAttention)
    monkeypatch.setattr(m, "get_tensor_model_parallel_world_size", lambda: 1)
    return m


def _build(m, num_layers: int, quant_config=None, model_type: str = "i2v"):
    return m.CausalWanModel(
        model_type=model_type,
        frame_seqlen=4,
        text_len=8,
        in_dim=2,
        dim=DIM,
        ffn_dim=2 * DIM,
        freq_dim=8,
        text_dim=8,
        out_dim=2,
        num_heads=NUM_HEADS,
        num_layers=num_layers,
        quant_config=quant_config,
    )


def _linears(model) -> dict[str, object]:
    return {path: mod for path, mod in model.named_modules() if isinstance(mod, _FakeLinear)}


def _expected_paths(num_layers: int) -> set[str]:
    per_block = (
        "self_attn.qkv",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "cross_attn.k_img",
        "cross_attn.v_img",
        "ffn.0",
        "ffn.2",
    )
    paths = {f"blocks.{i}.{name}" for i in range(num_layers) for name in per_block}
    return paths | {"img_emb.fc1", "img_emb.fc2"}


@pytest.mark.parametrize("num_layers", [1, 2])
def test_every_gemm_linear_gets_the_quant_config(dz, num_layers):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    fp8_config = Fp8Config()
    linears = _linears(_build(dz, num_layers, quant_config=fp8_config))

    assert set(linears) == _expected_paths(num_layers)
    assert all(mod.quant_config is fp8_config for mod in linears.values())


def test_prefix_matches_module_path(dz):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    linears = _linears(_build(dz, 2, quant_config=Fp8Config()))

    assert {mod.prefix for mod in linears.values()} == set(linears)
    assert all(mod.prefix == path for path, mod in linears.items())


def test_t2v_has_no_image_projections(dz):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    linears = _linears(_build(dz, 1, quant_config=Fp8Config(), model_type="t2v"))

    assert set(linears) == {
        "blocks.0.self_attn.qkv",
        "blocks.0.self_attn.o",
        "blocks.0.cross_attn.q",
        "blocks.0.cross_attn.k",
        "blocks.0.cross_attn.v",
        "blocks.0.cross_attn.o",
        "blocks.0.ffn.0",
        "blocks.0.ffn.2",
    }


def test_default_is_unquantized_and_prefixes_still_set(dz):
    linears = _linears(_build(dz, 1))

    assert set(linears) == _expected_paths(1)
    assert all(mod.quant_config is None for mod in linears.values())
    assert all(mod.prefix == path for path, mod in linears.items())


def test_plain_linear_projections_stay_full_precision(dz):
    model = _build(dz, 1, quant_config=None)

    plain = {path for path, mod in model.named_modules() if isinstance(mod, nn.Linear)}
    assert {
        "text_embedding.0",
        "text_embedding.2",
        "time_embedding.0",
        "time_embedding.2",
        "time_projection.1",
        "head.head",
    } <= plain
    # head.head runs once per forward (not TP-critical), so it stays a plain
    # nn.Linear and takes no quant_config -- not an accuracy decision.
    assert not any(isinstance(mod, _FakeLinear) for mod in model.head.modules())
