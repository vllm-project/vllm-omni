# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SigLIP2 vision-tower fallbacks for quantization and attention shapes it cannot use."""

from types import SimpleNamespace

import pytest
from vllm.config.quantization import _ONLINE_SHORTHANDS
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_omni.model_executor.models.hunyuan_image3 import siglip2

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("shorthand", "expected"),
    [
        # MXFP8 scales every 32 input elements: SigLIP2's fc2 (in 4304) cannot take it.
        ("mxfp8", 32),
        # Per-tensor scales carry no contracted-dim block, so the tower stays quantized.
        ("fp8_per_tensor", None),
    ],
)
def test_linear_scale_block_size_reads_online_config(shorthand, expected):
    if shorthand not in _ONLINE_SHORTHANDS:
        pytest.skip(f"vLLM has no online '{shorthand}' shorthand")
    config = OnlineQuantizationConfig(_ONLINE_SHORTHANDS[shorthand])

    assert siglip2._linear_scale_block_size(config) == expected


@pytest.mark.parametrize("quant_config", [None, SimpleNamespace(), SimpleNamespace(args=SimpleNamespace(linear=None))])
def test_linear_scale_block_size_without_block_scales(quant_config):
    assert siglip2._linear_scale_block_size(quant_config) is None


def _attention_with_backend(backend, head_dim):
    """A Siglip2Attention whose __init__ ran up to the backend retarget, without
    building the TP linears (which need an initialized process group)."""

    class _FakeMMEncoderAttention:
        def __init__(self, *args, **kwargs):
            self.attn_backend = backend
            self.is_flash_attn_backend = backend == AttentionBackendEnum.FLASH_ATTN
            self._fa_version = 2

    return _FakeMMEncoderAttention, SimpleNamespace(
        hidden_size=head_dim * 16, num_attention_heads=16, attention_dropout=0.0
    )


@pytest.mark.parametrize(
    ("backend", "head_dim", "expected"),
    [
        # SigLIP2: 1152 / 16 = 72, not 64-byte aligned for the XPU flash kernel.
        (AttentionBackendEnum.FLASH_ATTN, 72, AttentionBackendEnum.TORCH_SDPA),
        (AttentionBackendEnum.FLASH_ATTN, 64, AttentionBackendEnum.FLASH_ATTN),
        (AttentionBackendEnum.TORCH_SDPA, 72, AttentionBackendEnum.TORCH_SDPA),
    ],
)
def test_siglip2_attention_retargets_unaligned_head_dim(monkeypatch, backend, head_dim, expected):
    fake_attn_cls, config = _attention_with_backend(backend, head_dim)
    monkeypatch.setattr(siglip2, "MMEncoderAttention", fake_attn_cls)
    monkeypatch.setattr(siglip2, "is_vit_use_data_parallel", lambda: False)
    monkeypatch.setattr(siglip2, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(siglip2, "QKVParallelLinear", lambda *a, **k: None)
    monkeypatch.setattr(siglip2, "RowParallelLinear", lambda *a, **k: None)

    attn = siglip2.Siglip2Attention(config, quant_config=None, prefix="vision_model.encoder.layers.0.self_attn")

    assert attn.attn.attn_backend == expected
    if expected != backend:
        assert attn.attn.is_flash_attn_backend is False
        assert attn.attn._fa_version is None
