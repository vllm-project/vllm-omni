# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SigLIP2 vision-tower fallbacks for quantization and attention shapes it cannot use."""

import inspect
from types import SimpleNamespace

import pytest
from vllm.config.quantization import _ONLINE_SHORTHANDS
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_omni.model_executor.models.hunyuan_image3 import siglip2

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _online_quant_config(shorthand):
    if shorthand not in _ONLINE_SHORTHANDS:
        pytest.skip(f"vLLM has no online '{shorthand}' shorthand")
    return OnlineQuantizationConfig(_ONLINE_SHORTHANDS[shorthand])


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
    assert siglip2._linear_scale_block_size(_online_quant_config(shorthand)) == expected


def _encoder_quant_config(monkeypatch, quant_config, hidden_size, intermediate_size):
    """The quant_config Siglip2VisionTransformer hands to its encoder.

    The embeddings and the encoder are stubbed out: building them needs an
    initialized process group, and the decision under test is taken before that.
    """
    captured = {}

    def _fake_encoder(config, quant_config=None, prefix=""):
        captured["quant_config"] = quant_config
        return SimpleNamespace()

    monkeypatch.setattr(siglip2, "Siglip2VisionEmbeddings", lambda config: SimpleNamespace())
    monkeypatch.setattr(siglip2, "Siglip2Encoder", _fake_encoder)

    siglip2.Siglip2VisionTransformer(
        {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "layer_norm_eps": 1e-6,
        },
        quant_config=quant_config,
    )
    return captured["quant_config"]


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size", "keeps_quantization"),
    [
        # HunyuanImage-3's tower: fc2 contracts over 4304 and 4304 % 32 == 16, so the
        # MXFP8 weight-scale block does not fit and the tower must stay in checkpoint dtype.
        (1152, 4304, False),
        # Both dims 32-aligned: the tower keeps MXFP8.
        (1152, 4608, True),
    ],
)
def test_vision_tower_drops_unusable_quantization(monkeypatch, hidden_size, intermediate_size, keeps_quantization):
    quant_config = _online_quant_config("mxfp8")

    passed = _encoder_quant_config(monkeypatch, quant_config, hidden_size, intermediate_size)

    assert (passed is quant_config) == keeps_quantization
    if not keeps_quantization:
        assert passed is None


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
    ("is_xpu", "backend", "head_dim", "expected"),
    [
        # SigLIP2 on XPU: 1152 / 16 = 72, not 64-byte aligned for the XPU flash kernel.
        (True, AttentionBackendEnum.FLASH_ATTN, 72, AttentionBackendEnum.TORCH_SDPA),
        (True, AttentionBackendEnum.FLASH_ATTN, 64, AttentionBackendEnum.FLASH_ATTN),
        (True, AttentionBackendEnum.TORCH_SDPA, 72, AttentionBackendEnum.TORCH_SDPA),
        # Other platforms select the ViT backend through supports_head_size(), so
        # head_dim 72 stays on flash attention there.
        (False, AttentionBackendEnum.FLASH_ATTN, 72, AttentionBackendEnum.FLASH_ATTN),
        (False, AttentionBackendEnum.ROCM_AITER_FA, 72, AttentionBackendEnum.ROCM_AITER_FA),
    ],
)
def test_siglip2_attention_retargets_unaligned_head_dim_on_xpu(monkeypatch, is_xpu, backend, head_dim, expected):
    fake_attn_cls, config = _attention_with_backend(backend, head_dim)
    monkeypatch.setattr(siglip2, "MMEncoderAttention", fake_attn_cls)
    monkeypatch.setattr(siglip2, "current_omni_platform", SimpleNamespace(is_xpu=lambda: is_xpu))
    monkeypatch.setattr(siglip2, "is_vit_use_data_parallel", lambda: False)
    monkeypatch.setattr(siglip2, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(siglip2, "QKVParallelLinear", lambda *a, **k: None)
    monkeypatch.setattr(siglip2, "RowParallelLinear", lambda *a, **k: None)

    attn = siglip2.Siglip2Attention(config, quant_config=None, prefix="vision_model.encoder.layers.0.self_attn")

    assert attn.attn.attn_backend == expected
    if expected != backend:
        assert attn.attn.is_flash_attn_backend is False
        assert attn.attn._fa_version is None


def test_mm_encoder_attention_state_the_retarget_writes():
    """Pin the vLLM state the XPU retarget depends on.

    `Siglip2Attention` has to patch the constructed `MMEncoderAttention` because the
    constructor takes no backend argument. Fail here, rather than at runtime on XPU,
    if a vLLM bump renames that state or adds a backend argument to use instead.
    """
    init_source = inspect.getsource(MMEncoderAttention.__init__)
    for name in ("attn_backend", "is_flash_attn_backend", "_fa_version"):
        assert f"self.{name} =" in init_source, f"MMEncoderAttention no longer sets {name}"

    parameters = inspect.signature(MMEncoderAttention.__init__).parameters
    backend_parameters = [name for name in parameters if "backend" in name]
    assert not backend_parameters, (
        f"MMEncoderAttention now takes {backend_parameters}; pass the backend instead of patching the instance"
    )
