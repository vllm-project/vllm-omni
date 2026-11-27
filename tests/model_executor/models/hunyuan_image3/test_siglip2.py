# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SigLIP2 vision tower falls back to the checkpoint dtype for dims that block-scaled
weight quantization cannot cover. Platform-independent."""

from types import SimpleNamespace

import pytest
from vllm.config.quantization import _ONLINE_SHORTHANDS
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig

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
