# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for #2686: pre-quantized methods must not apply
quant config to vision / audio encoders.

For modelopt FP8/FP4/MXFP8 checkpoints the Thinker LM is the only
quantized component.  Vision and audio encoder weights are BF16 with no
FP8 scale tensors — passing quant_config to them causes FP8 kernels to
run on BF16 weights, producing garbage embeddings.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm_omni.quantization.component_config import (
    PRE_QUANTIZED_METHODS,
    ComponentQuantizationConfig,
    resolve_encoder_quant_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ---------------------------------------------------------------------------
# resolve_encoder_quant_config — the core routing logic for encoder quant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(PRE_QUANTIZED_METHODS))
def test_pre_quantized_returns_none(method: str) -> None:
    """visual_quant_config and audio_quant_config must be None for
    pre-quantized methods (modelopt, modelopt_fp4, modelopt_mxfp8)."""
    mock_config = MagicMock()
    mock_config.get_name.return_value = method

    assert resolve_encoder_quant_config(mock_config) is None


@pytest.mark.parametrize("method", ["fp8", "awq", "gptq", "bitsandbytes"])
def test_non_pre_quantized_preserves_config(method: str) -> None:
    """Non-pre-quantized methods should pass through the original config."""
    mock_config = MagicMock()
    mock_config.get_name.return_value = method

    assert resolve_encoder_quant_config(mock_config) is mock_config


def test_none_input_returns_none() -> None:
    """No quantization → None for encoders."""
    assert resolve_encoder_quant_config(None) is None


def test_component_config_passed_through() -> None:
    """ComponentQuantizationConfig should be returned as-is so the caller
    can call .resolve() with the appropriate prefix."""
    inner = MagicMock()
    inner.get_name.return_value = "modelopt"  # would be None if not Component
    component = ComponentQuantizationConfig(
        component_configs={"language_model": inner},
        default_config=None,
    )

    result = resolve_encoder_quant_config(component)
    assert result is component


# ---------------------------------------------------------------------------
# PRE_QUANTIZED_METHODS constant — exhaustiveness check
# ---------------------------------------------------------------------------


def test_pre_quantized_methods_contains_expected() -> None:
    """Guard against accidental removal of a known pre-quantized method."""
    expected = {"modelopt", "modelopt_fp4", "modelopt_mxfp8"}
    assert PRE_QUANTIZED_METHODS == expected
