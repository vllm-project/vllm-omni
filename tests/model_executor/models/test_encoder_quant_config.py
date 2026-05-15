# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for #2686: pre-quantized methods must not apply
quant config to vision / audio encoders.

For modelopt FP8/FP4/MXFP8 checkpoints the Thinker LM is the only
quantized component.  Vision and audio encoder weights are BF16 with no
FP8 scale tensors — passing quant_config to them causes FP8 kernels to
run on BF16 weights, producing garbage embeddings.

**Import note (pytest + ``tests/model_executor`` only):** :mod:`vllm_omni.quantization`
:file:`__init__` pulls ``factory``/``inc`` and reproduces vLLM CustomOp double
registration.  We load the same :file:`component_config.py` via
:data:`importlib` under a *non-package* name so the package ``__init__`` is
never executed for this file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_CC_MODULE_NAME = "_vllm_omni_tests_component_config_encoder_quant"


def _load_component_config():
    if _CC_MODULE_NAME in sys.modules:
        return sys.modules[_CC_MODULE_NAME]
    root = Path(__file__).resolve().parents[3]
    path = root / "vllm_omni" / "quantization" / "component_config.py"
    if not path.is_file():
        raise FileNotFoundError(f"expected component_config at {path}")
    spec = importlib.util.spec_from_file_location(_CC_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to create import spec for component_config")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_CC_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


_cc = _load_component_config()
PRE_QUANTIZED_METHODS = _cc.PRE_QUANTIZED_METHODS
ComponentQuantizationConfig = _cc.ComponentQuantizationConfig
resolve_encoder_quant_config = _cc.resolve_encoder_quant_config

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
