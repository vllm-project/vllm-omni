# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for INC/AutoRound quantization via the unified framework."""

import pytest
from vllm.model_executor.layers.quantization.inc import INCConfig

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS
from vllm_omni.quantization.factory import build_quantization_config
from vllm_omni.quantization.inc_config import OmniINCConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# AutoRound checkpoint quantization_config as it appears in a transformer's
# config.json. Trailing keys are metadata INCConfig must filter out.
_AUTOROUND_CKPT = {
    "quant_method": "auto-round",
    "bits": 4,
    "group_size": 128,
    "sym": True,
    "packing_format": "auto_round:auto_gptq",
    # from_config should ignore these fields.
    "autoround_version": "0.12.0",
    "batch_size": 1,
    "iters": 0,
}


@pytest.mark.parametrize("quant_method", ["auto-round", "auto_round", "inc"])
def test_checkpoint_resolves_to_omni_inc_config(quant_method):
    """Guards the removed OmniINCConfig.maybe_upgrade(): the early build path
    must yield OmniINCConfig, not a bare INCConfig."""
    config = build_quantization_config({**_AUTOROUND_CKPT, "quant_method": quant_method})
    assert type(config) is OmniINCConfig
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4
    assert config.group_size == 128


def test_bits_mapped_to_weight_bits():
    """The 'bits' key from checkpoints should be mapped to 'weight_bits'."""
    config = build_quantization_config({"quant_method": "auto-round", "bits": 4, "group_size": 128, "sym": True})
    assert type(config) is OmniINCConfig
    assert config.weight_bits == 4


def test_initializer_folds_bits_alias():
    """__init__ folds 'bits' -> 'weight_bits' (accepts either spelling), but they need to agree."""
    # Check direct initialization
    assert OmniINCConfig(bits=4, group_size=128).weight_bits == 4
    assert OmniINCConfig(weight_bits=4, group_size=128).weight_bits == 4
    # Check init through build_quantization_config
    assert build_quantization_config({"method": "inc", "bits": 4, "group_size": 128}).weight_bits == 4


def test_initializer_folds_bits_alias_must_match():
    """Ensure bits and weight_bits cannot be different."""
    with pytest.raises(ValueError, match="Conflicting bit widths"):
        OmniINCConfig(bits=4, weight_bits=8, group_size=128)


def test_checkpoint_metadata_filtered():
    """Checkpoint metadata keys (autoround_version, batch_size, iters) must be filtered."""
    config = build_quantization_config(_AUTOROUND_CKPT)
    assert type(config) is OmniINCConfig
    assert config.weight_bits == 4


def test_early_build_from_method_and_quant_config():
    """The (method_str, checkpoint_dict) form used by build_vllm_config."""
    config = build_quantization_config("auto-round", _AUTOROUND_CKPT)
    assert type(config) is OmniINCConfig
    assert config.weight_bits == 4


def test_autoround_in_supported_methods():
    """auto-round and inc should appear in SUPPORTED_QUANTIZATION_METHODS."""
    assert "auto-round" in SUPPORTED_QUANTIZATION_METHODS
    assert "inc" in SUPPORTED_QUANTIZATION_METHODS


def test_autodetect_from_transformer_config_resolves_to_omni_inc():
    """TransformerConfig auto-detect + OmniDiffusionConfig propagation."""
    tf_config = TransformerConfig.from_dict({"quantization_config": _AUTOROUND_CKPT})
    assert tf_config.quant_method == "auto-round"
    assert type(tf_config.quant_config) is OmniINCConfig

    od_config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
    assert type(od_config.quantization_config) is OmniINCConfig
    assert od_config.quantization_config.weight_bits == 4
