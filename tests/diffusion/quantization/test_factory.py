# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for quantization factory focusing on:
- Ensuring vLLM Omni's overrides properly hook into vLLM's quantization registry
- Quantization name resolution
"""

import pytest
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config

from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig
from vllm_omni.quantization.factory import (
    METHOD_KEY,
    QUANT_METHOD_KEY,
    _normalize_quant_method_alias,
    get_quantization_method,
)
from vllm_omni.quantization.inc_config import OmniINCConfig
from vllm_omni.quantization.int8_config import DiffusionInt8Config
from vllm_omni.quantization.mxfp4_config import (
    DiffusionMXFP4Config,
    DiffusionMXFP4DualScaleMixedConfig,
)
from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config
from vllm_omni.quantization.torchao_config import OmniTorchAOConfig, OmniTorchAOFloat8WeightOnlyConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Canonical override names must resolve to their omni config class through vLLM's
# registry. Each name equals the config's get_name(); no aliases are registered.
_CANONICAL_METHOD_TO_CLASS = {
    "int8": DiffusionInt8Config,
    "bitsandbytes": DiffusionBitsAndBytesConfig,
    "mxfp8": DiffusionMXFP8Config,
    "mxfp4": DiffusionMXFP4Config,
    "mxfp4_dualscale": DiffusionMXFP4DualScaleMixedConfig,
    "inc": OmniINCConfig,
    "torchao": OmniTorchAOConfig,
    "torchao_float8_weight_only": OmniTorchAOFloat8WeightOnlyConfig,
}

# AutoRound checkpoints are NOT registered as names (matching vLLM, which keeps
# auto-round out of the registry). Both spellings are claimed for inc via
# OmniINCConfig.override_quantization_method instead.
_AUTO_ROUND_ALIASES = ["auto-round", "auto_round"]


### Tests for override resolution & alias handling
@pytest.mark.parametrize("method, expected_cls", list(_CANONICAL_METHOD_TO_CLASS.items()))
def test_vllm_registry_resolves_override_to_omni_config(method, expected_cls):
    """Ensure overrides resolve to the vLLM Omni."""
    resolved = get_quantization_config(method)
    assert resolved is expected_cls


@pytest.mark.parametrize("alias", _AUTO_ROUND_ALIASES)
def test_auto_round_aliases_are_claimed_for_inc_not_registered(alias):
    """Ensure autoround aliases are not in the registry, but the config class handles them."""
    # Ensure we don't double register the alias into the registry itself
    with pytest.raises(ValueError):
        get_quantization_config(alias)

    # But OmniINCConfig still leverages the override hook with the alias correctly
    claimed = OmniINCConfig.override_quantization_method({"quant_method": alias}, None)
    assert claimed == "inc"


### Tests for name normalization correctness
@pytest.mark.parametrize("method", sorted(m for m in QUANTIZATION_METHODS if "-" in m))
def test_hyphenated_canonical_names_unchanged(method):
    # e.g., ensure "compressed-tensors" doesn't become "compressed_tensors"
    assert _normalize_quant_method_alias(method) == method


def test_non_alias_name_returned_verbatim():
    """Ensure nonaliased names doesn't cause case folding, hyphen to underscore, etc."""
    assert _normalize_quant_method_alias("Compressed-Tensors") == "Compressed-Tensors"


@pytest.mark.parametrize("spelling", ["auto-round", "auto_round"])
def test_auto_round_spellings_fold_to_inc(spelling):
    """Ensure autoround collapses to 'inc' and that alias is handled properly."""
    assert _normalize_quant_method_alias(spelling) == "inc"


### Checks for get / set quantization method behaviors
@pytest.mark.parametrize("key", [METHOD_KEY, QUANT_METHOD_KEY])
def test_get_quantization_method_reads_either_key(key):
    assert get_quantization_method({key: "int8"}) == "int8"


def test_get_quantization_method_missing_is_none():
    assert get_quantization_method({"activation_scheme": "dynamic"}) is None


def test_get_quantization_method_agreeing_aliases_ok():
    assert get_quantization_method({METHOD_KEY: "int8", QUANT_METHOD_KEY: "int8"}) == "int8"


def test_get_quantization_method_conflicting_aliases_raise():
    with pytest.raises(ValueError, match="Conflicting quantization method keys"):
        get_quantization_method({METHOD_KEY: "int8", QUANT_METHOD_KEY: "fp8"})
