# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for quantization factory focusing on:
- Ensuring vLLM Omni's overrides properly hook into vLLM's quantization registry
- Quantization name resolution
"""

from collections.abc import Callable
from functools import partial
from multiprocessing.reduction import ForkingPickler
from typing import NamedTuple

import pytest
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig
from vllm_omni.quantization.component_config import ComponentQuantizationConfig
from vllm_omni.quantization.factory import (
    METHOD_KEY,
    QUANT_METHOD_KEY,
    _normalize_quant_method_alias,
    build_quantization_config,
    get_quantization_method,
)
from vllm_omni.quantization.inc_config import OmniINCConfig
from vllm_omni.quantization.int8_config import DiffusionInt8Config
from vllm_omni.quantization.mxfp4_config import (
    DiffusionMXFP4Config,
    DiffusionMXFP4DualScaleMixedConfig,
)
from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config
from vllm_omni.quantization.svdquant_config import DiffusionSVDQuantConfig
from vllm_omni.quantization.torchao_config import OmniTorchAOConfig, OmniTorchAOFloat8WeightOnlyConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_torchao_config() -> QuantizationConfig:
    pytest.importorskip("torchao")
    return OmniTorchAOConfig(torchao_config={})


def _make_torchao_float8_config() -> QuantizationConfig:
    pytest.importorskip("torchao")
    return OmniTorchAOFloat8WeightOnlyConfig()


class ConfigCase(NamedTuple):
    config_cls: type[QuantizationConfig]
    constructor: Callable[[], QuantizationConfig]


# Canonical names, registry classes, and minimal valid constructors.
_CONFIG_CASES = {
    "int8": ConfigCase(DiffusionInt8Config, DiffusionInt8Config),
    "bitsandbytes": ConfigCase(DiffusionBitsAndBytesConfig, DiffusionBitsAndBytesConfig),
    "mxfp8": ConfigCase(DiffusionMXFP8Config, DiffusionMXFP8Config),
    "mxfp4": ConfigCase(DiffusionMXFP4Config, DiffusionMXFP4Config),
    "mxfp4_dualscale": ConfigCase(DiffusionMXFP4DualScaleMixedConfig, DiffusionMXFP4DualScaleMixedConfig),
    "svdquant": ConfigCase(DiffusionSVDQuantConfig, DiffusionSVDQuantConfig),
    "inc": ConfigCase(OmniINCConfig, partial(OmniINCConfig, weight_bits=4, group_size=128)),
    "torchao": ConfigCase(OmniTorchAOConfig, _make_torchao_config),
    "torchao_float8_weight_only": ConfigCase(OmniTorchAOFloat8WeightOnlyConfig, _make_torchao_float8_config),
    "fp8": ConfigCase(Fp8Config, Fp8Config),
}

# AutoRound checkpoints are NOT registered as names (matching vLLM, which keeps
# auto-round out of the registry). Both spellings are claimed for inc via
# OmniINCConfig.override_quantization_method instead.
_AUTO_ROUND_ALIASES = ["auto-round", "auto_round"]


### Tests for override resolution & alias handling
@pytest.mark.parametrize("method, case", _CONFIG_CASES.items())
def test_vllm_registry_resolves_config_class(method: str, case: ConfigCase) -> None:
    resolved = get_quantization_config(method)
    assert resolved is case.config_cls


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


### Checks for MP serialization
def test_per_component_config_preserves_built_config():
    """Ensure per component configs maintain prebuilt configs."""
    transformer_cfg = build_quantization_config("fp8")
    config = build_quantization_config({"transformer": transformer_cfg, "vae": None})

    assert isinstance(config, ComponentQuantizationConfig)
    assert config.resolve("transformer") is transformer_cfg
    assert config.resolve("vae") is None


def test_component_config_survives_multiprocessing_serialization():
    """Ensure component configs survive mp serialization."""
    config = ComponentQuantizationConfig({"transformer": Fp8Config(), "vae": None})

    restored = ForkingPickler.loads(ForkingPickler.dumps(config))

    assert restored.resolve("transformer").get_name() == "fp8"
    assert restored.resolve("vae") is None


@pytest.mark.parametrize("case", _CONFIG_CASES.values(), ids=_CONFIG_CASES)
def test_quantization_configs_survive_multiprocessing_serialization(
    case: ConfigCase,
) -> None:
    config = case.constructor()
    restored = ForkingPickler.loads(ForkingPickler.dumps(config))

    assert type(restored) is type(config)
    assert restored.get_name() == config.get_name()


def test_explicit_method_cannot_override_checkpoint_method():
    """Ensure that we raise if a checkpoint format & provided method are in conflict."""
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        build_quantization_config("fp8", {QUANT_METHOD_KEY: "modelopt", "quant_algo": "NVFP4"})
