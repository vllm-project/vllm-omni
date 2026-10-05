# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for quantization factory focusing on:
- Ensuring vLLM Omni's overrides properly hook into vLLM's quantization registry
- Quantization name resolution
"""

import subprocess
import sys
from collections.abc import Callable
from functools import partial
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from pytest_mock import MockerFixture
from safetensors.torch import save_file
from transformers import LlamaConfig
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig
from vllm_omni.quantization.component_config import ComponentQuantizationConfig
from vllm_omni.quantization.factory import (
    METHOD_KEY,
    QUANT_METHOD_KEY,
    _normalize_quant_method_alias,
    build_quantization_config,
    get_quantization_method,
    get_stage_quantization_config,
)
from vllm_omni.quantization.fp8_config import OmniFp8Config
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
    # NOTE: vLLM keeps its own mxfp4 (MoE W4A16) / mxfp8 (ModelOpt) and diffusion resolves
    # DiffusionMXFP4Config / DiffusionMXFP8Config per stage for now
    "mxfp8": ConfigCase(ModelOptMxFp8Config, DiffusionMXFP8Config),
    "mxfp4": ConfigCase(Mxfp4Config, DiffusionMXFP4Config),
    "mxfp4_dualscale": ConfigCase(DiffusionMXFP4DualScaleMixedConfig, DiffusionMXFP4DualScaleMixedConfig),
    "svdquant": ConfigCase(DiffusionSVDQuantConfig, DiffusionSVDQuantConfig),
    "inc": ConfigCase(OmniINCConfig, partial(OmniINCConfig, weight_bits=4, group_size=128)),
    "torchao": ConfigCase(OmniTorchAOConfig, _make_torchao_config),
    "torchao_float8_weight_only": ConfigCase(OmniTorchAOFloat8WeightOnlyConfig, _make_torchao_float8_config),
    "fp8": ConfigCase(OmniFp8Config, OmniFp8Config),
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


def test_mxfp4_resolves_per_stage_type():
    """Ensure diffusion mxfp4 builds Omni's W4A4 config while other stages keep vLLM's MXFP4.

    We need this because omni's mxfp4 for diffusion and vLLM's mxfp4 for AR are different."""
    assert type(build_quantization_config("mxfp4")) is DiffusionMXFP4Config
    assert type(build_quantization_config("mxfp4", is_diffusion=False)) is Mxfp4Config


def test_mxfp8_resolves_per_stage_type():
    """Ensure diffusion mxfp8 builds Omni's config while other stages keep vLLM's ModelOpt MXFP8.

    We need this because vLLM's mxfp8 supports platforms that omni's diffusion mxfp8 does not."""
    checkpoint = {QUANT_METHOD_KEY: "mxfp8"}
    assert type(build_quantization_config("mxfp8", checkpoint)) is DiffusionMXFP8Config
    assert type(build_quantization_config("mxfp8", checkpoint, is_diffusion=False)) is ModelOptMxFp8Config


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


@pytest.mark.parametrize(
    "checkpoint_config",
    [
        {QUANT_METHOD_KEY: "modelopt", "quant_algo": "NVFP4"},
        {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4"}},
    ],
)
def test_explicit_method_cannot_override_checkpoint_method(checkpoint_config):
    """Ensure that we raise if a checkpoint format & provided method are in conflict."""
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        build_quantization_config("fp8", checkpoint_config)


@pytest.mark.parametrize("method", [None, "modelopt"])
def test_legacy_modelopt_metadata_without_method_key_is_detected(method):
    """Ensure producer/quant_algo ModelOpt checkpoint metadata resolves without a method key,
    whether or not --quantization modelopt is passed explicitly (as vLLM accepts)."""
    legacy = {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "FP8"}}
    config = build_quantization_config(method, legacy)
    assert config is not None
    assert config.get_name() == "modelopt"


@pytest.mark.parametrize(
    "method, quant_algo, expected",
    [
        ("fp8", "FP8", "modelopt"),
        ("fp4", "NVFP4", "modelopt_fp4"),
        ("nvfp4", "NVFP4", "modelopt_fp4"),
    ],
)
def test_generic_request_adopts_modelopt_checkpoint(method, quant_algo, expected):
    """Ensure generic methods (fp8, fp4, nvfp4) resolve to the matching ModelOpt checkpoint config."""
    checkpoint = {QUANT_METHOD_KEY: "modelopt", "quant_algo": quant_algo}
    assert build_quantization_config(method, checkpoint).get_name() == expected


def test_non_modelopt_metadata_without_method_key_stays_unquantized():
    """Ensure a checkpoint mapping with no method key and no ModelOpt markers is unquantized."""
    assert build_quantization_config(None, {"foo": "bar"}) is None
    with pytest.raises(ValueError, match="must have a"):
        build_quantization_config({"foo": "bar"})


def test_stage_quantization_config_uses_model_revision(mocker: MockerFixture) -> None:
    """Ensure stage quant config uses the model revision."""
    read_checkpoint_config = mocker.patch(
        "vllm_omni.quantization.factory.read_checkpoint_quantization_config",
        return_value=None,
    )
    get_hf_config = mocker.patch(
        "vllm_omni.config.config_factory.StageConfigFactory.get_hf_config",
        return_value=None,
    )

    result = get_stage_quantization_config(
        "model",
        None,
        revision="revision",
        stage_type="llm",
        trust_remote_code=True,
        hf_config_name=None,
    )

    assert result is None
    read_checkpoint_config.assert_called_once_with(model="model", revision="revision")
    get_hf_config.assert_called_once_with(model="model", trust_remote_code=True, revision="revision")


@pytest.mark.parametrize("method", ["mxfp4", "mxfp8"])
def test_registered_config_handles_linear_on_cuda(mocker: MockerFixture, monkeypatch, method: str) -> None:
    """Ensure the registered config handles linear layers on CUDA, like vLLM's own class."""
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    for platform_check in ("is_npu", "is_rocm", "is_xpu"):
        monkeypatch.setattr(current_omni_platform, platform_check, lambda: False)
    config = get_quantization_config(method).from_config({QUANT_METHOD_KEY: method})

    assert config.get_quant_method(mocker.Mock(spec=LinearBase), "layer") is not None


def test_import_vllm_omni_loads_quack_fp8_patch() -> None:
    """Ensure importing vllm_omni loads the quack FP8 patch (no quantization.factory <-> omni_config cycle)."""
    code = "import sys, vllm_omni; assert 'vllm_omni.quantization.quack_fp8' in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_stage_quantization_config_detects_unquantized_modules(tmp_path: Path) -> None:
    """Ensure an AWQ LLM stage config skips modules whose checkpoint weights are unquantized."""
    awq = {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True}
    LlamaConfig(quantization_config=awq).save_pretrained(tmp_path)
    weights = {"quantized.qweight": torch.zeros(1, dtype=torch.int32), "unquantized.weight": torch.zeros(1)}
    save_file(weights, str(tmp_path / "model.safetensors"))

    config = get_stage_quantization_config(
        str(tmp_path),
        None,
        revision=None,
        stage_type="llm",
        trust_remote_code=False,
        hf_config_name=None,
    )

    assert config.modules_to_not_convert == ["unquantized"]
