# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for component routing for quantization."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    PRE_QUANTIZED_METHODS,
)
from vllm_omni.quantization.component_config import (
    ComponentQuantizationConfig,
)
from vllm_omni.quantization.inc_config import OmniINCConfig

pytestmark = [pytest.mark.core_model]


# ---------------------------------------------------------------------------
# Helpers: lightweight mock quant configs
# ---------------------------------------------------------------------------


class _MockQuantConfig(QuantizationConfig):
    """Minimal mock that only implements get_name()."""

    def __init__(self, name: str, **attrs):
        self._name = name
        for k, v in attrs.items():
            setattr(self, k, v)

    def get_name(self) -> str:
        return self._name

    def get_quant_method(self, layer, prefix):
        return MagicMock()

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]

    def get_min_capability(self):
        return 0

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def get_config_filenames(self):
        return []


def _make_inc_config(block_names="thinker.model.layers,talker.model.layers", extra_config=None):
    """Create a real OmniINCConfig with block_name_to_quantize."""
    return OmniINCConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        block_name_to_quantize=block_names,
        extra_config=extra_config or {},
    )


THINKER_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "thinker.lm_head.": "language_model.lm_head.",
        "thinker.model.": "language_model.model.",
        "thinker.": "",
    }
)

TALKER_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "talker.codec_head.": "language_model.lm_head.",
        "talker.model.": "language_model.model.",
        "talker.thinker_to_talker_proj.": "thinker_to_talker_proj.",
        "talker.": "",
    }
)


# ===================================================================
# 1. OmniINCConfig.apply_vllm_mapper
# ===================================================================


class TestApplyVllmMapper:
    def test_inc_csv_string_normalized_to_list(self):
        """CSV string block_name_to_quantize is split into a list."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        cfg.apply_vllm_mapper(THINKER_MAPPER)
        assert isinstance(cfg.block_name_to_quantize, list)

    def test_thinker_blocks_remapped(self):
        """thinker.model.layers -> language_model.model.layers after apply_vllm_mapper."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        cfg.apply_vllm_mapper(THINKER_MAPPER)
        assert any("language_model.model.layers" in b for b in cfg.block_name_to_quantize)

    def test_cross_stage_blocks_kept_unchanged(self):
        """Blocks not matching any mapper prefix are kept unchanged (harmless)."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        cfg.apply_vllm_mapper(THINKER_MAPPER)
        # talker.model.layers doesn't match any thinker mapper prefix → stays as-is
        assert "talker.model.layers" in cfg.block_name_to_quantize

    def test_talker_remap(self):
        """talker.model.layers -> language_model.model.layers with talker mapper."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        cfg.apply_vllm_mapper(TALKER_MAPPER)
        assert any("language_model.model.layers" in b for b in cfg.block_name_to_quantize)
        # thinker.model.layers doesn't match talker mapper → stays as-is
        assert "thinker.model.layers" in cfg.block_name_to_quantize

    def test_extra_config_keys_remapped(self):
        """Regex keys in extra_config get their escaped-dot prefixes remapped."""
        extra = {
            r".*thinker\.model\.layers\.0\.mlp\.gate.*": {"bits": 16, "data_type": "float"},
        }
        cfg = _make_inc_config("thinker.model.layers", extra_config=extra)
        cfg.apply_vllm_mapper(THINKER_MAPPER)
        # The key should now reference the vLLM runtime path
        assert any("language_model" in k for k in cfg.extra_config)
        # Original thinker\.model prefix should be replaced
        assert not any(r"thinker\.model" in k for k in cfg.extra_config)

    def test_single_block_name(self):
        """Only one block name (not CSV) still works."""
        cfg = _make_inc_config("thinker.model.layers")
        cfg.apply_vllm_mapper(THINKER_MAPPER)
        assert any("language_model.model.layers" in b for b in cfg.block_name_to_quantize)

    def test_already_list_block_names(self):
        """block_name_to_quantize already a list (not CSV string) works."""
        cfg = _make_inc_config(["thinker.model.layers", "talker.model.layers"])
        cfg.apply_vllm_mapper(THINKER_MAPPER)
        assert isinstance(cfg.block_name_to_quantize, list)
        assert any("language_model.model.layers" in b for b in cfg.block_name_to_quantize)

    def test_mutates_in_place(self):
        """apply_vllm_mapper mutates the config in place (same as upstream INCConfig)."""
        cfg = _make_inc_config("thinker.model.layers")
        original_id = id(cfg)
        cfg.apply_vllm_mapper(THINKER_MAPPER)
        assert id(cfg) == original_id

    # -- Stage prefix tests (runtime prefix = container + internal name) --

    def test_thinker_block_has_stage_prefix(self):
        """Mapped block name must start with 'thinker.' so runtime startswith() works."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        cfg.apply_vllm_mapper(THINKER_MAPPER)
        assert "thinker.language_model.model.layers" in cfg.block_name_to_quantize

    def test_talker_block_has_stage_prefix(self):
        """Mapped block name must start with 'talker.' so runtime startswith() works."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        cfg.apply_vllm_mapper(TALKER_MAPPER)
        assert "talker.language_model.model.layers" in cfg.block_name_to_quantize

    def test_thinker_block_matches_runtime_prefix(self):
        """Simulates get_layer_config's startswith() check for FusedMoE layers."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        cfg.apply_vllm_mapper(THINKER_MAPPER)
        runtime_prefix = "thinker.language_model.model.layers.0.mlp.experts"
        assert any(runtime_prefix.startswith(b) for b in cfg.block_name_to_quantize)

    def test_talker_block_matches_runtime_prefix(self):
        """Simulates get_layer_config's startswith() check for talker FusedMoE."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        cfg.apply_vllm_mapper(TALKER_MAPPER)
        runtime_prefix = "talker.language_model.model.layers.0.mlp.experts"
        assert any(runtime_prefix.startswith(b) for b in cfg.block_name_to_quantize)

    def test_extra_config_plain_key_has_stage_prefix(self):
        """Plain extra_config keys are remapped with stage prefix."""
        extra = {
            "talker.model.layers.0.mlp.shared_expert_gate": {"bits": 16},
        }
        cfg = _make_inc_config("talker.model.layers", extra_config=extra)
        cfg.apply_vllm_mapper(TALKER_MAPPER)
        assert "talker.language_model.model.layers.0.mlp.shared_expert_gate" in cfg.extra_config

    def test_extra_config_regex_key_still_works(self):
        """Regex extra_config keys use re.search so no stage prefix needed."""
        import re

        extra = {
            r".*thinker\.model\.layers\.0\.mlp\.gate.*": {"bits": 16},
        }
        cfg = _make_inc_config("thinker.model.layers", extra_config=extra)
        cfg.apply_vllm_mapper(THINKER_MAPPER)
        runtime_name = "thinker.language_model.model.layers.0.mlp.gate"
        matched = any(re.search(k, runtime_name) for k in cfg.extra_config)
        assert matched


# ===================================================================
# 2. OmniINCConfig upgrade helpers
# ===================================================================


class TestOmniINCConfigUpgrade:
    def test_maybe_upgrade_none(self):
        assert OmniINCConfig.maybe_upgrade(None) is None

    def test_maybe_upgrade_non_inc(self):
        """Non-INC configs are passed through unchanged."""
        cfg = _MockQuantConfig("fp8")
        assert OmniINCConfig.maybe_upgrade(cfg) is cfg

    def test_maybe_upgrade_already_omni(self):
        """Already OmniINCConfig is returned as-is."""
        cfg = _make_inc_config()
        assert OmniINCConfig.maybe_upgrade(cfg) is cfg

    def test_maybe_upgrade_vanilla_inc(self):
        """Vanilla INCConfig is promoted to OmniINCConfig."""
        from vllm.model_executor.layers.quantization.inc import INCConfig

        vanilla = INCConfig(weight_bits=4, group_size=128, sym=True)
        upgraded = OmniINCConfig.maybe_upgrade(vanilla)
        assert isinstance(upgraded, OmniINCConfig)
        assert upgraded.weight_bits == 4
        assert upgraded.group_size == 128


# ===================================================================
# 2. Three-branch thinker routing (simulated)
# ===================================================================


def _simulate_thinker_routing(quant_config):
    """Simulate the three-branch routing in thinker __init__.

    Returns (visual_quant_config, language_quant_config, wrapped_vllm_quant).
    """
    if isinstance(quant_config, ComponentQuantizationConfig):
        visual_quant_config = quant_config.resolve("visual")
        language_quant_config = quant_config.resolve("language_model")
        return visual_quant_config, language_quant_config, quant_config
    elif quant_config is not None:
        if quant_config.get_name() in PRE_QUANTIZED_METHODS:
            return quant_config, quant_config, quant_config
        else:
            language_quant_config = quant_config
            wrapped = ComponentQuantizationConfig(
                component_configs={"language_model": quant_config},
                default_config=None,
            )
            return None, language_quant_config, wrapped
    else:
        return None, None, None


class TestThinkerRouting:
    def test_none(self):
        vis, lang, wrapped = _simulate_thinker_routing(None)
        assert vis is None
        assert lang is None
        assert wrapped is None

    @pytest.mark.parametrize("method", ["modelopt", "modelopt_fp4", "modelopt_mxfp8"])
    def test_pre_quantized_all_components(self, method):
        """Pre-quantized methods pass config to all components."""
        cfg = _MockQuantConfig(method)
        vis, lang, wrapped = _simulate_thinker_routing(cfg)
        assert vis is cfg
        assert lang is cfg
        assert wrapped is cfg

    def test_fp8_dynamic_language_only(self):
        """fp8 dynamic: visual=None, language gets original config."""
        cfg = _MockQuantConfig("fp8")
        vis, lang, wrapped = _simulate_thinker_routing(cfg)
        assert vis is None
        assert lang is cfg
        assert isinstance(wrapped, ComponentQuantizationConfig)
        assert wrapped.resolve("language_model") is cfg
        assert wrapped.resolve("visual") is None

    def test_inc_autoround_language_only(self):
        """INC/AutoRound: not in _PRE_QUANTIZED_METHODS -> wrapped like fp8."""
        cfg = _MockQuantConfig("inc")
        vis, lang, wrapped = _simulate_thinker_routing(cfg)
        assert vis is None
        assert lang is cfg
        assert isinstance(wrapped, ComponentQuantizationConfig)

    def test_component_config_passthrough(self):
        """Explicit ComponentQuantizationConfig is used directly."""
        inner_fp8 = _MockQuantConfig("fp8")
        inner_modelopt = _MockQuantConfig("modelopt")
        cqc = ComponentQuantizationConfig(
            component_configs={
                "visual": inner_modelopt,
                "language_model": inner_fp8,
            }
        )
        vis, lang, wrapped = _simulate_thinker_routing(cqc)
        assert vis is inner_modelopt
        assert lang is inner_fp8
        assert wrapped is cqc


# ===================================================================
# 3. Talker visual routing (init_multi_modal guard)
# ===================================================================


def _simulate_talker_visual_routing(quant_config):
    """Simulate the talker init_multi_modal visual routing."""
    if quant_config is not None and quant_config.get_name() in PRE_QUANTIZED_METHODS:
        return quant_config
    return None


class TestTalkerVisualRouting:
    def test_none(self):
        assert _simulate_talker_visual_routing(None) is None

    @pytest.mark.parametrize("method", ["modelopt", "modelopt_fp4", "modelopt_mxfp8"])
    def test_pre_quantized_passes_through(self, method):
        """Pre-quantized methods pass quant config to visual."""
        cfg = _MockQuantConfig(method)
        assert _simulate_talker_visual_routing(cfg) is cfg

    def test_fp8_blocked(self):
        """fp8 dynamic must NOT be passed to visual."""
        cfg = _MockQuantConfig("fp8")
        assert _simulate_talker_visual_routing(cfg) is None

    def test_inc_blocked(self):
        """INC/AutoRound must NOT be passed to visual (not in _PRE_QUANTIZED_METHODS)."""
        cfg = _MockQuantConfig("inc")
        assert _simulate_talker_visual_routing(cfg) is None


# ===================================================================
# 4. ComponentQuantizationConfig.resolve
# ===================================================================


class TestComponentResolve:
    def test_longest_prefix_match(self):
        a = _MockQuantConfig("a")
        b = _MockQuantConfig("b")
        cqc = ComponentQuantizationConfig(component_configs={"language_model": a, "language_model.model": b})
        assert cqc.resolve("language_model.model.layers.0") is b
        assert cqc.resolve("language_model.lm_head") is a

    def test_no_match_returns_default(self):
        a = _MockQuantConfig("a")
        default = _MockQuantConfig("default")
        cqc = ComponentQuantizationConfig(
            component_configs={"language_model": a},
            default_config=default,
        )
        assert cqc.resolve("visual") is default

    def test_no_match_no_default_returns_none(self):
        a = _MockQuantConfig("a")
        cqc = ComponentQuantizationConfig(
            component_configs={"language_model": a},
        )
        assert cqc.resolve("visual") is None

    def test_get_name(self):
        cqc = ComponentQuantizationConfig(component_configs={})
        assert cqc.get_name() == "component"

    def test_get_quant_method_delegates(self):
        """get_quant_method dispatches to the resolved config."""
        inner = _MockQuantConfig("fp8")
        cqc = ComponentQuantizationConfig(
            component_configs={"language_model": inner},
        )
        layer = MagicMock()
        result = cqc.get_quant_method(layer, "language_model.model.layers.0.mlp")
        assert result is not None  # delegates to inner.get_quant_method

    def test_get_quant_method_returns_none_for_unmatched(self):
        """get_quant_method returns None when no config matches."""
        inner = _MockQuantConfig("fp8")
        cqc = ComponentQuantizationConfig(
            component_configs={"language_model": inner},
        )
        layer = MagicMock()
        result = cqc.get_quant_method(layer, "visual.blocks.0.mlp")
        assert result is None

    def test_min_capability(self):
        a = _MockQuantConfig("a")
        a.get_min_capability = lambda: 80
        b = _MockQuantConfig("b")
        b.get_min_capability = lambda: 70
        cqc = ComponentQuantizationConfig(component_configs={"x": a, "y": b})
        assert cqc.get_min_capability() == 70

    def test_min_capability_empty(self):
        cqc = ComponentQuantizationConfig(component_configs={})
        assert cqc.get_min_capability() == 0
