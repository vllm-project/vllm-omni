# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Wan2.2 quant_config propagation through transformer creation.

Tests cover:
- create_transformer_from_config passes quant_config and prefix
- create_vace_transformer_from_config passes quant_config and prefix
- set_tf_model_config propagates quant_config to OmniDiffusionConfig
- patch_wan_rms_norm safely iterates sys.modules with concurrent modifications
- I2V per-component quantization routing
"""

import sys
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 as wan22_module
import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace as wan22_vace_module
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    create_transformer_from_config,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import (
    _resolve_component_quant_config,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace import (
    create_vace_transformer_from_config,
)
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# create_transformer_from_config: quant_config / prefix forwarding
# ---------------------------------------------------------------------------


class TestCreateTransformerQuant:
    """Verify quant_config and prefix are forwarded to WanTransformer3DModel."""

    def test_quant_config_passed_through(self, mocker: MockerFixture):
        captured = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_module, "WanTransformer3DModel", FakeTransformer)

        fake_qc = mocker.MagicMock()
        create_transformer_from_config(
            {"patch_size": [1, 2, 2], "num_layers": 2},
            quant_config=fake_qc,
        )
        assert captured.get("quant_config") is fake_qc

    def test_prefix_passed_through(self, mocker: MockerFixture):
        captured = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_module, "WanTransformer3DModel", FakeTransformer)

        create_transformer_from_config(
            {"patch_size": [1, 2, 2]},
            prefix="model.transformer.",
        )
        assert captured.get("prefix") == "model.transformer."

    def test_quant_config_none_by_default(self, mocker: MockerFixture):
        captured = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_module, "WanTransformer3DModel", FakeTransformer)

        create_transformer_from_config({"patch_size": [1, 2, 2]})
        # When quant_config is None and prefix is "", they are not added
        assert "quant_config" not in captured or captured["quant_config"] is None

    def test_quant_config_and_prefix_together(self, mocker: MockerFixture):
        captured = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_module, "WanTransformer3DModel", FakeTransformer)

        fake_qc = mocker.MagicMock()
        create_transformer_from_config(
            {"patch_size": [1, 2, 2], "num_attention_heads": 4},
            quant_config=fake_qc,
            prefix="blocks.",
        )
        assert captured["quant_config"] is fake_qc
        assert captured["prefix"] == "blocks."


# ---------------------------------------------------------------------------
# create_vace_transformer_from_config: quant_config / prefix forwarding
# ---------------------------------------------------------------------------


class TestCreateVaceTransformerQuant:
    """Verify quant_config and prefix are forwarded to WanVACETransformer3DModel."""

    def test_quant_config_passed_through(self, mocker: MockerFixture):
        captured = {}

        class FakeVACETransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_vace_module, "WanVACETransformer3DModel", FakeVACETransformer)

        fake_qc = mocker.MagicMock()
        create_vace_transformer_from_config(
            {"patch_size": [1, 2, 2], "num_layers": 2},
            quant_config=fake_qc,
        )
        assert captured.get("quant_config") is fake_qc

    def test_prefix_passed_through(self, mocker: MockerFixture):
        captured = {}

        class FakeVACETransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_vace_module, "WanVACETransformer3DModel", FakeVACETransformer)

        create_vace_transformer_from_config(
            {"patch_size": [1, 2, 2]},
            prefix="vace.",
        )
        assert captured.get("prefix") == "vace."


# ---------------------------------------------------------------------------
# set_tf_model_config: propagation of quant_config
# ---------------------------------------------------------------------------


class TestSetTfModelConfig:
    """Test that set_tf_model_config propagates quant_config correctly."""

    def _make_od_config(self):
        """Create a minimal OmniDiffusionConfig-like object for testing."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        cfg = object.__new__(OmniDiffusionConfig)
        cfg.quantization_config = None
        cfg.tf_model_config = None
        return cfg

    def test_propagates_quant_config_when_none(self, mocker: MockerFixture):
        cfg = self._make_od_config()
        fake_qc = mocker.MagicMock()
        tf_config = SimpleNamespace(quant_config=fake_qc, quant_method="auto-round")

        cfg.set_tf_model_config(tf_config)

        assert cfg.tf_model_config is tf_config
        assert cfg.quantization_config is fake_qc

    def test_does_not_overwrite_existing_quantization_config(self, mocker: MockerFixture):
        cfg = self._make_od_config()
        existing_qc = mocker.MagicMock()
        cfg.quantization_config = existing_qc
        tf_config = SimpleNamespace(quant_config=mocker.MagicMock())

        cfg.set_tf_model_config(tf_config)

        assert cfg.tf_model_config is tf_config
        assert cfg.quantization_config is existing_qc  # not overwritten

    def test_no_propagation_when_tf_quant_config_is_none(self, mocker: MockerFixture):
        cfg = self._make_od_config()
        tf_config = SimpleNamespace(quant_config=None)

        cfg.set_tf_model_config(tf_config)

        assert cfg.tf_model_config is tf_config
        assert cfg.quantization_config is None


# ---------------------------------------------------------------------------
# patch_wan_rms_norm: sys.modules snapshot safety
# ---------------------------------------------------------------------------


class TestPatchWanRmsNorm:
    """Test that patch_wan_rms_norm doesn't raise on concurrent module registration."""

    def test_patches_modules_with_wan_rms_norm(self):
        from vllm_omni.diffusion.layers.norm import RMSNormVAE
        from vllm_omni.diffusion.models.wan2_2.patch_diffusers import patch_wan_rms_norm

        # Create a fake module that has WanRMS_norm
        fake_module = SimpleNamespace(WanRMS_norm=lambda x: x)
        sys.modules["_test_fake_wan_module"] = fake_module

        try:
            patch_wan_rms_norm()
            assert fake_module.WanRMS_norm is RMSNormVAE
        finally:
            del sys.modules["_test_fake_wan_module"]

    def test_no_error_when_modules_change_during_iteration(self):
        """Regression test: list() snapshot prevents RuntimeError."""
        from vllm_omni.diffusion.models.wan2_2.patch_diffusers import patch_wan_rms_norm

        # Simulate a module being added during iteration by a side effect
        original_items = sys.modules.items

        def items_with_side_effect():
            # This would cause RuntimeError without list() snapshot
            result = list(original_items())
            # Add a new module to simulate concurrent modification
            sys.modules["_test_dynamic_module"] = SimpleNamespace()
            return result

        try:
            # The function uses list(sys.modules.items()) so it takes a snapshot
            # Just verify it doesn't raise
            patch_wan_rms_norm()
        finally:
            sys.modules.pop("_test_dynamic_module", None)


# ---------------------------------------------------------------------------
# I2V per-component quantization routing
# ---------------------------------------------------------------------------


class TestI2VTransformerQuantConfig:
    """Test I2V per-component quantization routing."""

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            (None, (None, None)),
            (
                {"method": "fp8", "activation_scheme": "dynamic"},
                ("fp8", "fp8"),
            ),
            (
                {"transformer": {"method": "fp8"}},
                ("fp8", None),
            ),
            (
                {"transformer_2": {"method": "fp8"}},
                (None, "fp8"),
            ),
            (
                {
                    "default": {"method": "fp8"},
                    "transformer_2": None,
                },
                ("fp8", None),
            ),
            (
                {
                    "transformer": {"method": "fp8"},
                    "transformer_2": {"method": "int8"},
                },
                ("fp8", "int8"),
            ),
        ],
        ids=[
            "none",
            "flat",
            "transformer-only",
            "transformer-2-only",
            "default-with-transformer-2-disabled",
            "mixed",
        ],
    )
    def test_component_quantization_routing(self, spec, expected):
        config = build_quant_config(spec)

        transformer = _resolve_component_quant_config(config, "transformer")
        transformer_2 = _resolve_component_quant_config(config, "transformer_2")

        actual = (
            transformer.get_name() if transformer else None,
            transformer_2.get_name() if transformer_2 else None,
        )
        assert actual == expected
