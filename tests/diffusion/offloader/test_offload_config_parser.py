# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for canonical offload config parser (RFC #7204)."""

import json
import math

import pytest

from vllm_omni.diffusion.offloader.config_parser import (
    VALID_COMPONENTS,
    VALID_MODES,
    VALID_TRANSFERS,
    CanonicalOffloadPolicy,
    ComponentOptions,
    HostWeightRuntimePolicy,
    OffloadMode,
    WeightTransfer,
    expand_shortcut,
    parse_hwr_policy,
    parse_offload_policy,
    validate_joint,
)

# ---------------------------------------------------------------------------
# expand_shortcut
# ---------------------------------------------------------------------------


class TestExpandShortcut:
    def test_none(self):
        assert expand_shortcut("none") == {"mode": "none"}

    def test_module(self):
        result = expand_shortcut("module")
        assert result["mode"] == "module"
        assert result["components"] == {"dit": {}}
        assert result["pin_memory"] is True

    def test_layer(self):
        result = expand_shortcut("layer")
        assert result["mode"] == "layer"
        assert result["components"]["dit"]["weight_transfer"] == "rank-local"
        assert result["components"]["dit"]["resident_layers"] == 0
        assert result["pin_memory"] is True

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown diffusion-offload shortcut"):
            expand_shortcut("bogus")


# ---------------------------------------------------------------------------
# parse_offload_policy — shortcut path
# ---------------------------------------------------------------------------


class TestParseOffloadShortcut:
    def test_none(self):
        p = parse_offload_policy(diffusion_offload="none")
        assert p.mode is OffloadMode.NONE
        assert p.is_active is False
        assert p.components == {}

    def test_module(self):
        p = parse_offload_policy(diffusion_offload="module")
        assert p.mode is OffloadMode.MODULE
        assert p.is_active is True
        assert "dit" in p.components
        assert p.dit_transfer() is WeightTransfer.RANK_LOCAL

    def test_layer(self):
        p = parse_offload_policy(diffusion_offload="layer")
        assert p.mode is OffloadMode.LAYER
        assert p.is_active is True
        assert p.dit_transfer() is WeightTransfer.RANK_LOCAL
        assert p.dit_resident_layers() == 0

    def test_mutual_exclusion_with_config(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            parse_offload_policy(
                diffusion_offload="module",
                diffusion_offload_config={"mode": "layer"},
            )


# ---------------------------------------------------------------------------
# parse_offload_policy — JSON config path
# ---------------------------------------------------------------------------


class TestParseOffloadJSON:
    def test_mode_none(self):
        p = parse_offload_policy(diffusion_offload_config={"mode": "none"})
        assert p.mode is OffloadMode.NONE
        assert p.is_active is False
        assert p.components == {}

    def test_mode_module_defaults(self):
        """Omitted components → {"dit":{}} default for new active configs."""
        p = parse_offload_policy(diffusion_offload_config={"mode": "module"})
        assert p.mode is OffloadMode.MODULE
        assert p.components == {"dit": ComponentOptions(WeightTransfer.RANK_LOCAL, 0)}

    def test_mode_layer_defaults(self):
        p = parse_offload_policy(diffusion_offload_config={"mode": "layer"})
        assert p.mode is OffloadMode.LAYER
        assert p.dit_transfer() is WeightTransfer.RANK_LOCAL
        assert p.dit_resident_layers() == 0

    def test_allgather_transfer(self):
        p = parse_offload_policy(
            diffusion_offload_config={
                "mode": "layer",
                "components": {"dit": {"weight_transfer": "allgather"}},
            }
        )
        assert p.dit_transfer() is WeightTransfer.ALLGATHER

    def test_resident_layers(self):
        p = parse_offload_policy(
            diffusion_offload_config={
                "mode": "layer",
                "components": {"dit": {"resident_layers": 4}},
            }
        )
        assert p.dit_resident_layers() == 4

    def test_text_encoder_component(self):
        p = parse_offload_policy(
            diffusion_offload_config={
                "mode": "layer",
                "components": {
                    "dit": {},
                    "text_encoder": {},
                },
            }
        )
        assert p.has_dit
        assert p.has_text_encoder

    def test_pin_memory_false(self):
        p = parse_offload_policy(diffusion_offload_config={"mode": "layer", "pin_memory": False})
        assert p.pin_memory is False

    def test_json_string_input(self):
        p = parse_offload_policy(diffusion_offload_config=json.dumps({"mode": "layer"}))
        assert p.mode is OffloadMode.LAYER

    def test_invalid_json_string(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_offload_policy(diffusion_offload_config="{not json")


# ---------------------------------------------------------------------------
# JSON config — validation rejection cases
# ---------------------------------------------------------------------------


class TestOffloadValidation:
    def test_missing_mode(self):
        with pytest.raises(ValueError, match="requires a 'mode' key"):
            parse_offload_policy(diffusion_offload_config={})

    def test_bad_mode(self):
        with pytest.raises(ValueError, match="must be one of"):
            parse_offload_policy(diffusion_offload_config={"mode": "invalid"})

    def test_module_disallows_allgather(self):
        with pytest.raises(ValueError, match="only valid in mode=layer"):
            parse_offload_policy(
                diffusion_offload_config={
                    "mode": "module",
                    "components": {"dit": {"weight_transfer": "allgather"}},
                }
            )

    def test_module_disallows_resident_layers(self):
        with pytest.raises(ValueError, match="only valid in mode=layer"):
            parse_offload_policy(
                diffusion_offload_config={
                    "mode": "module",
                    "components": {"dit": {"resident_layers": 1}},
                }
            )

    def test_none_rejects_components(self):
        with pytest.raises(ValueError, match="does not accept"):
            parse_offload_policy(diffusion_offload_config={"mode": "none", "components": {"dit": {}}})

    def test_unknown_component(self):
        with pytest.raises(ValueError, match="Unknown component names"):
            parse_offload_policy(
                diffusion_offload_config={
                    "mode": "layer",
                    "components": {"vae": {}},
                }
            )

    def test_bad_transfer_value(self):
        with pytest.raises(ValueError, match="weight_transfer must be"):
            parse_offload_policy(
                diffusion_offload_config={
                    "mode": "layer",
                    "components": {"dit": {"weight_transfer": "pcie"}},
                }
            )

    def test_negative_resident(self):
        with pytest.raises(ValueError, match="resident_layers must be"):
            parse_offload_policy(
                diffusion_offload_config={
                    "mode": "layer",
                    "components": {"dit": {"resident_layers": -1}},
                }
            )

    def test_bool_resident_rejected(self):
        with pytest.raises(TypeError, match="resident_layers must be"):
            parse_offload_policy(
                diffusion_offload_config={
                    "mode": "layer",
                    "components": {"dit": {"resident_layers": True}},
                }
            )

    def test_text_encoder_rejects_resident(self):
        with pytest.raises(ValueError, match="only supported for components.dit"):
            parse_offload_policy(
                diffusion_offload_config={
                    "mode": "layer",
                    "components": {"text_encoder": {"resident_layers": 1}},
                }
            )

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown top-level keys"):
            parse_offload_policy(diffusion_offload_config={"mode": "layer", "foo": "bar"})

    def test_non_dict_components_rejected(self):
        with pytest.raises(TypeError):
            parse_offload_policy(diffusion_offload_config={"mode": "layer", "components": ["dit"]})

    def test_pin_memory_must_be_bool(self):
        with pytest.raises(TypeError, match="pin_memory must be a boolean"):
            parse_offload_policy(diffusion_offload_config={"mode": "layer", "pin_memory": "yes"})


# ---------------------------------------------------------------------------
# parse_offload_policy — legacy boolean flags path
# ---------------------------------------------------------------------------


class TestLegacyFlags:
    def test_no_flag_defaults_to_none(self):
        p = parse_offload_policy()
        assert p.mode is OffloadMode.NONE
        assert p.is_active is False

    def test_cpu_offload(self):
        p = parse_offload_policy(enable_cpu_offload=True)
        assert p.mode is OffloadMode.MODULE

    def test_layerwise_offload(self):
        p = parse_offload_policy(enable_layerwise_offload=True)
        assert p.mode is OffloadMode.LAYER

    def test_dlo_allgather(self):
        p = parse_offload_policy(
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
        )
        assert p.mode is OffloadMode.LAYER
        assert p.dit_transfer() is WeightTransfer.ALLGATHER

    def test_dlo_no_allgather_resident(self):
        p = parse_offload_policy(
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=False,
            dlo_resident_layers=3,
        )
        assert p.mode is OffloadMode.LAYER
        assert p.dit_transfer() is WeightTransfer.RANK_LOCAL
        assert p.dit_resident_layers() == 3

    def test_priority_dlo_over_layerwise(self):
        p = parse_offload_policy(
            enable_distributed_layerwise_offload=True,
            enable_layerwise_offload=True,
        )
        assert p.mode is OffloadMode.LAYER  # DLO path

    def test_priority_layerwise_over_cpu(self):
        p = parse_offload_policy(
            enable_layerwise_offload=True,
            enable_cpu_offload=True,
        )
        assert p.mode is OffloadMode.LAYER

    def test_pin_cpu_memory_false(self):
        p = parse_offload_policy(enable_cpu_offload=True, pin_cpu_memory=False)
        assert p.pin_memory is False


# ---------------------------------------------------------------------------
# HostWeightRuntimePolicy
# ---------------------------------------------------------------------------


class TestParseHWR:
    def test_legacy_default_disabled(self):
        hwr = parse_hwr_policy()
        assert hwr.mode == "disabled"
        assert hwr.root is None
        assert hwr.registration_limit_gib == 0.0

    def test_canonical_preferred(self):
        hwr = parse_hwr_policy(hwr_config={"mode": "preferred", "root": "/var/cache/hwr"})
        assert hwr.mode == "preferred"
        assert hwr.root == "/var/cache/hwr"

    def test_requires_root_for_preferred(self):
        with pytest.raises(ValueError, match="requires.*root"):
            parse_hwr_policy(hwr_config={"mode": "preferred"})

    def test_requires_root_for_required(self):
        with pytest.raises(ValueError, match="requires.*root"):
            parse_hwr_policy(hwr_config={"mode": "required"})

    def test_bad_mode(self):
        with pytest.raises(ValueError, match="hwr.mode must be"):
            parse_hwr_policy(hwr_config={"mode": "bogus"})

    def test_negative_limit(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            parse_hwr_policy(
                hwr_config={
                    "mode": "disabled",
                    "registration_limit_gib": -1,
                }
            )

    def test_infinite_limit_rejected(self):
        with pytest.raises(ValueError, match="must be finite"):
            parse_hwr_policy(
                hwr_config={
                    "mode": "disabled",
                    "registration_limit_gib": math.inf,
                }
            )

    def test_bool_limit_rejected(self):
        with pytest.raises(TypeError, match="registration_limit_gib must be"):
            parse_hwr_policy(
                hwr_config={
                    "mode": "disabled",
                    "registration_limit_gib": True,
                }
            )

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown top-level keys"):
            parse_hwr_policy(hwr_config={"mode": "disabled", "extra": 42})

    def test_to_dict(self):
        hwr = HostWeightRuntimePolicy(mode="preferred", root="/tmp/h", registration_limit_gib=4.5)
        assert hwr.to_dict() == {
            "mode": "preferred",
            "root": "/tmp/h",
            "registration_limit_gib": 4.5,
        }


# ---------------------------------------------------------------------------
# Joint validation (offload × HWR)
# ---------------------------------------------------------------------------


class TestJointValidation:
    def test_disabled_hwr_with_anything_is_ok(self):
        off = parse_offload_policy(diffusion_offload="none")
        validate_joint(off, HostWeightRuntimePolicy(mode="disabled"))  # no raise
        off = parse_offload_policy(diffusion_offload="module")
        validate_joint(off, HostWeightRuntimePolicy(mode="disabled"))  # no raise

    def test_preferred_hwr_with_layer_is_ok(self):
        off = parse_offload_policy(diffusion_offload="layer")
        hwr = HostWeightRuntimePolicy(mode="preferred", root="/tmp/h")
        validate_joint(off, hwr)  # no raise

    def test_preferred_hwr_rejects_mode_none(self):
        off = parse_offload_policy(diffusion_offload="none")
        hwr = HostWeightRuntimePolicy(mode="preferred", root="/tmp/h")
        with pytest.raises(ValueError, match="requires active offload"):
            validate_joint(off, hwr)

    def test_preferred_hwr_rejects_mode_module(self):
        off = parse_offload_policy(diffusion_offload="module")
        hwr = HostWeightRuntimePolicy(mode="preferred", root="/tmp/h")
        with pytest.raises(ValueError, match="requires mode=layer"):
            validate_joint(off, hwr)

    def test_preferred_hwr_rejects_allgather(self):
        off = parse_offload_policy(
            diffusion_offload_config={
                "mode": "layer",
                "components": {"dit": {"weight_transfer": "allgather"}},
            }
        )
        hwr = HostWeightRuntimePolicy(mode="preferred", root="/tmp/h")
        with pytest.raises(ValueError, match="allgather"):
            validate_joint(off, hwr)

    def test_registration_limit_requires_pin_memory(self):
        off = parse_offload_policy(diffusion_offload_config={"mode": "layer", "pin_memory": False})
        hwr = HostWeightRuntimePolicy(mode="preferred", root="/tmp/h", registration_limit_gib=8.0)
        with pytest.raises(ValueError, match="requires pin_memory=true"):
            validate_joint(off, hwr)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_policy_to_dict_roundtrip(self):
        p = parse_offload_policy(
            diffusion_offload_config={
                "mode": "layer",
                "components": {
                    "dit": {"weight_transfer": "rank-local", "resident_layers": 2},
                    "text_encoder": {},
                },
                "pin_memory": False,
            }
        )
        d = p.to_dict()
        assert d["mode"] == "layer"
        assert d["pin_memory"] is False
        assert d["components"]["dit"]["resident_layers"] == 2
        # Round-trip back through parser should still validate
        p2 = parse_offload_policy(diffusion_offload_config=d)
        assert p2.mode is OffloadMode.LAYER

    def test_is_active_property(self):
        assert CanonicalOffloadPolicy(mode=OffloadMode.NONE).is_active is False
        assert CanonicalOffloadPolicy(mode=OffloadMode.MODULE).is_active is True
        assert CanonicalOffloadPolicy(mode=OffloadMode.LAYER).is_active is True

    def test_component_option_defaults(self):
        opts = ComponentOptions()
        assert opts.weight_transfer is WeightTransfer.RANK_LOCAL
        assert opts.resident_layers == 0

    def test_sets_constants(self):
        assert VALID_MODES == {"none", "module", "layer"}
        assert VALID_TRANSFERS == {"rank-local", "allgather"}
        assert VALID_COMPONENTS == {"dit", "text_encoder"}
