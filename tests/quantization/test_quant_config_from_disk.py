# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for rebuilding a quantization config from a checkpoint's config.json.

ModelOpt writes ``quant_method: "modelopt"`` for every algorithm it exports and
distinguishes them with ``quant_algo``. Anything that selects a config class
from ``quant_method`` alone therefore cannot tell NVFP4 from FP8, which is what
these tests pin.
"""

from __future__ import annotations

import pytest

from vllm_omni.quantization.factory import build_quant_config, resolve_quant_config_from_disk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _nvfp4_disk_config(**overrides):
    """The shape ModelOpt 0.45 writes into a transformer's config.json."""
    config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "quant_type": "NVFP4",
        "producer": {"name": "modelopt", "version": "0.45.0"},
        "ignore": ["blocks.0*"],
    }
    config.update(overrides)
    return config


def test_an_nvfp4_checkpoint_resolves_to_nvfp4_not_fp8():
    """The bug this module exists for.

    ``quant_method`` is "modelopt" for every ModelOpt export, so resolving by
    that key alone picks FP8 for an NVFP4 checkpoint. Only ``quant_algo``
    separates them.
    """
    resolved = resolve_quant_config_from_disk(None, _nvfp4_disk_config())

    assert type(resolved).__name__ == "ModelOptNvFp4Config"
    assert resolved.get_name() == "modelopt_fp4"


def test_an_fp8_checkpoint_still_resolves_to_fp8():
    """The fix must not push everything ModelOpt down the NVFP4 path."""
    resolved = resolve_quant_config_from_disk(
        None,
        {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            "producer": {"name": "modelopt", "version": "0.45.0"},
        },
    )

    assert type(resolved).__name__ == "ModelOptFp8Config"
    assert resolved.get_name() == "modelopt"


def test_the_exclusion_list_survives_the_rebuild():
    """A config that resolved but lost its exclusions would be worse than none.

    Every excluded layer would be built quantized, and each would then fail on
    a scale tensor the checkpoint never shipped for it.
    """
    resolved = resolve_quant_config_from_disk(
        None, _nvfp4_disk_config(ignore=["blocks.0*", "head*", "patch_embedding"])
    )

    assert resolved.is_layer_excluded("blocks.0.self_attn.q")
    assert resolved.is_layer_excluded("patch_embedding")
    assert not resolved.is_layer_excluded("blocks.5.self_attn.q")


def test_a_checkpoint_without_quant_algo_is_rejected_by_name():
    """A ModelOpt config missing quant_algo must say so, not fail elsewhere."""
    with pytest.raises(ValueError, match="quant_algo"):
        resolve_quant_config_from_disk(None, {"quant_method": "modelopt", "producer": {"name": "modelopt"}})


def test_a_serialized_checkpoint_rebuilds_through_the_same_path():
    """The disk-marks-serialized branch is one of four call sites.

    All four rebuild the same way, so a fix applied to one of them would leave
    an NVFP4 checkpoint resolving to FP8 through any of the other three.
    """
    active = build_quant_config(_nvfp4_disk_config())
    # Same method, but the checkpoint declares itself serialized.
    resolved = resolve_quant_config_from_disk(active, _nvfp4_disk_config(is_checkpoint_nvfp4_serialized=True))

    assert type(resolved).__name__ == "ModelOptNvFp4Config"


def test_differing_ignored_layers_rebuild_through_the_same_path():
    active = build_quant_config(_nvfp4_disk_config())
    resolved = resolve_quant_config_from_disk(active, _nvfp4_disk_config(ignored_layers=["blocks.7.ffn.0"]))

    assert type(resolved).__name__ == "ModelOptNvFp4Config"


def test_an_explicit_nvfp4_flag_is_not_reported_as_mismatching_its_own_checkpoint():
    """The same root cause, seen from the other side.

    An active config built from this very checkpoint reports get_name() as
    "modelopt_fp4", while the checkpoint's quant_method reads "modelopt".
    Comparing those two strings directly rejects a pairing that is in fact
    identical, so a user who passes --quantization explicitly cannot load the
    checkpoint the flag names.
    """
    active = build_quant_config(_nvfp4_disk_config())
    assert active.get_name() == "modelopt_fp4"

    resolved = resolve_quant_config_from_disk(active, _nvfp4_disk_config())

    assert resolved is active or type(resolved).__name__ == "ModelOptNvFp4Config"


def test_a_mismatch_between_flag_and_checkpoint_is_still_refused():
    """The rebuild must not paper over a genuinely wrong --quantization flag."""
    active = build_quant_config({"quant_method": "modelopt", "quant_algo": "FP8"})
    with pytest.raises(ValueError, match="declares quant_method"):
        resolve_quant_config_from_disk(active, {"quant_method": "bitsandbytes"})


def test_a_disk_config_without_a_method_is_passed_through_untouched():
    active = build_quant_config(_nvfp4_disk_config())
    assert resolve_quant_config_from_disk(active, {"no_method_here": 1}) is active
    assert resolve_quant_config_from_disk(active, None) is active


def test_a_string_disk_config_still_auto_detects():
    resolved = resolve_quant_config_from_disk(None, "modelopt_fp4")
    assert resolved.get_name() == "modelopt_fp4"
