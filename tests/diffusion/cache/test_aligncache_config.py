# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest

from vllm_omni.diffusion.cache.aligncache.config import AlignCacheConfig, normalize_cache_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_defaults_merge_hash_immutable():
    default = AlignCacheConfig.parse()
    explicit = AlignCacheConfig.parse({"align_cache": {"predictor": {"beta": {"mid": 0.2}}}})
    assert explicit.predictor.beta.mid == 0.2
    assert explicit.predictor.beta.late == default.predictor.beta.late
    assert default.config_hash == AlignCacheConfig.parse({}).config_hash
    assert explicit.config_hash != default.config_hash
    with pytest.raises(dataclasses.FrozenInstanceError):
        default.mode = "trace"


@pytest.mark.parametrize(
    "value", ["{", "[]", "null", "1", '"text"', [], 1, True, {"align_cache": []}, {"typo": 1}, {"rel_l1_thresh": 0.2}]
)
def test_bad_entry_inputs(value):
    with pytest.raises(ValueError, match="cache_config"):
        normalize_cache_config(value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "fake"},
        {"profile": "future"},
        {"policy": "qwen"},
        {"schema_version": 2},
        {"schema_version": True},
        {"warmup_steps": True},
        {"tail_steps": -1},
        {"max_cache_bytes": 0},
        {"base_threshold": float("nan")},
        {"base_threshold": float("inf")},
        {"base_threshold": -0.1},
        {"unexpected": 0},
        {"phase": {"early_end": 0.9}},
        {"phase": {"scales": {"oops": 1}}},
        {"feedback": {"age_caps": {"mid": 1.5}}},
        {"predictor": {"stride": 32}},
        {"predictor": {"online_scale": 1}},
        {"feedback": {"eps": 0}},
        {"feedback": {"scale_low": {"mid": 2}}},
        {"anchor": {"blend": {"late": 1.1}}},
    ],
)
def test_bad_fields(overrides):
    with pytest.raises(ValueError, match="align_cache"):
        AlignCacheConfig.parse({"align_cache": overrides})


@pytest.mark.parametrize(
    "value", [{1: 1, "bad": 2}, {"align_cache": {1: 2, "bad": 3}}, {"align_cache": {"base_threshold": 10**1000}}]
)
def test_pathological_inputs_have_diagnostic_errors(value):
    with pytest.raises(ValueError, match="cache_config|align_cache"):
        AlignCacheConfig.parse(value)


def test_quantization_switch_is_explicit_and_qwen_only():
    assert AlignCacheConfig.parse().quantization == "none"
    assert AlignCacheConfig.parse({"align_cache": {"profile": "qwen21_reference_v1"}}).quantization == "none"
    qwen = AlignCacheConfig.parse({"align_cache": {"profile": "qwen21_reference_v1", "quantization": "fp8"}})
    assert qwen.quantization == "fp8" and qwen.policy == "qwen_flow_v1"
    assert qwen.config_hash != AlignCacheConfig.parse({"align_cache": {"profile": "qwen21_reference_v1"}}).config_hash
    for profile in (
        "wan21_reference_v1",
        "wan21_14b_reference_v1",
        "wan22_t2v_reference_v1",
        "wan22_i2v_reference_v1",
        "wan22_ti2v_reference_v1",
        "minimax_h3_reference_v1",
    ):
        with pytest.raises(ValueError, match="quantization"):
            AlignCacheConfig.parse({"align_cache": {"profile": profile, "quantization": "fp8"}})
    for value in ("int8", "FP8", True, 8):
        with pytest.raises(ValueError, match="quantization"):
            AlignCacheConfig.parse({"align_cache": {"profile": "qwen21_reference_v1", "quantization": value}})
