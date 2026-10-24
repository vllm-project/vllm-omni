# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from copy import deepcopy

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.cache.teacache.wan_profile import WAN_MODEL_ID, WAN_MODEL_REVISION, stage_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def profile():
    return {
        "model_id": WAN_MODEL_ID,
        "revision": WAN_MODEL_REVISION,
        "dtype": "bfloat16",
        "pp_size": 2,
        "cfg_size": 2,
        "stages": {
            "0": {"coefficients": [0, 0, 0, 1, 0], "threshold": 0.1, "max_input_distance": 0.5, "warmup_steps": 2},
            "1": {"coefficients": [0, 0, 0, 2, 0], "threshold": 0.2, "max_input_distance": 0.3, "warmup_steps": 3},
        },
    }


def resolve(value, **overrides):
    args = dict(model=WAN_MODEL_ID, revision=WAN_MODEL_REVISION, pp_size=2, cfg_size=2, pp_rank=0)
    args.update(overrides)
    return stage_config(value, **args)


def test_stage_specific_profile_round_trip():
    from vllm_omni.diffusion.data import DiffusionCacheConfig

    value = DiffusionCacheConfig.from_dict({"wan_profile": profile()}).wan_profile
    first, last = resolve(value), resolve(value, pp_rank=1)
    assert first.coefficients != last.coefficients
    assert (first.warmup_steps, last.warmup_steps) == (2, 3)
    assert (first.max_input_distance, last.max_input_distance) == (0.5, 0.3)
    assert first.clamp_negative_prediction and last.clamp_negative_prediction


@pytest.mark.parametrize(
    "key,value", [("revision", "main"), ("model_id", "other"), ("dtype", "float32"), ("pp_size", 1), ("cfg_size", 1)]
)
def test_profile_rejects_model_and_topology_mismatch(key, value):
    data = profile()
    data[key] = value
    with pytest.raises(ValueError):
        resolve(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold", float("nan")),
        ("max_input_distance", float("inf")),
        ("warmup_steps", -1),
        ("warmup_steps", 1.5),
        ("coefficients", [0, 0, 0, 0, float("nan")]),
    ],
)
def test_invalid_remote_stage_is_rejected_too(field, value):
    data = deepcopy(profile())
    data["stages"]["1"][field] = value
    with pytest.raises(ValueError):
        resolve(data, pp_rank=0)


def test_unknown_local_provenance_rejected(tmp_path):
    with pytest.raises(ValueError, match="metadata"):
        resolve(profile(), model=str(tmp_path))
    metadata = tmp_path / ".cache/huggingface/download/model_index.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(WAN_MODEL_REVISION + "\nexample-etag\n")
    assert resolve(profile(), model=str(tmp_path)).warmup_steps == 2


def test_production_hook_warmup_clamp_range_and_reset():
    cfg = TeaCacheConfig(
        transformer_type="WanTransformer3DModel",
        coefficients=[0, 0, 0, 0, -1],
        rel_l1_thresh=0.2,
        warmup_steps=2,
        max_input_distance=0.5,
        clamp_negative_prediction=True,
    )
    hook = TeaCacheHook(cfg)
    state = TeaCacheState()
    state.previous_modulated_input = torch.ones(4)
    assert hook._should_compute_full_transformer(state, torch.ones(4))
    state.cnt = 1
    assert hook._should_compute_full_transformer(state, torch.ones(4))
    state.cnt = 2
    assert not hook._should_compute_full_transformer(state, torch.ones(4))
    assert hook._should_compute_full_transformer(state, torch.full((4,), 2.0))
    assert state.accumulated_rel_l1_distance == 0
    assert hook._should_compute_full_transformer(state, torch.full((4,), float("nan")))
    state.reset()
    assert hook._should_compute_full_transformer(state, torch.ones(4))


def test_other_models_keep_negative_polynomial_absolute_value():
    hook = TeaCacheHook(TeaCacheConfig(coefficients=[0, 0, 0, 0, -1]))
    state = TeaCacheState()
    state.cnt = 2
    state.previous_modulated_input = torch.ones(4)
    assert hook._should_compute_full_transformer(state, torch.ones(4))
