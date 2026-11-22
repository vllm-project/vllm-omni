# SPDX-License-Identifier: Apache-2.0
"""Reference cache capacity, checkpoint defaults, and session separation."""

from types import SimpleNamespace

import pytest
import torch
from test_cookbook import manifest

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import (
    COSMOS3_NANO_SIM_BIMANUAL_ARTIFACT_FIELDS,
    Cosmos3NanoSimBimanualManifest,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
    append_dense_kv_history,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def config(**overrides):
    original = manifest()
    artifact = {
        key: getattr(original, key)
        for key in COSMOS3_NANO_SIM_BIMANUAL_ARTIFACT_FIELDS - {"conditioning", "fixed_step_sampler_config"}
    }
    artifact["conditioning"] = original.conditioning.model_dump(mode="json")
    artifact["fixed_step_sampler_config"] = {
        "sample_type": original.sample_type,
        "t_list": list(original.t_list),
        "num_train_timesteps": original.num_train_timesteps,
    }
    return SimpleNamespace(tf_model_config={"cosmos3_nano_sim_bimanual": artifact}, model_config=overrides)


def test_override_preserves_checkpoint_and_separates_sessions():
    baseline = Cosmos3NanoSimBimanualManifest.from_od_config(config())
    options = config(kv_cache_inference_size=64, attention_sink_size=8)
    effective = Cosmos3NanoSimBimanualManifest.from_od_config(options)
    assert (effective.window_frames, effective.sink_frames) == (55, 8)
    assert effective.digest != baseline.digest
    assert effective.chunk_size == baseline.chunk_size and effective.t_list == baseline.t_list
    assert options.tf_model_config["cosmos3_nano_sim_bimanual"]["window_frames"] == baseline.window_frames


def test_retained_history_matches_reference_frame_indices():
    effective = Cosmos3NanoSimBimanualManifest.from_od_config(config(kv_cache_inference_size=64, attention_sink_size=8))
    history = None
    for frame in range(226):
        expected = list(range(min(frame, 8))) + list(range(max(8, frame - 55), frame))
        if history is not None:
            assert history[0][0].flatten().tolist() == expected
        value = torch.tensor([[[frame]]])
        history = append_dense_kv_history(
            history,
            [(value, value)],
            tokens_per_frame=1,
            sink_frames=effective.sink_frames,
            window_frames=effective.window_frames,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"attention_sink_size": 8},
        {"kv_cache_inference_size": True},
        {"kv_cache_inference_size": 1},
        {"kv_cache_inference_size": 64, "attention_sink_size": -1},
        {"kv_cache_inference_size": 64, "attention_sink_size": 63},
    ],
)
def test_invalid_override_rejected(overrides):
    with pytest.raises(ValueError):
        Cosmos3NanoSimBimanualManifest.from_od_config(config(**overrides))
