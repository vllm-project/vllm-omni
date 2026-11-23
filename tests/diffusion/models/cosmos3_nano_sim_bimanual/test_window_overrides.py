# SPDX-License-Identifier: Apache-2.0
"""Reference cache capacity, checkpoint defaults, and session separation."""

from types import SimpleNamespace

import pytest
import torch
from test_action_schema import manifest

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import (
    COSMOS3_NANO_SIM_BIMANUAL_ARTIFACT_FIELDS,
    Cosmos3NanoSimBimanualManifest,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
    append_dense_kv_history,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def config(artifact_values=None, **overrides):
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
    artifact.update(artifact_values or {})
    return SimpleNamespace(tf_model_config={"cosmos3_nano_sim_bimanual": artifact}, model_config=overrides)


@pytest.mark.parametrize(("cache_size", "sinks"), [(16, 0), (64, 8), (128, 32)])
def test_override_preserves_checkpoint_and_separates_sessions(cache_size, sinks):
    baseline = Cosmos3NanoSimBimanualManifest.from_od_config(config())
    options = config(kv_cache_inference_size=cache_size, attention_sink_size=sinks)
    effective = Cosmos3NanoSimBimanualManifest.from_od_config(options)
    assert (effective.window_frames, effective.sink_frames) == (cache_size - sinks - 1, sinks)
    assert effective.digest != baseline.digest
    assert effective.chunk_size == baseline.chunk_size and effective.t_list == baseline.t_list
    assert options.tf_model_config["cosmos3_nano_sim_bimanual"]["window_frames"] == baseline.window_frames


@pytest.mark.parametrize(("cache_size", "sinks"), [(16, 0), (30, 3), (64, 8), (128, 32)])
def test_checkpoint_window_matches_reference_capacity(cache_size, sinks):
    checkpoint = Cosmos3NanoSimBimanualManifest.from_od_config(
        config({"window_frames": cache_size, "sink_frames": sinks})
    )
    override = Cosmos3NanoSimBimanualManifest.from_od_config(
        config(kv_cache_inference_size=cache_size, attention_sink_size=sinks)
    )
    assert (checkpoint.window_frames, checkpoint.sink_frames) == (cache_size - sinks - 1, sinks)
    assert checkpoint.digest == override.digest


def test_override_replaces_checkpoint_window():
    effective = Cosmos3NanoSimBimanualManifest.from_od_config(
        config({"window_frames": 64, "sink_frames": 8}, kv_cache_inference_size=32, attention_sink_size=4)
    )
    assert (effective.window_frames, effective.sink_frames) == (27, 4)


def test_checkpoint_full_history_is_unbounded():
    effective = Cosmos3NanoSimBimanualManifest.from_od_config(config({"window_frames": None, "sink_frames": 0}))
    assert (effective.window_frames, effective.sink_frames) == (None, 0)


@pytest.mark.parametrize(
    ("window", "sinks"),
    [(0, 0), (1, 0), (8, 8), (9, 8), (None, 8), (True, 0), (64, -1)],
)
def test_invalid_checkpoint_window_rejected(window, sinks):
    with pytest.raises(ValueError):
        Cosmos3NanoSimBimanualManifest.from_od_config(config({"window_frames": window, "sink_frames": sinks}))


@pytest.mark.parametrize(
    "options",
    [
        {"artifact_values": {"window_frames": 64, "sink_frames": 8}},
        {"kv_cache_inference_size": 64, "attention_sink_size": 8},
    ],
)
def test_retained_history_matches_reference_frame_indices(options):
    effective = Cosmos3NanoSimBimanualManifest.from_od_config(config(**options))
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
        {"kv_cache_inference_size": 64, "attention_sink_size": True},
    ],
)
def test_invalid_override_rejected(overrides):
    with pytest.raises(ValueError):
        Cosmos3NanoSimBimanualManifest.from_od_config(config(**overrides))
