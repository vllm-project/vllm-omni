# SPDX-License-Identifier: Apache-2.0
"""Check frame schedules and full-history admission without model weights."""

from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml
from test_cookbook import manifest
from test_rollout import SOURCE, fake_pipeline, request

from vllm_omni.diffusion.models.cosmos3.resolution import VIDEO_RES_SIZE_INFO
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import Cosmos3NanoSimBimanualResolutionPolicy
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.inference_config import (
    Cosmos3NanoSimBimanualInferenceConfig,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
    append_dense_kv_history,
)
from vllm_omni.diffusion.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    ARDiffusionRequestRejectedError,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
SCHEDULES = [[1.0, 0.9375, 5 / 6, 0.625], [1.0, 5 / 6]]


def settings(**overrides):
    return Cosmos3NanoSimBimanualInferenceConfig.from_od_config(
        SimpleNamespace(model_config={"inference_overrides": overrides}), manifest()
    )


def methods():
    tree = ast.parse((SOURCE / "pipeline_cosmos3_nano_sim_bimanual.py").read_text())
    cls = next(node for node in tree.body if getattr(node, "name", "") == "Cosmos3NanoSimBimanualPipeline")
    selected = {"_denoise_chunk", "_kv_spec_for_geometry", "_append_dense_kv", "_transformer_forward"}
    body = [node for node in cls.body if getattr(node, "name", "") in selected]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)] + body, type_ignores=[]
    )
    namespace = {
        "torch": torch,
        "ARDiffusionKVCacheSpec": ARDiffusionKVCacheSpec,
        "ARDiffusionKVBranchSpec": ARDiffusionKVBranchSpec,
        "ARDiffusionCrossAttentionKVSpec": ARDiffusionCrossAttentionKVSpec,
        "append_dense_kv_history": append_dense_kv_history,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return type("PipelineMethods", (), {name: namespace[name] for name in selected})


def test_defaults_preserve_artifact_and_explicit_settings_change_identity():
    artifact = manifest()
    before = artifact.digest
    default = Cosmos3NanoSimBimanualInferenceConfig.from_od_config(SimpleNamespace(), artifact)
    assert default.frame_sigma_schedules == (artifact.t_list,)
    assert (default.window_frames, default.sink_frames) == (artifact.window_frames, artifact.sink_frames)
    for frame in (0, 1, 225):
        assert default.sigmas_for_frame(frame) == artifact.t_list
    overridden = settings(frame_sigma_schedules=SCHEDULES, history_mode="full", max_num_frames=901)
    assert overridden.sigmas_for_frame(0) == tuple(SCHEDULES[0])
    assert overridden.sigmas_for_frame(1) == overridden.sigmas_for_frame(225) == tuple(SCHEDULES[1])
    assert overridden.window_frames == 226 and overridden.sink_frames == 0
    assert overridden.digest != default.digest
    assert artifact.digest == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"frame_sigma_schedules": []},
        {"frame_sigma_schedules": [[1, 0]]},
        {"frame_sigma_schedules": [[1, float("nan")]]},
        {"frame_sigma_schedules": [[1, 0.5, 0.75]]},
        {"frame_sigma_schedules": [[True, 0.5]]},
        {"history_mode": "full"},
        {"history_mode": "full", "max_num_frames": 900},
        {"history_mode": "full", "max_num_frames": 901, "attention_sink_size": 1},
        {"history_mode": "full", "max_num_frames": 901, "kv_cache_inference_size": 96},
        {"kv_cache_inference_size": False},
        {"max_num_frames": 901},
        {"frame_sigma_schedule": SCHEDULES},
    ],
)
def test_invalid_overrides_are_rejected(overrides):
    with pytest.raises(ValueError):
        settings(**overrides)


@pytest.mark.parametrize("frame,steps", [(0, 4), (1, 2), (3, 2), (225, 2)])
def test_actual_scheduler_uses_selected_sigmas_and_existing_rng(frame, steps):
    pipe = methods()()
    pipe.inference_config = settings(frame_sigma_schedules=SCHEDULES)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(stochastic_sampling=True)
    pipe._set_mixed_precision_step = Mock()
    pipe._reset_mixed_precision = Mock()
    initial = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 2, 2)
    generator = torch.Generator().manual_seed(42 + frame)
    generator_state = generator.get_state()
    calls = []

    def velocity(x, timestep):
        calls.append(timestep.item())
        return x * 0.25

    output = pipe._denoise_chunk(velocity, initial, generator=generator, chunk_start=frame)
    sigmas = [*SCHEDULES[min(frame, 1)], 0]
    expected = initial.clone()
    reference_rng = torch.Generator().set_state(generator_state)
    for current, following in zip(sigmas, sigmas[1:]):
        x0 = expected - current * (expected * 0.25)
        noise = torch.randn(expected.shape, generator=reference_rng)
        expected = (1 - following) * x0 + following * noise
    torch.testing.assert_close(output, expected)
    assert calls == pytest.approx([sigma * 1000 for sigma in sigmas[:-1]])
    assert len(calls) == steps
    assert torch.equal(generator.get_state(), reference_rng.get_state())
    pipe._reset_mixed_precision.assert_called_once()


def test_full_history_reaches_dense_paged_and_batched_paths():
    pipe = methods()()
    pipe.manifest = manifest()
    pipe.inference_config = settings(history_mode="full", max_num_frames=901)
    pipe._MAIN_BRANCH = "main"
    pipe._SESSION_CAPACITY = 1
    pipe.transformer = Mock(num_hidden_layers=1, num_kv_heads_local=1, head_dim=8)
    geometry = SimpleNamespace(tokens_per_frame=lambda _: 1)
    spec = pipe._kv_spec_for_geometry(geometry)
    assert spec.window_frames == 226 and spec.sink_frames == 0
    state = SimpleNamespace(dense_kv_by_branch={})
    for frame in range(226):
        value = torch.tensor([[[float(frame)]]])
        pipe._append_dense_kv(state, [(value, value)], geometry)
    assert state.dense_kv_by_branch["main"][0][0].flatten().tolist() == list(range(226))
    pipe._ar_diffusion_kv_state = None
    pipe._transformer_forward(
        state,
        torch.zeros(1, 1, 2, 1, 1),
        torch.zeros(1),
        geometry=geometry,
        text_kv=[],
        real_text_kv_len=1,
        frame_start=1,
        fps=30,
        action_latents=None,
        action_domain_ids=None,
        condition_vision=True,
        null_action_frame_indexes=(),
        commit_current=False,
        frame_causal=True,
    )
    assert pipe.transformer.call_args.kwargs["history_window"] == (0, 226)


def test_full_rollout_boundary_and_chunk_start_selection():
    pipe, state = fake_pipeline()
    pipe.manifest = replace(pipe.manifest, chunk_size=2)
    pipe.inference_config = settings(frame_sigma_schedules=SCHEDULES, history_mode="full", max_num_frames=901)
    pipe.forward(request(901, output_type="latent"))
    assert state.next_frame_idx == 226
    assert [c.kwargs["chunk_start"] for c in pipe._denoise_chunk.call_args_list] == list(range(1, 226, 2))
    pipe, _ = fake_pipeline()
    pipe.inference_config = settings(history_mode="full", max_num_frames=901)
    with pytest.raises(ARDiffusionRequestRejectedError, match="configured capacity is 226"):
        pipe.forward(request(905, output_type="latent"))
    pipe._denoise_chunk.assert_not_called()
    pipe._ensure_text_kv.assert_not_called()
    pipe._initial_condition_latent.assert_not_called()


def test_full_history_limit_applies_across_ticks():
    pipe, state = fake_pipeline()
    pipe.inference_config = settings(history_mode="full", max_num_frames=17)
    pipe.forward(request(17, output_type="latent", chunk_only=True, num_latent_frames=4, close_session=False))
    assert state.next_frame_idx == 5
    pipe._denoise_chunk.reset_mock()
    pipe._initial_condition_latent.reset_mock()
    with pytest.raises(ARDiffusionRequestRejectedError, match="configured capacity is 5"):
        pipe.forward(request(33, output_type="latent", chunk_only=True, num_latent_frames=4, reset=False))
    assert state.next_frame_idx == 5
    pipe._denoise_chunk.assert_not_called()
    pipe._initial_condition_latent.assert_not_called()


def test_override_changes_session_fingerprint():
    pipe, _ = fake_pipeline()
    pipe.forward(request(17, output_type="latent", chunk_only=True, num_latent_frames=4, close_session=False))
    pipe.inference_config = settings(frame_sigma_schedules=SCHEDULES)
    with pytest.raises(ARDiffusionRequestRejectedError, match="inference_id"):
        pipe.forward(request(33, output_type="latent", chunk_only=True, num_latent_frames=4, reset=False))


def test_revision_deployment_is_opt_in():
    root = SOURCE.parents[3]
    original = yaml.safe_load((root / "vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml").read_text())
    selected = yaml.safe_load((root / "vllm_omni/deploy/cosmos3_nano_sim_bimanual_i4.yaml").read_text())
    assert "inference_overrides" not in original["stages"][0]["model_config"]
    config = Cosmos3NanoSimBimanualInferenceConfig.from_od_config(
        SimpleNamespace(model_config=selected["stages"][0]["model_config"]), manifest()
    )
    assert config.frame_sigma_schedules == tuple(map(tuple, SCHEDULES))
    assert config.window_frames == 226 and config.history_mode == "full"
    model_config = selected["stages"][0]["model_config"]
    policy = Cosmos3NanoSimBimanualResolutionPolicy(
        default_resolution=model_config["default_resolution"], max_pixels=model_config["max_pixels"]
    )
    for width, height in VIDEO_RES_SIZE_INFO["480"].values():
        policy.resolve(height, width)
