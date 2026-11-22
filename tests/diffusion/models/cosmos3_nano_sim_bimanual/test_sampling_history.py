# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check checkpoint schedules and full-history admission without model weights."""

from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_action_schema import manifest
from test_rollout import SOURCE, fake_pipeline, request

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
    append_dense_kv_history,
)
from vllm_omni.diffusion.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from vllm_omni.errors import OmniClientError

ARDiffusionRequestRejectedError = OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
SCHEDULES = [[1.0, 0.9375, 5 / 6, 0.625], [1.0, 5 / 6]]


def methods():
    tree = ast.parse((SOURCE / "pipeline_cosmos3_nano_sim_bimanual.py").read_text())
    cls = next(node for node in tree.body if getattr(node, "name", "") == "Cosmos3NanoSimBimanualPipeline")
    selected = {
        "_denoise_chunk",
        "_append_dense_kv",
        "_ensure_text_kv",
        "_transformer_forward",
    }
    body = [node for node in cls.body if getattr(node, "name", "") in selected]
    body += [node for node in tree.body if getattr(node, "name", "") == "_admission_int"]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)] + body, type_ignores=[]
    )
    namespace = {
        "torch": torch,
        "ARDiffusionRequestRejectedError": ARDiffusionRequestRejectedError,
        "append_dense_kv_history": append_dense_kv_history,
        "logger": Mock(),
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return type("PipelineMethods", (), {name: namespace[name] for name in selected})


@pytest.mark.parametrize("frame", [0, 1, 3, 225])
@pytest.mark.parametrize("sigmas", SCHEDULES)
def test_actual_scheduler_uses_checkpoint_sigmas_and_existing_rng(frame, sigmas):
    pipe = methods()()
    pipe.manifest = replace(manifest(), t_list=tuple(sigmas))
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

    output = pipe._denoise_chunk(velocity, initial, generator=generator)
    sigmas = [*sigmas, 0]
    expected = initial.clone()
    reference_rng = torch.Generator().set_state(generator_state)
    for current, following in zip(sigmas, sigmas[1:]):
        x0 = expected - current * (expected * 0.25)
        noise = torch.randn(expected.shape, generator=reference_rng)
        expected = (1 - following) * x0 + following * noise
    torch.testing.assert_close(output, expected)
    assert calls == pytest.approx([sigma * 1000 for sigma in sigmas[:-1]])
    assert len(calls) == len(sigmas) - 1
    assert torch.equal(generator.get_state(), reference_rng.get_state())
    pipe._reset_mixed_precision.assert_called_once()


def test_full_history_reaches_dense_batched_path():
    pipe = methods()()
    pipe.manifest = replace(manifest(), window_frames=None, sink_frames=0)
    pipe._MAIN_BRANCH = "main"
    pipe.transformer = Mock()
    geometry = SimpleNamespace(tokens_per_frame=lambda _: 1)
    state = SimpleNamespace(dense_kv_by_branch={})
    for frame in range(226):
        value = torch.tensor([[[float(frame)]]])
        pipe._append_dense_kv(state, [(value, value)], geometry)
    assert state.dense_kv_by_branch["main"][0][0].flatten().tolist() == list(range(226))
    state.dense_attention = None
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
    assert pipe.transformer.call_args.kwargs["history_window"] == (0, None)


def test_dense_text_kv_is_not_capped_by_manifest_cache_length():
    pipe = methods()()
    pipe._MAIN_BRANCH = "main"
    cached = [(torch.ones(1, 513, 1), torch.ones(1, 513, 1))]
    pipe.transformer = Mock()
    pipe.transformer.encode_und_kv.return_value = (cached, 513)
    state = SimpleNamespace(text_kv_by_branch={})

    actual = pipe._ensure_text_kv(state, torch.ones(1, 513), torch.ones(1, 513))

    assert actual is cached
    assert state.text_kv_by_branch["main"] is cached


def test_full_rollout_uses_checkpoint_chunk_without_history_limit():
    pipe, state = fake_pipeline()
    pipe.manifest = replace(pipe.manifest, window_frames=None, sink_frames=0)
    pipe.forward(request(901, output_type="latent"))
    assert state.next_frame_idx == 226
    assert pipe._denoise_chunk.call_count == len(range(1, 226, 2))


def test_checkpoint_schedule_changes_session_fingerprint():
    pipe, _ = fake_pipeline()
    pipe.forward(request(9, output_type="latent", chunk_only=True, num_latent_frames=2, close_session=False))
    pipe.manifest = replace(pipe.manifest, t_list=(1.0, 0.9, 0.8, 0.6))
    with pytest.raises(ARDiffusionRequestRejectedError, match="manifest_id|sampler_id|inference_id"):
        pipe.forward(request(17, output_type="latent", chunk_only=True, num_latent_frames=2, reset=False))
