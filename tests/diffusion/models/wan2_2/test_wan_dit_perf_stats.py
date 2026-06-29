from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

WanDiTForwardStats = importlib.import_module("vllm_omni.diffusion.metrics.perf").WanDiTForwardStats
pipeline_wan_module = importlib.import_module("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2")
Wan22Pipeline = pipeline_wan_module.Wan22Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_parallel_config(**overrides):
    values = {
        "tensor_parallel_size": 1,
        "sequence_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "cfg_parallel_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_wan_pipeline_for_stats(**parallel_overrides):
    pipeline = object.__new__(Wan22Pipeline)
    pipeline.od_config = SimpleNamespace(parallel_config=_make_parallel_config(**parallel_overrides))
    pipeline.transformer_config = SimpleNamespace(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=4,
        ffn_dim=32,
        num_layers=6,
    )
    pipeline.transformer = SimpleNamespace(start_layer=1, end_layer=4, config=pipeline.transformer_config)
    pipeline.transformer_2 = None
    pipeline._last_dit_forward_stats = None
    return pipeline


def test_wan_records_model_specific_stats_from_latent_runtime_shapes(monkeypatch):
    pipeline = _make_wan_pipeline_for_stats(tensor_parallel_size=2, sequence_parallel_size=4, pipeline_parallel_size=2)
    latents = torch.zeros((2, 16, 5, 8, 10))
    prompt_embeds = torch.zeros((2, 77, 8))
    timesteps = torch.arange(4)

    monkeypatch.setattr(pipeline_wan_module, "get_classifier_free_guidance_world_size", lambda: 1)

    pipeline._record_dit_forward_stats(
        latents=latents,
        prompt_embeds=prompt_embeds,
        timesteps=timesteps,
        do_true_cfg=False,
    )

    stats = pipeline.get_dit_forward_stats(req=object(), output=object())

    assert stats is not None
    assert isinstance(stats, WanDiTForwardStats)
    assert stats.batch_size == 2
    assert stats.latent_num_frames == 5
    assert stats.latent_height == 8
    assert stats.latent_width == 10
    assert stats.patch_size == (1, 2, 2)
    assert stats.context_len == 77
    assert stats.hidden_dim == 8
    assert stats.ffn_dim == 32
    assert stats.num_layers == 3
    assert stats.num_steps == 4
    assert stats.tensor_parallel_size == 2
    assert stats.sequence_parallel_size == 4
    assert stats.pipeline_parallel_size == 2
    assert stats.forwards_per_step_per_gpu == 1
    assert stats.logical_seq_len == 5 * 4 * 5
    assert stats.padded_seq_len == 100
    assert stats.local_seq_len == 25


def test_wan_counts_cfg_forwards_per_gpu(monkeypatch):
    pipeline = _make_wan_pipeline_for_stats()
    monkeypatch.setattr(pipeline_wan_module, "get_classifier_free_guidance_world_size", lambda: 1)

    pipeline._record_dit_forward_stats(
        latents=torch.zeros((1, 16, 1, 2, 2)),
        prompt_embeds=torch.zeros((1, 5, 8)),
        timesteps=torch.arange(7),
        do_true_cfg=True,
    )
    sequential_stats = pipeline.get_dit_forward_stats(req=object(), output=object())

    monkeypatch.setattr(pipeline_wan_module, "get_classifier_free_guidance_world_size", lambda: 2)
    pipeline._record_dit_forward_stats(
        latents=torch.zeros((1, 16, 1, 2, 2)),
        prompt_embeds=torch.zeros((1, 5, 8)),
        timesteps=torch.arange(7),
        do_true_cfg=True,
    )
    parallel_stats = pipeline.get_dit_forward_stats(req=object(), output=object())

    assert sequential_stats is not None
    assert sequential_stats.forwards_per_step_per_gpu == 2
    assert parallel_stats is not None
    assert parallel_stats.forwards_per_step_per_gpu == 1


def test_wan_counts_mixed_guidance_forwards_by_timestep(monkeypatch):
    pipeline = _make_wan_pipeline_for_stats()
    monkeypatch.setattr(pipeline_wan_module, "get_classifier_free_guidance_world_size", lambda: 1)

    pipeline._record_dit_forward_stats(
        latents=torch.zeros((1, 16, 1, 2, 2)),
        prompt_embeds=torch.zeros((1, 5, 8)),
        timesteps=torch.tensor([8.0, 4.0, 2.0]),
        do_true_cfg=True,
        guidance_low=1.0,
        guidance_high=2.0,
        boundary_timestep=5.0,
        has_negative_prompt=True,
    )

    stats = pipeline.get_dit_forward_stats(req=object(), output=object())

    assert stats is not None
    assert stats.forwards_per_step_per_gpu == 1
    assert stats.num_forwards_per_gpu == 5


def test_wan_collects_fail_closed_when_shape_is_incomplete():
    pipeline = _make_wan_pipeline_for_stats()

    pipeline._record_dit_forward_stats(
        latents=torch.zeros((1, 16, 1, 2)),
        prompt_embeds=torch.zeros((1, 5, 8)),
        timesteps=torch.arange(7),
        do_true_cfg=False,
    )

    assert pipeline.get_dit_forward_stats(req=object(), output=object()) is None
