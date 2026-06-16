from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

HunyuanVideoDiTForwardStats = importlib.import_module("vllm_omni.diffusion.metrics.perf").HunyuanVideoDiTForwardStats
pipeline_hunyuan_module = importlib.import_module("vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5")
HunyuanVideo15Pipeline = pipeline_hunyuan_module.HunyuanVideo15Pipeline

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


def _make_hunyuan_pipeline_for_stats(**parallel_overrides):
    pipeline = object.__new__(HunyuanVideo15Pipeline)
    pipeline.od_config = SimpleNamespace(parallel_config=_make_parallel_config(**parallel_overrides))
    pipeline.transformer = SimpleNamespace(
        inner_dim=8,
        ffn_dim=32,
        patch_size_t=1,
        patch_size=2,
        transformer_blocks=[object(), object(), object()],
    )
    pipeline._last_dit_forward_stats = None
    return pipeline


def test_hunyuan_video_records_model_specific_stats_from_runtime_shapes(monkeypatch):
    pipeline = _make_hunyuan_pipeline_for_stats(
        tensor_parallel_size=2,
        sequence_parallel_size=4,
        pipeline_parallel_size=2,
    )
    monkeypatch.setattr(pipeline_hunyuan_module, "get_classifier_free_guidance_world_size", lambda: 1)

    pipeline._record_dit_forward_stats(
        latents=torch.zeros((2, 16, 5, 8, 10)),
        prompt_embeds=torch.zeros((2, 77, 8)),
        prompt_embeds_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        prompt_embeds_2=torch.zeros((2, 8, 8)),
        prompt_embeds_mask_2=torch.tensor([[1, 1], [1, 0]]),
        image_embeds=torch.zeros((2, 6, 8)),
        image_embeds_mask=torch.tensor([[1, 1, 1], [1, 0, 0]]),
        timesteps=torch.arange(4),
        do_true_cfg=False,
    )

    stats = pipeline.get_dit_forward_stats(req=object(), output=object())

    assert stats is not None
    assert isinstance(stats, HunyuanVideoDiTForwardStats)
    assert stats.batch_size == 2
    assert stats.latent_num_frames == 5
    assert stats.latent_height == 8
    assert stats.latent_width == 10
    assert stats.patch_size_t == 1
    assert stats.patch_size == 2
    assert stats.context_len == 3 + 2 + 3
    assert stats.hidden_dim == 8
    assert stats.ffn_dim == 32
    assert stats.num_layers == 3
    assert stats.num_steps == 4
    assert stats.tensor_parallel_size == 2
    assert stats.sequence_parallel_size == 4
    assert stats.pipeline_parallel_size == 2
    assert stats.forwards_per_step_per_gpu == 1


def test_hunyuan_video_counts_cfg_forwards_per_gpu(monkeypatch):
    pipeline = _make_hunyuan_pipeline_for_stats()
    monkeypatch.setattr(pipeline_hunyuan_module, "get_classifier_free_guidance_world_size", lambda: 1)

    pipeline._record_dit_forward_stats(
        latents=torch.zeros((1, 16, 1, 2, 2)),
        prompt_embeds=torch.zeros((1, 5, 8)),
        prompt_embeds_mask=None,
        prompt_embeds_2=torch.zeros((1, 7, 8)),
        prompt_embeds_mask_2=None,
        image_embeds=torch.zeros((1, 0, 8)),
        image_embeds_mask=None,
        timesteps=torch.arange(7),
        do_true_cfg=True,
    )
    sequential_stats = pipeline.get_dit_forward_stats(req=object(), output=object())

    monkeypatch.setattr(pipeline_hunyuan_module, "get_classifier_free_guidance_world_size", lambda: 2)
    pipeline._record_dit_forward_stats(
        latents=torch.zeros((1, 16, 1, 2, 2)),
        prompt_embeds=torch.zeros((1, 5, 8)),
        prompt_embeds_mask=None,
        prompt_embeds_2=torch.zeros((1, 7, 8)),
        prompt_embeds_mask_2=None,
        image_embeds=torch.zeros((1, 0, 8)),
        image_embeds_mask=None,
        timesteps=torch.arange(7),
        do_true_cfg=True,
    )
    parallel_stats = pipeline.get_dit_forward_stats(req=object(), output=object())

    assert sequential_stats is not None
    assert sequential_stats.forwards_per_step_per_gpu == 2
    assert parallel_stats is not None
    assert parallel_stats.forwards_per_step_per_gpu == 1


def test_hunyuan_video_collects_fail_closed_when_shape_is_incomplete():
    pipeline = _make_hunyuan_pipeline_for_stats()

    pipeline._record_dit_forward_stats(
        latents=torch.zeros((1, 16, 1, 2)),
        prompt_embeds=torch.zeros((1, 5, 8)),
        prompt_embeds_mask=None,
        prompt_embeds_2=torch.zeros((1, 7, 8)),
        prompt_embeds_mask_2=None,
        image_embeds=torch.zeros((1, 0, 8)),
        image_embeds_mask=None,
        timesteps=torch.arange(7),
        do_true_cfg=False,
    )

    assert pipeline.get_dit_forward_stats(req=object(), output=object()) is None
