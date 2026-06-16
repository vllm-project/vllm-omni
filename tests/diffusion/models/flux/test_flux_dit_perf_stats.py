from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.metrics.perf import FluxDiTForwardStats
from vllm_omni.diffusion.models.flux.pipeline_flux import FluxPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_flux_pipeline_for_stats(**parallel_overrides):
    parallel_config = SimpleNamespace(tensor_parallel_size=1)
    for key, value in parallel_overrides.items():
        setattr(parallel_config, key, value)

    pipeline = object.__new__(FluxPipeline)
    pipeline.od_config = SimpleNamespace(parallel_config=parallel_config)
    pipeline.transformer = SimpleNamespace(
        inner_dim=8,
        transformer_blocks=[object(), object()],
        single_transformer_blocks=[object(), object(), object()],
    )
    pipeline._last_dit_forward_stats = None
    return pipeline


def test_flux_records_model_specific_stats_from_runtime_shapes(monkeypatch):
    import vllm_omni.diffusion.models.flux.pipeline_flux as pipeline_flux_module

    pipeline = _make_flux_pipeline_for_stats(tensor_parallel_size=2)
    monkeypatch.setattr(pipeline_flux_module, "get_classifier_free_guidance_world_size", lambda: 1)

    pipeline._record_dit_forward_stats(
        latents=torch.zeros((2, 16, 8)),
        prompt_embeds=torch.zeros((2, 77, 8)),
        timesteps=torch.arange(4),
        do_true_cfg=False,
    )

    stats = pipeline.get_dit_forward_stats(req=object(), output=object())

    assert stats == FluxDiTForwardStats(
        batch_size=2,
        image_seq_len=16,
        text_seq_len=77,
        hidden_dim=8,
        ffn_dim=32,
        num_double_layers=2,
        num_single_layers=3,
        num_steps=4,
        forwards_per_step_per_gpu=1,
        tensor_parallel_size=2,
    )


def test_flux_counts_sequential_and_parallel_cfg_forwards(monkeypatch):
    import vllm_omni.diffusion.models.flux.pipeline_flux as pipeline_flux_module

    pipeline = _make_flux_pipeline_for_stats()
    monkeypatch.setattr(pipeline_flux_module, "get_classifier_free_guidance_world_size", lambda: 1)

    pipeline._record_dit_forward_stats(
        latents=torch.zeros((1, 16, 8)),
        prompt_embeds=torch.zeros((1, 77, 8)),
        timesteps=torch.arange(4),
        do_true_cfg=True,
    )
    sequential_stats = pipeline.get_dit_forward_stats(req=object(), output=object())

    monkeypatch.setattr(pipeline_flux_module, "get_classifier_free_guidance_world_size", lambda: 2)
    pipeline._record_dit_forward_stats(
        latents=torch.zeros((1, 16, 8)),
        prompt_embeds=torch.zeros((1, 77, 8)),
        timesteps=torch.arange(4),
        do_true_cfg=True,
    )
    parallel_stats = pipeline.get_dit_forward_stats(req=object(), output=object())

    assert sequential_stats is not None
    assert sequential_stats.forwards_per_step_per_gpu == 2
    assert parallel_stats is not None
    assert parallel_stats.forwards_per_step_per_gpu == 1


def test_flux_clears_stale_dit_forward_stats_at_request_start():
    pipeline = _make_flux_pipeline_for_stats()
    pipeline._last_dit_forward_stats = object()

    pipeline._clear_dit_forward_stats()

    assert pipeline.get_dit_forward_stats(req=object(), output=object()) is None
