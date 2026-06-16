# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.metrics.perf import (
    DiTForwardStats,
    FluxDiTForwardStats,
    HunyuanVideoDiTForwardStats,
    WanDiTForwardStats,
    collect_flux_dit_forward_stats,
    estimate_diffusion_forward_flops_per_gpu,
    estimate_dit_flops_per_gpu,
    estimate_dit_perf_stats,
    estimate_flux_dit_forward_flops_per_gpu,
    estimate_hunyuan_video_dit_forward_flops_per_gpu,
    estimate_wan_dit_forward_flops_per_gpu,
    to_perf_stats,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _stats(**overrides) -> DiTForwardStats:
    values = {
        "batch_size": 2,
        "seq_len": 3,
        "context_len": 5,
        "hidden_dim": 7,
        "ffn_dim": 11,
        "num_layers": 13,
        "num_steps": 17,
        "forwards_per_step_per_gpu": 19,
    }
    values.update(overrides)
    return DiTForwardStats(**values)


def test_dit_flops_estimator_matches_hand_computed_terms() -> None:
    stats = _stats()

    breakdown = estimate_dit_flops_per_gpu(stats)

    qkv_out = 8 * 7 * 7 * 3
    cross_attn_proj = (4 * 7 * 7 * 3) + (4 * 7 * 7 * 5)
    mlp = 4 * 7 * 11 * 3
    self_attn = 4 * 3 * 3 * 7
    cross_attn = 4 * 3 * 5 * 7
    per_layer = qkv_out + cross_attn_proj + mlp + self_attn + cross_attn
    total = 2 * per_layer * 13 * 17 * 19

    assert breakdown.qkv_out_flops == qkv_out
    assert breakdown.cross_attn_proj_flops == cross_attn_proj
    assert breakdown.mlp_flops == mlp
    assert breakdown.self_attn_flops == self_attn
    assert breakdown.cross_attn_flops == cross_attn
    assert breakdown.flops_per_layer == per_layer
    assert breakdown.total_flops_per_gpu == total


@pytest.mark.parametrize(
    ("field", "expected_multiplier"),
    [
        ("batch_size", 2),
        ("num_layers", 2),
        ("num_steps", 2),
        ("forwards_per_step_per_gpu", 2),
    ],
)
def test_dit_flops_estimator_scales_linearly_for_request_multipliers(
    field: str,
    expected_multiplier: int,
) -> None:
    values = {
        "batch_size": 1,
        "num_layers": 1,
        "num_steps": 1,
        "forwards_per_step_per_gpu": 1,
    }
    base = _stats(**values)
    values[field] = 2
    doubled = _stats(**values)

    base_flops = estimate_dit_flops_per_gpu(base).total_flops_per_gpu
    doubled_flops = estimate_dit_flops_per_gpu(doubled).total_flops_per_gpu

    assert doubled_flops == base_flops * expected_multiplier


def test_dit_flops_estimator_exposes_self_attention_quadratic_sequence_term() -> None:
    short = _stats(batch_size=1, seq_len=3, num_layers=1, num_steps=1, forwards_per_step_per_gpu=1)
    long = _stats(batch_size=1, seq_len=6, num_layers=1, num_steps=1, forwards_per_step_per_gpu=1)

    short_breakdown = estimate_dit_flops_per_gpu(short)
    long_breakdown = estimate_dit_flops_per_gpu(long)

    assert long_breakdown.self_attn_flops == short_breakdown.self_attn_flops * 4


def test_dit_flops_estimator_context_len_does_not_change_self_attention() -> None:
    short_context = _stats(batch_size=1, context_len=5, num_layers=1, num_steps=1, forwards_per_step_per_gpu=1)
    long_context = _stats(batch_size=1, context_len=10, num_layers=1, num_steps=1, forwards_per_step_per_gpu=1)

    short_breakdown = estimate_dit_flops_per_gpu(short_context)
    long_breakdown = estimate_dit_flops_per_gpu(long_context)

    assert long_breakdown.self_attn_flops == short_breakdown.self_attn_flops
    assert long_breakdown.cross_attn_proj_flops > short_breakdown.cross_attn_proj_flops
    assert long_breakdown.cross_attn_flops > short_breakdown.cross_attn_flops


def test_dit_flops_to_perf_stats_sets_only_flops_counter() -> None:
    breakdown = estimate_dit_flops_per_gpu(_stats())

    perf_stats = to_perf_stats(breakdown)

    assert perf_stats.num_flops_per_gpu == breakdown.total_flops_per_gpu
    assert perf_stats.num_read_bytes_per_gpu == 0
    assert perf_stats.num_write_bytes_per_gpu == 0


def test_dit_hardware_mfu_uses_runtime_padded_sequence_length_not_goodput() -> None:
    useful = _stats(batch_size=1, seq_len=7, num_layers=1, num_steps=1, forwards_per_step_per_gpu=1)
    padded = _stats(batch_size=1, seq_len=8, num_layers=1, num_steps=1, forwards_per_step_per_gpu=1)

    useful_flops = estimate_dit_flops_per_gpu(useful).total_flops_per_gpu
    padded_flops = estimate_dit_flops_per_gpu(padded).total_flops_per_gpu

    assert padded_flops > useful_flops


class _FakeTransformer:
    inner_dim = 128
    transformer_blocks = [object(), object(), object()]
    single_transformer_blocks = [object(), object(), object(), object()]


def test_flux_dit_stats_use_runtime_tensors_and_transformer_shape() -> None:
    latents = torch.empty(2, 9, 64)
    prompt_embeds = torch.empty(2, 5, 4096)
    timesteps = torch.arange(4)

    stats = collect_flux_dit_forward_stats(
        latents=latents,
        prompt_embeds=prompt_embeds,
        timesteps=timesteps,
        transformer=_FakeTransformer(),
        do_true_cfg=False,
        cfg_parallel_world_size=1,
    )

    assert stats == FluxDiTForwardStats(
        batch_size=2,
        image_seq_len=9,
        text_seq_len=5,
        hidden_dim=128,
        ffn_dim=512,
        num_double_layers=3,
        num_single_layers=4,
        num_steps=4,
        forwards_per_step_per_gpu=1,
        tensor_parallel_size=1,
    )


def test_flux_dit_stats_count_sequential_true_cfg_as_two_forwards() -> None:
    stats = collect_flux_dit_forward_stats(
        latents=torch.empty(1, 9, 64),
        prompt_embeds=torch.empty(1, 5, 4096),
        timesteps=torch.arange(4),
        transformer=_FakeTransformer(),
        do_true_cfg=True,
        cfg_parallel_world_size=1,
    )

    assert stats is not None
    assert stats.forwards_per_step_per_gpu == 2


def test_flux_dit_stats_count_cfg_parallel_true_cfg_as_one_forward_per_gpu() -> None:
    stats = collect_flux_dit_forward_stats(
        latents=torch.empty(1, 9, 64),
        prompt_embeds=torch.empty(1, 5, 4096),
        timesteps=torch.arange(4),
        transformer=_FakeTransformer(),
        do_true_cfg=True,
        cfg_parallel_world_size=2,
    )

    assert stats is not None
    assert stats.forwards_per_step_per_gpu == 1


def test_flux_dit_stats_return_none_when_required_transformer_shape_is_missing() -> None:
    stats = collect_flux_dit_forward_stats(
        latents=torch.empty(1, 9, 64),
        prompt_embeds=torch.empty(1, 5, 4096),
        timesteps=torch.arange(4),
        transformer=object(),
        do_true_cfg=False,
        cfg_parallel_world_size=1,
    )

    assert stats is None


def test_flux_dit_stats_return_none_when_cfg_parallel_world_size_is_invalid() -> None:
    stats = collect_flux_dit_forward_stats(
        latents=torch.empty(1, 9, 64),
        prompt_embeds=torch.empty(1, 5, 4096),
        timesteps=torch.arange(4),
        transformer=_FakeTransformer(),
        do_true_cfg=True,
        cfg_parallel_world_size=0,
    )

    assert stats is None


def test_estimate_dit_perf_stats_wraps_estimated_flops_only() -> None:
    stats = _stats()

    perf_stats = estimate_dit_perf_stats(stats)

    assert perf_stats.num_flops_per_gpu == estimate_dit_flops_per_gpu(stats).total_flops_per_gpu
    assert perf_stats.num_read_bytes_per_gpu == 0
    assert perf_stats.num_write_bytes_per_gpu == 0


def test_estimate_flux_dit_forward_flops_per_gpu_counts_dual_and_single_blocks() -> None:
    stats = FluxDiTForwardStats(
        batch_size=2,
        image_seq_len=3,
        text_seq_len=5,
        hidden_dim=7,
        ffn_dim=11,
        num_double_layers=2,
        num_single_layers=3,
        num_steps=4,
        forwards_per_step_per_gpu=2,
    )

    joint_seq_len = 3 + 5
    dual_layer_flops = 8 * 7 * 7 * (3 + 5) + 4 * 7 * 11 * (3 + 5) + 4 * joint_seq_len * joint_seq_len * 7
    single_layer_flops = 8 * 7 * 7 * joint_seq_len + 4 * 7 * 11 * joint_seq_len + 4 * joint_seq_len * joint_seq_len * 7
    expected = 2 * ((2 * dual_layer_flops) + (3 * single_layer_flops)) * 4 * 2

    assert estimate_flux_dit_forward_flops_per_gpu(stats) == expected
    assert estimate_diffusion_forward_flops_per_gpu(stats) == expected


def test_estimate_wan_dit_forward_flops_per_gpu_uses_post_patch_local_sequence() -> None:
    stats = WanDiTForwardStats(
        batch_size=1,
        latent_num_frames=5,
        latent_height=8,
        latent_width=10,
        patch_size=(1, 2, 5),
        context_len=6,
        hidden_dim=4,
        ffn_dim=8,
        num_layers=7,
        num_steps=3,
        tensor_parallel_size=2,
        sequence_parallel_size=2,
    )

    logical_seq_len = 5 * 4 * 2
    local_seq_len = logical_seq_len // 2
    per_layer_flops = (
        8 * 4 * 4 * local_seq_len
        + (4 * 4 * 4 * local_seq_len + 4 * 4 * 4 * 6)
        + 4 * 4 * 8 * local_seq_len
        + 4 * local_seq_len * logical_seq_len * 4
        + 4 * local_seq_len * 6 * 4
    )
    expected = (per_layer_flops // 2) * 7 * 3

    assert stats.logical_seq_len == logical_seq_len
    assert stats.padded_seq_len == logical_seq_len
    assert stats.local_seq_len == local_seq_len
    assert estimate_wan_dit_forward_flops_per_gpu(stats) == expected


def test_estimate_wan_dit_forward_flops_per_gpu_can_use_explicit_forward_count() -> None:
    per_forward = WanDiTForwardStats(
        batch_size=1,
        latent_num_frames=1,
        latent_height=2,
        latent_width=2,
        patch_size=(1, 1, 1),
        context_len=2,
        hidden_dim=4,
        ffn_dim=8,
        num_layers=1,
        num_steps=1,
        forwards_per_step_per_gpu=1,
    )
    mixed_cfg = WanDiTForwardStats(
        batch_size=1,
        latent_num_frames=1,
        latent_height=2,
        latent_width=2,
        patch_size=(1, 1, 1),
        context_len=2,
        hidden_dim=4,
        ffn_dim=8,
        num_layers=1,
        num_steps=3,
        forwards_per_step_per_gpu=2,
        num_forwards_per_gpu=5,
    )

    assert estimate_wan_dit_forward_flops_per_gpu(mixed_cfg) == (
        5 * estimate_wan_dit_forward_flops_per_gpu(per_forward)
    )


def test_estimate_hunyuan_video_dit_forward_flops_per_gpu_uses_effective_context_len() -> None:
    stats = HunyuanVideoDiTForwardStats(
        batch_size=1,
        latent_num_frames=4,
        latent_height=6,
        latent_width=8,
        patch_size_t=2,
        patch_size=2,
        context_len=9,
        hidden_dim=6,
        ffn_dim=24,
        num_layers=3,
        num_steps=5,
        forwards_per_step_per_gpu=2,
        sequence_parallel_size=3,
    )

    assert stats.logical_seq_len == 2 * 3 * 4
    assert stats.padded_seq_len == 24
    assert stats.local_seq_len == 8
    expected = estimate_wan_dit_forward_flops_per_gpu(
        WanDiTForwardStats(
            batch_size=1,
            latent_num_frames=4,
            latent_height=6,
            latent_width=8,
            patch_size=(2, 2, 2),
            context_len=9,
            hidden_dim=6,
            ffn_dim=24,
            num_layers=3,
            num_steps=5,
            forwards_per_step_per_gpu=2,
            sequence_parallel_size=3,
        )
    )
    assert estimate_hunyuan_video_dit_forward_flops_per_gpu(stats) == expected


def test_flux_pipeline_get_dit_forward_stats_returns_cached_stats() -> None:
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.models.flux.pipeline_flux import FluxPipeline

    pipeline = object.__new__(FluxPipeline)
    stats = _stats()
    pipeline._last_dit_forward_stats = stats

    assert pipeline.get_dit_forward_stats(req=object(), output=DiffusionOutput(output=None)) is stats
