# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for multiview topology and the sparse Ulysses boundary."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import multiview_parallel as parallel
from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
    MultiviewAttentionContext,
    MultiviewLayout,
    padded_multiview_flex_attention,
)

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"cfg_parallel_size": 2},
        {"ulysses_degree": 8},
        {"tensor_parallel_size": 8},
        {"cfg_parallel_size": 2, "ulysses_degree": 2, "tensor_parallel_size": 2},
        {"use_hsdp": True, "hsdp_shard_size": 8},
        {"use_hsdp": True, "cfg_parallel_size": 2, "ulysses_degree": 4},
    ],
)
def test_supported_topologies(config):
    parallel.validate_multiview_parallel_config(
        SimpleNamespace(**config), num_attention_heads=32, num_key_value_heads=8, intermediate_size=12288
    )


@pytest.mark.parametrize(
    "config,error",
    [
        ({"cfg_parallel_size": 3}, "cfg_parallel_size"),
        ({"ulysses_degree": 0}, "positive"),
        ({"tensor_parallel_size": 0}, "positive"),
        ({"ring_degree": 2}, "ring_degree"),
        ({"allgather_degree": 2}, "allgather_degree"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size"),
        ({"vae_patch_parallel_size": 2}, "vae_patch_parallel_size"),
        ({"ulysses_degree": 2, "sequence_parallel_size": 4}, "sequence_parallel_size"),
        ({"ulysses_degree": 2, "ulysses_mode": "advanced_uaa"}, "strict"),
        ({"tensor_parallel_size": 2, "use_hsdp": True}, "cannot be combined"),
        ({"tensor_parallel_size": 4, "ulysses_degree": 4}, "KV heads"),
        ({"ulysses_degree": 3}, "query heads"),
    ],
)
def test_rejects_unsupported_topologies(config, error):
    with pytest.raises(ValueError, match=error):
        parallel.validate_multiview_parallel_config(
            SimpleNamespace(**config), num_attention_heads=32, num_key_value_heads=8, intermediate_size=12288
        )


def test_head_and_mlp_constraints_use_checkpoint_dimensions():
    with pytest.raises(ValueError, match="KV heads"):
        parallel.validate_multiview_parallel_config(
            SimpleNamespace(tensor_parallel_size=2),
            num_attention_heads=28,
            num_key_value_heads=7,
            intermediate_size=64,
        )
    with pytest.raises(ValueError, match="intermediate_size"):
        parallel.validate_multiview_parallel_config(
            SimpleNamespace(tensor_parallel_size=2),
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=63,
        )


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("scope", ["all_views", "same_view", "decomposed"])
@pytest.mark.parametrize("und_len", [3, 7])
def test_ulysses_preserves_global_mask_and_strips_only_cp_padding(monkeypatch, world_size, scope, und_len):
    # 66 GEN tokens: shard boundaries cut cameras and CP4/8 need tail padding.
    layout = MultiviewLayout(
        11,
        33,
        1,
        1,
        attention_scope=scope,
        max_und_tokens=64,
        decomposed_temporal_window_seconds=0.5,
        control_attends_sensor=True,
    )
    torch.manual_seed(7)
    q = torch.randn(1, layout.gen_tokens, 32, 8)
    k = torch.randn(1, layout.gen_tokens, 8, 8)
    v = torch.randn_like(k)
    ku = torch.randn(1, und_len, 8, 8)
    vu = torch.randn_like(ku)
    expected = padded_multiview_flex_attention(q, k, v, ku, vu, MultiviewAttentionContext(layout, {}))
    local_len = (layout.gen_tokens + world_size - 1) // world_size
    pad = local_len * world_size - layout.gen_tokens

    def padded(t):
        # Synthetic rows must never enter the real sparse attention context.
        return torch.cat((t, t.new_full((1, pad, *t.shape[2:]), 1000)), dim=1)

    full = [padded(t) for t in (q, k, v)]
    expected_padded = torch.cat((expected, expected.new_zeros(1, pad, 32, 8)), dim=1)
    for rank in range(world_size):
        exchanges = []

        def exchange(tensor, group, scatter, gather):
            assert group is sentinel
            index = len(exchanges)
            exchanges.append((scatter, gather))
            if index < 3:
                torch.testing.assert_close(tensor, full[index][:, rank * local_len : (rank + 1) * local_len])
                return full[index].chunk(world_size, dim=2)[rank].contiguous()
            assert (scatter, gather) == (1, 2)
            torch.testing.assert_close(tensor, expected_padded.chunk(world_size, dim=2)[rank], atol=1e-5, rtol=1e-5)
            return expected_padded[:, rank * local_len : (rank + 1) * local_len].contiguous()

        sentinel = object()
        monkeypatch.setattr(parallel, "_all_to_all", exchange)
        local = [t[:, rank * local_len : (rank + 1) * local_len] for t in full]
        actual = parallel.multiview_ulysses_attention(
            *local, ku, vu, MultiviewAttentionContext(layout, {}), group=sentinel, rank=rank, world_size=world_size
        )
        assert actual.shape == (1, local_len, 32, 8)
        assert exchanges == [(2, 1), (2, 1), (2, 1), (1, 2)]


def test_ulysses_rejects_unsharded_inputs_before_communication(monkeypatch):
    layout = MultiviewLayout(2, 4, 1, 1, max_und_tokens=64)
    q = torch.zeros(1, layout.gen_tokens, 4, 8)
    ku = torch.zeros(1, 3, 4, 8)

    def unexpected(*args):
        pytest.fail("Invalid input entered a collective")

    monkeypatch.setattr(parallel, "_all_to_all", unexpected)
    with pytest.raises(ValueError, match="sequence-sharding hooks"):
        parallel.multiview_ulysses_attention(
            q, q, q, ku, ku, MultiviewAttentionContext(layout, {}), group=object(), rank=0, world_size=2
        )


def _gloo_worker(rank, world_size, rendezvous, cp):
    from datetime import timedelta

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        # CP2 reproduces CFG2 x CP2; CP4 also tests non-divisible GEN length.
        groups = [dist.new_group(list(range(start, start + cp))) for start in range(0, world_size, cp)]
        group = groups[rank // cp]
        cp_rank = rank % cp
        torch.manual_seed(100 + rank // cp)
        layout = MultiviewLayout(11, 33, 1, 1, max_und_tokens=64)
        q = torch.randn(1, layout.gen_tokens, 8, 8)
        k = torch.randn(1, layout.gen_tokens, 4, 8)
        v = torch.randn_like(k)
        ku = torch.randn(1, 3 + rank // cp, 4, 8)
        vu = torch.randn_like(ku)
        expected = padded_multiview_flex_attention(q, k, v, ku, vu, MultiviewAttentionContext(layout, {}))
        pad = (-layout.gen_tokens) % cp
        padded = [torch.cat((t, t.new_full((1, pad, *t.shape[2:]), 1000)), dim=1) for t in (q, k, v)]
        actual = parallel.multiview_ulysses_attention(
            *(t.chunk(cp, dim=1)[cp_rank].contiguous() for t in padded),
            ku,
            vu,
            MultiviewAttentionContext(layout, {}),
            group=group,
            rank=cp_rank,
            world_size=cp,
        )
        expected = torch.cat((expected, expected.new_zeros(1, pad, 8, 8)), dim=1)
        torch.testing.assert_close(actual, expected.chunk(cp, dim=1)[cp_rank], atol=1e-5, rtol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp", [2, 4])
def test_real_collectives_use_cp_subgroup(tmp_path, cp):
    """Run the shared all-to-all implementation, including its inverse, on CPU."""
    import torch.distributed as dist

    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    torch.multiprocessing.spawn(_gloo_worker, args=(4, (tmp_path / "rendezvous").as_uri(), cp), nprocs=4)
