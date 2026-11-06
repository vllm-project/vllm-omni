# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WorldKV switches execute independently of removed Deep Forcing policies."""

from unittest.mock import patch

import pytest
import torch

from tests.diffusion.ar_diffusion.test_worldkv import worldkv
from tests.diffusion.ar_diffusion.test_worldkv_retrieval import append, pose

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    "name", ["rebase_sink", "rebase_pc", "rebase_retrieval", "pc_budget_frames", "pc_recent_frames"]
)
def test_other_policy_keys_are_rejected(name):
    with pytest.raises(TypeError, match=name):
        worldkv.WorldKVConfig.parse({name: 1}, window_frames=18, sink_frames=9, chunk_frames=3)


def test_compression_does_not_implicitly_enable_retrieval():
    with pytest.raises(ValueError, match="requires retrieval"):
        worldkv.WorldKVConfig(world_keep_ratio=0.5)
    with pytest.raises(ValueError, match="positive bank_bytes"):
        worldkv.WorldKVConfig(retrieval_frames=2, world_keep_ratio=0.5)


@pytest.mark.parametrize("retrieval,ratio", [(0, 1.0), (2, 1.0), (2, 0.5)])
@pytest.mark.parametrize("score", ["centroid", "pairwise_mean"])
def test_retrieval_and_compression_switches_execute_only_when_enabled(retrieval, ratio, score):
    config = worldkv.WorldKVConfig(
        window_frames=8,
        sink_frames=2,
        retrieval_frames=retrieval,
        bank_bytes=1 << 20 if retrieval else 0,
        world_keep_ratio=ratio,
        world_score=score,
    )
    memory = worldkv.WorldKV(config, num_layers=2, tokens_per_frame=2)
    with (
        patch.object(worldkv, "camera_distances", wraps=worldkv.camera_distances) as rank,
        patch.object(torch.nn.functional, "normalize", wraps=torch.nn.functional.normalize) as normalize,
    ):
        for start in range(0, 40, 2):
            append(memory, start, camera=pose())
            assert memory.bank_nbytes <= config.bank_bytes
            # Local canonical KV retains absolute positions regardless of bank policy.
            for layer, rows in enumerate(memory.history):
                assert rows.positions[:4].tolist() == [0, 0, 1, 1]
                assert rows.positions[-4:].tolist() == [start, start, start + 1, start + 1]
                assert rows.positions.numel() <= (8 - retrieval) * 2
                for entry in memory.bank.values():
                    bank_rows = entry.layers[layer]
                    expected = 4 if ratio == 1 else 3
                    assert bank_rows.positions.numel() == expected
        assert bool(rank.call_count) == bool(retrieval)
        assert bool(normalize.call_count) == (ratio < 1)
    assert bool(memory.bank) == bool(retrieval)
    memory.close()
    assert not memory.bank and memory.history == [None, None]


@pytest.mark.parametrize(
    "name", ["reuse_attention_kv", "prefetch_kv", "prefetch_bank_allocation", "layerwise_bank_copy"]
)
@pytest.mark.parametrize("value", [1, "false", None])
def test_transfer_optimization_flags_require_booleans(name, value):
    with pytest.raises(ValueError, match=name):
        worldkv.WorldKVConfig(**{name: value})


def test_attention_window_cache_is_charged_to_gpu_reservation():
    baseline = worldkv.WorldKVConfig(window_frames=8, sink_frames=2)
    cached = worldkv.WorldKVConfig(window_frames=8, sink_frames=2, reuse_attention_kv=True)
    geometry = (2, 4, 2, 8, 2)
    extra = cached.reservation_bytes(*geometry, chunk_frames=2) - baseline.reservation_bytes(*geometry, chunk_frames=2)
    assert extra == 2 * 8 * 4 * 2 * 2 * 8 * 2


def test_layerwise_copy_charges_inflight_pending_storage_to_reservation():
    baseline = worldkv.WorldKVConfig(window_frames=8, sink_frames=2, retrieval_frames=2, bank_bytes=9000)
    staged = worldkv.WorldKVConfig(
        window_frames=8, sink_frames=2, retrieval_frames=2, bank_bytes=9000, layerwise_bank_copy=True
    )
    geometry = (2, 4, 2, 8, 2)
    extra = staged.reservation_bytes(*geometry, chunk_frames=2) - baseline.reservation_bytes(*geometry, chunk_frames=2)
    assert extra == 2 * 4 * 2 * (2 * 2 * 8 * 2)
