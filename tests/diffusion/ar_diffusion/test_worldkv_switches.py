# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WorldKV switches execute independently of removed Deep Forcing policies."""

from unittest.mock import patch

import pytest
import torch

from tests.diffusion.ar_diffusion.test_video_memory import vm
from tests.diffusion.ar_diffusion.test_video_retrieval import append, pose

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    "name", ["rebase_sink", "rebase_pc", "rebase_retrieval", "pc_budget_frames", "pc_recent_frames"]
)
def test_other_policy_keys_are_rejected(name):
    with pytest.raises(TypeError, match=name):
        vm.VideoMemoryConfig.parse({name: 1})


def test_compression_does_not_implicitly_enable_retrieval():
    with pytest.raises(ValueError, match="requires retrieval"):
        vm.VideoMemoryConfig(world_keep_ratio=0.5)
    with pytest.raises(ValueError, match="positive bank_bytes"):
        vm.VideoMemoryConfig(retrieval_frames=2, world_keep_ratio=0.5)


@pytest.mark.parametrize("retrieval,ratio", [(0, 1.0), (2, 1.0), (2, 0.5)])
@pytest.mark.parametrize("score", ["centroid", "pairwise_mean"])
def test_retrieval_and_compression_switches_execute_only_when_enabled(retrieval, ratio, score):
    config = vm.VideoMemoryConfig(
        window_frames=8,
        sink_frames=2,
        retrieval_frames=retrieval,
        bank_bytes=1 << 20 if retrieval else 0,
        world_keep_ratio=ratio,
        world_score=score,
    )
    memory = vm.VideoMemory(config, num_layers=2, tokens_per_frame=2)
    with (
        patch.object(vm, "camera_distances", wraps=vm.camera_distances) as rank,
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
