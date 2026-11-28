# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real vision modules and hook dispatch with Gloo, without checkpoint downloads."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Qwen3VLVisionConfig

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLVisionModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RecordingHook(ModelHook):
    """Record the same registry dispatch used by layerwise offloading."""

    def __init__(self, events, name):
        self.events = events
        self.name = name

    def pre_forward(self, module, *args, **kwargs):
        self.events.append((self.name, "pre"))
        return args, kwargs

    def post_forward(self, module, output):
        self.events.append((self.name, "post"))
        return output


def _vision_worker(rank, size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=size, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(10168)
        config = Qwen3VLVisionConfig(
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=16,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1],
        )
        model = MiniMaxH3Qwen3VLVisionModel(config).eval()
        events = []
        names = []
        modules = [(f"block{index}", block) for index, block in enumerate(model.blocks)]
        modules += [(f"deep{index}", merger) for index, merger in enumerate(model.deepstack_merger_list)]
        modules += [("final", model.merger)]
        for name, module in modules:
            names.append(name)
            HookRegistry.get_or_create(module).register_hook("vision_sp_test", _RecordingHook(events, name))

        generator = torch.Generator().manual_seed(10779)
        # Multi-frame/multi-image, uneven groups, and ranks with no local tokens.
        for geometry in ([[1, 2, 2], [2, 2, 6]], [[1, 4, 8]]):
            grid = torch.tensor(geometry, dtype=torch.long)
            count = sum(t * h * w for t, h, w in geometry)
            pixels = torch.randn(count, 24, generator=generator)
            with torch.inference_mode():
                model.sequence_parallel_group = None
                reference = model(pixels, grid)
                expected_events = list(events)
                assert all(expected_events.count((name, phase)) == 1 for name in names for phase in ("pre", "post"))
                events.clear()
                model.sequence_parallel_group = dist.group.WORLD
                actual = model(pixels, grid)
                assert events == expected_events
                events.clear()
                repeated = model(pixels, grid)
                assert events == expected_events
                events.clear()

            assert len(actual[1]) == len(reference[1]) == 2
            for expected, output, again in zip(
                [reference[0], *reference[1]], [actual[0], *actual[1]], [repeated[0], *repeated[1]], strict=True
            ):
                assert output.shape == expected.shape == (count // 4, 32)
                assert torch.equal(output.contiguous().view(torch.uint8), again.contiguous().view(torch.uint8))
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
                peers = [torch.empty_like(output) for _ in range(size)]
                dist.all_gather(peers, output)
                assert all(torch.equal(peer, output) for peer in peers)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("size", [2, 4, 8])
def test_vision_sp_real_model_outputs_and_hooks(tmp_path, size):
    mp.spawn(_vision_worker, args=(size, (tmp_path / "vision-model").as_uri()), nprocs=size, join=True)
