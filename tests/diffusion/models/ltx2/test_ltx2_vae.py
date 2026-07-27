# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LTX-2.3 VAE tiling and distributed decode behavior."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestLTXOutputRank:
    @pytest.mark.parametrize(
        ("distributed_vae_state", "expected_decode_calls"),
        [(False, 0), (True, 1), (RuntimeError("unavailable"), 0)],
    )
    def test_non_output_rank_only_enters_collective_vae_decode(
        self,
        monkeypatch,
        distributed_vae_state,
        expected_decode_calls,
    ):
        import vllm_omni.diffusion.models.ltx2.pipeline_ltx2 as ltx

        class FakeVae:
            dtype = torch.float32
            config = SimpleNamespace(timestep_conditioning=False)

            def __init__(self):
                self.decode_calls = 0

            def is_distributed_enabled(self):
                if isinstance(distributed_vae_state, Exception):
                    raise distributed_vae_state
                return distributed_vae_state

            def decode(self, *_args, **_kwargs):
                self.decode_calls += 1
                return (torch.ones(1, 1),)

        monkeypatch.setattr(ltx.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(ltx.torch.distributed, "get_rank", lambda: 1)
        pipe = object.__new__(ltx.LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.vae = FakeVae()

        output = pipe._decode_output(
            latents=torch.ones(1, 1),
            audio_latents=torch.ones(1, 1),
            output_type="np",
            connector_prompt_embeds=torch.ones(1, 1),
            generator=None,
            device=torch.device("cpu"),
            decode_timestep=0.0,
            decode_noise_scale=None,
            prompt_batch_size=1,
        )

        assert pipe.vae.decode_calls == expected_decode_calls
        assert output.output[0].numel() == 0
        assert output.output[1].numel() == 0


class TestLTX23VaeDistributedDecode:
    """Test LTX-2.3 distributed VAE helpers without loading weights."""

    def test_ltx23_video_vae_is_distributed_tile_only_class(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
            DistributedAutoencoderKLLTX2Video,
        )
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin

        assert issubclass(DistributedAutoencoderKLLTX2Video, DistributedVaeMixin)
        assert not hasattr(DistributedAutoencoderKLLTX2Video, "patch_split")

    def test_ltx23_vae_executor_gathers_known_tile_shapes_and_returns_empty_on_non_rank0(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import LTX2VaeExecutor
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
            DistributedOperator,
            GridSpec,
            TileTask,
        )

        z = torch.zeros(1, 1, 1, 1, 1)
        tile_output_shapes = {
            0: (1, 1, 1, 2, 2),
            1: (1, 1, 1, 2, 1),
            2: (1, 1, 1, 1, 2),
            3: (1, 1, 1, 1, 1),
        }
        tasks = [
            TileTask(0, (0, 0), z, workload=4),
            TileTask(1, (0, 1), z, workload=2),
            TileTask(2, (1, 0), z, workload=2),
            TileTask(3, (1, 1), z, workload=1),
        ]
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(2, 2),
            tile_spec={
                "max_tile_output_shape": (1, 1, 1, 2, 2),
                "tile_output_shapes": tile_output_shapes,
            },
            output_dtype=torch.float32,
        )
        seen = {}

        def exec_tile(task):
            return torch.full(tile_output_shapes[task.tile_id], float(task.tile_id + 1))

        def merge_tiles(coord_tensor_map, passed_grid_spec):
            seen["merged_shapes"] = {coord: tuple(tile.shape) for coord, tile in coord_tensor_map.items()}
            assert passed_grid_spec is grid_spec
            return torch.stack(
                [
                    coord_tensor_map[(0, 0)].flatten()[0],
                    coord_tensor_map[(0, 1)].flatten()[0],
                    coord_tensor_map[(1, 0)].flatten()[0],
                    coord_tensor_map[(1, 1)].flatten()[0],
                ]
            )

        operator = DistributedOperator(split=lambda _z: (tasks, grid_spec), exec=exec_tile, merge=merge_tiles)

        rank0_executor = object.__new__(LTX2VaeExecutor)
        rank0_executor.parallel_size = 2
        rank0_executor.world_size = 2
        rank0_executor.rank = 0

        def gather_rank0(local_tile_tensor):
            assigned = rank0_executor._balance_tasks(tasks, 2)
            rank1_results = [(task.tile_id, exec_tile(task)) for task in assigned[1]]
            rank1_tile_tensor = rank0_executor._pack_local_tiles_without_meta(
                rank1_results,
                list(local_tile_tensor.shape),
                z.device,
                torch.float32,
            )
            seen["rank0_gather_shape"] = tuple(local_tile_tensor.shape)
            return [local_tile_tensor, rank1_tile_tensor]

        def fail_final_sync(*_args, **_kwargs):
            raise AssertionError("broadcast_result=False should not sync the final result")

        rank0_executor.gather_tensors = gather_rank0
        rank0_executor._sync_final_result = fail_final_sync

        rank0_result = rank0_executor.execute(z, operator, broadcast_result=False)

        torch.testing.assert_close(rank0_result, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert seen["rank0_gather_shape"] == (2, 1, 1, 1, 2, 2)
        assert seen["merged_shapes"] == {
            (0, 0): (1, 1, 1, 2, 2),
            (0, 1): (1, 1, 1, 2, 1),
            (1, 0): (1, 1, 1, 1, 2),
            (1, 1): (1, 1, 1, 1, 1),
        }

        non_rank0_executor = object.__new__(LTX2VaeExecutor)
        non_rank0_executor.parallel_size = 2
        non_rank0_executor.world_size = 2
        non_rank0_executor.rank = 1

        def gather_rank1(local_tile_tensor):
            seen["rank1_gather_shape"] = tuple(local_tile_tensor.shape)
            return None

        def fail_non_rank0_merge(*_args, **_kwargs):
            raise AssertionError("non-rank0 should not merge gathered tiles")

        non_rank0_executor.gather_tensors = gather_rank1
        non_rank0_executor._sync_final_result = fail_final_sync

        empty_result = non_rank0_executor.execute(
            z,
            DistributedOperator(
                split=lambda _z: (tasks, grid_spec),
                exec=exec_tile,
                merge=fail_non_rank0_merge,
            ),
            broadcast_result=False,
        )

        assert tuple(empty_result.shape) == (0,)
        assert seen["rank1_gather_shape"] == (2, 1, 1, 1, 2, 2)


class TestLTX23VaeTiling:
    """Test LTX-2.3 video VAE tile helpers without loading weights."""

    def test_ltx23_video_vae_tile_split_uses_native_ltx23_tile_geometry(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
            DistributedAutoencoderKLLTX2Video,
        )

        vae = SimpleNamespace(
            spatial_compression_ratio=32,
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=448,
            tile_sample_stride_width=448,
            temporal_compression_ratio=8,
            dtype=torch.float32,
        )

        z = torch.zeros(1, 2, 5, 16, 24)
        tasks, grid_spec = DistributedAutoencoderKLLTX2Video.tile_split(vae, z)

        assert grid_spec.grid_shape == (2, 2)
        assert grid_spec.split_dims == (3, 4)
        assert grid_spec.tile_spec["sample_height"] == 512
        assert grid_spec.tile_spec["sample_width"] == 768
        assert grid_spec.tile_spec["blend_height"] == 64
        assert grid_spec.tile_spec["blend_width"] == 64
        assert grid_spec.tile_spec["max_tile_output_shape"] == (1, 3, 33, 512, 512)
        assert grid_spec.tile_spec["tile_output_shapes"] == {
            0: (1, 3, 33, 512, 512),
            1: (1, 3, 33, 512, 320),
            2: (1, 3, 33, 64, 512),
            3: (1, 3, 33, 64, 320),
        }
        assert [task.grid_coord for task in tasks] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert [tuple(task.tensor.shape) for task in tasks] == [
            (1, 2, 5, 16, 16),
            (1, 2, 5, 16, 10),
            (1, 2, 5, 2, 16),
            (1, 2, 5, 2, 10),
        ]
        assert [task.workload for task in tasks] == [5 * 16 * 16, 5 * 16 * 10, 5 * 2 * 16, 5 * 2 * 10]

    def test_ltx23_video_vae_tile_merge_blends_and_crops_like_tiled_decode(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
            DistributedAutoencoderKLLTX2Video,
        )
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import GridSpec

        class FakeVae:
            def __init__(self):
                self.blend_calls = []

            def clear_cache(self):
                pass

            def blend_v(self, _previous, current, blend_height):
                self.blend_calls.append(("v", blend_height))
                return current

            def blend_h(self, _previous, current, blend_width):
                self.blend_calls.append(("h", blend_width))
                return current

        fake_vae = FakeVae()
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(2, 2),
            tile_spec={
                "sample_height": 10,
                "sample_width": 10,
                "blend_height": 1,
                "blend_width": 2,
                "tile_sample_stride_height": 5,
                "tile_sample_stride_width": 5,
            },
        )
        tiles = {
            (0, 0): torch.full((1, 3, 2, 6, 6), 1.0),
            (0, 1): torch.full((1, 3, 2, 6, 6), 2.0),
            (1, 0): torch.full((1, 3, 2, 6, 6), 3.0),
            (1, 1): torch.full((1, 3, 2, 6, 6), 4.0),
        }

        merged = DistributedAutoencoderKLLTX2Video.tile_merge(fake_vae, tiles, grid_spec)

        assert merged.shape == (1, 3, 2, 10, 10)
        assert fake_vae.blend_calls == [("h", 2), ("v", 1), ("v", 1), ("h", 2)]
        torch.testing.assert_close(merged[:, :, :, :5, :5], torch.ones(1, 3, 2, 5, 5))
        torch.testing.assert_close(merged[:, :, :, :5, 5:], torch.full((1, 3, 2, 5, 5), 2.0))
        torch.testing.assert_close(merged[:, :, :, 5:, :5], torch.full((1, 3, 2, 5, 5), 3.0))
        torch.testing.assert_close(merged[:, :, :, 5:, 5:], torch.full((1, 3, 2, 5, 5), 4.0))

    def test_ltx23_video_vae_tiled_decode_dispatches_to_tile_operator(self):
        from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_ltx2

        z = torch.zeros(1, 2, 1, 16, 24)
        expected = torch.ones(1, 3, 1, 512, 768)
        seen = {}

        class FakeExecutor:
            def execute(self, tensor, operator, broadcast_result=True):
                seen["tensor"] = tensor
                seen["operator"] = operator
                seen["broadcast_result"] = broadcast_result
                return expected

        vae = SimpleNamespace(distributed_executor=FakeExecutor(), is_distributed_enabled=lambda: True)
        vae.tile_split = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tile_split.__get__(vae)
        vae.tile_exec = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tile_exec.__get__(vae)
        vae.tile_merge = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tile_merge.__get__(vae)

        output = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tiled_decode(
            vae,
            z,
            temb=torch.tensor(0.5),
            return_dict=False,
        )

        assert len(output) == 1
        assert output[0] is expected
        assert seen["tensor"] is z
        assert seen["broadcast_result"] is False
        assert seen["operator"].split.__name__ == "tile_split"
        assert seen["operator"].merge.__name__ == "tile_merge"
