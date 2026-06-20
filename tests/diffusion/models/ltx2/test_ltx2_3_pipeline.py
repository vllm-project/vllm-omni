# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LTX-2.3 pipeline integration.

These tests verify:
- Pipeline is properly registered in the diffusion registry
- Post-process function is registered
- Cache-DiT enablers are registered
- Pipeline does NOT inherit from LTX2Pipeline
- Vocoder sample rate detection logic
- Re-export module works correctly
"""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_ltx23_pipeline(sequence_parallel_size: int = 1):
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

    pipeline = object.__new__(LTX23Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.audio_vae_temporal_compression_ratio = 4
    pipeline.audio_vae_mel_compression_ratio = 4
    pipeline.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=sequence_parallel_size))
    pipeline.audio_vae = SimpleNamespace(
        latents_mean=torch.tensor(0.0),
        latents_std=torch.tensor(1.0),
    )
    return pipeline


class TestPipelineIndependence:
    """Verify LTX23Pipeline is fully independent from LTX2Pipeline."""

    def test_ltx23_pipeline_does_not_inherit_from_ltx2(self):
        """LTX23Pipeline must NOT inherit from LTX2Pipeline."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

        assert not issubclass(LTX23Pipeline, LTX2Pipeline), (
            "LTX23Pipeline should be fully independent and not inherit from LTX2Pipeline"
        )

    def test_ltx23_pipeline_is_nn_module(self):
        """LTX23Pipeline must be an nn.Module."""
        import torch.nn as nn

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

        assert issubclass(LTX23Pipeline, nn.Module)

    def test_ltx23_pipeline_has_progress_bar(self):
        """LTX23Pipeline must mix in ProgressBarMixin."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

        assert issubclass(LTX23Pipeline, ProgressBarMixin)

    def test_ltx23_pipeline_has_cfg_parallel_mixin(self):
        """LTX23Pipeline must use the shared CFG parallel implementation."""
        from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

        assert issubclass(LTX23Pipeline, CFGParallelMixin)

    def test_ltx23_pipeline_declares_offload_components(self):
        """LTX23Pipeline must expose LTX-2.3-specific modules to offload discovery."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

        pipe = object.__new__(LTX23Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.transformer = torch.nn.Linear(1, 1)
        pipe.text_encoder = torch.nn.Linear(1, 1)
        pipe.connectors = torch.nn.Linear(1, 1)
        pipe.vae = torch.nn.Linear(1, 1)
        pipe.audio_vae = torch.nn.Linear(1, 1)
        pipe.vocoder = torch.nn.Linear(1, 1)

        modules = ModuleDiscovery.discover(pipe)

        assert LTX23Pipeline._dit_modules == ["transformer"]
        assert LTX23Pipeline._encoder_modules == ["text_encoder", "connectors"]
        assert LTX23Pipeline._vae_modules == ["vae", "audio_vae"]
        assert LTX23Pipeline._resident_modules == ["vocoder"]
        assert modules.dit_names == ["transformer"]
        assert modules.encoder_names == ["text_encoder", "connectors"]
        assert modules.resident_names == ["vocoder"]
        assert len(modules.vaes) == 2

    def test_ltx23_pipeline_has_diffusion_pipeline_profiler_mixin(self):
        """LTX23Pipeline must support lightweight stage timing."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin

        assert issubclass(LTX23Pipeline, DiffusionPipelineProfilerMixin)


class TestLTX23VaeDecodeParallel:
    """Test LTX-2.3 video VAE tiled parallel helpers without loading weights."""

    def test_ltx23_video_vae_is_distributed_tile_only_class(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
            DistributedAutoencoderKLLTX2Video,
        )
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin

        assert issubclass(DistributedAutoencoderKLLTX2Video, DistributedVaeMixin)
        assert not hasattr(DistributedAutoencoderKLLTX2Video, "patch_split")

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


class TestCFGParallelHelpers:
    """Test LTX-2.3 CFG helper math without loading model weights."""

    def test_combine_cfg_noise_matches_x0_space_formula(self):
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

        pipe = object.__new__(LTX23Pipeline)
        video_sample = torch.tensor([[[1.0, -2.0]]])
        audio_sample = torch.tensor([[[0.5, 3.0]]])
        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2]]])
        audio_neg = torch.tensor([[[0.1, 0.4]]])
        video_sigma = torch.tensor(0.25)
        audio_sigma = torch.tensor(0.5)
        scale = 4.0

        video_combined, audio_combined = pipe.combine_cfg_noise(
            (video_pos, audio_pos),
            (video_neg, audio_neg),
            scale,
            video_latents=video_sample,
            audio_latents=audio_sample,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

        x0_video_cond = video_sample - video_pos * video_sigma
        x0_video_uncond = video_sample - video_neg * video_sigma
        x0_video_guided = x0_video_cond + (scale - 1) * (x0_video_cond - x0_video_uncond)
        expected_video = (video_sample - x0_video_guided) / video_sigma

        x0_audio_cond = audio_sample - audio_pos * audio_sigma
        x0_audio_uncond = audio_sample - audio_neg * audio_sigma
        x0_audio_guided = x0_audio_cond + (scale - 1) * (x0_audio_cond - x0_audio_uncond)
        expected_audio = (audio_sample - x0_audio_guided) / audio_sigma
        assert torch.allclose(video_combined, expected_video)
        assert torch.allclose(audio_combined, expected_audio)

    def test_two_rank_cfg_parallel_smoke_uses_rank_local_branch_and_x0_formula(self, monkeypatch):
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2_3 as ltx23

        pipe = object.__new__(ltx23.LTX23Pipeline)
        video_sample = torch.tensor([[[1.0, -2.0]]])
        audio_sample = torch.tensor([[[0.5, 3.0, -1.0]]])
        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2, 0.3]]])
        audio_neg = torch.tensor([[[0.1, 0.4, -0.5]]])
        video_sigma = torch.tensor(0.25)
        audio_sigma = torch.tensor(0.5)
        scale = 4.0

        class FakeCfgGroup:
            def all_gather(self, tensor, separate_tensors=True):
                assert separate_tensors
                if tensor.shape == video_pos.shape:
                    return [video_pos, video_neg]
                return [audio_pos, audio_neg]

        monkeypatch.setattr(ltx23, "get_classifier_free_guidance_world_size", lambda: 2)
        monkeypatch.setattr(ltx23, "get_cfg_group", lambda: FakeCfgGroup())

        expected_video = ltx23.LTX23Pipeline._combine_x0_space_cfg(
            video_sample,
            video_pos,
            video_neg,
            video_sigma,
            scale,
        )
        expected_audio = ltx23.LTX23Pipeline._combine_x0_space_cfg(
            audio_sample,
            audio_pos,
            audio_neg,
            audio_sigma,
            scale,
        )

        for rank, expected_branch in ((0, "positive"), (1, "negative")):
            calls = []
            monkeypatch.setattr(ltx23, "get_classifier_free_guidance_rank", lambda rank=rank: rank)

            def fake_predict_noise(**kwargs):
                calls.append(kwargs["branch"])
                if kwargs["branch"] == "positive":
                    return video_pos, audio_pos
                return video_neg, audio_neg

            object.__setattr__(pipe, "predict_noise", fake_predict_noise)
            video_combined, audio_combined = pipe.predict_noise_with_parallel_cfg(
                true_cfg_scale=scale,
                positive_kwargs={"branch": "positive"},
                negative_kwargs={"branch": "negative"},
                cfg_normalize=False,
                video_latents=video_sample,
                audio_latents=audio_sample,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
            )

            assert calls == [expected_branch]
            torch.testing.assert_close(video_combined, expected_video)
            torch.testing.assert_close(audio_combined, expected_audio)

        assert "_cfg_video_latents" not in pipe.__dict__
        assert "_cfg_audio_latents" not in pipe.__dict__


class TestCFGParallelForwardPath:
    """Test the LTX-2.3 CFG-parallel denoising path without loading model weights."""

    @pytest.mark.parametrize(("cfg_rank", "expected_prompt_value"), [(0, 1.0), (1, 0.0)])
    def test_forward_cfg_parallel_steps_video_and_audio_scheduler(
        self,
        monkeypatch,
        cfg_rank,
        expected_prompt_value,
    ):
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2_3 as ltx23
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = object.__new__(ltx23.LTX23Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.device = torch.device("cpu")
        pipe.tokenizer_max_length = 1
        pipe.vae_spatial_compression_ratio = 32
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1
        pipe.audio_sampling_rate = 1
        pipe.audio_hop_length = 1
        pipe.audio_vae_temporal_compression_ratio = 1
        pipe.audio_vae_mel_compression_ratio = 1
        pipe.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1))
        pipe.tokenizer = SimpleNamespace(padding_side="left")
        pipe.vae = SimpleNamespace(
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
            config=SimpleNamespace(scaling_factor=1.0),
        )
        pipe.audio_vae = SimpleNamespace(
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
            config=SimpleNamespace(mel_bins=2, latent_channels=1),
        )

        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2]]])
        audio_neg = torch.tensor([[[0.1, 0.4]]])

        class FakeCfgGroup:
            def all_gather(self, tensor, separate_tensors=True):
                assert separate_tensors
                if torch.equal(tensor, video_pos) or torch.equal(tensor, video_neg):
                    return [video_pos, video_neg]
                if torch.equal(tensor, audio_pos) or torch.equal(tensor, audio_neg):
                    return [audio_pos, audio_neg]
                raise AssertionError(f"Unexpected gathered tensor: {tensor}")

        monkeypatch.setattr(ltx23, "get_classifier_free_guidance_world_size", lambda: 2)
        monkeypatch.setattr(ltx23, "get_classifier_free_guidance_rank", lambda: cfg_rank)
        monkeypatch.setattr(ltx23, "get_cfg_group", lambda: FakeCfgGroup())

        def fake_retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas=None, mu=None):
            scheduler.sigmas = torch.tensor([0.25, 0.25], device=device)
            return torch.tensor([1.0, 0.5], device=device), 2

        monkeypatch.setattr(ltx23, "retrieve_timesteps", fake_retrieve_timesteps)

        class FakeScheduler:
            def __init__(self, name="video", calls=None):
                self.name = name
                self.calls = [] if calls is None else calls
                self.config = {
                    "max_image_seq_len": 4096,
                    "base_image_seq_len": 1024,
                    "base_shift": 0.95,
                    "max_shift": 2.05,
                }
                self.sigmas = torch.tensor([0.25, 0.25])

            def __deepcopy__(self, memo):
                return FakeScheduler("audio", self.calls)

            def step(self, noise_pred, t, latents, return_dict=False, generator=None):
                self.calls.append((self.name, noise_pred.clone(), t.clone(), latents.clone()))
                return (latents - noise_pred,)

        class FakeConnectors:
            def to(self, device):
                return self

            def __call__(self, prompt_embeds, prompt_attention_mask, padding_side):
                assert padding_side == "left"
                assert prompt_embeds.shape[0] == 2
                return prompt_embeds, prompt_embeds, prompt_attention_mask

        class FakeRope:
            def prepare_video_coords(self, batch_size, num_frames, height, width, device, fps):
                return torch.zeros(batch_size, num_frames * height * width, 3, device=device)

            def prepare_audio_coords(self, batch_size, num_frames, device):
                return torch.zeros(batch_size, num_frames, 1, device=device)

        class FakeTransformer:
            def __init__(self):
                self.config = SimpleNamespace(in_channels=2)
                self.rope = FakeRope()
                self.audio_rope = FakeRope()
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                expected_prompt = torch.full((1, 1, 1), expected_prompt_value)
                torch.testing.assert_close(kwargs["encoder_hidden_states"], expected_prompt)
                torch.testing.assert_close(kwargs["audio_encoder_hidden_states"], expected_prompt)
                assert kwargs["hidden_states"].shape == (1, 1, 2)
                assert kwargs["audio_hidden_states"].shape == (1, 1, 2)
                if cfg_rank == 0:
                    return video_pos, audio_pos
                return video_neg, audio_neg

        class DummyProgress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self):
                pass

        pipe.scheduler = FakeScheduler()
        pipe.connectors = FakeConnectors()
        pipe.transformer = FakeTransformer()
        object.__setattr__(pipe, "progress_bar", lambda total: DummyProgress())

        def fake_encode_prompt(**kwargs):
            return (
                torch.ones(1, 1, 1),
                torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, 1, 1),
                torch.ones(1, 1, dtype=torch.bool),
            )

        object.__setattr__(pipe, "encode_prompt", fake_encode_prompt)

        video_latents = torch.tensor([[[1.0, -2.0]]])
        audio_latents = torch.tensor([[[0.5, 3.0]]])
        req = OmniDiffusionRequest(
            prompts=[{"prompt": "prompt", "negative_prompt": "negative"}],
            sampling_params=OmniDiffusionSamplingParams(
                height=32,
                width=32,
                num_frames=1,
                frame_rate=1.0,
                num_inference_steps=2,
                guidance_scale=4.0,
                latents=video_latents,
                audio_latents=audio_latents,
                output_type="latent",
            ),
            request_id="ltx23-cfg-parallel-forward-test",
        )

        output = pipe.forward(req)

        expected_video_noise = ltx23.LTX23Pipeline._combine_x0_space_cfg(
            video_latents,
            video_pos,
            video_neg,
            pipe.scheduler.sigmas[0],
            4.0,
        )
        expected_audio_noise = ltx23.LTX23Pipeline._combine_x0_space_cfg(
            audio_latents,
            audio_pos,
            audio_neg,
            pipe.scheduler.sigmas[0],
            4.0,
        )
        scheduler_call_names = [call[0] for call in pipe.scheduler.calls]
        assert scheduler_call_names == ["video", "audio", "video", "audio"]
        assert len(pipe.transformer.calls) == 2
        torch.testing.assert_close(pipe.scheduler.calls[0][1], expected_video_noise)
        torch.testing.assert_close(pipe.scheduler.calls[1][1], expected_audio_noise)
        torch.testing.assert_close(pipe.scheduler.calls[2][1], expected_video_noise)
        torch.testing.assert_close(pipe.scheduler.calls[3][1], expected_audio_noise)
        torch.testing.assert_close(pipe.scheduler.calls[2][3], video_latents - expected_video_noise)
        torch.testing.assert_close(pipe.scheduler.calls[3][3], audio_latents - expected_audio_noise)

        video_out, audio_out = output.output
        torch.testing.assert_close(video_out, (video_latents - 2 * expected_video_noise).reshape(1, 2, 1, 1, 1))
        torch.testing.assert_close(audio_out, (audio_latents - 2 * expected_audio_noise).reshape(1, 1, 1, 2))


class TestRegistryIntegration:
    """Verify all LTX-2.3 pipeline variants are registered."""

    def test_pipeline_models_registered(self):
        """LTX-2.3 pipeline variants must be in _DIFFUSION_MODELS."""
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_MODELS, f"{name} not found in _DIFFUSION_MODELS"

    def test_pipeline_module_paths(self):
        """Registry entries must point to the correct modules."""
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        # T2V -> pipeline_ltx2_3
        assert _DIFFUSION_MODELS["LTX23Pipeline"] == ("ltx2", "pipeline_ltx2_3", "LTX23Pipeline")

        # I2V -> pipeline_ltx2_3_image2video
        assert _DIFFUSION_MODELS["LTX23ImageToVideoPipeline"] == (
            "ltx2",
            "pipeline_ltx2_3_image2video",
            "LTX23ImageToVideoPipeline",
        )

    def test_post_process_funcs_registered(self):
        """Pipeline variants must map to get_ltx2_post_process_func."""
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_POST_PROCESS_FUNCS, f"{name} not in _DIFFUSION_POST_PROCESS_FUNCS"
            assert _DIFFUSION_POST_PROCESS_FUNCS[name] == "get_ltx2_post_process_func"

    def test_cache_dit_for_ltx2_does_not_have_custom_enablers_registered(self):
        """Pipeline variants are *not* registered in CUSTOM_DIT_ENABLERS."""
        from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS

        # NOTE: We used to have custom enablers for this model, but refactored to handle
        # it more generically. Now we only need to ensure it has git cache adapter config.
        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name not in CUSTOM_DIT_ENABLERS, f"{name} not in CUSTOM_DIT_ENABLERS"

    def test_ltx2_transformer_has_dit_cache_config(self):
        """Ensure LTX2 has a Cache DiT adapter config and that it uses separate CFG."""
        from vllm_omni.diffusion.cache.cache_dit_backend import CacheDiTAdapterConfig
        from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel

        adapter_config = getattr(LTX2VideoTransformer3DModel, "_cache_dit_adapter_config")
        assert isinstance(adapter_config, CacheDiTAdapterConfig)
        assert adapter_config.has_separate_cfg


class TestVocoderSampleRateDetection:
    """Test _detect_vocoder_output_sample_rate logic."""

    def test_detects_48khz_from_config(self):
        """Should detect output_sampling_rate=48000 from vocoder/config.json."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000, "input_sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result == 48000

    def test_returns_none_for_no_output_sr(self):
        """Should return None if vocoder config has no output_sampling_rate."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result is None

    def test_returns_none_for_missing_directory(self):
        """Should return None if vocoder directory doesn't exist."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        result = _detect_vocoder_output_sample_rate("/nonexistent/path")
        assert result is None


class TestPostProcessFunction:
    """Test the post-process function factory."""

    def test_post_process_includes_audio_sample_rate(self):
        """Post-process func should include audio_sample_rate when detected."""
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import get_ltx2_post_process_func

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000}, f)

            # Create a minimal od_config mock
            class MockConfig:
                model = tmpdir

            func = get_ltx2_post_process_func(MockConfig())

            video = torch.zeros(1, 3, 4, 64, 64)
            audio = torch.zeros(1, 1, 48000)
            result = func((video, audio))

            assert isinstance(result, dict)
            assert "video" in result
            assert "audio" in result
            assert result["audio_sample_rate"] == 48000

    def test_post_process_without_vocoder_config(self):
        """Post-process func should work without vocoder config (no audio_sample_rate key)."""
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import get_ltx2_post_process_func

        class MockConfig:
            model = "/nonexistent/path"

        func = get_ltx2_post_process_func(MockConfig())

        video = torch.zeros(1, 3, 4, 64, 64)
        audio = torch.zeros(1, 1, 16000)
        result = func((video, audio))

        assert isinstance(result, dict)
        assert "video" in result
        assert "audio" in result
        assert "audio_sample_rate" not in result


class TestReExportModule:
    """Test that pipeline_ltx2_3_image2video.py correctly re-exports."""

    def test_i2v_classes_importable(self):
        """I2V classes must be importable from the re-export module."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        assert LTX23ImageToVideoPipeline is not None

    def test_post_process_func_importable(self):
        """get_ltx2_post_process_func must be importable from re-export module."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import get_ltx2_post_process_func

        assert callable(get_ltx2_post_process_func)

    def test_i2v_classes_are_same_as_direct_import(self):
        """Re-exported classes must be the same objects as direct imports."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline as Direct
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import (
            LTX23ImageToVideoPipeline as ReExported,
        )

        assert Direct is ReExported


class TestInitExports:
    """Test that __init__.py exports all LTX-2.3 classes."""

    def test_all_ltx23_classes_exported(self):
        """All LTX-2.3 pipeline classes must be in the ltx2 package __all__."""
        from vllm_omni.diffusion.models import ltx2

        expected_classes = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected_classes:
            assert hasattr(ltx2, name), f"{name} not exported from ltx2 package"
            assert name in ltx2.__all__, f"{name} not in ltx2.__all__"


class TestAudioLatentSPPadding:
    def test_prepare_audio_latents_pads_generated_dummy_length_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=2)

        latents, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=8,
            num_mel_bins=64,
            audio_latent_length=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert original_num_frames == 1
        assert padded_num_frames == 2
        assert latents.shape == (1, 2, 128)

    def test_prepare_audio_latents_pads_provided_packed_sequence_dim_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.arange(40, dtype=torch.float32).view(1, 10, 4)

        padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=2,
            num_mel_bins=8,
            audio_latent_length=10,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        assert original_num_frames == 10
        assert padded_num_frames == 12
        assert padded.shape == (1, 12, 4)
        torch.testing.assert_close(padded[:, :10], latents)
        torch.testing.assert_close(padded[:, 10:], torch.zeros(1, 2, 4))

    def test_prepare_audio_latents_accepts_already_padded_4d_latents_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.arange(96, dtype=torch.float32).view(1, 2, 12, 4)

        audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)
        padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=2,
            num_mel_bins=16,
            audio_latent_length=audio_latent_length,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        assert audio_latent_length == 10
        assert original_num_frames == 10
        assert padded_num_frames == 12
        assert padded.shape == (1, 12, 8)
        torch.testing.assert_close(padded, pipeline._pack_audio_latents(latents))

    def test_resolve_audio_latent_length_preserves_legacy_4d_shape_inference(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.zeros(1, 2, 13, 4)

        audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)

        assert audio_latent_length == 13

    def test_prepare_audio_latents_rejects_incompatible_provided_length(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.zeros(1, 11, 4)

        with pytest.raises(ValueError, match="incompatible audio frame count"):
            pipeline.prepare_audio_latents(
                batch_size=1,
                num_channels_latents=2,
                num_mel_bins=8,
                audio_latent_length=10,
                dtype=torch.float32,
                device=torch.device("cpu"),
                latents=latents,
            )
