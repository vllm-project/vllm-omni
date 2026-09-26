# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import AutoencoderKL, AutoencoderKLFlux2
from torch import nn

from vllm_omni.config.omni_config import OmniStageDiffusionParallelConfig
from vllm_omni.diffusion import registry
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl as kl_module
from vllm_omni.diffusion.distributed.autoencoders import distributed_vae_executor as executor_module
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2 import DistributedAutoencoderKLFlux2
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeExecutor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.parallel]


class _RecordingDecoder(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1, bias=False, dtype=dtype)
        with torch.no_grad():
            self.conv.weight.copy_(torch.arange(1, 4, dtype=dtype).reshape(3, 1, 1, 1))
        self.batch_sizes = []

    def forward(self, z):
        self.batch_sizes.append(z.shape[0])
        return self.conv(z)


def _make_vae(vae_type=DistributedAutoencoderKLFlux2, dtype=torch.float32):
    """Keep the native decode methods; replace only the learned decoder."""
    vae = vae_type.__new__(vae_type)
    nn.Module.__init__(vae)
    vae.register_to_config(use_post_quant_conv=False)
    vae.decoder = _RecordingDecoder(dtype)
    vae.post_quant_conv = None
    vae.use_slicing = False
    vae.use_tiling = False
    return vae


@dataclass
class _WorldGroup:
    device_group: dist.ProcessGroup


def _batch_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(executor_module, "get_world_group", lambda: _WorldGroup(dist.group.WORLD))
            for name in (
                "get_data_parallel_world_size",
                "get_pipeline_parallel_world_size",
                "get_classifier_free_guidance_world_size",
            ):
                patch.setattr(kl_module, name, lambda: 1)
            for vae_type, native_type in (
                (DistributedAutoencoderKL, AutoencoderKL),
                (DistributedAutoencoderKLFlux2, AutoencoderKLFlux2),
            ):
                for dtype, autocast in ((torch.float32, False), (torch.bfloat16, False), (torch.float32, True)):
                    vae = _make_vae(vae_type, dtype)
                    vae.init_distributed()
                    vae.set_parallel_size(4, mode="batch")
                    vae.use_tiling = True
                    assert not vae.is_distributed_enabled()  # Spatial encode/decode stays disabled.
                    vae.use_tiling = False
                    # Reuse the same instance with changing batches and contents.
                    for batch, return_dict, expected_sizes in (
                        (2, False, (1, 1, 0, 0)),
                        (5, True, (2, 1, 1, 1)),
                        (1, True, (1, 1, 1, 1)),
                    ):
                        z = torch.arange(batch * 6, dtype=dtype).reshape(batch, 1, 2, 3) + batch
                        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                            expected = native_type.decode(vae, z, return_dict=False)[0]
                            vae.decoder.batch_sizes.clear()
                            if vae_type is DistributedAutoencoderKL and not return_dict:
                                result = vae.decode(z, False, torch.Generator().manual_seed(0))
                            else:
                                result = vae.decode(z, return_dict=return_dict)
                        actual = result.sample if return_dict else result[0]
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        assert actual.dtype == expected.dtype
                        assert sum(vae.decoder.batch_sizes) == expected_sizes[rank]
                        assert not vae.use_tiling

                    # Degree one bypasses collectives and keeps native slicing.
                    vae.set_parallel_size(1, mode="batch")
                    vae.use_slicing = True
                    z = torch.ones(2, 1, 2, 3, dtype=dtype)
                    vae.decoder.batch_sizes.clear()
                    if vae_type is DistributedAutoencoderKL:
                        actual = vae.decode(z, False, torch.Generator().manual_seed(0))[0]
                    else:
                        actual = vae.decode(z, return_dict=False)[0]
                    assert vae.decoder.batch_sizes == [1, 1]
                    expected = native_type.decode(vae, z, return_dict=False)[0]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires the CPU Gloo backend")
def test_batch_decode_matches_native_on_all_ranks(tmp_path):
    mp.spawn(_batch_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)


def test_batch_decode_without_distributed_initialization_uses_native(monkeypatch, mocker):
    vae = _make_vae()
    executor = mocker.Mock(spec=DistributedVaeExecutor)
    executor.parallel_mode = "batch"
    executor.parallel_size = executor.world_size = 4
    vae.distributed_executor = executor
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    z = torch.ones(2, 1, 2, 3)
    expected = AutoencoderKLFlux2.decode(vae, z, return_dict=False)[0]
    torch.testing.assert_close(vae.decode(z).sample, expected, rtol=0, atol=0)
    executor.execute.assert_not_called()


@pytest.mark.parametrize("config_type", [DiffusionParallelConfig, OmniStageDiffusionParallelConfig])
@pytest.mark.parametrize("axis", ["data_parallel_size", "pipeline_parallel_size", "cfg_parallel_size"])
def test_batch_mode_rejects_separate_request_or_pipeline_groups(config_type, axis):
    with pytest.raises(ValueError, match="batch.*DP, PP, and CFG"):
        config_type(vae_parallel_mode="batch", **{axis: 2})


@pytest.mark.parametrize("config_type", [DiffusionParallelConfig, OmniStageDiffusionParallelConfig])
def test_batch_mode_accepts_tensor_parallel_group(config_type):
    config = config_type(tensor_parallel_size=4, vae_patch_parallel_size=4, vae_parallel_mode="batch")
    assert config.world_size == 4


def test_batch_mode_rejects_implicitly_resolved_data_parallelism():
    config = DiffusionParallelConfig(vae_parallel_mode="batch", data_parallel_size=None)
    with pytest.raises(ValueError, match="batch.*DP"):
        config.resolve_data_parallel_size(2)


@pytest.mark.parametrize("tiling", [False, True])
def test_registry_preserves_requested_tiling_in_batch_mode(monkeypatch, mocker, tiling):
    vae = _make_vae()
    vae.distributed_executor = mocker.Mock(spec=DistributedVaeExecutor)

    class Pipeline(nn.Module):
        def __init__(self, od_config):
            super().__init__()
            self.vae = vae

    monkeypatch.setattr(registry.DiffusionModelRegistry, "_try_load_model_cls", lambda _: Pipeline)
    config = OmniDiffusionConfig(
        model_class_name="test_batch_vae",
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=4, vae_patch_parallel_size=4, vae_parallel_mode="batch"
        ),
        vae_use_tiling=tiling,
        vae_use_slicing=True,
        vae_fast_path="off",
    )
    model = registry.initialize_model(config)
    assert model.vae.use_tiling is tiling
    assert model.vae.use_slicing is True
    assert config.vae_use_tiling is tiling
    vae.distributed_executor.set_parallel_size.assert_called_once_with(4, mode="batch")


@pytest.mark.parametrize("vae_type", [nn.Identity, DistributedAutoencoderKLWan])
def test_registry_rejects_unsupported_batch_vae(monkeypatch, vae_type):
    class Pipeline(nn.Module):
        def __init__(self, od_config):
            super().__init__()
            self.vae = vae_type.__new__(vae_type)
            nn.Module.__init__(self.vae)

    monkeypatch.setattr(registry.DiffusionModelRegistry, "_try_load_model_cls", lambda _: Pipeline)
    config = OmniDiffusionConfig(
        model_class_name="unsupported_batch_vae",
        parallel_config=DiffusionParallelConfig(vae_parallel_mode="batch"),
    )
    with pytest.raises(ValueError, match="batch.*AutoencoderKL"):
        registry.initialize_model(config)


@pytest.mark.parametrize("mode", ["tile", "batch"])
def test_klein_uses_distributed_vae_only_for_batch_mode(monkeypatch, tmp_path, mode):
    from vllm_omni.diffusion.models.flux2_klein import pipeline_flux2_klein as module

    class StopAtVaeLoadError(Exception):
        pass

    def load_component(factory, *args, subfolder, **kwargs):
        if subfolder == "text_encoder":
            return nn.Identity()
        expected = DistributedAutoencoderKLFlux2 if mode == "batch" else AutoencoderKLFlux2
        assert factory.__self__ is expected
        raise StopAtVaeLoadError

    monkeypatch.setattr(module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module, "prefetch_subfolders", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "from_pretrained_with_prefetch", load_component)
    monkeypatch.setattr(module.FlowMatchEulerDiscreteScheduler, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.Qwen2TokenizerFast, "from_pretrained", lambda *args, **kwargs: None)
    config = OmniDiffusionConfig(model=str(tmp_path), parallel_config=DiffusionParallelConfig(vae_parallel_mode=mode))
    with pytest.raises(StopAtVaeLoadError):
        module.Flux2KleinPipeline(od_config=config)
