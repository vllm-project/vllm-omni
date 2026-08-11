import os
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageCausalConv3d

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.distributed.autoencoders import qwenimage_spatial_shard
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
    DistributedAutoencoderKLQwenImage,
)
from vllm_omni.diffusion.distributed.autoencoders.spatial_shard import (
    prepare_pipeline_spatial_shard_decode,
)
from vllm_omni.diffusion.offloader.module_collector import PipelineModules
from vllm_omni.platforms import current_omni_platform


@dataclass
class _FakeExecutor:
    group: object
    parallel_size: int
    parallel_mode: str


def _tiny_qwenimage_vae() -> DistributedAutoencoderKLQwenImage:
    return DistributedAutoencoderKLQwenImage(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        input_channels=3,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )


@pytest.mark.core_model
@pytest.mark.cpu
def test_qwenimage_auto_spatial_shard_gate_is_shape_and_topology_aware(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.spatial_shard.dist.get_world_size",
        lambda group=None: 4,
    )
    vae = DistributedAutoencoderKLQwenImage.__new__(DistributedAutoencoderKLQwenImage)
    vae.use_tiling = True
    vae.distributed_executor = _FakeExecutor(group=object(), parallel_size=4, parallel_mode="auto")
    vae.is_distributed_enabled = lambda: True

    landscape = torch.zeros((1, 16, 1, 60, 104))
    portrait = torch.zeros((1, 16, 1, 104, 60))

    assert vae._spatial_shard_decode_split_dim(landscape, 4) == "width"
    assert vae._spatial_shard_decode_split_dim(portrait, 4) == "height"
    assert vae._spatial_shard_decode_enabled(landscape) is True

    vae.distributed_executor.parallel_size = 2
    assert vae._spatial_shard_decode_enabled(landscape) is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_pipeline_dispatches_qwenimage_spatial_preparation(monkeypatch: pytest.MonkeyPatch):
    vae = DistributedAutoencoderKLQwenImage.__new__(DistributedAutoencoderKLQwenImage)
    prepared = []
    vae._prepare_spatial_shard_decode = lambda: prepared.append(vae)

    monkeypatch.setattr(
        "vllm_omni.diffusion.offloader.module_collector.ModuleDiscovery.discover",
        lambda pipeline: PipelineModules(
            dits=[],
            dit_names=[],
            encoders=[],
            encoder_names=[],
            vaes=[torch.nn.Identity(), vae],
        ),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.spatial_shard.dist.is_initialized",
        lambda: True,
    )

    prepare_pipeline_spatial_shard_decode(torch.nn.Identity())

    assert prepared == [vae]


@pytest.mark.core_model
@pytest.mark.cpu
@torch.no_grad()
def test_qwenimage_installed_decoder_preserves_cache_discovery_state_and_direct_path():
    torch.manual_seed(0)
    vae = _tiny_qwenimage_vae().eval()
    latents = torch.randn(1, 4, 2, 4, 5)
    parameter_names = {name for name, _ in vae.named_parameters()}
    state_dict = {name: value.clone() for name, value in vae.state_dict().items()}
    reference = vae.decode(latents, return_dict=False)[0]
    causal_conv_count = sum(isinstance(module, QwenImageCausalConv3d) for module in vae.decoder.modules())

    qwenimage_spatial_shard.install_qwenimage_spatial_shard_decode(vae, group=object(), split_dim="height")
    qwenimage_spatial_shard.install_qwenimage_spatial_shard_decode(vae, group=object(), split_dim="width")
    vae.clear_cache()

    assert causal_conv_count > 0
    assert len(vae._feat_map) == causal_conv_count
    assert sum(isinstance(module, QwenImageCausalConv3d) for module in vae.decoder.modules()) == causal_conv_count
    assert any(
        isinstance(module, qwenimage_spatial_shard.QwenImageDistCausalConv3d) for module in vae.decoder.modules()
    )
    assert {name for name, _ in vae.named_parameters()} == parameter_names
    vae.load_state_dict(state_dict, strict=True)
    assert torch.equal(vae.decode(latents, return_dict=False)[0], reference)


@pytest.mark.core_model
@pytest.mark.cpu
@torch.no_grad()
def test_qwenimage_installed_decoder_direct_path_can_be_compiled():
    torch.manual_seed(1)
    vae = _tiny_qwenimage_vae().eval()
    latents = torch.randn(1, 4, 1, 3, 5)
    qwenimage_spatial_shard.install_qwenimage_spatial_shard_decode(vae, group=object())
    eager = vae.decode(latents, return_dict=False)[0]

    compiled_decode = torch.compile(vae.decode, backend="eager", fullgraph=False)
    compiled = compiled_decode(latents, return_dict=False)[0]

    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)


def _qwenimage_spatial_shard_worker(rank: int, master_port: str, results) -> None:
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    device = current_omni_platform.get_torch_device(rank)
    current_omni_platform.set_device(device)
    backend = current_omni_platform.dist_backend
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank, backend=backend)
    initialize_model_parallel(sequence_parallel_size=2, ulysses_degree=2, backend=backend)

    try:
        torch.manual_seed(2)
        vae = _tiny_qwenimage_vae().to(device=device, dtype=torch.float32).eval()
        vae.init_distributed()
        generator = torch.Generator(device=device).manual_seed(3)
        landscape = torch.randn((1, 4, 2, 5, 7), generator=generator, device=device)
        portrait = torch.randn((1, 4, 2, 7, 5), generator=generator, device=device)

        with torch.inference_mode():
            vae.use_tiling = False
            vae.set_parallel_size(1, mode="tile")
            landscape_reference = vae.decode(landscape, return_dict=False)[0]
            portrait_reference = vae.decode(portrait, return_dict=False)[0]

            vae.enable_tiling(
                tile_sample_min_height=32,
                tile_sample_min_width=32,
                tile_sample_stride_height=24,
                tile_sample_stride_width=24,
            )
            vae.set_parallel_size(2, mode="tile")
            tile_before_install = vae.decode(landscape, return_dict=False)[0]

            vae.set_parallel_size(2, mode="auto")
            assert vae._spatial_shard_decode_split_dim(landscape, 2) == "width"
            landscape_auto = vae.decode(landscape, return_dict=False)[0]
            assert vae._spatial_shard_decode_split_dim(portrait, 2) == "height"
            portrait_auto = vae.decode(portrait).sample

            vae.set_parallel_size(2, mode="spatial_shard_height")
            landscape_height = vae.decode(landscape, return_dict=False)[0]
            vae.set_parallel_size(2, mode="spatial_shard_width")
            portrait_width = vae.decode(portrait, return_dict=False)[0]

            vae.set_parallel_size(2, mode="tile")
            tile_after_install = vae.decode(landscape, return_dict=False)[0]

        assert tile_after_install.shape == tile_before_install.shape
        assert torch.equal(tile_after_install, tile_before_install)
        assert all(
            buffer is None
            for module in vae.decoder.modules()
            for name, buffer in module._buffers.items()
            if name in {"_halo_recv_top_buf", "_halo_recv_bottom_buf"}
        )

        diffs = {
            "landscape_auto": (landscape_auto - landscape_reference).abs().max().item(),
            "portrait_auto": (portrait_auto - portrait_reference).abs().max().item(),
            "landscape_height": (landscape_height - landscape_reference).abs().max().item(),
            "portrait_width": (portrait_width - portrait_reference).abs().max().item(),
        }
        results[rank] = {
            "diffs": diffs,
            "landscape_shape": tuple(landscape_auto.shape),
            "portrait_shape": tuple(portrait_auto.shape),
        }
    finally:
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_qwenimage_spatial_shard_matches_reference_broadcasts_and_switches_paths():
    manager = mp.get_context("spawn").Manager()
    results = manager.dict()
    mp.spawn(
        _qwenimage_spatial_shard_worker,
        args=(str(get_open_port()), results),
        nprocs=2,
        join=True,
    )

    assert set(results) == {0, 1}
    assert results[0]["landscape_shape"] == results[1]["landscape_shape"] == (1, 3, 5, 40, 56)
    assert results[0]["portrait_shape"] == results[1]["portrait_shape"] == (1, 3, 5, 56, 40)
    for rank_results in results.values():
        assert max(rank_results["diffs"].values()) <= 2e-5
