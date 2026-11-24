# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for GLM-Image VAE patch parallelism support."""

from dataclasses import dataclass
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.distributed.autoencoders import distributed_vae_executor as executor_module
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# GLM-Image vae/config.json topology (4 blocks -> 8x, 16 latent channels, no quant
# convs) at reduced width. sample_size scales tile_latent_min_size from 128 to 8,
# so an 8x8 latent stands in for a 1024px image and 12x12 for a 1536px image.
_GLM_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "block_out_channels": [8, 16, 16, 16],
    "layers_per_block": 1,
    "norm_num_groups": 8,
    "mid_block_add_attention": False,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "sample_size": 64,
}
_PARITY_CASES = (
    # (latent side, strategy selected at vae_patch_parallel_size=2, decoder calls per rank)
    (8, "patch", 1),
    (12, "tile", 2),
)


@dataclass
class _WorldGroup:
    device_group: dist.ProcessGroup
    cpu_group: dist.ProcessGroup


def _decode_parity_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(executor_module, "get_world_group", lambda: _WorldGroup(dist.group.WORLD, dist.group.WORLD))
            torch.manual_seed(0)
            vae = DistributedAutoencoderKL.from_config(_GLM_VAE_CONFIG).eval()
            vae.use_tiling = True
            assert vae.tile_latent_min_size == 8

            decoder_calls = []
            vae.decoder.register_forward_pre_hook(lambda _module, _args: decoder_calls.append(1))

            for latent_side, strategy, calls_per_rank in _PARITY_CASES:
                z = torch.randn(1, 16, latent_side, latent_side, generator=torch.Generator().manual_seed(latent_side))
                generator = torch.Generator().manual_seed(42)
                with torch.no_grad():
                    vae.set_parallel_size(1)
                    assert not vae.is_distributed_enabled()
                    baseline = vae.decode(z, return_dict=False, generator=generator)[0]

                    vae.set_parallel_size(2)
                    assert vae.is_distributed_enabled()
                    assert vae._strategy_select(z)[0] == getattr(vae, f"{strategy}_split")
                    decoder_calls.clear()
                    parallel = vae.decode(z, return_dict=False, generator=generator)[0]
                    assert len(decoder_calls) == calls_per_rank

                    tasks, grid_spec = getattr(vae, f"{strategy}_split")(z)
                    single_process = getattr(vae, f"{strategy}_merge")(
                        {task.grid_coord: vae.tile_exec(task) for task in tasks}, grid_spec
                    )

                if rank != 0:
                    assert parallel.numel() == 0
                    continue
                assert parallel.shape == baseline.shape == (1, 3, latent_side * 8, latent_side * 8)
                # Splitting across ranks must not change the result of the split algorithm.
                torch.testing.assert_close(parallel, single_process, rtol=0, atol=0)
                if strategy == "tile":
                    # Same tiles and blending as diffusers tiled_decode at size 1.
                    torch.testing.assert_close(parallel, baseline, rtol=0, atol=0)
                # Halo-patch error versus full decode depends on trained weights and
                # image size, so it is not bounded with this random-weight VAE.
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires the CPU Gloo backend")
def test_glm_image_vae_decode_parity_pp1_vs_pp2(tmp_path):
    mp.spawn(_decode_parity_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)


def test_glm_image_pipeline_uses_distributed_vae_class():
    from vllm_omni.diffusion.models.glm_image import pipeline_glm_image as glm_pipeline_module

    assert glm_pipeline_module.DistributedAutoencoderKL is DistributedAutoencoderKL
    assert issubclass(DistributedAutoencoderKL, DistributedVaeMixin)


def test_glm_image_pipeline_loads_distributed_vae(mocker):
    from vllm_omni.diffusion.models.glm_image.pipeline_glm_image import GlmImagePipeline

    mock_vae = mocker.MagicMock()
    mock_vae.config.block_out_channels = [128, 256, 512, 512]
    mock_vae.eval.return_value = mock_vae
    mock_vae.to.return_value = mock_vae

    mock_scheduler = mocker.MagicMock()
    mock_text_encoder = mocker.MagicMock()
    mock_text_encoder.eval.return_value = mock_text_encoder
    mock_text_encoder.to.return_value = mock_text_encoder
    mock_tokenizer = mocker.MagicMock()
    mock_transformer = mocker.MagicMock()
    mock_transformer.patch_size = 2

    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.DistributedAutoencoderKL.from_pretrained",
        return_value=mock_vae,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.FlowMatchEulerDiscreteScheduler.from_pretrained",
        return_value=mock_scheduler,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.T5EncoderModel.from_pretrained",
        return_value=mock_text_encoder,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.ByT5Tokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.GlmImageTransformer2DModel",
        return_value=mock_transformer,
    )
    mocker.patch("vllm_omni.diffusion.models.glm_image.pipeline_glm_image.os.path.exists", return_value=True)
    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.download_weights_from_hf_specific",
        return_value="/tmp/glm-image",
    )

    od_config = mocker.MagicMock()
    od_config.model = "/tmp/glm-image"
    od_config.revision = None
    od_config.parallel_config = mocker.MagicMock()
    od_config.quantization_config = None
    od_config.enable_diffusion_pipeline_profiler = False

    pipeline = GlmImagePipeline(od_config=od_config)

    assert pipeline.vae is mock_vae
    DistributedAutoencoderKL.from_pretrained.assert_called_once_with(
        "/tmp/glm-image",
        subfolder="vae",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
