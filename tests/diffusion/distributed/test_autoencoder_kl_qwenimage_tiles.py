import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
    DistributedAutoencoderKLQwenImage,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_qwen_tiles_reassemble_seven_rgba_images_with_partial_edges():
    vae = DistributedAutoencoderKLQwenImage.__new__(DistributedAutoencoderKLQwenImage)
    torch.nn.Module.__init__(vae)
    vae.register_parameter("test_weight", torch.nn.Parameter(torch.empty(0)))
    vae.encoder = torch.nn.Identity()
    vae.decoder = torch.nn.Identity()
    vae.spatial_compression_ratio = 2
    vae.tile_sample_min_height = 4
    vae.tile_sample_min_width = 4
    vae.tile_sample_stride_height = 2
    vae.tile_sample_stride_width = 2
    source = torch.arange(7 * 4 * 1 * 3 * 5, dtype=torch.float32).reshape(7, 4, 1, 3, 5)

    tasks, grid = vae.tile_split(source)
    assert grid.grid_shape == (3, 5)
    assert tasks[-1].tensor[0].shape == (7, 4, 1, 1, 1)
    decoded = {
        task.grid_coord: task.tensor[0].repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        for task in tasks
    }

    result = vae.tile_merge(decoded, grid)
    assert result.shape == (7, 4, 1, 6, 10)
    torch.testing.assert_close(result, source.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4))
