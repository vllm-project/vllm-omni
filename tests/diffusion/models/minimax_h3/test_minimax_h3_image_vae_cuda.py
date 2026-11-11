# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-graph tile decode of the MiniMax-H3-Image VAE keeps the eager output."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.minimax_h3.image_vae import MiniMaxH3ImageVAE, _BatchedTileGraphs

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


def test_graph_tile_decode_is_bitwise_eager():
    config = {
        "_class_name": "MiniMaxH3ImageVAE",
        "latent_channels": 24,
        "latents_mean": [0.1] * 24,
        "latents_std": [2.0] * 24,
        "vae_tile_size": 64,
        "vae_tile_overlap_min": 16,
        "vit_decoder_2d": True,
        "vit_decoder_kwargs": {"dim_head": 64, "heads": 4, "num_layers": 2, "rope_dim_ratio": 0.5, "rope_theta": 100.0},
    }
    torch.manual_seed(0)
    vae = MiniMaxH3ImageVAE(config).cuda().eval()
    for parameter in vae.decoder.parameters():
        parameter.data.normal_(0, 0.02)
    latent = torch.randn(1, 24, 1, 10, 14, device="cuda")
    eager = vae.decode_latent(latent)
    graphed = vae.decode_latent(latent, use_cuda_graph=True)
    again = vae.decode_latent(latent, use_cuda_graph=True)
    assert torch.equal(eager, graphed)
    assert torch.equal(graphed, again)
    # One tile shape, replayed over parallel stream replicas (one on ROCm).
    assert len(vae._decode_graphs.entries) == 1
    replicas = len(next(iter(vae._decode_graphs.entries.values())))
    assert replicas == 1 if torch.version.hip else replicas > 1


def test_concurrent_graph_replicas_match_eager_at_release_width():
    """Parallel replica replay must not share a capture stream's cuBLAS workspace.

    The toy decoder above is too small to exercise the workspace; at the release
    decoder width, replicas captured on one shared side stream race when map()
    replays them concurrently and return corrupted tiles.
    """
    if torch.version.hip:
        pytest.skip("ROCm replays one graph replica")
    config = {
        "_class_name": "MiniMaxH3ImageVAE",
        "latent_channels": 24,
        "latents_mean": [0.1] * 24,
        "latents_std": [2.0] * 24,
        "vae_tile_size": 256,
        "vae_tile_overlap_min": 64,
        "vit_decoder_2d": True,
        "vit_decoder_kwargs": {
            "dim_head": 64,
            "heads": 16,
            "num_layers": 4,
            "ffn_activation_fn": "silu",
            "ffn_use_gated": True,
            "norm_affine": True,
            "norm_type": "rms_norm",
            "qk_norm_affine": False,
            "qk_norm_type": "rms_norm",
            "rope_dim_ratio": 0.5,
            "rope_theta": 100.0,
        },
    }
    torch.manual_seed(0)
    vae = MiniMaxH3ImageVAE(config).cuda().eval()
    for parameter in vae.decoder.parameters():
        parameter.data.normal_(0, 0.02)
    # 1024x1024 pixels: more 256px tiles than graph replicas, so replays overlap.
    latent = torch.randn(1, 24, 1, 64, 64, device="cuda")
    with torch.inference_mode():
        eager = vae.decode_latent(latent)
        replays = [vae.decode_latent(latent, use_cuda_graph=True) for _ in range(4)]
    assert len(next(iter(vae._decode_graphs.entries.values()))) > 1
    for replay in replays:
        assert torch.equal(replay, eager)


@pytest.mark.parametrize("batch_size", [2, 3, 8, 32])
def test_batched_graph_tiles_keep_order_tail_and_output_storage(batch_size):
    with torch.inference_mode():
        graphs = _BatchedTileGraphs(
            lambda value: value.square() + 2, autocast_dtype=torch.bfloat16, tile_batch_size=batch_size
        )
        tiles = [torch.full((2, 3, 1, 4, 4), index, device="cuda") for index in range(7)]
        outputs = graphs.map(tiles)
        second = graphs.map([tile + 3 for tile in tiles])
        assert len(outputs) == len(second) == len(tiles)
        for tile, first, repeated in zip(tiles, outputs, second):
            assert torch.equal(first, tile.square() + 2)
            assert torch.equal(repeated, (tile + 3).square() + 2)
        assert graphs.map([]) == []


def test_batched_graph_tiles_keep_mixed_shapes():
    with torch.inference_mode():
        graphs = _BatchedTileGraphs(lambda value: value + 2, autocast_dtype=torch.bfloat16, tile_batch_size=3)
        tiles = [torch.ones(1, 3, 1, 4, width, device="cuda") for width in (4, 8, 4)]
        for tile, output in zip(tiles, graphs.map(tiles)):
            assert torch.equal(output, tile + 2)


def test_tile_batch_size_changes_rebuild_graphs():
    config = {
        "_class_name": "MiniMaxH3ImageVAE",
        "latent_channels": 24,
        "latents_mean": [0.1] * 24,
        "latents_std": [2.0] * 24,
        "vae_tile_size": 64,
        "vae_tile_overlap_min": 16,
        "vit_decoder_2d": True,
        "vit_decoder_kwargs": {"dim_head": 64, "heads": 4, "num_layers": 2, "rope_dim_ratio": 0.5, "rope_theta": 100.0},
    }
    torch.manual_seed(0)
    vae = MiniMaxH3ImageVAE(config).cuda().eval()
    for parameter in vae.decoder.parameters():
        parameter.data.normal_(0, 0.02)
    latent = torch.randn(1, 24, 1, 10, 14, device="cuda")
    previous = None
    for batch_size in (1, 2, 3, 1):
        first = vae.decode_latent(latent, use_cuda_graph=True, tile_batch_size=batch_size)
        graphs = vae._decode_graphs
        assert graphs is not previous
        assert graphs.tile_batch_size == batch_size
        second = vae.decode_latent(latent, use_cuda_graph=True, tile_batch_size=batch_size)
        assert vae._decode_graphs is graphs
        assert torch.equal(first, second)
        previous = graphs
