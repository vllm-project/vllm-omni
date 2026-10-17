# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run with torchrun --nproc-per-node=2 -m pytest this_file.py.

Compare against the same spatial partition, independently of the numerical
changes caused by partitioning a full-frame cuDNN convolution.
"""

import copy
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan
from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import (
    spatial_shard_decode,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    install_wan_vae_fastpath,
    is_installed,
    uninstall_wan_vae_fastpath,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.full_model,
    *hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=2),
]


@pytest.fixture(scope="module", autouse=True)
def process_group():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2 or not torch.cuda.is_available():
        pytest.skip("requires torchrun with at least two CUDA ranks")
    current_omni_platform.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    # Keep the FP32 check independent of cuDNN choosing TF32 for one layout.
    old_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32 = old_tf32
    dist.destroy_process_group()


# Representative combinations cover both axes, levels, and Wan decoder variants
# without a Cartesian product of otherwise identical decode checks.
@pytest.mark.parametrize(
    "axis,level,variant,dtype",
    [
        ("height", "lossless", "wan21", torch.float32),
        ("width", "lossless", "patch2", torch.bfloat16),
        ("width", "channels_last", "wan22", torch.bfloat16),
        ("height", "channels_last", "patch2", torch.float32),
    ],
)
@torch.inference_mode()
def test_spatial_fastpath_parity_and_lifecycle(axis, level, variant, dtype):
    torch.manual_seed(41)
    patch = variant == "patch2"
    config = dict(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=4,
        dim_mult=[1, 1],
        num_res_blocks=1,
        temperal_downsample=[False, True],
        is_residual=variant != "wan21",
        patch_size=2 if patch else None,
        in_channels=12 if patch else 3,
        out_channels=12 if patch else 3,
    )
    reference = OmniAutoencoderKLWan(**config).eval().to(device="cuda", dtype=dtype)
    candidate = copy.deepcopy(reference)
    if variant == "wan21":
        # Existing direct installation must be rebound around spatial wrappers.
        assert install_wan_vae_fastpath(candidate, level=level).installed
    else:
        candidate.distributed_executor = SimpleNamespace(parallel_mode=f"spatial_shard_{axis}", parallel_size=2)
        report = install_wan_vae_fastpath(candidate, level=level)
        assert not report.installed and "deferred" in report.reason
    # Uneven shard extents, first/cached frames, shape changes and fresh sessions.
    for frames, height, width in [(6, 7, 9), (1, 8, 10), (2, 7, 9)]:
        z = torch.randn(1, 4, frames, height, width, device="cuda", dtype=dtype)
        dist.broadcast(z, 0)
        expected = spatial_shard_decode(
            reference, z, group=dist.group.WORLD, split_dim=axis, temporal_chunk_size=4
        ).sample
        actual = spatial_shard_decode(
            candidate, z, group=dist.group.WORLD, split_dim=axis, temporal_chunk_size=4
        ).sample
        assert is_installed(candidate)
        one_frame = spatial_shard_decode(reference, z, group=dist.group.WORLD, split_dim=axis).sample
        if dist.get_rank() == 0:
            # Batching must preserve causal history across the four-frame chunk
            # and short tail, allowing convolution-algorithm rounding changes.
            torch.testing.assert_close(expected, one_frame, atol=0.04 if dtype == torch.bfloat16 else 1e-4, rtol=0)
            if level == "lossless":
                assert torch.equal(actual, expected), (actual.float() - expected.float()).abs().max().item()
            else:
                error = (actual.float() - expected.float()).abs()
                assert error.max() <= (0.04 if dtype == torch.bfloat16 else 1e-4)
                assert error.square().mean().sqrt() <= (0.005 if dtype == torch.bfloat16 else 1e-5)
        else:
            assert actual.numel() == expected.numel() == 0
        chunks: list[torch.Tensor] = []
        spatial_shard_decode(
            candidate, z, group=dist.group.WORLD, split_dim=axis, temporal_chunk_size=4, on_chunk=chunks.append
        )
        if dist.get_rank() == 0:
            assert torch.equal(torch.cat(chunks, dim=2), actual)
        else:
            assert not chunks
        # A subsequent small request takes ordinary temporal dispatch, while
        # the decoder remains spatially patched and only rank 0 owns output.
        candidate.enable_tiling()
        assert height <= candidate.tile_sample_min_height // candidate.spatial_compression_ratio
        assert width <= candidate.tile_sample_min_width // candidate.spatial_compression_ratio
        small = candidate.decode(z).sample
        if dist.get_rank() == 0:
            torch.testing.assert_close(small, one_frame, atol=0.04 if dtype == torch.bfloat16 else 1e-4, rtol=0)
        else:
            assert small.shape == (0,)
    wrapper = candidate.decoder.forward
    uninstall_wan_vae_fastpath(candidate)
    assert candidate.decoder.forward is wrapper
    assert not is_installed(candidate)
    actual = spatial_shard_decode(candidate, z, group=dist.group.WORLD, split_dim=axis, temporal_chunk_size=4).sample
    assert torch.equal(actual, expected)
    assert install_wan_vae_fastpath(candidate, level=level).installed


@pytest.mark.parametrize("ndim,axis,dtype", [(4, "height", torch.float32), (5, "width", torch.bfloat16)])
@torch.inference_mode()
def test_deferred_spatial_bias_preserves_invalid_extent(ndim, axis, dtype):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard as spatial

    torch.manual_seed(44)
    cls = WanCausalConv3d if ndim == 5 else torch.nn.Conv2d
    source = cls(4, 4, kernel_size=3, padding=1).cuda().to(dtype=dtype)
    source.bias.fill_(0.25)
    wrapper = spatial.WanDistCausalConv3d if ndim == 5 else spatial.WanDistConv2d
    conv = wrapper(source, dist.group.WORLD, split_dim=axis)
    conv.input_memory_format = torch.channels_last_3d if ndim == 5 else torch.channels_last
    conv.to(memory_format=conv.input_memory_format)
    extent = (4, 5) if axis == "height" else (5, 4)
    shape = (1, 4, 2, *extent) if ndim == 5 else (2, 4, *extent)
    x = torch.randn(shape, device="cuda", dtype=dtype).to(memory_format=conv.input_memory_format)
    token = spatial._SPATIAL_SHARD_CONTEXT.set(
        spatial.SpatialShardContext(
            input_extent=7,
            local_input_extent=4,
            split_dim=axis,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )
    )
    try:
        reference = conv(x)
        actual, bias = conv(x, return_bias=True)
        if dist.get_rank() == 1:
            assert bias is None
            invalid = actual.narrow(-2 if axis == "height" else -1, 3, 1)
            assert torch.count_nonzero(invalid) == 0
        else:
            assert bias is not None
            actual = actual + bias.view(1, -1, *([1] * (ndim - 2)))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    finally:
        spatial._SPATIAL_SHARD_CONTEXT.reset(token)


@pytest.mark.parametrize("ndim,axis,dtype", [(4, "height", torch.float32), (5, "width", torch.bfloat16)])
@torch.inference_mode()
def test_paired_halo_copies(ndim, axis, dtype):
    from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath.triton_spatial_halo import launch_halo_pair

    shape = (2, 7, 3, 9, 11) if ndim == 5 else (6, 7, 9, 11)
    memory_format = torch.channels_last_3d if ndim == 5 else torch.channels_last
    source = torch.randn(shape, device="cuda", dtype=dtype).contiguous(memory_format=memory_format)
    target = torch.full_like(source, 42)
    expected = target.clone()
    dim = ndim - (2 if axis == "height" else 1)
    top = source.narrow(dim, 1, 2)
    bottom = source.narrow(dim, source.shape[dim] - 3, 2)
    send_top = torch.empty(top.shape, dtype=dtype, device="cuda", memory_format=memory_format)
    send_bottom = torch.empty_like(send_top)
    assert launch_halo_pair(top, bottom, send_top, send_bottom)
    assert torch.equal(send_top, top)
    assert torch.equal(send_bottom, bottom)
    assert launch_halo_pair(
        send_top, send_bottom, target.narrow(dim, 0, 2), target.narrow(dim, target.shape[dim] - 2, 2)
    )
    expected.narrow(dim, 0, 2).copy_(top)
    expected.narrow(dim, expected.shape[dim] - 2, 2).copy_(bottom)
    assert torch.equal(target, expected)


@pytest.mark.parametrize(
    "axis,dtype,shape",
    [("width", torch.bfloat16, (2, 2, 7, 9)), ("height", torch.float32, (1, 1, 1, 8))],
)
@torch.inference_mode()
def test_local_attention_queries_with_uneven_and_empty_shards(axis, dtype, shape):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock

    from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard as spatial

    torch.manual_seed(32)
    baseline = WanAttentionBlock(8).eval().to(device="cuda", dtype=dtype)
    candidate = copy.deepcopy(baseline)
    spatial._patch_attention_block(baseline, dist.group.WORLD, axis)
    spatial._patch_attention_block(candidate, dist.group.WORLD, axis)
    from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath.forwards import CFG_ATTR, FastPathConfig

    setattr(candidate, CFG_ATTR, FastPathConfig(channels_last=True))
    batch, frames, height, width = shape
    full = torch.randn(batch, 8, frames, height, width, device="cuda", dtype=dtype)
    dist.broadcast(full, 0)
    dim = spatial._spatial_dim(axis)
    full_extent = full.shape[dim]
    local, _ = spatial.split_for_parallel_decode(full, upsample_count=0, split_dim=axis, group=dist.group.WORLD)
    token = spatial._SPATIAL_SHARD_CONTEXT.set(
        spatial.SpatialShardContext(full_extent, local.shape[dim], axis, dist.get_rank(), dist.get_world_size())
    )
    try:
        expected = baseline(local)
        actual = candidate(local)
        torch.testing.assert_close(actual, expected, atol=0.02 if dtype == torch.bfloat16 else 2e-6, rtol=0)
        valid = spatial._local_valid_extent(local.shape[dim])
        if valid < local.shape[dim]:
            assert torch.count_nonzero(actual.narrow(dim, valid, local.shape[dim] - valid)) == 0
    finally:
        spatial._SPATIAL_SHARD_CONTEXT.reset(token)
