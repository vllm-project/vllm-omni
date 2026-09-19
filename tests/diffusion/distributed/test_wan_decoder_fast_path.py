# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The exact Wan decoder fast path must be bit-identical to the plain streaming decode.

Two decoders are driven frame by frame through ``WanStreamingDecoder`` over two chunks of one session and
a fresh session (the persistent buffers' causal front frames go zeros -> cache -> zeros), on CPU: plain
diffusers modules in one process, and the spatially sharded wrappers across two gloo ranks with real halo
exchange, each in fp32 and under bf16 autocast (where the parameter cast applies). Every output must
``torch.equal`` the untouched decoder's on every rank.
"""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanDecoder3d

from vllm_omni.diffusion.distributed.autoencoders.wan_decoder_fast_path import install_wan_decoder_fast_path
from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import (
    WanDistCausalConv3d,
    WanDistConv2d,
    install_wan_spatial_shard_decode,
)
from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder


def _make_vae(seed: int = 0) -> SimpleNamespace:
    torch.manual_seed(seed)
    # Three levels: the default temporal-upsample list gives one upsample2d and one upsample3d resample.
    decoder = WanDecoder3d(dim=8, z_dim=4, dim_mult=[1, 2, 4], num_res_blocks=1, attn_scales=[])
    post_quant_conv = WanCausalConv3d(4, 4, 1)
    for p in list(decoder.parameters()) + list(post_quant_conv.parameters()):
        p.data.uniform_(-0.5, 0.5)
    decoder.eval()
    count = sum(1 for m in decoder.modules() if m.__class__.__name__ == "WanCausalConv3d")
    return SimpleNamespace(decoder=decoder, post_quant_conv=post_quant_conv, _cached_conv_counts={"decoder": count})


def _run(vae, latents: list[torch.Tensor], autocast: bool) -> list[torch.Tensor]:
    decoder = WanStreamingDecoder(vae)
    outs = []
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else torch.no_grad()
    with torch.no_grad(), ctx:
        state = decoder.new_decode_state("a")
        outs.append(decoder.decode_chunk(latents[0], state))
        outs.append(decoder.decode_chunk(latents[1], state))
        fresh = decoder.new_decode_state("b")
        outs.append(decoder.decode_chunk(latents[2], fresh))
    return [o.float() for o in outs]


def _latents() -> list[torch.Tensor]:
    torch.manual_seed(1)
    return [torch.randn(1, 4, 2, 6, 10), torch.randn(1, 4, 1, 6, 10), torch.randn(1, 4, 2, 6, 10)]


def _check_pair(reference, candidate, autocast: bool, sharded: bool, level: str = "exact") -> None:
    dtype = torch.bfloat16 if autocast else None
    counts = install_wan_decoder_fast_path(candidate, conv_dtype=dtype, level=level)
    assert counts["level"] == level
    assert counts["upsamples"] == 2
    convs = sum(1 for m in candidate.decoder.modules() if isinstance(m, torch.nn.Conv3d | torch.nn.Conv2d)) + 1
    assert counts["conv_params_cast"] == (convs if autocast else 0)
    dist_convs = sum(1 for m in candidate.decoder.modules() if isinstance(m, WanDistCausalConv3d | WanDistConv2d))
    assert counts["persistent_input_buffers"] == dist_convs
    assert (dist_convs > 0) == sharded
    if level == "fused":
        assert (
            counts["fused_norm_silu"]
            == 2 * sum(1 for m in candidate.decoder.modules() if m.__class__.__name__ == "WanResidualBlock") + 1
        )
        assert counts["channels_last_convs"] == convs
        assert counts["resample_forwards"] == 2
        with pytest.raises(ValueError, match="already installed"):
            install_wan_decoder_fast_path(candidate, conv_dtype=dtype, level="exact")
    # Idempotent: a second install is a no-op that reports the same counts.
    assert install_wan_decoder_fast_path(candidate, conv_dtype=dtype, level=level) is counts
    latents = _latents()
    expected = _run(reference, latents, autocast)
    actual = _run(candidate, latents, autocast)
    assert len(expected) == len(actual) == 3
    for e, a in zip(expected, actual):
        assert e.shape == a.shape
        if level == "exact":
            assert torch.equal(e, a)
        else:
            # The fused level changes layouts only on CPU (the kernels need CUDA); the eager norm's reduction
            # order may differ with the layout, so the outputs are close, not identical.
            assert torch.allclose(e, a, atol=1e-2 if autocast else 1e-5, rtol=1e-2 if autocast else 1e-4), (
                (e - a).abs().max()
            )
    if sharded:
        bufs = [m._input_buf for m in candidate.decoder.modules() if isinstance(m, WanDistCausalConv3d | WanDistConv2d)]
        assert all(b is not None for b in bufs)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("autocast", [False, True])
def test_fast_path_is_bit_identical(autocast: bool) -> None:
    reference = _make_vae()
    candidate = copy.deepcopy(reference)
    _check_pair(reference, candidate, autocast, sharded=False)


def _swap_norms_to_rmsnorm_vae(vae) -> int:
    """Ensure the decoder uses RMSNormVAE and return the number of matching norms.

    Collecting pipeline tests can already apply patch_wan_rms_norm process-wide before this test runs.
    """
    from vllm_omni.diffusion.layers.norm import RMSNormVAE

    count = 0
    for module in list(vae.decoder.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, RMSNormVAE):
                count += 1
                continue
            if child.__class__.__name__ == "WanRMS_norm":
                images = child.gamma.dim() == 3  # (dim, 1, 1) for the attention block's 4D norm
                norm = RMSNormVAE(child.gamma.shape[0], channel_first=child.channel_first, images=images, bias=False)
                norm.gamma.data.copy_(child.gamma.data.reshape(norm.gamma.shape))
                setattr(module, name, norm)
                count += 1
    return count


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("norm", ["default", "RMSNormVAE"])
@pytest.mark.parametrize("autocast", [False, True])
def test_fused_level_plumbing_on_cpu(autocast: bool, norm: str) -> None:
    reference = _make_vae()
    if norm == "RMSNormVAE":
        assert _swap_norms_to_rmsnorm_vae(reference) > 0
    candidate = copy.deepcopy(reference)
    _check_pair(reference, candidate, autocast, sharded=False, level="fused")


def _sharded_worker(rank: int, world_size: int, port: str, autocast: bool, return_dict, level: str = "exact") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)
        reference = _make_vae()
        candidate = copy.deepcopy(reference)
        for vae in (reference, candidate):
            install_wan_spatial_shard_decode(vae, dist.group.WORLD, split_dim="width", dst=None)
        _check_pair(reference, candidate, autocast, sharded=True, level=level)
        return_dict[rank] = "ok"
    except BaseException as exc:  # report, the parent asserts
        return_dict[rank] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        dist.destroy_process_group()


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("level", ["exact", "fused"])
@pytest.mark.parametrize("autocast", [False, True])
def test_sharded_fast_path_on_every_rank(autocast: bool, level: str) -> None:
    manager = mp.get_context("spawn").Manager()
    return_dict = manager.dict()
    port = str(29610 + int(autocast) + 2 * (level == "fused"))
    mp.spawn(_sharded_worker, args=(2, port, autocast, return_dict, level), nprocs=2, join=True)
    for rank in range(2):
        assert return_dict.get(rank) == "ok", f"rank {rank}: {return_dict.get(rank)}"


@pytest.mark.core_model
@pytest.mark.cpu
def test_conv_dtype_requires_half() -> None:
    with pytest.raises(ValueError, match="conv_dtype"):
        install_wan_decoder_fast_path(_make_vae(), conv_dtype=torch.float32)
