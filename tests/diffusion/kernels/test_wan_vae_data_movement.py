# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import AutoencoderKLWan

from vllm_omni.diffusion.distributed.autoencoders import wan_vae_data_movement
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_data_movement import (
    _cache_payload,
    _causal_padding,
    _is_dynamic_spatial_conv,
    _run_cached_causal_conv,
    install_wan_vae_data_movement,
)
from vllm_omni.diffusion.kernels.wan_vae_data_movement import cat_pad_5d, dup_up3d_add

pytestmark = pytest.mark.core_model
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _dense_5d(shape: tuple[int, ...], dtype: torch.dtype, channels_last: bool) -> torch.Tensor:
    tensor = torch.randn(shape, device="cuda", dtype=dtype)
    if channels_last:
        return tensor.contiguous(memory_format=torch.channels_last_3d)
    return tensor.contiguous()


def _reference_cat_pad(
    x: torch.Tensor,
    cache: torch.Tensor | None,
    padding: tuple[int, ...],
) -> torch.Tensor:
    reference_padding = list(padding)
    if cache is not None:
        x = torch.cat([cache, x], dim=2)
        reference_padding[4] -= cache.shape[2]
    return F.pad(x, reference_padding)


@pytest.mark.cuda
@cuda_only
@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize(
    "channels,frames,height,width,cache_frames,padding",
    [
        (96, 1, 10, 14, 0, (1, 1, 1, 1, 2, 0)),
        (96, 1, 10, 14, 1, (1, 1, 1, 1, 2, 0)),
        (96, 1, 10, 14, 2, (1, 1, 1, 1, 2, 0)),
        (64, 1, 10, 14, 2, (0, 0, 0, 0, 2, 0)),
        (48, 4, 10, 14, 2, (1, 1, 1, 1, 2, 0)),
    ],
)
def test_cat_pad_5d_is_bitwise_and_layout_exact(
    dtype: torch.dtype,
    channels_last: bool,
    channels: int,
    frames: int,
    height: int,
    width: int,
    cache_frames: int,
    padding: tuple[int, ...],
) -> None:
    torch.cuda.manual_seed(0)
    x = _dense_5d((1, channels, frames, height, width), dtype, channels_last)
    cache = None
    if cache_frames:
        pad_height, pad_width = padding[2], padding[0]
        buffer = _dense_5d(
            (1, channels, cache_frames, height + 2 * pad_height, width + 2 * pad_width),
            dtype,
            channels_last,
        )
        cache = buffer[:, :, :, pad_height : pad_height + height, pad_width : pad_width + width]

    reference = _reference_cat_pad(x, cache, padding)
    output = cat_pad_5d(x, cache, padding)
    assert output is not None
    assert output.shape == reference.shape
    assert output.stride() == reference.stride()
    assert torch.equal(output, reference)

    pair = cat_pad_5d(x, cache, padding, keep_cache_frames=2)
    assert pair is not None
    output_with_cache, next_cache = pair
    assert torch.equal(output_with_cache, reference)
    pad_height, pad_width = padding[2], padding[0]
    keep_frames = min(2, reference.shape[2])
    expected_cache = reference[
        :,
        :,
        -keep_frames:,
        pad_height : pad_height + height,
        pad_width : pad_width + width,
    ]
    assert next_cache.shape == expected_cache.shape
    assert torch.equal(next_cache, expected_cache)
    assert next_cache.is_contiguous(memory_format=torch.channels_last_3d) is channels_last


@pytest.mark.cuda
@cuda_only
@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("main_layout", ["contiguous", "channels_last", "permuted"])
@pytest.mark.parametrize(
    "in_channels,out_channels,frames,height,width,factor_temporal,factor_spatial,drop_first",
    [
        (128, 64, 1, 10, 14, 2, 2, False),
        (128, 64, 1, 10, 14, 2, 2, True),
        (64, 32, 2, 10, 14, 1, 2, False),
    ],
)
def test_dup_up3d_add_is_bitwise_and_layout_exact(
    dtype: torch.dtype,
    main_layout: str,
    in_channels: int,
    out_channels: int,
    frames: int,
    height: int,
    width: int,
    factor_temporal: int,
    factor_spatial: int,
    drop_first: bool,
) -> None:
    torch.cuda.manual_seed(0)
    repeats = out_channels * factor_temporal * factor_spatial * factor_spatial // in_channels
    source = _dense_5d((1, in_channels, frames, height, width), dtype, main_layout == "channels_last")
    out_frames = frames * factor_temporal - (factor_temporal - 1 if drop_first else 0)
    shape = (1, out_channels, out_frames, height * factor_spatial, width * factor_spatial)
    if main_layout == "channels_last":
        main = _dense_5d(shape, dtype, True)
    elif main_layout == "permuted":
        main = torch.randn(
            (shape[0], shape[2], shape[1], shape[3], shape[4]),
            device="cuda",
            dtype=dtype,
        ).permute(0, 2, 1, 3, 4)
    else:
        main = _dense_5d(shape, dtype, False)

    shortcut = source.repeat_interleave(repeats, dim=1)
    shortcut = shortcut.view(
        1,
        out_channels,
        factor_temporal,
        factor_spatial,
        factor_spatial,
        frames,
        height,
        width,
    )
    shortcut = shortcut.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    shortcut = shortcut.view(shape[0], shape[1], frames * factor_temporal, shape[3], shape[4])
    if drop_first:
        shortcut = shortcut[:, :, factor_temporal - 1 :]
    reference = main + shortcut

    output = dup_up3d_add(
        main,
        source,
        factor_temporal,
        factor_spatial,
        repeats,
        drop_first,
    )
    assert output is not None
    assert output.shape == reference.shape
    assert output.stride() == reference.stride()
    assert torch.equal(output, reference)


@pytest.mark.cuda
@cuda_only
@torch.no_grad()
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("temporal_only", [False, True])
def test_cached_conv_chunk_loop_matches_reference(monkeypatch, channels_last: bool, temporal_only: bool) -> None:
    from diffusers.models.autoencoders.autoencoder_kl_wan import CACHE_T, WanCausalConv3d

    torch.cuda.manual_seed(0)
    channels = 64
    if temporal_only:
        conv = WanCausalConv3d(channels, 2 * channels, (3, 1, 1), padding=(1, 0, 0))
    else:
        conv = WanCausalConv3d(channels, channels, 3, padding=1)
    conv = conv.to(device="cuda", dtype=torch.float32)
    chunks = [_dense_5d((1, channels, 1, 10, 14), torch.float32, channels_last) for _ in range(4)]

    def run(force_reference: bool, start):
        cache = [start]
        outputs = []
        original = wan_vae_data_movement.cat_pad_5d
        if force_reference:
            monkeypatch.setattr(wan_vae_data_movement, "cat_pad_5d", None)
        try:
            for chunk in chunks:
                outputs.append(_run_cached_causal_conv(conv, chunk, cache, 0))
        finally:
            monkeypatch.setattr(wan_vae_data_movement, "cat_pad_5d", original)
        return outputs, cache[0]

    for start in (None, "Rep"):
        fused_outputs, fused_cache = run(False, start)
        reference_outputs, reference_cache = run(True, start)
        for output, reference in zip(fused_outputs, reference_outputs, strict=True):
            assert torch.equal(output, reference)
        payload = _cache_payload(fused_cache)
        assert payload is not None and payload.shape[2] == CACHE_T
        assert torch.equal(payload, reference_cache[:, :, -CACHE_T:])


@pytest.mark.cpu
def test_installer_is_idempotent_and_preserves_state_dict_keys() -> None:
    vae = OmniAutoencoderKLWan(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=4,
        dim_mult=[1, 1],
        num_res_blocks=1,
        temperal_downsample=[False, False],
        is_residual=True,
    )
    keys_before = set(vae.state_dict())
    assert install_wan_vae_data_movement(vae)
    assert install_wan_vae_data_movement(vae)
    assert set(vae.state_dict()) == keys_before


@pytest.mark.cpu
def test_dynamic_spatial_conv_exposes_direct_path_compatibility_contract() -> None:
    conv = torch.nn.Conv3d(2, 2, 3)
    conv._vllm_omni_dynamic_spatial_shard_conv = True
    conv._source_padding = (1, 1, 1, 1, 2, 0)

    assert _is_dynamic_spatial_conv(conv)
    assert _causal_padding(conv) == conv._source_padding


@pytest.mark.cuda
@cuda_only
@torch.no_grad()
@pytest.mark.parametrize("channels_last", [False, True])
def test_tiny_wan_decoder_is_bitwise_exact(channels_last: bool) -> None:
    config = dict(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=4,
        dim_mult=[1, 1],
        num_res_blocks=1,
        temperal_downsample=[False, True],
        is_residual=True,
    )
    torch.manual_seed(0)
    reference = AutoencoderKLWan(**config).eval().to("cuda")
    candidate = OmniAutoencoderKLWan(**config).eval().to("cuda")
    candidate.load_state_dict(reference.state_dict())
    if channels_last:
        for model in (reference, candidate):
            for module in model.decoder.modules():
                if isinstance(module, torch.nn.Conv3d):
                    module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last_3d)

    latents = torch.randn(1, 4, 3, 4, 4, device="cuda")
    if channels_last:
        latents = latents.contiguous(memory_format=torch.channels_last_3d)
    expected = reference.decode(latents, return_dict=False)[0]
    output = candidate.decode(latents, return_dict=False)[0]
    assert output.stride() == expected.stride()
    assert torch.equal(output, expected)
