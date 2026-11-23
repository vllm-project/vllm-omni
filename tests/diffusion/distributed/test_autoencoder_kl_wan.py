# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref
from contextlib import contextmanager, nullcontext

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_wan as wan_vae_module
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
    OmniAutoencoderKLWan,
    _requires_gfx11_math_sdpa,
)
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import TileTask

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyOmniAutoencoderKLWan(OmniAutoencoderKLWan):
    def __init__(self, *, dtype: torch.dtype, head_dim: int | None = None):
        torch.nn.Module.__init__(self)
        self.register_parameter("dummy_weight", torch.nn.Parameter(torch.ones(1, dtype=dtype)))
        if head_dim is not None:
            self.attention = WanAttentionBlock(head_dim)


class _QKVProjection(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels


class WanAttentionBlock(torch.nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.to_qkv = _QKVProjection(head_dim * 3)


def test_wan_vae_execution_context_handles_fp32():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.float32)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.float32


def test_wan_vae_execution_context_handles_bf16():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.bfloat16


def test_wan_vae_execution_context_uses_platform_autocast(mocker):
    platform = mocker.Mock()
    platform.create_autocast_context.return_value = nullcontext()
    mocker.patch.object(wan_vae_module, "current_omni_platform", platform)

    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)

    with model._execution_context():
        pass
    platform.create_autocast_context.assert_called_once_with(
        device_type=model.dummy_weight.device.type,
        dtype=torch.bfloat16,
        enabled=True,
    )


@pytest.mark.parametrize(
    ("is_rocm", "capability", "dtype", "head_dim", "expected"),
    [
        (True, (11, 0), torch.bfloat16, 384, True),
        (True, (11, 5), torch.float16, 384, True),
        (True, (9, 4), torch.bfloat16, 384, False),
        (False, (11, 0), torch.bfloat16, 384, False),
        (True, (11, 0), torch.float32, 384, False),
        (True, (11, 0), torch.bfloat16, 256, False),
    ],
)
def test_gfx11_math_sdpa_guard_is_narrow(
    mocker,
    is_rocm: bool,
    capability: tuple[int, int],
    dtype: torch.dtype,
    head_dim: int,
    expected: bool,
) -> None:
    platform = mocker.Mock()
    platform.is_rocm.return_value = is_rocm
    platform.get_device_capability.return_value = capability
    mocker.patch.object(wan_vae_module, "current_omni_platform", platform)
    model = _DummyOmniAutoencoderKLWan(dtype=dtype, head_dim=head_dim)

    assert (
        _requires_gfx11_math_sdpa(
            model,
            device=torch.device("cuda", 0),
            dtype=dtype,
        )
        is expected
    )


def test_wan_vae_execution_context_enters_math_sdpa(mocker) -> None:
    events = []

    @contextmanager
    def record_context(name):
        events.append(("enter", name))
        yield
        events.append(("exit", name))

    platform = mocker.Mock()
    platform.create_autocast_context.return_value = record_context("autocast")
    mocker.patch.object(wan_vae_module, "current_omni_platform", platform)
    mocker.patch.object(wan_vae_module, "_requires_gfx11_math_sdpa", return_value=True)
    sdpa = mocker.patch.object(
        wan_vae_module,
        "sdpa_kernel",
        side_effect=lambda backend: record_context(backend),
    )
    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16, head_dim=384)

    with model._execution_context():
        events.append(("body", None))

    sdpa.assert_called_once_with(wan_vae_module.SDPBackend.MATH)
    assert events == [
        ("enter", "autocast"),
        ("enter", wan_vae_module.SDPBackend.MATH),
        ("body", None),
        ("exit", wan_vae_module.SDPBackend.MATH),
        ("exit", "autocast"),
    ]


@pytest.mark.parametrize("fail_second_chunk", [False, True])
def test_tile_decode_releases_chunks_and_history_on_completion_or_failure(fail_second_chunk):
    model = DistributedAutoencoderKLWan.__new__(DistributedAutoencoderKLWan)
    torch.nn.Module.__init__(model)
    model._cached_conv_counts = {"decoder": 1, "encoder": 0}
    model.post_quant_conv = torch.nn.Identity()
    cache_refs = []
    chunk_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    class Decoder(torch.nn.Module):
        temperal_upsample = []

        def forward(self, tile, *, feat_cache, feat_idx, first_chunk):
            assert all(ref() is None for ref in chunk_refs)
            assert feat_idx == [0]
            if first_chunk:
                assert feat_cache == [None]
            else:
                # A premature per-chunk clear would lose causal history.
                torch.testing.assert_close(feat_cache[0], torch.ones_like(tile))
            feat_cache[0] = tile.clone()
            cache_refs.append(weakref.ref(feat_cache[0]))
            if fail_second_chunk and not first_chunk:
                raise RuntimeError("injected tile decode failure")
            decoded = tile * 2
            chunk_refs.append(weakref.ref(decoded))
            return decoded

    model.decoder = Decoder()
    task = TileTask(0, (0, 0), [torch.ones(1, 3, 1, 2, 2), torch.full((1, 3, 1, 2, 2), 2.0)])
    if fail_second_chunk:
        with pytest.raises(RuntimeError, match="injected tile decode failure"):
            model.tile_exec(task)
    else:
        output = model.tile_exec(task)
        torch.testing.assert_close(output[:, :, :1], task.tensor[0] * 2)
        torch.testing.assert_close(output[:, :, 1:], task.tensor[1] * 2)
        assert all(ref() is None for ref in cache_refs + chunk_refs)
    assert model._feat_map == [None]


@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "strided", "mixed"])
@pytest.mark.parametrize("channels", [3, 12])
def test_tile_preallocation_matches_raw_chunk_concatenation(layout, channels):
    model = DistributedAutoencoderKLWan.__new__(DistributedAutoencoderKLWan)
    torch.nn.Module.__init__(model)
    model._cached_conv_counts = {"decoder": 1, "encoder": 0}
    model.post_quant_conv = torch.nn.Identity()

    def make_chunk(index):
        frames = 1 if index == 0 else 4
        x = torch.arange(channels * frames * 2 * 3, dtype=torch.float32).reshape(1, channels, frames, 2, 3)
        if layout == "channels_last" or (layout == "mixed" and index == 0):
            return x.contiguous(memory_format=torch.channels_last_3d)
        if layout == "strided":
            return x.transpose(-1, -2)
        return x

    class Decoder(torch.nn.Module):
        temperal_upsample = [False, True, True]

        def forward(self, tile, *, feat_cache, feat_idx, first_chunk):
            return make_chunk(int(tile.item()))

    model.decoder = Decoder()
    task = TileTask(0, (0, 0), [torch.tensor(index) for index in range(3)])
    expected = torch.cat([make_chunk(index) for index in range(3)], dim=2)
    output = model.tile_exec(task)
    torch.testing.assert_close(output, expected, rtol=0, atol=0, check_stride=True)
    # Twelve channels represent patchified RGB; neither clamping nor unpatchify
    # belongs here because tile blending must happen first.
    assert output.shape[1] == channels
    assert output.max() > 1
    assert model._feat_map == [None]


@pytest.mark.parametrize("invalid", ["empty_task", "empty_chunk", "short_chunk", "long_chunk", "spatial_shape"])
def test_tile_preallocation_rejects_invalid_decoder_schedule_and_clears_cache(invalid):
    model = DistributedAutoencoderKLWan.__new__(DistributedAutoencoderKLWan)
    torch.nn.Module.__init__(model)
    model._cached_conv_counts = {"decoder": 1, "encoder": 0}
    model.post_quant_conv = torch.nn.Identity()

    class Decoder(torch.nn.Module):
        temperal_upsample = [True, True]

        def forward(self, tile, *, feat_cache, feat_idx, first_chunk):
            feat_cache[0] = tile.clone()
            frames = 1 if first_chunk else {"short_chunk": 3, "long_chunk": 5}.get(invalid, 4)
            if invalid == "empty_chunk":
                frames = 0
            width = 3 if invalid == "spatial_shape" and not first_chunk else 2
            return torch.zeros(1, 3, frames, 2, width)

    model.decoder = Decoder()
    task = TileTask(0, (0, 0), [] if invalid == "empty_task" else [torch.ones(1), torch.ones(1)])
    with pytest.raises(RuntimeError, match="Wan tile decoder"):
        model.tile_exec(task)
    assert model._feat_map == [None]


@torch.no_grad()
def test_tile_preallocation_matches_wan_decoder_chunks():
    model = DistributedAutoencoderKLWan(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=4,
        dim_mult=[1, 1],
        num_res_blocks=1,
        temperal_downsample=[False, True],
        is_residual=True,
    ).eval()
    task = TileTask(0, (0, 0), [torch.randn(1, 4, 1, 2, 2) for _ in range(3)])
    model.clear_cache()
    chunks = []
    for index, latent in enumerate(task.tensor):
        model._conv_idx = [0]
        chunks.append(
            model.decoder(
                model.post_quant_conv(latent),
                feat_cache=model._feat_map,
                feat_idx=model._conv_idx,
                first_chunk=index == 0,
            )
        )
    expected = torch.cat(chunks, dim=2)
    torch.testing.assert_close(model.tile_exec(task), expected, rtol=0, atol=0, check_stride=True)
    assert all(cache is None for cache in model._feat_map)
