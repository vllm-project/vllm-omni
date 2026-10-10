# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import weakref

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.magi2.turbo_vae import (
    Magi2TurboVAEDecoder,
    extract_turbo_decoder_state_dict,
    turbo_unpatchify,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


class RepeatDecoder(nn.Module):
    def forward(self, value, *, is_first_chunk):
        del is_first_chunk
        return value.repeat_interleave(4, dim=2) * 1.25


def make_decoder(device, dtype):
    decoder = object.__new__(Magi2TurboVAEDecoder)
    nn.Module.__init__(decoder)
    decoder.z_dim = 2
    decoder.first_chunk_size = 3
    decoder.step_size = 2
    decoder.temporal_compression_ratio = 4
    decoder.decoder = RepeatDecoder()
    decoder.register_buffer("latent_mean", torch.zeros(2, device=device))
    decoder.register_buffer("latent_std", torch.ones(2, device=device))
    return decoder.to(dtype=dtype)


@pytest.mark.cpu
def test_turbo_unpatchify_matches_wan_channel_order():
    # Four patch channels for one output channel. MAGI's order interleaves
    # width before height when reconstructing each 2x2 patch.
    patched = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1, 1)
    output = turbo_unpatchify(patched, patch_size=2)
    expected = torch.tensor([[[[[0.0, 2.0], [1.0, 3.0]]]]])
    torch.testing.assert_close(output, expected)


@pytest.mark.cpu
def test_extract_turbo_decoder_prefers_ema_and_drops_training_heads():
    decoder_weight = torch.ones(1)
    checkpoint = {
        "state_dict": {"module.decoder.old.weight": torch.zeros(1)},
        "ema_state_dict": {
            "module.decoder.conv_in.conv.weight": decoder_weight,
            "module.aligned_feature_projection_heads.0.weight": torch.empty(1),
        },
    }
    assert extract_turbo_decoder_state_dict(checkpoint) == {"decoder.conv_in.conv.weight": decoder_weight}


@pytest.mark.cpu
def test_turbo_temporal_tiles_preserve_chunk_overlap_contract():
    class RepeatDecoder(nn.Module):
        def forward(self, value: torch.Tensor, *, is_first_chunk: bool) -> torch.Tensor:
            del is_first_chunk
            return value.repeat_interleave(4, dim=2)

    decoder = object.__new__(Magi2TurboVAEDecoder)
    nn.Module.__init__(decoder)
    decoder.z_dim = 1
    decoder.first_chunk_size = 3
    decoder.step_size = 2
    decoder.temporal_compression_ratio = 4
    decoder.decoder = RepeatDecoder()
    decoder.use_tiling = False
    decoder.register_buffer("latent_mean", torch.zeros(1), persistent=False)
    decoder.register_buffer("latent_std", torch.ones(1), persistent=False)

    latent = torch.arange(7, dtype=torch.float32).view(1, 1, 7, 1, 1)
    actual = decoder.decode(latent)
    expected = latent.repeat_interleave(4, dim=2)

    torch.testing.assert_close(actual, expected)

    decoder.latent_mean = torch.zeros(1, device="meta")
    decoder.latent_std = torch.ones(1, device="meta")
    prepared, _ = decoder._prepare_latent(latent)
    assert prepared.device.type == "meta"


@pytest.mark.parametrize("frames", [1, 3, 4, 7, 10, 35])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.cuda
def test_offload_matches_device_output(frames, dtype):
    decoder = make_decoder("cuda", dtype)
    # Batch > 1 and sliced input exercise strided source/crop layouts.
    latent = torch.randn(2, 2, frames, 8, 18, device="cuda", dtype=dtype)[..., ::2]
    expected = decoder.decode(latent).cpu()
    actual = decoder.decode(latent, output_offload=True)
    assert actual.device.type == "cpu"
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A subsequent decode may reuse allocator storage; prior CPU output is owned.
    decoder.decode(latent + 1, output_offload=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.cuda
def test_offload_drains_after_decode_error(monkeypatch):
    decoder = make_decoder("cuda", torch.float32)
    latent = torch.randn(1, 2, 10, 64, 64, device="cuda")
    decode = decoder._decode_chunk
    calls = 0

    def fail_on_third(task):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected decode error")
        return decode(task)

    monkeypatch.setattr(decoder, "_decode_chunk", fail_on_third)
    with pytest.raises(RuntimeError, match="injected decode error"):
        decoder.decode(latent, output_offload=True)
    monkeypatch.setattr(decoder, "_decode_chunk", decode)
    torch.testing.assert_close(
        decoder.decode(latent, output_offload=True), decoder.decode(latent).cpu(), rtol=0, atol=0
    )


@pytest.mark.cpu
def test_cpu_offload_keeps_sync_fallback(monkeypatch):
    decoder = make_decoder("cpu", torch.float32)

    def unexpected(*args):
        raise AssertionError("CPU should not enter CUDA offload")

    monkeypatch.setattr(decoder, "_decode_offloaded", unexpected)
    latent = torch.randn(1, 2, 10, 2, 2)
    torch.testing.assert_close(decoder.decode(latent, output_offload=True), decoder.decode(latent))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.cuda
def test_pinned_staging_is_bounded(monkeypatch):
    decoder = make_decoder("cuda", torch.float32)
    allocations = []
    empty = torch.empty

    def track(*args, **kwargs):
        result = empty(*args, **kwargs)
        if kwargs.get("pin_memory"):
            allocations.append(weakref.ref(result))
            assert sum(ref() is not None for ref in allocations) <= 2
        return result

    monkeypatch.setattr(torch, "empty", track)
    decoder.decode(torch.randn(1, 2, 35, 8, 8, device="cuda"), output_offload=True)
    assert allocations
    assert all(ref() is None for ref in allocations)
