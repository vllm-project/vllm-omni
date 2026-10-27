# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.hy_nomas import decoder as decoder_module
from vllm_omni.diffusion.models.hy_nomas.decoder import (
    PixelUMMDecoder,
    _window_starts,
    local_attention_layout,
    patchify,
    unpatchify,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _SDPA(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), enable_gqa=True
        ).transpose(1, 2)


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    # CPU math coverage does not initialize a CUDA/distributed backend.
    monkeypatch.setattr(decoder_module, "Attention", _SDPA)


def tiny_decoder(**kwargs):
    return PixelUMMDecoder(
        hidden_size=16,
        intermediate_size=24,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        temporal_window_latents=3,
        temporal_overlap_latents=1,
        **kwargs,
    ).eval()


def test_patchify_preserves_tubelet_order():
    pixels = torch.arange(3 * 8 * 32 * 48).reshape(1, 3, 8, 32, 48)
    tokens = patchify(pixels)
    assert torch.equal(tokens[0, 0], pixels[0, :, :4, :16, :16].permute(1, 2, 3, 0).flatten())
    assert torch.equal(unpatchify(tokens, (2, 2, 3)), pixels)
    with pytest.raises(ValueError, match="multiple"):
        patchify(pixels[:, :, :7])


def test_spatial_neighborhood_includes_partial_edge_tiles():
    permutation, inverse, regions = local_attention_layout((2, 7, 9), 2, "cpu")
    assert torch.equal(permutation[inverse], torch.arange(126))
    assert sum(end - start for (start, end), _ in regions) == 126
    # A corner attends a full 3 x 3 neighborhood, including the second-nearest tile.
    assert len(regions[0][1]) == 9
    assert regions[-1][0][1] == 126
    assert all(0 <= start < end <= 126 for _, neighbors in regions for start, end in neighbors)


def test_window_noise_is_absolute_and_does_not_advance_generator():
    model = tiny_decoder()
    latent = torch.zeros(1, 32, 5, 1, 1)
    all_noise = model._make_noise(latent, 0, 5, 13)
    overlap = model._make_noise(latent, 2, 4, 13)
    assert torch.equal(overlap, all_noise[:, :, 8:16])
    assert _window_starts(8, 3, 2) == [0, 2, 4, 5]


def test_streaming_stable_prefix_matches_full_decode(monkeypatch):
    model = tiny_decoder()

    # A window-sensitive decoder makes wrong overlap placement observable without
    # a released 8B checkpoint or stochastic attention kernels.
    def decode_window(latent, noise):
        frames = latent[:, :1].repeat_interleave(4, dim=2)
        return frames + latent[:, :1].mean(dim=2, keepdim=True) + noise[:, :1, :, :1, :1] * 0.01

    monkeypatch.setattr(model.decoder, "forward", decode_window)
    latents = torch.arange(8).reshape(1, 1, 8, 1, 1).expand(1, 32, 8, 1, 1).float()
    generator = torch.Generator().manual_seed(23)
    before = generator.get_state().clone()
    complete = model.decode(latents, generator=generator, return_dict=False)[0]
    session = model.create_streaming_decode_session(to_cpu=True, generator=generator)
    first = session.push(latents[:, :, :4], pixel_frames=13)
    old_prefix = first.clone()
    second = session.push(latents[:, :, 4:], pixel_frames=15)
    tail = session.finish()
    output = torch.cat([first, second, tail], dim=2)
    expected = torch.cat([complete[:, :, :13], complete[:, :, 16:31]], dim=2)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.equal(first, old_prefix)
    assert torch.equal(generator.get_state(), before)
    assert session.stable_pixel_frames == 28
    torch.testing.assert_close(session.full_video(), expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="after finish"):
        session.push(latents[:, :, :1])


def test_decoder_load_rejects_missing_and_unknown_tensors():
    model = tiny_decoder()
    weights = list(model.state_dict().items())
    assert model.load_weights(weights) == set(model.state_dict())
    with pytest.raises(ValueError, match="Missing"):
        model.load_weights(weights[:-1])
    with pytest.raises(ValueError, match="Unexpected"):
        model.load_weights([("invalid", torch.ones(1))])


def reference_pair(**kwargs):
    reference = pytest.importorskip("hymm.models.vae.pixelumm_decoder.model")
    wrapper = pytest.importorskip("hymm.models.vae.pixelumm_decoder.autoencoder")
    native = tiny_decoder(**kwargs)
    arch = reference.GenExpertArch(**vars(native.decoder.arch))
    original = wrapper.PixelUMMDCAELatentDecoder(
        reference.DCAEVPixelUMMDecoder(arch, local_attention_tile=native.decoder.local_attention_tile),
        **{
            name: getattr(native.config, name)
            for name in (
                "temporal_window_latents",
                "temporal_overlap_latents",
                "decoder_tile_pixels",
                "decoder_tile_overlap_pixels",
                "denoising_step_list",
                "num_steps",
                "seed",
            )
        },
        tail_hold_latents=native.tail_hold_latents,
    ).eval()
    original.load_state_dict(native.state_dict(), strict=True)
    return original, native


@torch.no_grad()
@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("tile", [0, 64])
def test_native_full_decode_matches_reference(steps, tile):
    original, native = reference_pair(num_steps=steps, decoder_tile_pixels=tile, decoder_tile_overlap_pixels=32)
    latent = torch.randn(1, 32, 5, 3, 3)
    expected = original.decode(latent, pixel_frames=17).sample
    actual = native.decode(latent, pixel_frames=17).sample
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@torch.no_grad()
@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1), (2, 3), (4, 1)])
def test_native_stream_matches_reference_at_every_push(chunks):
    original, native = reference_pair()
    sessions = [
        model.create_streaming_decode_session(to_cpu=True, retain_history=False) for model in (original, native)
    ]
    outputs: list[list[torch.Tensor]] = [[], []]
    for index, count in enumerate(chunks):
        latent = torch.randn(1, 32, count, 1, 1)
        for output, session in zip(outputs, sessions):
            session.final_segment = index == len(chunks) - 1
            output.append(session.push(latent, pixel_frames=count * 4 - (3 if index == 0 else 0)))
        torch.testing.assert_close(outputs[1][-1], outputs[0][-1], rtol=2e-5, atol=2e-6)
        assert sessions[0].available_pixel_frames == sessions[1].available_pixel_frames
    for output, session in zip(outputs, sessions):
        output.append(session.finish())
    for expected, actual in zip(*outputs):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert sum(part.shape[2] for part in outputs[1] if part.numel()) == sum(chunks) * 4 - 3


@torch.no_grad()
def test_bounded_qk_matches_unsplit_math(monkeypatch):
    model = tiny_decoder()
    latent = torch.randn(1, 32, 2, 2, 2)
    pixels = torch.randn(1, 3, 8, 64, 64)
    expected = model.decoder(latent, pixels)
    monkeypatch.setattr(decoder_module, "_QK_CHUNK_INPUT_BYTES", 64)
    actual = model.decoder(latent, pixels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_local_gqa_matches_reference_visibility(monkeypatch):
    reference = pytest.importorskip("hymm.models.vae.pixelumm_decoder.model")
    original, native = reference_pair(local_attention_tile_pixels=16)

    def dense_attention(query, key, value, block_mask):
        queries = torch.arange(query.shape[2])[:, None]
        keys = torch.arange(key.shape[2])[None, :]
        mask = block_mask.mask_mod(0, 0, queries, keys)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, enable_gqa=True)

    monkeypatch.setattr(reference, "_flex_attention", dense_attention)
    latent = torch.randn(1, 32, 2, 2, 2)
    pixels = torch.randn(1, 3, 8, 64, 64)
    expected = original.decoder(latent, pixels)
    actual = native.decoder(latent, pixels)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_native_factory_keeps_checkpoint_names_and_bf16_placement():
    from tests.diffusion.models.hy_nomas.test_hy_nomas import DECODER_CONFIG
    from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import build_decoder

    model = build_decoder(DECODER_CONFIG, torch.device("meta"), backend="native")
    parameters = dict(model.named_parameters())
    assert len(parameters) == 476
    assert all(name.startswith("decoder.") for name in parameters)
    assert all(value.device.type == "meta" and value.dtype == torch.bfloat16 for value in parameters.values())
    assert model.config.scaling_factor == 1.0 and model.config.shift_factor is None
