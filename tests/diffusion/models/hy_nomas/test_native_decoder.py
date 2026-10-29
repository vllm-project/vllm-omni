# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.hy_nomas import decoder as decoder_module
from vllm_omni.diffusion.models.hy_nomas.decoder import PixelUMMDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "config,message",
    [({"head_dim": dim}, "head_dim must be a positive multiple of 8") for dim in (0, -8, 12)]
    + [
        ({"num_heads": 0}, "must be positive"),
        ({"num_kv_heads": 0}, "must be positive"),
        ({"num_heads": -2}, "must be positive"),
        ({"num_kv_heads": -1}, "must be positive"),
        ({"num_heads": 3, "num_kv_heads": 2}, "must be divisible"),
    ],
)
def test_native_architecture_rejects_invalid_head_geometry(config, message):
    with pytest.raises(ValueError, match=message):
        decoder_module.PixelUMMArchitecture(**config)


@pytest.mark.parametrize("head_dim", [8, 16, 128])
def test_native_mrope_matches_axis_rotation(head_dim):
    arch = decoder_module.PixelUMMArchitecture(head_dim=head_dim)
    grid = (2, 3, 4)
    sections = arch.rope_sections
    assert sum(sections) == head_dim and all(width % 2 == 0 for width in sections)
    cos, sin = decoder_module.mrope_tables(arch, grid, torch.device("cpu"))
    assert cos.shape == sin.shape == (1, math.prod(grid), 1, head_dim)
    values = torch.randn(1, math.prod(grid), 2, head_dim, generator=torch.Generator().manual_seed(42))
    expected = torch.empty_like(values)
    # Independent scalar rotation pairs check axis order, frequencies and pairing.
    for token, position in enumerate(itertools.product(*(range(size) for size in grid))):
        offset = 0
        for coordinate, width, theta in zip(
            position, sections, (arch.rope_theta_t, arch.rope_theta_hw, arch.rope_theta_hw)
        ):
            half = width // 2
            for pair in range(half):
                angle = coordinate * theta ** (-2 * pair / width)
                first = values[:, token, :, offset + pair]
                second = values[:, token, :, offset + half + pair]
                expected[:, token, :, offset + pair] = first * math.cos(angle) - second * math.sin(angle)
                expected[:, token, :, offset + half + pair] = second * math.cos(angle) + first * math.sin(angle)
            offset += width
    actual = decoder_module.apply_mrope(values, cos, sin, sections)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


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
@pytest.mark.parametrize("steps,tile", [(1, 0), (2, 64)])
def test_native_full_decode_matches_reference(steps, tile):
    original, native = reference_pair(num_steps=steps, decoder_tile_pixels=tile, decoder_tile_overlap_pixels=32)
    latent = torch.randn(1, 32, 5, 3, 3)
    expected = original.decode(latent, pixel_frames=17).sample
    actual = native.decode(latent, pixel_frames=17).sample
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@torch.no_grad()
def test_native_stream_matches_reference_at_every_push():
    chunks = (1, 3, 1)
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
    original, native = reference_pair(local_attention_tile_pixels=48)

    def dense_attention(query, key, value, block_mask):
        queries = torch.arange(query.shape[2])[:, None]
        keys = torch.arange(key.shape[2])[None, :]
        mask = block_mask.mask_mod(0, 0, queries, keys)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, enable_gqa=True)

    monkeypatch.setattr(reference, "_flex_attention", dense_attention)
    # Partial edge tiles exercise neighborhood boundaries as well as GQA.
    latent = torch.randn(1, 32, 1, 2, 5)
    pixels = torch.randn(1, 3, 4, 64, 160)
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
