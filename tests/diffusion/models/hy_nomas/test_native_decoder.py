# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.models.hy_nomas.test_hy_nomas import DECODER_CONFIG
from vllm_omni.diffusion.models.hy_nomas import decoder as decoder_module
from vllm_omni.diffusion.models.hy_nomas import decoder_kernels, pixelumm, pixelumm_model
from vllm_omni.diffusion.models.hy_nomas.decoder import PixelUMMDecoder
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import build_decoder

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.diffusion,
    pytest.mark.usefixtures("single_rank_parallel"),
]


@pytest.mark.parametrize(
    "config,message",
    [
        ({"head_dim": 0}, "head_dim must be a positive multiple of 8"),
        ({"head_dim": -8}, "head_dim must be a positive multiple of 8"),
        ({"head_dim": 12}, "head_dim must be a positive multiple of 8"),
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
    thetas = (arch.rope_theta_t, arch.rope_theta_hw, arch.rope_theta_hw)
    assert sum(sections) == head_dim
    for width in sections:
        assert width % 2 == 0
    cos, sin = decoder_module.mrope_tables(arch, grid, torch.device("cpu"))
    assert cos.shape == sin.shape == (1, math.prod(grid), 1, head_dim)
    values = torch.randn(1, math.prod(grid), 2, head_dim, generator=torch.Generator().manual_seed(42))
    expected = torch.empty_like(values)
    # Independent scalar rotation pairs check axis order, frequencies and pairing.
    positions = itertools.product(range(grid[0]), range(grid[1]), range(grid[2]))
    for token, position in enumerate(positions):
        offset = 0
        for coordinate, width, theta in zip(position, sections, thetas):
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
    attn_backend = None

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
    model = PixelUMMDecoder(
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
    # vLLM's parallel linear layers allocate uninitialized weights.
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * parameter.shape[-1] ** -0.5)
    return model


def reference_pair(**kwargs):
    """The in-tree reference port and a native decoder sharing the same weights."""
    native = tiny_decoder(**kwargs)
    arch = pixelumm_model.GenExpertArch(**vars(native.decoder.arch))
    network = pixelumm_model.DCAEVPixelUMMDecoder(arch, local_attention_tile=native.decoder.local_attention_tile)
    config = native.config
    original = pixelumm.PixelUMMDCAELatentDecoder(
        network,
        temporal_window_latents=config.temporal_window_latents,
        temporal_overlap_latents=config.temporal_overlap_latents,
        decoder_tile_pixels=config.decoder_tile_pixels,
        decoder_tile_overlap_pixels=config.decoder_tile_overlap_pixels,
        denoising_step_list=config.denoising_step_list,
        num_steps=config.num_steps,
        seed=config.seed,
        tail_hold_latents=native.tail_hold_latents,
    ).eval()
    native.load_weights(original.state_dict().items())
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
        original.create_streaming_decode_session(to_cpu=True),
        native.create_streaming_decode_session(to_cpu=True, retain_history=False, rolling=False),
    ]
    outputs: list[list[torch.Tensor]] = [[], []]
    for index, count in enumerate(chunks):
        latent = torch.randn(1, 32, count, 1, 1)
        # The first segment starts with one frame per latent instead of four.
        pixel_frames = count * 4
        if index == 0:
            pixel_frames -= 3
        for output, session in zip(outputs, sessions):
            session.final_segment = index == len(chunks) - 1
            output.append(session.push(latent, pixel_frames=pixel_frames))
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
def test_native_rolling_stream_preserves_output_and_bounds_state():
    """Repeated eviction preserves absolute noise, cropped segments and the final tail."""
    original, native = reference_pair()
    sessions = [
        original.create_streaming_decode_session(to_cpu=True),
        native.create_streaming_decode_session(to_cpu=True, retain_history=False, rolling=True),
        native.create_streaming_decode_session(to_cpu=True, retain_history=False, rolling=False),
    ]
    generator = torch.Generator().manual_seed(42)
    chunks = (1, 3, 1, 4) * 4
    for index, count in enumerate(chunks):
        latent = torch.randn(1, 32, count, 1, 1, generator=generator)
        outputs = []
        for session in sessions:
            session.final_segment = index == len(chunks) - 1
            outputs.append(session.push(latent, pixel_frames=count * 4 - index % 4))
        torch.testing.assert_close(outputs[1], outputs[0], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(outputs[1], outputs[2], rtol=0, atol=0)
        rolling = sessions[1]
        assert rolling.available_pixel_frames == sessions[0].available_pixel_frames
        assert rolling.stable_pixel_frames == sessions[0].stable_pixel_frames
        assert rolling.buffer.shape[2] <= 2 * rolling.window
        assert rolling.buffer.untyped_storage().nbytes() == rolling.buffer.numel() * rolling.buffer.element_size()
        assert len(rolling.push_latent_frames) <= 2 * rolling.window
    assert sessions[1].buffer_latent_start > 0
    tails = [session.finish() for session in sessions]
    torch.testing.assert_close(tails[1], tails[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(tails[1], tails[2], rtol=0, atol=0)
    assert sessions[1].buffer.numel() == 0 and sessions[1].pending is None
    assert not sessions[1].push_latent_frames and not sessions[1].push_pixel_frames


@torch.no_grad()
def test_local_gqa_matches_reference_visibility():
    # The reference applies its FlexAttention block mask; CPU runs it eagerly.
    original, native = reference_pair(local_attention_tile_pixels=48)
    # Partial edge tiles exercise neighborhood boundaries as well as GQA.
    latent = torch.randn(1, 32, 1, 2, 5)
    pixels = torch.randn(1, 3, 4, 64, 160)
    expected = original.decoder(latent, pixels)
    actual = native.decoder(latent, pixels)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_native_factory_keeps_checkpoint_names_and_bf16_placement():
    model = build_decoder(DECODER_CONFIG, torch.device("meta"), backend="native")
    parameters = dict(model.named_parameters())
    # Each of 36 layers stacks q/k/v and gate/up: 5 checkpoint projections become 2 parameters.
    assert len(parameters) == 476 - 3 * 36
    for name, value in parameters.items():
        assert name.startswith("decoder.")
        assert value.device.type == "meta"
        assert value.dtype == torch.bfloat16
    assert model.config.scaling_factor == 1.0
    assert model.config.shift_factor is None


def test_native_load_weights_stacks_qkv_and_mlp_projections():
    original, native = reference_pair()
    checkpoint = original.state_dict()
    layer = native.decoder.layers[0]
    prefix = "decoder.layers.0"
    qkv = torch.cat([checkpoint[f"{prefix}.attn.{name}_proj.weight"] for name in "qkv"])
    gate_up = torch.cat([checkpoint[f"{prefix}.mlp.{name}_proj.weight"] for name in ("gate", "up")])
    torch.testing.assert_close(layer.attn.qkv_proj.weight, qkv, rtol=0, atol=0)
    torch.testing.assert_close(layer.mlp.gate_up_proj.weight, gate_up, rtol=0, atol=0)
    del checkpoint["decoder.head.norm.weight"]
    with pytest.raises(ValueError, match="missing 1 tensors"):
        tiny_decoder().load_weights(checkpoint.items())


@torch.inference_mode()
def test_fused_kernels_leave_cpu_to_eager():
    x = torch.randn(1, 16, 8, 128, dtype=torch.bfloat16)
    weight = torch.ones(64, dtype=torch.bfloat16)
    cos, sin = decoder_module.mrope_tables(decoder_module.PixelUMMArchitecture(), (1, 4, 4), torch.device("cpu"))
    # CPU never reaches Triton or torch.compile; the callers' eager math runs.
    assert decoder_kernels.split_norm_rope(x, cos, sin, weight, weight, 1e-6) is None
