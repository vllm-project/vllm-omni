# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contracts of the MiniMax-H3-Image pipeline that do not need real weights."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import image_vae
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_image as h3i
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3DiTArchConfig
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_RELEASE_PDD = {
    "num_inference_steps": 6,
    "sigmas": [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0],
}


@pytest.mark.parametrize(
    ("size", "area", "canvas"),
    [
        ((1024, 1024), 2048 * 2048, (2048, 2048)),
        # Matches the reference implementation's log for a 16:9 request.
        ((1280, 720), 2048 * 2048, (2720, 1536)),
        # At or above the area: only aligned to the 32px grid.
        ((4736, 3552), 2048 * 2048, (4736, 3552)),
        ((4000, 3000), 2048 * 2048, (4000, 3008)),
        ((1024, 1024), None, (1024, 1024)),
        ((1024, 1024), 32 * 32, (1024, 1024)),
    ],
)
def test_generation_canvas(size, area, canvas):
    assert h3i.minimax_h3_image_generation_canvas(*size, area) == canvas


@pytest.mark.parametrize("value", [1023, 4096 * 4096 + 1, True, 2.0e6, "4194304"])
def test_min_generation_area_rejects_out_of_range(value):
    with pytest.raises(OmniClientError):
        h3i.minimax_h3_validate_min_generation_area(value)


@pytest.mark.parametrize(
    ("size", "bounds", "expected"),
    [
        ((1024, 1024), (1024, 2048), None),
        ((1536, 2048), (1024, 2048), None),
        ((512, 768), (1024, 2048), (1024, 1536)),
        ((3000, 4000), (1024, 2048), (2048, 2720)),
        ((1500, 2000), (1024, 1024), (1024, 1376)),
    ],
)
def test_reference_short_edge_bounds(size, bounds, expected):
    assert h3i.minimax_h3_image_reference_size(*size, min_short_edge=bounds[0], max_short_edge=bounds[1]) == expected


def test_reference_grid_is_even_16px_cells():
    assert h3i.minimax_h3_image_reference_grid(1024, 1024) == (64, 64)
    assert h3i.minimax_h3_image_reference_grid(1050, 1000) == (62, 64)


def test_release_pdd_grid_and_run_sigmas():
    grid = h3i.minimax_h3_image_pdd_grid(_RELEASE_PDD)
    assert len(grid) == 7
    sigmas = h3i.minimax_h3_image_run_sigmas(grid)
    assert len(sigmas) == 7 and sigmas[0] == 1.0 and sigmas[-1] == 0.0
    # Each sigma survives the FP32 timestep round trip unchanged.
    timesteps = torch.tensor([1.0 - s for s in sigmas[:-1]], dtype=torch.float32)
    assert sigmas[:-1] == (1.0 - timesteps).tolist()


@pytest.mark.parametrize(
    "config",
    [
        {"num_inference_steps": 2, "sigmas": [1.0, 0.5]},
        {"num_inference_steps": 2, "sigmas": [1.0, 0.5, 0.1]},
        {"num_inference_steps": 2, "sigmas": [0.9, 0.5, 0.0]},
        {"num_inference_steps": 2, "sigmas": [1.0, 1.0, 0.0]},
        {"num_inference_steps": True, "sigmas": [1.0, 0.0]},
        {"num_inference_steps": 1, "sigmas": "1,0"},
    ],
)
def test_pdd_grid_rejects_invalid_configs(config):
    with pytest.raises(ValueError):
        h3i.minimax_h3_image_pdd_grid(config)


class _FakeTokenizer:
    special = {"<|vision_start|>": 900, "<|vision_end|>": 901, "<|image_pad|>": 902}

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    def convert_tokens_to_ids(self, token):
        return self.special[token]


def test_presentation_labels_references_and_pads_to_eight():
    ids, tags, text_len = h3i.minimax_h3_image_presentation(_FakeTokenizer(), prompt="hi", image_token_counts=[3, 1])
    label1, label2 = [ord(c) for c in "Picture 1: "], [ord(c) for c in "Picture 2: "]
    expected = label1 + [900, 902, 902, 902, 901] + label2 + [900, 902, 901] + [ord("h"), ord("i")]
    assert text_len == len(expected) == tags.numel()
    assert ids.numel() % 8 == 0
    assert ids[:text_len].tolist() == expected
    assert ids[text_len:].tolist() == [0] * (ids.numel() - text_len)
    visual = [0] * 5
    assert tags.tolist() == [1] * len(label1) + visual + [1] * len(label2) + [0] * 3 + [1, 1]


def test_presentation_truncates_each_text_segment():
    ids, _, text_len = h3i.minimax_h3_image_presentation(_FakeTokenizer(), prompt="x" * 5000, image_token_counts=[])
    assert text_len == 4096 and ids.numel() == 4096


def test_packed_layout_accepts_audio_free_targets():
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=10,
        latent_t=1,
        latent_h=8,
        latent_w=8,
        audio_t=0,
        ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 6}],
        seq_len=64,
    )
    assert packed["audio_pos"].numel() == 0
    assert int((~packed["update_mask"]).sum()) == 4 * 6 // 4
    assert packed["img_pos"].numel() == 16 + 6
    with pytest.raises(ValueError, match="visual or audio"):
        minimax_h3_packed_sequence_ref2va_blocks(
            text_len=10, latent_t=0, latent_h=8, latent_w=8, audio_t=0, ref_blocks=[], seq_len=64
        )


def test_arch_config_audio_enabled_contract():
    assert MiniMaxH3DiTArchConfig.from_mapping({"audio_enabled": False}).audio_enabled is False
    assert MiniMaxH3DiTArchConfig.from_mapping({}).audio_enabled is True
    with pytest.raises(ValueError, match="bool"):
        MiniMaxH3DiTArchConfig.from_mapping({"audio_enabled": 0})
    with pytest.raises(ValueError, match="pdd_steps"):
        MiniMaxH3DiTArchConfig.from_mapping({"audio_enabled": False, "pdd_steps": 32})


def test_text_encoder_root_resolution(tmp_path):
    base = tmp_path / "base"
    (base / "text_encoder").mkdir(parents=True)
    release = tmp_path / "release"
    release.mkdir()
    index: dict[str, Any] = {
        "text_encoder": [
            "transformers",
            "Qwen3VLForConditionalGeneration",
            {"pretrained_model_name_or_path": str(base), "subfolder": "text_encoder"},
        ]
    }
    assert h3i.MiniMaxH3ImagePipeline._resolve_text_encoder_root(index, release) == str(base)
    assert h3i.MiniMaxH3ImagePipeline._resolve_text_encoder_root({}, release) == str(release)
    # The Hub release bundles text_encoder/ while model_index.json still names
    # the base repository: the bundled copy wins and nothing is downloaded.
    (release / "text_encoder").mkdir()
    assert h3i.MiniMaxH3ImagePipeline._resolve_text_encoder_root(index, release) == str(release)
    index["text_encoder"][2]["subfolder"] = "encoder"
    with pytest.raises(ValueError, match="unsupported"):
        h3i.MiniMaxH3ImagePipeline._resolve_text_encoder_root(index, tmp_path)


def test_post_process_truncates_like_the_released_writer():
    post = h3i.get_minimax_h3_image_post_process_func(None)
    pixels = torch.tensor([0.0, 0.5, 254.9 / 255, 1.0]).view(1, 1, 1, 4).expand(1, 3, 1, 4)
    array = post(pixels, output_type="np")
    assert array.dtype.name == "uint8"
    assert array[0, 0, :, 0].tolist() == [0, 127, 254, 255]


@pytest.mark.parametrize(
    ("length", "starts"),
    [
        (256, [0]),
        (200, [0]),
        (1024, [0, 192, 384, 576, 768]),
        (2048, [0, 176, 352, 528, 704, 880, 1056, 1232, 1408, 1600, 1792]),
    ],
)
def test_vae_tile_split(length, starts):
    got, lengths, overlaps = image_vae._split_tiles(length, 256, 64)
    assert got == starts
    if length <= 256:
        assert lengths == [length] and overlaps == []
    else:
        assert got[-1] + lengths[-1] == length
        assert all(o >= 64 and o % 16 == 0 for o in overlaps)


def test_vae_tiles_reassemble_identity():
    # Encoder inputs are cropped to the 16px grid.
    x = torch.arange(3 * 608 * 528, dtype=torch.float32).view(1, 3, 608, 528)
    y_idx, y_len, y_ov = image_vae._split_tiles(608, 256, 64)
    x_idx, x_len, x_ov = image_vae._split_tiles(528, 256, 64)
    rows = [[x[..., t : t + h, s : s + w] for s, w in zip(x_idx, x_len)] for t, h in zip(y_idx, y_len)]
    torch.testing.assert_close(image_vae._assemble_tiles(rows, y_ov, x_ov), x, rtol=1e-6, atol=0)


def _tiny_vae_config() -> dict:
    return {
        "_class_name": "MiniMaxH3ImageVAE",
        "latent_channels": 24,
        "latents_mean": [0.1] * 24,
        "latents_std": [2.0] * 24,
        "vae_tile_size": 32,
        "vae_tile_overlap_min": 16,
        "vit_decoder_2d": True,
        "vit_decoder_kwargs": {
            "dim_head": 16,
            "heads": 2,
            "num_layers": 1,
            "rope_dim_ratio": 0.5,
            "rope_theta": 100.0,
            "norm_type": "rms_norm",
            "qk_norm_type": "rms_norm",
        },
    }


def test_vae_round_trip_shapes_and_fp32_statistics(tmp_path):
    from PIL import Image
    from safetensors.torch import save_file

    config = _tiny_vae_config()
    with torch.device("cpu"):
        model = image_vae.MiniMaxH3ImageVAE(config)
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(tmp_path / "diffusion_pytorch_model.safetensors"),
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        # Built inside a BF16 default-dtype context, as the diffusion loader does.
        vae = image_vae.MiniMaxH3ImageVAE.from_pretrained(tmp_path)
    finally:
        torch.set_default_dtype(previous)
    assert all(p.dtype == torch.float32 for p in vae.parameters())
    assert vae.latents_std.dtype == torch.float32 and vae.pixel_std.dtype == torch.float32
    image = Image.fromarray((torch.rand(70, 50, 3) * 255).to(torch.uint8).numpy())
    latent = vae.encode_image_latent(image, generator=torch.Generator().manual_seed(0))
    # Cropped top-left to the 16px grid: 64x48 pixels -> 4x3 latent cells.
    assert latent.shape == (1, 24, 1, 4, 3)
    assert torch.equal(latent, latent.half().float())
    again = vae.encode_image_latent(image, generator=torch.Generator().manual_seed(0))
    assert torch.equal(latent, again)
    pixels = vae.decode_latent(vae.normalize_latent(latent))
    assert pixels.shape == (1, 3, 64, 48)
    assert pixels.dtype == torch.float32 and 0.0 <= pixels.min() and pixels.max() <= 1.0


def test_vae_rejects_non_image_configs():
    config = _tiny_vae_config()
    config["vit_decoder_2d"] = False
    with pytest.raises(ValueError, match="vit_decoder_2d"):
        image_vae.MiniMaxH3ImageVAE(config)
    config = _tiny_vae_config()
    config["vit_decoder_kwargs"]["norm_type"] = "layer_norm"
    with pytest.raises(ValueError, match="norm_type"):
        image_vae.MiniMaxH3ImageVAE(config)


def _reference_float_resize(pixels: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """The released resize: quantize, PIL LANCZOS, return ``uint8 / 255``."""
    import numpy as np
    from PIL import Image

    out = []
    for image in pixels:
        array = (image * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        resized = Image.fromarray(array).resize((width, height), Image.Resampling.LANCZOS)
        out.append(torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1))
    return torch.stack(out).float().div_(255)


def test_uint8_output_path_matches_the_released_float_path():
    torch.manual_seed(0)
    pixels = torch.rand(1, 3, 96, 64)
    post = h3i.get_minimax_h3_image_post_process_func(None)
    expected = post(_reference_float_resize(pixels, 50, 70), output_type="np")
    resized = h3i.minimax_h3_resize_image_output(h3i.minimax_h3_quantize_image(pixels), 50, 70)
    assert resized.dtype == torch.uint8
    assert (post(resized, output_type="np") == expected).all()
    # Without a resize the writer quantizes the decoded floats directly.
    assert (post(h3i.minimax_h3_quantize_image(pixels), output_type="np") == post(pixels, output_type="np")).all()


@pytest.mark.parametrize(("world", "count"), [(4, 121), (4, 6), (3, 7), (4, 3)])
def test_tiles_spread_over_ranks_keep_their_order(monkeypatch, world, count):
    tiles = [torch.full((1, 2), float(index)) for index in range(count)]

    def fn(tile):
        return tile * 10 + 1

    expected = [fn(tile) for tile in tiles]
    results = []
    for rank in range(world):
        monkeypatch.setattr(image_vae, "_group_world", lambda group, rank=rank: (world, rank))

        def all_gather(out, inp, group=None):
            # Every rank stacks its round-robin share, padded to the same length.
            per_rank = inp.shape[0]
            for peer in range(world):
                local = [fn(tiles[index]) for index in range(peer, count, world)]
                out[peer * per_rank : peer * per_rank + len(local)] = torch.stack(local)

        monkeypatch.setattr(image_vae.dist, "all_gather_into_tensor", all_gather)
        results.append(image_vae._run_tiles(tiles, fn, group=object()))
    for got in results:
        assert len(got) == count
        assert all(torch.equal(a, b) for a, b in zip(got, expected))


@pytest.mark.parametrize(
    ("size", "target"),
    [
        ((256, 256), (128, 128)),
        ((340, 192), (160, 90)),
        ((200, 200), (99, 101)),
        ((64, 40), (128, 77)),
        ((256, 64), (7, 64)),
    ],
)
def test_device_lanczos_matches_pillow(size, target):
    import numpy as np
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.image_resize import resize_lanczos_uint8

    array = np.random.default_rng(2).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    expected = np.asarray(Image.fromarray(array).resize(target, Image.Resampling.LANCZOS))
    got = resize_lanczos_uint8(torch.from_numpy(array.copy()).permute(2, 0, 1)[None], *target)
    assert np.array_equal(got[0].permute(1, 2, 0).numpy(), expected)


def test_post_process_accepts_hwc_uint8():
    post = h3i.get_minimax_h3_image_post_process_func(None)
    chw = h3i.minimax_h3_quantize_image(torch.rand(1, 3, 20, 30))
    hwc = chw.permute(0, 2, 3, 1).contiguous()
    assert (post(hwc, output_type="np") == post(chw, output_type="np")).all()


def test_rope_follows_position_device_when_buffer_is_offloaded():
    # Model CPU offload keeps the DiT's inv_freq on the host until its forward
    # hook runs, but the denoise loop builds the rope table before that forward.
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Rope

    rope = MiniMaxH3Rope(16)
    rope.inv_freq.copy_(torch.arange(16, dtype=torch.float32))
    positions = torch.zeros(1, 4, 3, device="meta")

    freqs = rope(positions)

    assert freqs.device.type == "meta" and freqs.shape == (4, 96)


def test_offload_plan_streams_encoder_and_keeps_graph_replayed_vae_resident():
    plan = h3i.MiniMaxH3ImagePipeline._offload_plan
    assert plan.encoder_block_attrs == {"text_encoder": ("vision.blocks", "text_model.layers")}
    assert plan.resident_dit_paths == frozenset({"transformer"})
    # The image VAE replays CUDA graphs that capture its weight addresses.
    assert "vae" not in plan.on_demand_component_paths
    assert h3i.MiniMaxH3ImagePipeline._resident_dit_layers_on_device is not None


@pytest.mark.parametrize(
    ("size", "ok"), [((1024, 8192), True), ((8192, 1024), True), ((32, 16384), False), ((9000, 1000), False)]
)
def test_reference_aspect_ratio_is_bounded(size, ok):
    # Short edges are resized to 1024px, so an unbounded aspect ratio turns a tiny
    # 32x16384 upload into a 1024x524288 Qwen/VAE input.
    from PIL import Image

    image = Image.new("RGB", size, (128, 64, 32))
    if ok:
        assert h3i._load_reference(image).size == size
    else:
        with pytest.raises(OmniClientError, match="aspect ratio"):
            h3i._load_reference(image)


def test_reference_loading_reads_path_upright_once(tmp_path):
    from PIL import Image

    path = tmp_path / "ref.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    Image.new("RGB", (64, 48), (10, 20, 30)).save(path, exif=exif)
    image = h3i._load_reference(str(path))
    assert image.size == (48, 64) and image.mode == "RGB"
    assert image.getexif().get(0x0112) in (None, 1)


class _RecordingEncoder:
    def __init__(self):
        self.calls: list[str] = []

    def load_to_device(self):
        self.calls.append("load")

    def offload_to_cpu(self):
        self.calls.append("offload")

    def encode_ids(self, ids, **vision):
        self.calls.append("encode")
        return torch.zeros(ids.shape[0], 4, dtype=torch.bfloat16)


@pytest.mark.parametrize("offloaded", [True, False])
def test_offloaded_text_encoder_is_released_after_the_encode(monkeypatch, offloaded):
    pipe = h3i.MiniMaxH3ImagePipeline.__new__(h3i.MiniMaxH3ImagePipeline)
    torch.nn.Module.__init__(pipe)
    pipe.text_encoder = _RecordingEncoder()
    pipe.tokenizer = None
    pipe._dit_rank = 0
    pipe.text_encoder_tp_size = 1
    pipe.device = torch.device("cpu")
    monkeypatch.setattr(
        h3i, "minimax_h3_image_presentation", lambda *a, **k: (torch.zeros(5, dtype=torch.long), torch.zeros(5), 3)
    )
    monkeypatch.setattr(h3i, "_broadcast_tensor", lambda tensor, **k: tensor)
    monkeypatch.setattr(h3i.MiniMaxH3ImagePipeline, "_uses_manual_component_offload", lambda self, c: offloaded)

    hidden, _ = pipe._encode_text("a prompt", [])

    assert hidden.shape == (3, 4)
    assert pipe.text_encoder.calls == (["load", "encode", "offload"] if offloaded else ["load", "encode"])
