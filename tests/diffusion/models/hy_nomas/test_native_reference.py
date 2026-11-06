# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoTokenizer

from tests.diffusion.models.hy_nomas.test_hy_nomas import TRANSFORMER_CONFIG
from vllm_omni.diffusion.models.hy_nomas.camera import prepare_camera_inputs
from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import HyNomasTransformer
from vllm_omni.diffusion.models.hy_nomas.layers import TimestepEmbedder
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import build_model_config
from vllm_omni.diffusion.models.hy_nomas.pixelumm import PixelUMMDCAELatentDecoder
from vllm_omni.diffusion.models.hy_nomas.pixelumm_model import DCAEVPixelUMMDecoder, GenExpertArch
from vllm_omni.diffusion.models.hy_nomas.text_encoder import _VIDEO_SYSTEM, TextEncoder

pytestmark = [pytest.mark.local_model, pytest.mark.cpu, pytest.mark.diffusion]


@torch.no_grad()
@pytest.mark.parametrize("tile", [0, 1])
def test_small_pixelumm_matches_independent_reference(tile):
    reference = pytest.importorskip("hymm.models.vae.pixelumm_decoder.model")
    options = dict(hidden_size=32, intermediate_size=64, num_layers=1, num_heads=4, num_kv_heads=2, head_dim=8)
    torch.manual_seed(42)
    expected_model = reference.DCAEVPixelUMMDecoder(
        reference.GenExpertArch(**options), local_attention_tile=tile
    ).eval()
    actual_model = DCAEVPixelUMMDecoder(GenExpertArch(**options), local_attention_tile=tile).eval()
    actual_model.load_state_dict(expected_model.state_dict(), strict=True)
    latents = torch.randn(1, 32, 2, 2, 3)
    noise = torch.randn(1, 3, 8, 64, 96)
    torch.testing.assert_close(actual_model(latents, noise), expected_model(latents, noise), rtol=0, atol=0)


@torch.no_grad()
def test_timestep_embedding_matches_reference():
    reference = pytest.importorskip("hymm.models.dit.modules.embed_layers")
    expected = reference.TimestepEmbedder(32, nn.SiLU)
    actual = TimestepEmbedder(32)
    actual.load_state_dict(expected.state_dict(), strict=True)
    values = torch.tensor([0.0, 206.897, 1000.0])
    torch.testing.assert_close(actual(values), expected(values), rtol=0, atol=0)


@pytest.mark.parametrize(
    "name",
    [
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "forward",
        "circular",
        "spiral",
        "dolly",
        "crane_up",
        "crane_down",
        "truck_left",
        "truck_right",
        "arc",
        "zigzag",
        "random_walk",
    ],
)
@pytest.mark.parametrize("frames", [49, 129])
def test_camera_presets_match_reference(name, frames):
    reference = pytest.importorskip("hymm.pipeline.bidirectional")
    options = dict(
        trajectory_source=name,
        num_frames=frames,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        normalize_relative_to_first_frame=True,
        normalize_framewise=True,
        normalize_displacement=True,
        normalize_displacement_per_video=True,
        rotation_scale=2.5,
        segment_boundaries_pixel=[0, 24, frames],
    )
    expected = reference.prepare_camera_inputs(**options)["norm_extrinsics"]
    actual = prepare_camera_inputs(**options)["norm_extrinsics"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "prompt",
    [
        "A forest beside a river",
        "",
        '{"segments":[{"contained_actions":{"a":{"start_time":"00:00","end_time":"00:01","action_description":"walk"}}}]}',
    ],
)
def test_real_tokenizer_matches_reference(prompt):
    reference = pytest.importorskip("hymm.models.text_encoder.encoder")
    constants = pytest.importorskip("configs.constants")
    root = Path(os.environ.get("HY_NOMAS_MODEL", "")) / "tokenizer"
    if not root.is_dir():
        pytest.skip("Set HY_NOMAS_MODEL to the locally cached Diffusers checkpoint.")
    tokenizer = AutoTokenizer.from_pretrained(root, padding_side="right", local_files_only=True)
    expected = reference.TextEncoder.__new__(reference.TextEncoder)
    nn.Module.__init__(expected)
    expected.tokenizer = tokenizer
    expected.use_template = expected.use_video_template = True
    expected.text_encoder_type = "qwen-2.5vl-7b"
    expected.prompt_template = deepcopy(constants.PROMPT_TEMPLATE["li-dit-encode-image-json"])
    expected.prompt_template_video = deepcopy(constants.PROMPT_TEMPLATE["li-dit-encode-video-json"])
    actual = TextEncoder.__new__(TextEncoder)
    nn.Module.__init__(actual)
    actual.tokenizer = tokenizer
    actual.prompt_template_video = {
        "template": [{"role": "system", "content": _VIDEO_SYSTEM}, {"role": "user", "content": "{}"}],
        "crop_start": -1,
    }
    expected_tokens = expected.text2tokens([prompt], data_type="video", max_length=1280)
    actual_tokens = actual.text2tokens([prompt], data_type="video", max_length=1280)
    assert actual.prompt_template_video == expected.prompt_template_video
    assert actual_tokens.keys() == expected_tokens.keys()
    for name in actual_tokens:
        assert torch.equal(actual_tokens[name], expected_tokens[name])


def test_native_dit_names_match_actual_checkpoint():
    root = Path(os.environ.get("HY_NOMAS_MODEL", "")) / "transformer"
    indices = list(root.glob("*.index.json"))
    if not indices:
        pytest.skip("Set HY_NOMAS_MODEL to the locally cached Diffusers checkpoint.")
    from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import _DIT_NAMES

    names = json.loads(indices[0].read_text())["weight_map"]
    mapped = {name for name, _ in _DIT_NAMES.apply((name, None) for name in names)}
    model = HyNomasTransformer(TRANSFORMER_CONFIG, device="meta")
    assert set(dict(model.named_parameters())) == mapped


def test_native_defaults_match_pinned_reference():
    reference = pytest.importorskip("configs.config")
    from omegaconf import OmegaConf

    root = Path(reference.__file__).parent
    base = OmegaConf.load(root / "config.yaml")
    inference = OmegaConf.load(root / "experiment" / "inference.yaml")
    del base["defaults"]
    del inference["defaults"]
    expected = reference._flatten_cfg(OmegaConf.merge(base, inference))
    actual = build_model_config()
    assert actual.text_len == expected.text_len
    assert actual.normalize_displacement_per_video == expected.normalize_displacement_per_video
    for name, value in actual.inference.items():
        assert value == expected.inference[name], name
    assert expected.camera.per_chunk_camera_attn and expected.camera.camera_chunk_size == 1


class WindowContext(nn.Module):
    # A window-dependent predictor isolates state bookkeeping, not neural accuracy.
    local_attention_tile = 0

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, latent, noise):
        values = latent[:, :3] + latent[:, :3].mean(2, keepdim=True)
        values = values.repeat_interleave(4, 2).repeat_interleave(32, 3).repeat_interleave(32, 4)
        return self.scale * noise + values


@torch.no_grad()
@pytest.mark.parametrize("lengths", [[1] * 81, [3, 1, 7, 2, 16, 5, 1], [2], [4], [5], [8, 1]])
@pytest.mark.parametrize("tail_hold", [0, 1])
def test_native_rolling_state_matches_independent_reference(lengths, tail_hold):
    reference = pytest.importorskip("hymm.models.vae.pixelumm_decoder.autoencoder")
    options = dict(
        seed=13,
        temporal_window_latents=4,
        temporal_overlap_latents=2,
        decoder_tile_pixels=0,
        tail_hold_latents=tail_hold,
    )
    expected = reference.PixelUMMDCAELatentDecoder(WindowContext(), **options).create_streaming_decode_session(
        to_cpu=True, retain_history=False, rolling=True
    )
    actual = PixelUMMDCAELatentDecoder(WindowContext(), **options).create_streaming_decode_session()
    generator = torch.Generator().manual_seed(301)
    for index, length in enumerate(lengths):
        latent = torch.randn((1, 32, length, 1, 1), generator=generator)
        pixels = length * 4 - index % 4
        actual.final_segment = expected.final_segment = index + 1 == len(lengths)
        torch.testing.assert_close(
            actual.push(latent, pixel_frames=pixels), expected.push(latent, pixel_frames=pixels), rtol=0, atol=0
        )
        assert actual.available_latent_frames == expected.available_latent_frames
        assert actual.stable_pixel_frames == expected.stable_pixel_frames
        assert actual.withheld_pixel_frames == expected.withheld_pixel_frames
        assert actual.buffer.shape[2] <= 2 * actual.window + actual.stride
        assert len(actual.push_latent_frames) <= 2 * actual.window
    torch.testing.assert_close(actual.finish(), expected.finish(), rtol=0, atol=0)
    assert actual.buffer.numel() == 0 and actual.pending is None
    assert not actual.push_latent_frames


@pytest.mark.parametrize("kind", ["json", "inline", "npz", "precomputed"])
def test_camera_file_and_recording_inputs_match_reference(tmp_path, kind):
    reference = pytest.importorskip("hymm.pipeline.bidirectional")
    from scipy.spatial.transform import Rotation

    flags = dict(
        normalize_relative_to_first_frame=True,
        normalize_framewise=True,
        normalize_displacement=True,
        normalize_displacement_per_video=False,
    )
    poses = np.repeat(np.eye(4)[None], 9, axis=0)
    poses[:, :3, 3] = np.arange(9)[:, None] * np.array([0.03, 0.01, 0.05])
    poses[:, :3, :3] = Rotation.from_euler("y", np.linspace(0, 0.2, 9)[:, None]).as_matrix()
    intrinsics = np.repeat(np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])[None], 9, axis=0)
    recording = {
        "frames": [
            {"position": dict(zip("xyz", position.tolist())), "quaternion": dict(zip("xyzw", quaternion.tolist()))}
            for position, quaternion in zip(poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat())
        ]
    }
    path = tmp_path / ("camera.json" if kind == "json" else "camera.npz")
    if kind == "json":
        payload = {
            str(index): {"extrinsic": pose.tolist(), "intrinsic": k.tolist()}
            for index, (pose, k) in enumerate(zip(poses, intrinsics))
        }
        path.write_text(json.dumps(payload))
    elif kind == "npz":
        np.savez(path, c2ws=poses, intrinsics=intrinsics)
    elif kind == "precomputed":
        prepare = pytest.importorskip("hymm.models.camera.geometry.demo_precomputed").prepare_turn
        arrays = prepare(recording, 49, flags)
        np.savez(
            path,
            camera_format="demo_camera_precomputed_v1",
            normalization_json=json.dumps(flags),
            segment_boundaries_pixel=np.array([0, 49]),
            turn_camera_offsets=np.array([0, 49]),
            turn_num_frames=np.array([49]),
            turn_has_camera=np.array([True]),
            **arrays,
        )
    source = json.dumps(recording) if kind == "inline" else str(path)
    options = dict(
        trajectory_source=source,
        num_frames=49,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        segment_boundaries_pixel=[0, 49],
        rotation_scale=2.5,
        **flags,
    )
    expected = reference.prepare_camera_inputs(**options)
    actual = prepare_camera_inputs(**options)
    torch.testing.assert_close(actual["norm_extrinsics"], expected["norm_extrinsics"], rtol=0, atol=0)
    assert actual.get("turn_camera_offsets") == expected.get("turn_camera_offsets")
