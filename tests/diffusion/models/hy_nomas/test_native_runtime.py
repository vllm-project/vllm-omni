# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tests.diffusion.models.hy_nomas.test_hy_nomas import TRANSFORMER_CONFIG
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.models.hy_nomas import pipeline_hy_nomas
from vllm_omni.diffusion.models.hy_nomas.attention import RangeAttention, build_range_mask
from vllm_omni.diffusion.models.hy_nomas.dit import HyNomasTransformer
from vllm_omni.diffusion.models.hy_nomas.pixelumm import PixelUMMDCAELatentDecoder
from vllm_omni.diffusion.models.hy_nomas.pixelumm_model import DCAEVPixelUMMDecoder, GenExpertArch
from vllm_omni.diffusion.models.hy_nomas.scheduler import EulerSolver

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _small_transformer():
    config = {
        **TRANSFORMER_CONFIG,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "num_attention_heads": 4,
        "attention_head_dim": 8,
        "text_embed_dim": 6,
        "rope_axes_dim": [2, 2, 4],
        "camera_embed_dim": 8,
        "camera_attention_dim": 32,
        "camera_num_attention_heads": 4,
        "camera_rope_axes_dim": [2, 2, 4],
    }
    return HyNomasTransformer(config, device="cpu", dtype=torch.float32).eval().requires_grad_(False)


def test_production_package_has_no_reference_imports():
    directory = Path(pipeline_hy_nomas.__file__).parent
    for path in directory.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            assert not any(name.split(".")[0] in ("hymm", "configs") for name in names), path


def test_import_and_real_small_modules_with_reference_imports_blocked():
    code = """
import importlib.abc
import sys
class BlockReference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('hymm', 'configs'):
            raise AssertionError('Runtime imported reference code: ' + fullname)
sys.meta_path.insert(0, BlockReference())
from tests.diffusion.models.hy_nomas.test_native_runtime import test_native_dit_cache_and_camera, test_real_pixelumm_stream_matches_full_decode
test_native_dit_cache_and_camera()
test_real_pixelumm_stream_matches_full_decode()
assert not any(name.split('.')[0] in ('hymm', 'configs') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@torch.no_grad()
def test_native_dit_cache_and_camera():
    torch.manual_seed(17)
    model = _small_transformer()
    cache = [{"k": torch.zeros(1, 20, 4, 8), "v": torch.zeros(1, 20, 4, 8)}]
    mask = torch.zeros(1, 20, dtype=torch.bool)
    inputs = dict(
        hidden_states=torch.randn(1, 4, 2, 1, 1),
        timestep=torch.tensor([300.0]),
        text_states=torch.randn(1, 3, 6),
        text_mask=torch.tensor([[1, 1, 0]]),
        kv_cache_list=cache,
        cache_mask_list=[mask],
    )
    original = cache[0]["k"].clone()
    first, written = model.forward_with_cache(**inputs)
    assert written == 0 and torch.equal(cache[0]["k"], original)
    camera = torch.randn(1, 2, 64)
    camera_output, written = model.forward_with_cache(**inputs, camera_embedding=camera)
    assert not torch.equal(first, camera_output)
    disabled, _ = model.forward_with_cache(**inputs, camera_embedding=camera, camera_has_mask=torch.tensor([False]))
    torch.testing.assert_close(disabled, first, rtol=0, atol=0)
    committed, written = model.forward_with_cache(**inputs, update_cache=True)
    assert written == 5 and mask[0, :5].tolist() == [True, True, True, True, False]
    torch.testing.assert_close(committed, first, rtol=0, atol=0)
    history = cache[0]["k"][:, :5].clone()
    next_output, _ = model.forward_with_cache(**inputs, cache_valid_len=5, global_frame_counter=2, cache_txt=False)
    assert torch.isfinite(next_output).all() and torch.equal(cache[0]["k"][:, :5], history)
    assert len(model.attention.attention.masks) <= 32
    model.clear_attention_cache()
    assert not model.attention.attention.masks and not model.camera_attention.attention.masks


@torch.no_grad()
def test_range_attention_holes_matches_dense_math():
    torch.manual_seed(9)
    q, k, v = torch.randn(1, 4, 2, 8), torch.randn(1, 6, 2, 8), torch.randn(1, 6, 2, 8)
    qm, km = torch.tensor([[True, False, True, True]]), torch.tensor([[True, False, True, True, False, True]])
    actual = RangeAttention()(q, k, v, AttentionMetadata(extra={"query_mask": qm, "key_mask": km}))
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=(qm[:, :, None] & km[:, None, :])[:, None]
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(actual[:, 1]) == 0


def test_range_mask_exact_rectangles_and_padding():
    mask = build_range_mask(
        3, 4, [(0, 2, 0, 2), (2, 3, 2, 4)], "cpu", np.array([True, False, True]), np.array([True, True, False, True])
    )
    actual = mask.mask_mod(torch.tensor(0), torch.tensor(0), torch.arange(4)[:, None], torch.arange(5)[None])
    expected = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.equal(actual, expected)


def test_range_mask_rejects_overlapping_rectangles():
    with pytest.raises(ValueError, match="must not overlap"):
        build_range_mask(3, 4, [(0, 2, 0, 2), (1, 3, 1, 3)], "cpu", np.ones(3, dtype=bool), np.ones(4, dtype=bool))


@pytest.mark.parametrize(
    "update",
    [
        {"num_layers": 0},
        {"num_attention_heads": 0},
        {"rope_axes_dim": [16, 56, 55]},
        {"camera_rope_axes_dim": [8, 28, 30]},
    ],
)
def test_invalid_model_geometry_fails_before_allocation(update):
    with pytest.raises(ValueError, match="dimensions|RoPE"):
        HyNomasTransformer({**TRANSFORMER_CONFIG, **update}, device="meta")


@torch.no_grad()
def test_real_pixelumm_stream_matches_full_decode():
    torch.manual_seed(11)
    arch = GenExpertArch(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, num_kv_heads=1, head_dim=8)
    model = PixelUMMDCAELatentDecoder(
        DCAEVPixelUMMDecoder(arch), seed=7, temporal_window_latents=4, temporal_overlap_latents=2, decoder_tile_pixels=0
    ).eval()
    latent = torch.randn(1, 32, 7, 1, 1)
    full = model.decode(latent).sample
    session = model.create_streaming_decode_session(to_cpu=True, retain_history=False, rolling=True)
    parts = []
    for start, end in ((0, 2), (2, 4), (4, 7)):
        session.final_segment = end == 7
        part = session.push(latent[:, :, start:end], pixel_frames=4 * (end - start))
        if part.numel():
            parts.append(part)
    parts.append(session.finish())
    torch.testing.assert_close(torch.cat(parts, dim=2), full, rtol=0, atol=0)
    assert session.buffer.numel() == 0 and session.pending is None


@pytest.mark.parametrize("steps", [4, 6, 8])
def test_euler_matches_explicit_released_schedule(steps):
    solver = EulerSolver()
    solver.set_timesteps(steps)
    expected = torch.linspace(1, 0, steps + 1)
    expected = 5 * expected / (1 + 4 * expected)
    assert torch.equal(solver.timesteps, expected[:-1] * 1000)
    sample = torch.tensor([1.25], dtype=torch.bfloat16)
    for index, timestep in enumerate(solver.timesteps):
        velocity = torch.tensor([0.5], dtype=torch.bfloat16)
        result = solver.step(velocity, timestep, sample).prev_sample
        reference = sample.float() + velocity.float() * (expected[index + 1] - expected[index])
        assert torch.equal(result, reference)
        sample = result
