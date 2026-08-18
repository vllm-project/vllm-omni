# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SenseNova-Vision dense-image decoders."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vllm_omni.model_executor.models.sensenova.decoders.dense_decoders import (
    decode_depth,
    decode_normal,
    decode_point_map,
    decode_segmentation,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_decode_segmentation_binary():
    arr = ((np.arange(16 * 16).reshape(16, 16) * 16) % 256).astype(np.uint8)
    mask = decode_segmentation(arr)
    assert mask.shape == (16, 16)
    assert set(np.unique(mask)).issubset({0, 1})


def test_decode_segmentation_binary_threshold():
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[:, :4] = 200
    mask = decode_segmentation(arr, threshold=127)
    assert (mask[:, :4] == 1).all()
    assert (mask[:, 4:] == 0).all()


def test_decode_segmentation_rgb_classes():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :4, 0] = 255  # red class 0
    rgb[:, 4:, 1] = 255  # green class 1
    class_define = [(255, 0, 0), (0, 255, 0)]
    mask = decode_segmentation(rgb, class_define=class_define)
    assert (mask[:, :4] == 0).all()
    assert (mask[:, 4:] == 1).all()


def test_decode_segmentation_rgb_black_background():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    class_define = [(255, 0, 0)]
    mask = decode_segmentation(rgb, class_define=class_define)
    assert mask[0, 0] == 0
    assert (mask[1:, :] == 1).all()
    assert (mask[0, 1:] == 1).all()


def test_decode_segmentation_accepts_pil(tmp_path):
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[2:, :] = (0, 255, 0)
    img = Image.fromarray(rgb)
    p = tmp_path / "mask.png"
    img.save(p)
    for source in (img, rgb, str(p)):
        mask = decode_segmentation(source, class_define=[(255, 0, 0), (0, 255, 0)])
        assert (mask[2:, :] == 1).all()


def test_decode_segmentation_invalid_palette():
    with pytest.raises(ValueError):
        decode_segmentation(np.zeros((4, 4, 3), dtype=np.uint8), class_define=[[1, 2]])


def test_decode_depth_grayscale():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:, :, :] = 128
    depth = decode_depth(arr)
    assert depth.shape == (8, 8)
    assert np.allclose(depth, 128.0 / 255.0)


def test_decode_depth_2d_input():
    arr = np.full((6, 6), 64, dtype=np.uint8)
    depth = decode_depth(arr)
    assert depth.shape == (6, 6)
    assert np.allclose(depth, 64.0 / 255.0)


def test_decode_depth_resize():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:] = 255
    depth = decode_depth(arr, size=(8, 8))
    assert depth.shape == (8, 8)
    assert np.allclose(depth, 1.0)

    img = Image.fromarray(arr)
    depth_pil = decode_depth(img, size=(8, 8))
    assert depth_pil.shape == (8, 8)
    assert np.allclose(depth_pil, 1.0)


def test_decode_normal_rgb():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:, :, 2] = 255  # +z normal encoded as (0,0,255) in RGB
    normals = decode_normal(arr)
    assert normals.shape == (4, 4, 3)
    # +z: map (0,0,255)/255*2-1 = (-1,-1,1), then flip x -> (1,-1,1)
    assert np.allclose(normals[:, :, 0], 1.0)
    assert np.allclose(normals[:, :, 1], -1.0)
    assert np.allclose(normals[:, :, 2], 1.0)


def test_decode_normal_no_flip_x():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:, :, 2] = 255
    normals = decode_normal(arr, flip_x=False)
    assert np.allclose(normals[:, :, 0], -1.0)


def test_decode_normal_resize():
    arr = np.full((4, 4, 3), 128, dtype=np.uint8)
    normals = decode_normal(arr, size=(2, 2))
    assert normals.shape == (2, 2, 3)

    img = Image.fromarray(arr)
    normals_pil = decode_normal(img, size=(2, 2))
    assert normals_pil.shape == (2, 2, 3)


def test_decode_point_map_float_array_passthrough():
    arr = np.random.default_rng(0).uniform(-1, 1, size=(4, 4, 3)).astype(np.float32)
    out = decode_point_map(arr)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.float32
    assert np.array_equal(out, arr)


def test_decode_point_map_uint8_scale():
    arr = np.full((4, 4, 3), 128, dtype=np.uint8)
    out = decode_point_map(arr)
    assert out.dtype == np.float32
    # (128/255*2-1) ~= 0.0039 (float32 tolerance)
    assert np.allclose(out, 128.0 / 255.0 * 2.0 - 1.0, atol=1e-6)


def test_decode_point_map_float_in_uint8_range_scaled():
    arr = np.full((4, 4, 3), 255.0, dtype=np.float32)
    out = decode_point_map(arr)
    assert np.allclose(out, 1.0)


def test_decode_point_map_invalid_shape():
    with pytest.raises(ValueError):
        decode_point_map(np.zeros((4, 4), dtype=np.float32))
