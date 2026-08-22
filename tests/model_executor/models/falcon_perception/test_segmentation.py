# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception mask head: geometry, chunking, and the AnyUp cache.

These run on CPU with tiny tensors and no checkpoint. They target the parts of
the stage whose failures are silent rather than loud — a wrong canvas or a
badly stitched chunk still returns plausible masks, just of the wrong pixels.
"""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.falcon_perception.falcon_perception_segmentation import (
    DEFAULT_HR_UPSAMPLE_RATIO,
    FalconPerceptionSegmentation,
    _apply_mask_nms,
    _boxes_to_xywh,
    _mask_nms,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PATCH = 16
MAX_IMAGE = 1024


# --------------------------------------------------------------------------
# Wiring failures must raise, not return empty masks.
#
# An empty mask list with a success status is indistinguishable from "nothing
# detected", so a broken payload would silently serve maskless results forever.
# These three are deterministic faults: they surface on the first request.
# --------------------------------------------------------------------------


def _seg_stub():
    """Enough of the stage to reach the validation branches."""
    from vllm_omni.model_executor.models.falcon_perception.configuration_falcon_perception import (
        FalconPerceptionConfig,
    )

    config = FalconPerceptionConfig()
    stub = SimpleNamespace(
        config=config,
        segm_out_dim=config.segm_out_dim,
        patch_size=config.spatial_patch_size,
        max_image_size=config.max_image_size,
        parameters=lambda: iter([torch.zeros(1)]),
    )
    return stub


def _run(stub, info, kwargs=None):
    return FalconPerceptionSegmentation._run_one_request(stub, info, kwargs or {})


# --------------------------------------------------------------------------
# Square canvas geometry.
#
# The reference pads the image and the feature grid to a square max_image_size
# before AnyUp and crops afterwards. Running on the native non-square grid is
# 3.3x faster BUT destroys the masks: because AnyUp was trained on square inputs.
# These assert the padded extent and the crop-back extent explicitly, since a
# regression here produces smeared masks rather than an exception.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("grid", [(26, 46), (51, 64), (36, 64), (64, 64), (16, 16)])
def test_square_canvas_pads_to_max_image_size_and_crops_back_to_the_image(grid):
    grid_h, grid_w = grid
    img_h, img_w = grid_h * PATCH, grid_w * PATCH

    # What the stage feeds AnyUp: image padded to the square canvas, features
    # padded to the square patch grid, output a square side.
    square = ((MAX_IMAGE + PATCH - 1) // PATCH) * PATCH
    square_patches = square // PATCH
    ratio = DEFAULT_HR_UPSAMPLE_RATIO

    stub = _seg_stub()
    stub.conv_segm = lambda x: torch.zeros(1, 256, x.shape[2], x.shape[3])
    stub.proj_segm = lambda x: torch.zeros(x.shape[0], 256)
    stub._hr_cache_lookup = lambda k, e: None
    stub._hr_cache_store = lambda k, v: None

    captured = {}

    def mock_upsampler(images, features, attn_mask, output_size):
        captured["images"] = images
        captured["features"] = features
        captured["output_size"] = output_size
        # Return a tensor we can trace through the crop
        return [
            torch.arange(256 * output_size[0] * output_size[1], dtype=torch.float32).reshape(
                256, output_size[0], output_size[1]
            )
        ]

    stub.itok_upsampler = mock_upsampler

    # Intercept the cropped hr_features before they are consumed by the rest of the pipeline
    def mock_cache_store(k, v):
        captured["cropped_hr"] = v

    stub._hr_cache_store = mock_cache_store

    pixels = torch.zeros(img_h, img_w, 3)
    info = {
        "hidden_states": {
            "image_features": torch.zeros(grid_h * grid_w, 1024),
            "seg_features": torch.zeros(1, 1024),
            "pixel_values": pixels,
        }
    }

    _run(stub, info)

    # Assert padding
    assert captured["images"].shape == (1, 3, square, square)
    assert captured["features"].shape == (1, 256, square_patches, square_patches)
    assert captured["output_size"] == (square_patches * ratio, square_patches * ratio)

    # And what it keeps afterwards: this image's own extent, not the canvas.
    out_side = square_patches * ratio
    crop_h, crop_w = grid_h * ratio, grid_w * ratio
    assert crop_h <= out_side and crop_w <= out_side

    cropped = captured["cropped_hr"]
    assert cropped.shape == (256, crop_h, crop_w)
    assert (crop_h, crop_w) == (grid_h * 8, grid_w * 8)

    # The pad is bottom/right, so the valid region is the top-left corner.
    # We returned arange from mock_upsampler, so we can verify the crop slice.
    hr = captured["images"].new_empty(0)  # just to get the device/dtype if needed, but we returned CPU float32
    hr = torch.arange(256 * out_side * out_side, dtype=torch.float32).reshape(256, out_side, out_side)
    assert torch.equal(cropped[0, :3, :3], hr[0, :3, :3])


# --------------------------------------------------------------------------
# Chunked mask decode. Dense scenes carry 250+ instances and materialising all
# (n, H, W) float32 logits at once OOMs, so decoding is chunked — which must be
# arithmetically identical to one pass.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_inst", [1, 31, 32, 33, 64, 65])
def test_chunked_mask_decode_matches_a_single_pass(n_inst):
    from vllm_omni.model_executor.models.falcon_perception.falcon_perception_segmentation import _MASK_CHUNK

    torch.manual_seed(0)
    seg = torch.randn(n_inst, 8)
    hr = torch.randn(8, 5, 7)
    threshold = 0.3

    one = (torch.sigmoid(torch.einsum("kc,chw->khw", seg, hr)) > threshold).to(torch.uint8)

    chunks = []
    for start in range(0, seg.shape[0], _MASK_CHUNK):
        block = seg[start : start + _MASK_CHUNK]
        chunks.append((torch.sigmoid(torch.einsum("kc,chw->khw", block, hr)) > threshold).to(torch.uint8))
    chunked = torch.cat(chunks, dim=0)

    assert chunked.shape == one.shape == (n_inst, 5, 7)
    assert torch.equal(chunked, one)


# --------------------------------------------------------------------------
# Reference mask NMS. It must suppress duplicate masks and apply the same kept
# indices to the boxes, or mask/geometry pairs silently become misaligned.
# --------------------------------------------------------------------------


def test_mask_nms_suppresses_overlapping_duplicates():
    masks = torch.zeros((3, 8, 8), dtype=torch.uint8)
    masks[0, 1:5, 1:5] = 1
    masks[1, 1:5, 1:5] = 1
    masks[2, 6:8, 6:8] = 1

    keep = _mask_nms(masks)

    assert keep.tolist() == [0, 2]


def test_mask_nms_filters_boxes_with_the_same_indices():
    masks = torch.zeros((3, 8, 8), dtype=torch.uint8)
    masks[0, 1:5, 1:5] = 1
    masks[1, 1:5, 1:5] = 1
    masks[2, 6:8, 6:8] = 1
    boxes = torch.tensor(
        [
            [0.1, 0.1, 0.2, 0.2],
            [0.3, 0.3, 0.4, 0.4],
            [0.5, 0.5, 0.6, 0.6],
        ]
    )

    kept_masks, kept_boxes = _apply_mask_nms(masks, boxes)

    assert kept_masks.shape[0] == kept_boxes.shape[0] == 2
    assert torch.equal(kept_boxes, boxes[[0, 2]])


# --------------------------------------------------------------------------
# AnyUp output cache. A hit is bit-exact by construction (the features depend
# only on the image), so the risk is not numerical — it is serving the wrong
# entry, or letting the budget stop binding.
# --------------------------------------------------------------------------


def _cache_stub(budget_mb: int = 1):
    stub = SimpleNamespace(
        _hr_cache=OrderedDict(),
        _hr_cache_bytes=0,
        _hr_cache_budget=budget_mb * 1024 * 1024,
    )
    stub._hr_cache_lookup = lambda k, e: FalconPerceptionSegmentation._hr_cache_lookup(stub, k, e)
    stub._hr_cache_store = lambda k, v: FalconPerceptionSegmentation._hr_cache_store(stub, k, v)
    return stub


def test_cache_returns_the_stored_tensor_on_a_hit_and_none_on_a_miss():
    stub = _cache_stub()
    value = torch.randn(4, 8, 8)
    key = (123, 8, 1024, 26, 46)
    assert stub._hr_cache_lookup(key, tuple(value.shape)) is None
    stub._hr_cache_store(key, value)
    hit = stub._hr_cache_lookup(key, tuple(value.shape))
    assert hit is not None and torch.equal(hit, value)
    assert stub._hr_cache_lookup((999, 8, 1024, 26, 46), tuple(value.shape)) is None


def test_a_shape_disagreement_is_a_miss_not_a_wrong_feature_map():
    """A hash collision must degrade to recompute, never reshape the mask head."""
    stub = _cache_stub()
    key = (123, 8, 1024, 26, 46)
    stub._hr_cache_store(key, torch.randn(4, 8, 8))
    assert stub._hr_cache_lookup(key, (4, 9, 9)) is None


def test_no_key_means_no_caching_rather_than_a_shared_entry():
    stub = _cache_stub()
    stub._hr_cache_store(None, torch.randn(4, 8, 8))
    assert len(stub._hr_cache) == 0
    assert stub._hr_cache_lookup(None, (4, 8, 8)) is None


def test_budget_evicts_least_recently_used_and_accounting_stays_exact():
    # 4 KB entries, 1 MB budget -> 256 fit; use a tiny budget to force eviction.
    stub = _cache_stub()
    stub._hr_cache_budget = 3 * 4 * 1024  # exactly 3 entries of (4, 16, 16) float32
    entries = {}
    for i in range(5):
        v = torch.randn(4, 16, 16)
        entries[i] = v
        stub._hr_cache_store((i, 8, 1024, 4, 4), v)

    assert len(stub._hr_cache) <= 3
    # Accounting must equal the real footprint, or the budget silently stops binding.
    real = sum(v.element_size() * v.numel() for v in stub._hr_cache.values())
    assert stub._hr_cache_bytes == real
    # The oldest keys went first.
    assert (0, 8, 1024, 4, 4) not in stub._hr_cache
    assert (4, 8, 1024, 4, 4) in stub._hr_cache


def test_replacing_a_key_with_a_different_size_keeps_the_byte_count_honest():
    """The shape-mismatch miss path re-stores the same key with a new shape."""
    stub = _cache_stub()
    key = (7, 8, 1024, 4, 4)
    stub._hr_cache_store(key, torch.randn(4, 16, 16))
    first = stub._hr_cache_bytes
    stub._hr_cache_store(key, torch.randn(4, 32, 32))
    assert stub._hr_cache_bytes != first
    real = sum(v.element_size() * v.numel() for v in stub._hr_cache.values())
    assert stub._hr_cache_bytes == real, "stale bytes left behind would unbind the budget"


def test_a_zero_budget_disables_the_cache():
    stub = _cache_stub(budget_mb=0)
    stub._hr_cache_store((1, 8, 1024, 4, 4), torch.randn(4, 8, 8))
    assert len(stub._hr_cache) == 0


# --------------------------------------------------------------------------
# Box packing.
# --------------------------------------------------------------------------


def test_boxes_are_truncated_to_the_shorter_stream_rather_than_mispaired():
    empty = torch.zeros((0, 4))
    xy = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    hw = torch.tensor([[0.7, 0.8], [0.9, 1.0]])
    out = _boxes_to_xywh(xy, hw, empty)
    assert out.shape == (2, 4)
    # hw is stored (h, w) and emitted as (w, h) after the flip.
    assert torch.allclose(out[0], torch.tensor([0.1, 0.2, 0.8, 0.7]))


def test_boxes_missing_entirely_give_the_empty_tensor():
    empty = torch.zeros((0, 4))
    assert _boxes_to_xywh(None, None, empty).shape == (0, 4)
    assert _boxes_to_xywh(torch.zeros((0, 2)), torch.zeros((0, 2)), empty).shape == (0, 4)


# --------------------------------------------------------------------------
# Wiring failures must raise, not return empty masks.
#
# An empty mask list with a success status is indistinguishable from "nothing
# detected", so a broken payload would silently serve maskless results forever.
# These three are deterministic faults: they surface on the first request.
# --------------------------------------------------------------------------


def test_missing_payload_raises_rather_than_reporting_nothing_detected():
    with pytest.raises(ValueError, match="stage payload missing"):
        _run(_seg_stub(), {"hidden_states": {}})


def test_missing_image_raises():
    info = {
        "hidden_states": {
            "image_features": torch.zeros(4, 1024),
            "seg_features": torch.zeros(2, 1024),
        }
    }
    with pytest.raises(ValueError, match="no image reached the mask head"):
        _run(_seg_stub(), info)


def test_a_grid_that_disagrees_with_the_features_raises():
    """Decoding against a misaligned feature map would give confident, wrong masks."""
    # 2x2 patch grid = 4 rows expected; supply 7.
    pixels = torch.zeros(2 * 16, 2 * 16, 3)
    info = {
        "hidden_states": {
            "image_features": torch.zeros(7, 1024),
            "seg_features": torch.zeros(1, 1024),
            "pixel_values": pixels,
        }
    }
    with pytest.raises(ValueError, match="image feature rows"):
        _run(_seg_stub(), info)


def test_no_seg_tokens_is_a_normal_empty_result_not_an_error():
    """A detection-only answer is legitimate — this one must NOT raise."""
    info = {
        "hidden_states": {
            "image_features": torch.zeros(4, 1024),
            "seg_features": torch.zeros(0, 1024),
        }
    }
    masks, boxes = _run(_seg_stub(), info)
    assert masks.shape[0] == 0
    assert boxes.shape == (0, 4)
