# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HiDream-O1-Image layout/bbox conditioning utilities.

All functions are pure (no learned parameters, no GPU), so these tests run
entirely on CPU without model weights.
"""

import json

import pytest
from PIL import Image as PILImage

from vllm_omni.diffusion.models.hidream_o1_image.utils_hidream_o1 import (
    DEFAULT_COLORS,
    MAX_BOX,
    add_outer_border_keep_size,
    create_layout_reference_images,
    draw_bbox_layout,
    load_layout_bboxes,
    parse_layout_bboxes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_W, _H = 512, 512  # synthetic output resolution for all tests


# ---------------------------------------------------------------------------
# load_layout_bboxes
# ---------------------------------------------------------------------------


class TestLoadLayoutBboxes:
    def test_load_from_json_string(self):
        data = [[0.1, 0.5, 0.1, 0.5]]
        result = load_layout_bboxes(json.dumps(data))
        assert result == data

    def test_load_from_json_file(self, tmp_path):
        data = [{"bbox": [0.1, 0.5, 0.1, 0.5], "text": "person"}]
        p = tmp_path / "bboxes.json"
        p.write_text(json.dumps(data))
        result = load_layout_bboxes(str(p))
        assert result == data

    def test_dict_wrapper_accepted(self):
        data = {"bboxes": [[0.0, 0.5, 0.0, 0.5]]}
        result = load_layout_bboxes(json.dumps(data))
        assert result == data


# ---------------------------------------------------------------------------
# parse_layout_bboxes
# ---------------------------------------------------------------------------


class TestParseLayoutBboxes:
    def test_relative_coords_converted_to_absolute(self):
        boxes = parse_layout_bboxes([[0.0, 0.5, 0.0, 0.5]], _W, _H)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]["bbox"]
        assert x1 == 0
        assert x2 == _W // 2
        assert y1 == 0
        assert y2 == _H // 2

    def test_percentage_coords_same_result_as_relative(self):
        boxes_rel = parse_layout_bboxes([[0.0, 0.5, 0.0, 0.5]], _W, _H)
        boxes_pct = parse_layout_bboxes([[0, 50, 0, 50]], _W, _H)
        assert boxes_rel[0]["bbox"] == boxes_pct[0]["bbox"]

    def test_xxyy_input_order_mapped_to_xyxy_output(self):
        # Input [x1, x2, y1, y2] = [0.1, 0.4, 0.2, 0.8]
        boxes = parse_layout_bboxes([[0.1, 0.4, 0.2, 0.8]], _W, _H)
        x1, y1, x2, y2 = boxes[0]["bbox"]
        assert x1 < x2
        assert y1 < y2
        assert x1 == round(0.1 * _W)
        assert x2 == round(0.4 * _W)
        assert y1 == round(0.2 * _H)
        assert y2 == round(0.8 * _H)

    def test_dict_items_accepted(self):
        boxes = parse_layout_bboxes([{"bbox": [0.0, 0.5, 0.0, 0.5], "text": "cat"}], _W, _H)
        assert boxes[0]["text"] == "cat"

    def test_dict_wrapper_unwrapped(self):
        raw = {"bboxes": [[0.0, 0.5, 0.0, 0.5]]}
        boxes = parse_layout_bboxes(raw, _W, _H)
        assert len(boxes) == 1

    def test_swapped_pairs_auto_sorted(self):
        # x1 > x2 in input — should be sorted so x1 < x2
        boxes = parse_layout_bboxes([[0.6, 0.1, 0.7, 0.2]], _W, _H)
        x1, y1, x2, y2 = boxes[0]["bbox"]
        assert x1 < x2
        assert y1 < y2

    def test_orig_idx_preserved(self):
        boxes = parse_layout_bboxes([[0.0, 0.3, 0.0, 0.3], [0.5, 0.9, 0.5, 0.9]], _W, _H)
        assert boxes[0]["_orig_idx"] == 0
        assert boxes[1]["_orig_idx"] == 1


# ---------------------------------------------------------------------------
# draw_bbox_layout
# ---------------------------------------------------------------------------


class TestDrawBboxLayout:
    def _simple_boxes(self, n: int) -> list[dict]:
        """Generate n non-overlapping boxes parsed to absolute coords."""
        step = 1.0 / (n + 1)
        raw = [[i * step, (i + 1) * step, 0.1, 0.9] for i in range(n)]
        return parse_layout_bboxes(raw, _W, _H)

    def test_returns_pil_image_of_correct_size(self):
        result = draw_bbox_layout(self._simple_boxes(2), _W, _H)
        assert isinstance(result, PILImage.Image)
        assert result.size == (_W, _H)

    def test_background_is_black(self):
        boxes = self._simple_boxes(1)
        # Use a very small image so we can check corner pixels easily
        result = draw_bbox_layout(boxes, 64, 64)
        pixels = list(result.getdata())
        # Top-left corner (0,0) should be black unless the box covers it
        assert pixels[0] == (0, 0, 0)

    def test_max_box_limit_respected(self):
        boxes = self._simple_boxes(MAX_BOX + 3)
        result, colors = draw_bbox_layout(boxes, _W, _H, return_color=True)
        # Colors beyond MAX_BOX rank should be None (not drawn)
        non_none = sum(1 for c in colors if c is not None)
        assert non_none <= MAX_BOX

    def test_return_color_gives_tuple(self):
        boxes = self._simple_boxes(2)
        result = draw_bbox_layout(boxes, _W, _H, return_color=True)
        assert isinstance(result, tuple) and len(result) == 2
        layout_img, color_list = result
        assert isinstance(layout_img, PILImage.Image)
        assert len(color_list) == 2

    def test_colors_are_from_default_palette(self):
        boxes = self._simple_boxes(3)
        _, color_list = draw_bbox_layout(boxes, _W, _H, return_color=True)
        for c in color_list:
            if c is not None:
                assert c in DEFAULT_COLORS


# ---------------------------------------------------------------------------
# add_outer_border_keep_size
# ---------------------------------------------------------------------------


class TestAddOuterBorderKeepSize:
    def test_output_size_unchanged(self):
        pil = PILImage.new("RGB", (64, 64), (128, 128, 128))
        result = add_outer_border_keep_size(pil, (255, 0, 0), width=4)
        assert result.size == pil.size

    def test_border_pixels_are_colored(self):
        pil = PILImage.new("RGB", (64, 64), (128, 128, 128))
        result = add_outer_border_keep_size(pil, (255, 0, 0), width=2)
        # Top-left corner pixel must be the border color
        assert result.getpixel((0, 0)) == (255, 0, 0)

    def test_center_pixel_unchanged(self):
        pil = PILImage.new("RGB", (64, 64), (100, 200, 50))
        result = add_outer_border_keep_size(pil, (255, 0, 0), width=4)
        assert result.getpixel((32, 32)) == (100, 200, 50)

    def test_zero_width_returns_original_colors(self):
        pil = PILImage.new("RGB", (32, 32), (77, 88, 99))
        result = add_outer_border_keep_size(pil, (255, 0, 0), width=0)
        assert result.getpixel((0, 0)) == (77, 88, 99)


# ---------------------------------------------------------------------------
# create_layout_reference_images
# ---------------------------------------------------------------------------


class TestCreateLayoutReferenceImages:
    def _refs(self, n: int) -> list[PILImage.Image]:
        return [PILImage.new("RGB", (64, 64), (i * 30, 0, 0)) for i in range(n)]

    def _bboxes(self, n: int):
        step = 1.0 / (n + 1)
        return [[i * step, (i + 1) * step, 0.1, 0.9] for i in range(n)]

    def test_output_length_is_refs_plus_one(self):
        refs = self._refs(2)
        result = create_layout_reference_images(refs, self._bboxes(2), _W, _H)
        assert len(result) == 3  # 2 refs + 1 layout

    def test_all_outputs_are_pil_images(self):
        refs = self._refs(2)
        result = create_layout_reference_images(refs, self._bboxes(2), _W, _H)
        for img in result:
            assert isinstance(img, PILImage.Image)

    def test_last_image_is_layout_black_background(self):
        refs = self._refs(1)
        result = create_layout_reference_images(refs, self._bboxes(1), _W, _H)
        layout_img = result[-1]
        assert layout_img.size == (_W, _H)
        # Top-left corner should be black (outside any bbox)
        assert layout_img.getpixel((0, 0)) == (0, 0, 0)

    def test_ref_images_have_colored_borders(self):
        refs = self._refs(1)
        result = create_layout_reference_images(refs, self._bboxes(1), _W, _H)
        bordered = result[0]
        # Border pixels (top-left corner) must not be the original fill color (0, 0, 0)
        border_pixel = bordered.getpixel((0, 0))
        # The border color comes from DEFAULT_COLORS, so it should be one of them
        assert border_pixel in DEFAULT_COLORS

    def test_single_ref_with_json_string_input(self):
        refs = self._refs(1)
        json_str = json.dumps([[0.1, 0.5, 0.1, 0.5]])
        result = create_layout_reference_images(refs, json_str, _W, _H)
        assert len(result) == 2
