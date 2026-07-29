# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure tensor/PIL helpers for HiDream-O1-Image.

Ported (near-verbatim where noted) from the reference repo
(github.com/HiDream-ai/HiDream-O1-Image, ``models/utils.py``). These
functions have no learned parameters and no vLLM-Omni-specific
dependencies, so they are ported as-is rather than re-derived.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from typing import Any

import einops
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

PATCH_SIZE = 32


def find_closest_resolution(width: int, height: int, patch_size: int = PATCH_SIZE) -> tuple[int, int]:
    """Snap an arbitrary (width, height) to the nearest patch-aligned resolution
    that preserves total pixel count area as closely as possible while keeping
    the aspect ratio close to the requested one.

    Ported from the reference repo's ``find_closest_resolution``. Both
    dimensions must be divisible by ``patch_size`` for patchify/depatchify to
    round-trip cleanly.
    """
    if width % patch_size == 0 and height % patch_size == 0:
        return width, height

    aspect_ratio = width / height
    area = width * height

    h = round(math.sqrt(area / aspect_ratio) / patch_size) * patch_size
    w = round(h * aspect_ratio / patch_size) * patch_size
    h = max(patch_size, h)
    w = max(patch_size, w)
    return w, h


def calculate_dimensions(max_size: int, ratio: float) -> tuple[int, int]:
    """Given a max side length and a target aspect ratio (width / height),
    return (width, height) patch-aligned dimensions whose longer side is
    ``max_size``.
    """
    if ratio >= 1.0:
        w = max_size
        h = round(max_size / ratio / PATCH_SIZE) * PATCH_SIZE
    else:
        h = max_size
        w = round(max_size * ratio / PATCH_SIZE) * PATCH_SIZE
    return max(PATCH_SIZE, w), max(PATCH_SIZE, h)


def resize_pilimage(pil_image: Image.Image, image_size: int, patch_size: int = PATCH_SIZE) -> Image.Image:
    """Resize ``pil_image`` so its longer side equals ``image_size``, then
    round both dimensions down to the nearest multiple of ``patch_size``.
    """
    w, h = pil_image.size
    ratio = w / h
    new_w, new_h = calculate_dimensions(image_size, ratio)
    return pil_image.resize((new_w, new_h), resample=Image.LANCZOS)


def patchify(image: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """``(B, C, H, W)`` pixel tensor -> ``(B, H/P * W/P, C*P*P)`` patch tokens.

    No VAE is used by this model; patches are raw pixel blocks.
    """
    return einops.rearrange(image, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size)


def depatchify(patches: torch.Tensor, h_patches: int, w_patches: int, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Inverse of :func:`patchify`: ``(B, H/P * W/P, C*P*P)`` -> ``(B, C, H, W)``."""
    return einops.rearrange(
        patches,
        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
        h=h_patches,
        w=w_patches,
        p1=patch_size,
        p2=patch_size,
    )


def get_rope_index_fix_point(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: torch.LongTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    skip_vision_start_token: list[int] | None = None,
    fix_point: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D MRoPE (T, H, W) position ids for a packed text+image sequence.

    This is a near-verbatim port of the reference repo's
    ``get_rope_index_fix_point`` (a modified version of the stock Qwen-VL
    ``get_rope_index``). The key difference from stock Qwen-VL: any image
    span whose corresponding ``skip_vision_start_token[i]`` flag is True
    (used for the target image being generated, and for every reference
    image) is anchored at a separate, fixed ``fix_point`` offset instead of
    being laid out contiguously right after the preceding text span. This
    keeps long text prompts and large/many images from colliding or
    extrapolating into position ranges the checkpoint's RoPE table wasn't
    trained on. Do not "simplify" this anchoring away -- see Phase 0 design
    note in the HiDream-O1-Image roadmap.

    Returns:
        position_ids: ``(3, batch, seq_len)`` long tensor (T, H, W channels).
        mrope_position_deltas: ``(batch, 1)`` long tensor (unused by this
            pipeline today, kept for parity with the upstream signature).
    """
    if skip_vision_start_token is None:
        skip_vision_start_token = []

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if image_grid_thw is None and video_grid_thw is None:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas

    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)
    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    image_index, video_index = 0, 0
    attention_mask = attention_mask.to(total_input_ids.device)
    for i, ids_row in enumerate(total_input_ids):
        ids_row = ids_row[attention_mask[i] == 1]
        vision_start_indices = torch.argwhere(ids_row == vision_start_token_id).squeeze(1)
        vision_tokens = ids_row[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum())
        video_nums = int((vision_tokens == video_token_id).sum())
        input_tokens = ids_row.tolist()
        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t, llm_grid_h, llm_grid_w = (
                int(t),
                int(h) // spatial_merge_size,
                int(w) // spatial_merge_size,
            )
            text_len = ed - st
            text_len -= skip_vision_start_token[image_index - 1] if skip_vision_start_token else 0
            text_len = max(0, text_len)

            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

            use_fix_point = bool(skip_vision_start_token[image_index - 1]) if skip_vision_start_token else False
            if use_fix_point:
                anchor = fix_point - st_idx if fix_point > 0 else fix_point
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + anchor + st_idx)
                fix_point = 0
            else:
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(int(llm_positions.max()) + 1 - len(total_input_ids[i]))

    mrope_position_deltas_t = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas_t


def build_packed_attention_metadata(
    token_types: torch.Tensor,
) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
    """Build BOTH masking representations from one ``token_types`` tensor.

    Per the Phase 0 design lock: vLLM-Omni's flash-attention backend only
    honors ``AttentionMetadata.full_attn_spans``; the SDPA backend only
    honors ``AttentionMetadata.attn_mask``. Building both from the same
    source data lets each backend pick the representation it understands,
    with no silent fallback to "no masking at all" on non-flash backends.

    Convention (matches the reference repo): ``token_types == 0`` is text
    (causal-only-over-text); any ``token_types > 0`` value is a
    generation/conditioning token (timestep, target-image patch, or
    reference-image patch) that attends bidirectionally over the whole
    sequence. This function does not distinguish between contiguous spans
    (single trailing image vs. multiple ref-image spans) -- it derives
    spans directly from contiguous runs of ``token_types > 0``, so it
    already generalizes to Phase 2's multi-reference-image case.

    Args:
        token_types: ``(batch, seq_len)`` tensor, 0 = causal/text, >0 = gen.

    Returns:
        dense_mask: ``(batch, 1, seq_len, seq_len)`` boolean mask, True =
            allowed to attend. Causal lower-triangular among text positions,
            full attention (every row True) for gen-token rows.
        full_attn_spans: per-sample list of half-open ``(start, end)``
            spans (in this sample's own sequence coordinates) covering
            contiguous runs of gen tokens, sorted by start.
    """
    batch_size, seq_len = token_types.shape
    device = token_types.device

    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(diagonal=0)
    dense_mask = causal.unsqueeze(0).repeat(batch_size, 1, 1)

    full_attn_spans: list[list[tuple[int, int]]] = [[] for _ in range(batch_size)]
    is_gen = token_types > 0
    for b in range(batch_size):
        dense_mask[b, is_gen[b]] = True  # gen rows attend to everything (incl. future gen positions)
        row = is_gen[b]
        idx = torch.nonzero(row, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        # Group contiguous indices into spans.
        breaks = torch.nonzero(idx[1:] - idx[:-1] > 1, as_tuple=False).squeeze(-1) + 1
        starts = torch.cat([idx[:1], idx[breaks]]) if breaks.numel() else idx[:1]
        ends = torch.cat([idx[breaks - 1] if breaks.numel() else idx[:0], idx[-1:]]) if breaks.numel() else idx[-1:]
        ends = ends + 1  # half-open
        full_attn_spans[b] = [(int(s), int(e)) for s, e in zip(starts.tolist(), ends.tolist())]
        full_attn_spans[b].sort(key=lambda x: x[0])

    return dense_mask.unsqueeze(1), full_attn_spans


# ---------------------------------------------------------------------------
# Phase 2 — Reference image preprocessing helpers
# ---------------------------------------------------------------------------

# Normalization transform matching the reference repo's ``TENSOR_TRANSFORM``
# (normalize raw [0,255] pixels to the [-1, 1] range expected by the model).
_REF_TRANSFORM = T.Compose(
    [
        T.ToTensor(),  # [0,255] HWC → [0,1] CHW
        T.Normalize([0.5], [0.5]),  # [0,1] → [-1,1] per channel
    ]
)


def adaptive_ref_max_size(k: int, output_max: int) -> int:
    """Return the max side-length for k reference images at the given output resolution.

    Mirrors the reference repo's pipeline.py adaptive sizing table so that
    total ref-image token count grows sub-linearly with k, keeping sequence
    length reasonable for personalization with many references.
    """
    if k == 1:
        return output_max
    if k == 2:
        return output_max * 48 // 64
    if k <= 4:
        return output_max // 2
    if k <= 8:
        return output_max * 24 // 64
    return output_max // 4


def preprocess_ref_patches(
    pil_list: list[Image.Image],
    max_size: int,
    patch_size: int = PATCH_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, list[int]]:
    """Convert a list of reference PIL images into patchified pixel tensors.

    Each image is resized (aspect-ratio preserving, snapped to ``patch_size``
    multiples), normalized to ``[-1, 1]``, then patchified. All patches are
    concatenated into a single ``[1, total_tokens, C*patch_size*patch_size]``
    tensor, matching the format that ``forward_generation`` expects for the
    ``vinputs`` concat.

    Args:
        pil_list: raw reference PIL images (RGB).
        max_size: maximum side length (from ``adaptive_ref_max_size``).
        patch_size: spatial patch size (always 32 for this model).
        dtype: target tensor dtype.
        device: target device.

    Returns:
        ref_patches: ``[1, total_tokens, C*P*P]`` float tensor.
        ref_image_lens: list of patch-token counts per image (one per ref).
    """
    patches_list = []
    ref_image_lens: list[int] = []
    for pil in pil_list:
        pil_r = resize_pilimage(pil.convert("RGB"), max_size, patch_size)
        x = _REF_TRANSFORM(pil_r)  # [C, H, W] in [-1, 1]
        w, h = pil_r.size
        hp, wp = h // patch_size, w // patch_size
        # patchify single image: [1, C, H, W] → [1, hp*wp, C*P*P]
        xp = patchify(x.unsqueeze(0), patch_size=patch_size)
        patches_list.append(xp)
        ref_image_lens.append(hp * wp)
    ref_patches = torch.cat(patches_list, dim=1).to(device=device, dtype=dtype)
    return ref_patches, ref_image_lens


# ---------------------------------------------------------------------------
# Phase 3 — Layout / bbox conditioning helpers
# ---------------------------------------------------------------------------
# Ported near-verbatim from the reference repo's models/utils.py.
# Layout conditioning reuses the Phase 2 ref-image path exactly:
# create_layout_reference_images() augments ref_pils with bordered versions +
# a layout image, which are then fed through _build_edit_sample() unchanged
# as token_type=2 conditioning tokens.

MAX_BOX = 5
DEFAULT_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 180, 0),
    (0, 0, 255),
    (204, 180, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
]


def _unwrap_boxes(data: Any) -> Any:
    if isinstance(data, dict):
        for key in ("layout_bboxes", "bboxes", "boxes", "bbox_list"):
            if key in data:
                return data[key]
    return data


def _as_bbox_and_text(item: Any) -> tuple[Sequence[float], str]:
    if isinstance(item, dict):
        bbox = item.get("bbox") or item.get("box")
        text = str(item.get("text") or item.get("label") or "")
        if bbox is None:
            raise ValueError(f"Missing bbox in layout item: {item!r}")
        return bbox, text
    if isinstance(item, (list, tuple)) and len(item) == 4:
        return item, ""
    raise ValueError(f"Unsupported layout bbox item: {item!r}")


def _xxyy_relative_to_absolute_bbox(bbox: Sequence[float], width: int, height: int) -> list[int]:
    """Convert xxyy relative coordinates to absolute xyxy pixel coordinates.

    Input order is [x1, x2, y1, y2] (xxyy, NOT xyxy). Values in [0, 1] are
    treated as relative; values in [0, 100] as percentages. Returns
    [x1, y1, x2, y2] in absolute pixels (xyxy order) with x1 < x2, y1 < y2.
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got: {bbox!r}")
    x1, x2, y1, y2 = [float(v) for v in bbox]
    max_abs = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_abs <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif max_abs <= 100.0:
        x1, x2 = x1 / 100.0 * width, x2 / 100.0 * width
        y1, y2 = y1 / 100.0 * height, y2 / 100.0 * height
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after scaling/clamping: {[x1, y1, x2, y2]!r}")
    return [x1, y1, x2, y2]


def _bbox_area(item: dict[str, Any]) -> int:
    x1, y1, x2, y2 = item["bbox"]
    return max(0, x2 - x1) * max(0, y2 - y1)


def get_render_params(image_width: int, image_height: int) -> tuple[int, int]:
    """Return (max_font_size, max_bbox_line_width) scaled to image resolution."""
    edge = math.sqrt(image_width * image_height)
    return int(edge * 0.07), int(edge * 0.05)


def load_layout_bboxes(layout_bboxes: str) -> Any:
    """Load layout boxes from a JSON string or a JSON file path."""
    if os.path.exists(layout_bboxes):
        with open(layout_bboxes, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(layout_bboxes)


def parse_layout_bboxes(
    layout_bboxes: Any,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Parse raw bbox data into absolute-pixel dicts for ``draw_bbox_layout``.

    Accepted input formats (all equivalent):
    - Plain list:        ``[[x1,x2,y1,y2], ...]``
    - List of dicts:     ``[{"bbox": [x1,x2,y1,y2], "text": "..."}, ...]``
    - Dict wrapper:      ``{"bboxes": [...]}`` (also "boxes", "bbox_list", "layout_bboxes")

    Coordinates are in **xxyy** order (not xyxy), values in [0, 1] (relative)
    or [0, 100] (percentage). Returns a list of dicts with an ``"bbox"`` key
    holding absolute xyxy pixel coordinates.
    """
    raw_boxes = _unwrap_boxes(layout_bboxes)
    if not isinstance(raw_boxes, list):
        raise ValueError(
            "layout_bboxes must be a list, or a dict containing one of: layout_bboxes/bboxes/boxes/bbox_list"
        )
    parsed: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_boxes):
        bbox, text = _as_bbox_and_text(item)
        parsed.append(
            {
                "bbox": _xxyy_relative_to_absolute_bbox(bbox, width, height),
                "color": "",
                "text": text,
                "image": None,
                "_orig_idx": idx,
            }
        )
    return parsed


def draw_bbox_layout(
    bbox_list: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    max_bbox: int = MAX_BOX,
    max_bbox_line_width: int | None = None,
    bbox_line_gap: int | None = None,
    return_color: bool = False,
) -> Image.Image | tuple[Image.Image, list[tuple[int, int, int] | None]]:
    """Draw a black layout image with colored bounding boxes.

    Boxes are sorted by area (largest first) and only the top ``max_bbox``
    are drawn. Each box gets a distinct color from ``DEFAULT_COLORS``.
    The ``_orig_idx`` field in each item maps the sorted color assignment
    back to the original box order so ``create_layout_reference_images``
    can draw matching borders on the corresponding ref images.

    If ``return_color=True``, returns ``(layout_image, color_list)`` where
    ``color_list[i]`` is the color assigned to the i-th *original* box.
    """
    if max_bbox_line_width is None:
        _, max_bbox_line_width = get_render_params(image_width, image_height)
    if bbox_line_gap is None:
        bbox_line_gap = max(1, max_bbox_line_width // max_bbox)

    image = Image.new("RGB", (image_width, image_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    color_list: list[tuple[int, int, int] | None] = [None] * len(bbox_list)
    sorted_bboxes = sorted(bbox_list, key=_bbox_area, reverse=True)[:max_bbox]

    for sorted_idx, item in enumerate(sorted_bboxes):
        color = DEFAULT_COLORS[sorted_idx % len(DEFAULT_COLORS)]
        orig_idx = int(item.get("_orig_idx", sorted_idx))
        if 0 <= orig_idx < len(color_list):
            color_list[orig_idx] = color
        line_width = max(max_bbox_line_width - sorted_idx * bbox_line_gap, 5)
        draw.rectangle([int(v) for v in item["bbox"]], outline=color, width=line_width)

    if return_color:
        return image, color_list
    return image


def add_outer_border_keep_size(
    pil: Image.Image,
    color: Sequence[int],
    width: int,
) -> Image.Image:
    """Draw a solid border inside ``pil`` without changing its canvas size."""
    img = pil.convert("RGB").copy()
    color_tuple = tuple(int(c) for c in color)
    width = max(0, int(width))
    if width == 0:
        return img
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for t in range(width):
        draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=color_tuple)
    return img


def create_layout_reference_images(
    ref_pils: list[Image.Image],
    layout_bboxes: Any,
    image_width: int,
    image_height: int,
) -> list[Image.Image]:
    """Augment ref images with colored borders and append a layout image.

    For each input ref PIL image, draws a colored border matching the color
    of the corresponding bounding box in the layout image (so the model can
    learn the spatial correspondence). Appends the layout image (black
    background with colored bbox outlines) as the final element.

    The returned list has length ``len(ref_pils) + 1`` and is passed directly
    to ``_build_edit_sample`` as a regular multi-ref conditioning set
    (token_type=2), with no changes to the downstream pipeline.

    Args:
        ref_pils: original reference PIL images (RGB).
        layout_bboxes: raw bbox data — JSON string, file path, list, or dict.
            Coordinates are in xxyy order, values in [0, 1] or [0, 100].
        image_width: output image width (used to compute absolute bbox coords
            and the layout image canvas size).
        image_height: output image height.

    Returns:
        List of PIL images: [bordered_ref_0, ..., bordered_ref_N-1, layout_image].
    """
    if isinstance(layout_bboxes, str):
        layout_bboxes = load_layout_bboxes(layout_bboxes)
    parsed_boxes = parse_layout_bboxes(layout_bboxes, image_width, image_height)
    layout_image, color_list = draw_bbox_layout(
        parsed_boxes,
        image_width=image_width,
        image_height=image_height,
        return_color=True,
    )
    output_refs: list[Image.Image] = []
    for idx, ref in enumerate(ref_pils):
        color = (
            color_list[idx]
            if idx < len(color_list) and color_list[idx] is not None
            else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        )
        line_width = int(math.sqrt(ref.width * ref.height) * 0.04)
        bordered = add_outer_border_keep_size(ref, color, line_width)
        output_refs.append(bordered)
    output_refs.append(layout_image)
    return output_refs
