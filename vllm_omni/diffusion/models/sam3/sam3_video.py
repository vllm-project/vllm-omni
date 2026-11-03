# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SAM3 video concept segmentation: detect every frame, track with memory, associate.

Ported from the Hugging Face Transformers ``Sam3VideoModel``
(``transformers/models/sam3_video/modeling_sam3_video.py``, Apache-2.0). Each
frame runs the detector for every text prompt and propagates the existing
masklets with the tracker; detections that match no masklet start new objects,
and hot-start heuristics drop masklets that are unmatched or duplicated during
their first frames. Detection NMS and hole filling run in-tree (Transformers
pulls them from a Hub kernel). One deliberate difference: on a frame that
removes an object, outputs are paired with the object ids tracked *before* the
removal, as in Meta's original implementation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes

from vllm_omni.diffusion.models.sam3.sam3_model import Sam3Detector, Sam3VisionNeck, _fields
from vllm_omni.diffusion.models.sam3.sam3_tracker import Sam3Tracker, Sam3TrackerObjectState

NO_OBJ_LOGIT = -10.0
NEVER_OCCLUDED = -1
# Larger than any frame index: marks objects removed by hot-start on this frame.
ALWAYS_OCCLUDED = 100000
REMOVED_SCORE = -1e4


@dataclass
class Sam3VideoConfig:
    low_res_mask_size: int = 288
    score_threshold_detection: float = 0.5
    det_nms_thresh: float = 0.1
    assoc_iou_thresh: float = 0.1
    trk_assoc_iou_thresh: float = 0.5
    new_det_thresh: float = 0.7
    recondition_on_trk_masks: bool = True
    hotstart_delay: int = 15
    hotstart_unmatch_thresh: int = 8
    hotstart_dup_thresh: int = 8
    suppress_unmatched_only_within_hotstart: bool = True
    init_trk_keep_alive: int = 30
    max_trk_keep_alive: int = 30
    min_trk_keep_alive: int = -1
    suppress_overlapping_based_on_recent_occlusion_threshold: float = 0.7
    decrease_trk_keep_alive_for_empty_masklets: bool = False
    fill_hole_area: int = 16
    max_num_objects: int = 10000
    recondition_every_nth_frame: int = 16
    high_conf_thresh: float = 0.8
    high_iou_thresh: float = 0.8

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Sam3VideoConfig:
        return cls(**_fields(cls, values))


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def mask_iou(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU of boolean masks [N, H, W] x [M, H, W] -> [N, M]."""
    flat_a = masks_a.flatten(1).unsqueeze(1)
    flat_b = masks_b.flatten(1).unsqueeze(0)
    intersection = (flat_a & flat_b).sum(dim=2).float()
    union = (flat_a | flat_b).sum(dim=2).float()
    return intersection / union.clamp(min=1)


def generic_nms(ious: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Greedy NMS over a precomputed IoU matrix; returns kept indices in descending score order."""
    ious_np = ious.float().cpu().numpy()
    order = scores.float().cpu().numpy().argsort()[::-1]
    kept = []
    while order.size > 0:
        index = int(order[0])
        kept.append(index)
        order = order[1:][ious_np[index, order[1:]] <= iou_threshold]
    return torch.tensor(kept, dtype=torch.long, device=scores.device)


def nms_masks(pred_probs, pred_masks, prob_threshold: float, iou_threshold: float) -> torch.Tensor:
    """Score-threshold then mask-NMS the detections; returns a keep mask over all of them."""
    is_valid = pred_probs > prob_threshold
    probs = pred_probs[is_valid]
    if probs.numel() == 0:
        return is_valid
    binary = pred_masks[is_valid] > 0
    kept = generic_nms(mask_iou(binary, binary), probs, iou_threshold)
    valid_inds = torch.where(is_valid, is_valid.cumsum(dim=0) - 1, -1)
    return torch.isin(valid_inds, kept)


def connected_component_areas(mask: torch.Tensor) -> torch.Tensor:
    """For a boolean [B, 1, H, W] mask, the 8-connected component area at each pixel (0 off the mask)."""
    host = mask.squeeze(1).to(torch.uint8).cpu().numpy()
    areas = np.zeros(host.shape, dtype=np.int32)
    for index, plane in enumerate(host):
        if not plane.any():
            continue
        _, labels, stats, _ = cv2.connectedComponentsWithStats(plane, connectivity=8)
        component_area = stats[:, cv2.CC_STAT_AREA].astype(np.int32)
        component_area[0] = 0  # background label
        areas[index] = component_area[labels]
    return torch.from_numpy(areas).to(mask.device).unsqueeze(1)


def fill_holes_in_mask_scores(mask: torch.Tensor, max_area: int) -> torch.Tensor:
    """Fill background holes and drop foreground sprinkles up to ``max_area`` pixels in [B, 1, H, W] logits."""
    if max_area <= 0:
        return mask
    background = mask <= 0
    mask = torch.where(background & (connected_component_areas(background) <= max_area), 0.1, mask)
    foreground = mask > 0
    fg_thresh = torch.sum(foreground, dim=(2, 3), keepdim=True, dtype=torch.int32)
    fg_thresh.floor_divide_(2).clamp_(max=max_area)
    return torch.where(foreground & (connected_component_areas(foreground) <= fg_thresh), -0.1, mask)


def _non_overlapping(pred_masks: torch.Tensor) -> torch.Tensor:
    """Keep only the highest-scoring object at every pixel of [N, 1, H, W] scores."""
    if pred_masks.size(0) == 1:
        return pred_masks
    owner = torch.argmax(pred_masks, dim=0, keepdim=True)
    keep = owner == torch.arange(pred_masks.size(0), device=pred_masks.device)[:, None, None, None]
    return torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))


def _per_prompt(fn, masks: torch.Tensor, prompt_ids: list[int], *args) -> torch.Tensor:
    ids = torch.tensor(prompt_ids, device=masks.device, dtype=torch.long)
    grouped = masks.clone()
    for prompt_id in ids.unique(sorted=True):
        indices = torch.nonzero(ids == prompt_id, as_tuple=True)[0]
        grouped[indices] = fn(grouped[indices], *(arg[indices] for arg in args)).to(grouped.dtype)
    return grouped


def _suppress_shrinkage(pred_masks: torch.Tensor) -> torch.Tensor:
    """Suppress objects whose area shrinks below 30% once overlaps are resolved."""
    if pred_masks.size(0) <= 1:
        return pred_masks
    resolved = _non_overlapping(pred_masks)
    area_before = torch.clamp((pred_masks > 0).sum(dim=(-1, -2)), min=1.0)
    area_after = (resolved > 0).sum(dim=(-1, -2))
    keep = (area_after / area_before >= 0.3)[..., None, None].expand_as(pred_masks)
    return torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))


def _object_wise_non_overlapping(masks: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Give each overlapping pixel of boolean [N, 1, H, W] masks to the object with the highest score."""
    single_score = torch.where(masks > 0, scores[..., None, None], 0.0)
    resolved = _non_overlapping(single_score)
    return torch.where(resolved > 0, masks, torch.clamp(masks.float(), max=0.0)).to(single_score.dtype)


# ---------------------------------------------------------------------------
# Request state
# ---------------------------------------------------------------------------


@dataclass
class Sam3VideoState:
    """All per-request tracking state of one video."""

    num_frames: int
    video_height: int
    video_width: int
    dtype: torch.dtype
    prompts: list[str]
    prompt_features: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    obj_ids: list[int] = field(default_factory=list)
    objects: list[Sam3TrackerObjectState] = field(default_factory=list)
    new_input_objects: set[int] = field(default_factory=set)
    obj_id_to_prompt_id: dict[int, int] = field(default_factory=dict)
    obj_id_to_score: dict[int, float] = field(default_factory=dict)
    obj_id_to_tracker_score_frame_wise: defaultdict = field(default_factory=lambda: defaultdict(dict))
    obj_id_to_last_occluded: dict[int, torch.Tensor] = field(default_factory=dict)
    max_obj_id: int = -1
    obj_first_frame_idx: dict[int, int] = field(default_factory=dict)
    unmatched_frame_inds: defaultdict = field(default_factory=lambda: defaultdict(list))
    overlap_pair_to_frame_inds: defaultdict = field(default_factory=lambda: defaultdict(list))
    trk_keep_alive: dict[int, int] = field(default_factory=dict)
    removed_obj_ids: set[int] = field(default_factory=set)
    suppressed_obj_ids: defaultdict = field(default_factory=lambda: defaultdict(set))
    hotstart_removed_obj_ids: set[int] = field(default_factory=set)
    # Tracker FPN features of the frame being processed.
    vision_feats: list[torch.Tensor] | None = None
    vision_pos_embeds: list[torch.Tensor] | None = None

    def add_object(self, obj_id: int) -> int:
        self.obj_ids.append(obj_id)
        self.objects.append(Sam3TrackerObjectState())
        return len(self.obj_ids) - 1

    def remove_object(self, obj_id: int) -> None:
        if obj_id not in self.obj_ids:
            return
        index = self.obj_ids.index(obj_id)
        self.obj_ids.pop(index)
        self.objects.pop(index)
        self.obj_id_to_prompt_id.pop(obj_id, None)
        self.new_input_objects = {i - (i > index) for i in self.new_input_objects if i != index}


@dataclass
class Sam3VideoFrameOutput:
    """Raw per-frame result: low-res mask logits per object and what to hide."""

    frame_idx: int
    obj_id_to_mask: dict[int, torch.Tensor]
    obj_id_to_score: dict[int, float]
    obj_id_to_tracker_score: dict[int, float]
    removed_obj_ids: set[int]
    suppressed_obj_ids: set[int]


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


class Sam3VideoSegmenter:
    """Drives the shared detector, tracker neck and tracker over one video."""

    def __init__(
        self,
        config: Sam3VideoConfig,
        detector: Sam3Detector,
        tracker: Sam3Tracker,
        tracker_neck: Sam3VisionNeck,
    ):
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.tracker_neck = tracker_neck
        # Tracked frames whose memory can still be read: the memory window and the object-pointer window.
        self.memory_horizon = max(tracker.num_maskmem, tracker.config.max_object_pointers_in_encoder)

    # -- per-frame building blocks ----------------------------------------------

    def _tracker_features(self, last_hidden_state: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        neck = self.tracker_neck(last_hidden_state)
        feature_maps = list(neck.fpn_hidden_states)
        feature_maps[0] = self.tracker.mask_decoder.conv_s0(feature_maps[0])
        feature_maps[1] = self.tracker.mask_decoder.conv_s1(feature_maps[1])
        feats = [feature.flatten(2).permute(2, 0, 1) for feature in feature_maps]
        pos = [encoding.flatten(2).permute(2, 0, 1) for encoding in neck.fpn_position_encoding]
        return feats, pos

    def _detect(self, state: Sam3VideoState, vision) -> tuple[dict[str, torch.Tensor], dict[int, int]]:
        cfg = self.config
        bboxes, masks, scores = [], [], []
        det_idx_to_prompt_id: dict[int, int] = {}
        for prompt_id, (prompt_features, prompt_mask) in enumerate(state.prompt_features):
            out = self.detector.detect(vision, prompt_features, prompt_mask)
            probs = out.pred_logits.sigmoid() * out.presence_logits.sigmoid()
            if cfg.det_nms_thresh > 0.0:
                keep = nms_masks(probs[0], out.pred_masks[0], cfg.score_threshold_detection, cfg.det_nms_thresh)
                probs[0][~keep] = 0.0
            positive = torch.where(probs > cfg.score_threshold_detection)
            count = positive[0].numel()
            if count:
                bboxes.append(out.pred_boxes[positive])
                masks.append(out.pred_masks[positive])
                scores.append(probs[positive])
                start = len(det_idx_to_prompt_id)
                det_idx_to_prompt_id.update({start + i: prompt_id for i in range(count)})
        if bboxes:
            return {"bbox": torch.cat(bboxes), "mask": torch.cat(masks), "scores": torch.cat(scores)}, (
                det_idx_to_prompt_id
            )
        device = state.prompt_features[0][0].device
        size = cfg.low_res_mask_size
        empty = {
            "bbox": torch.zeros(0, 4, device=device),
            "mask": torch.zeros(0, size, size, device=device),
            "scores": torch.zeros(0, device=device),
        }
        return empty, det_idx_to_prompt_id

    def _propagate(self, state: Sam3VideoState, frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.config.low_res_mask_size
        device = state.vision_feats[-1].device
        if not state.obj_ids:
            return torch.zeros(0, size, size, device=device), torch.zeros(0, device=device)
        masks, scores = self.tracker.track_frame(
            state.objects,
            state.new_input_objects,
            frame_idx,
            state.vision_feats,
            state.vision_pos_embeds,
            state.num_frames,
            reverse=False,
            dtype=state.dtype,
            run_mem_encoder=False,
        )
        masks = fill_holes_in_mask_scores(masks.squeeze(1).unsqueeze(1), self.config.fill_hole_area).squeeze(1)
        return masks, scores.squeeze(1) if scores.ndim > 1 else scores

    def _associate(self, det_masks, det_scores, trk_masks, trk_obj_ids, det_prompt_ids, trk_prompt_ids):
        cfg = self.config
        device = det_masks.device
        trk_ids = torch.tensor(trk_obj_ids, dtype=torch.long, device=device)
        if trk_masks.size(0) == 0:
            return list(range(det_masks.size(0))), [], {}, {}, []
        if det_masks.size(0) == 0:
            nonempty = (trk_masks > 0).any(dim=(1, 2))
            return [], trk_ids[nonempty].tolist(), {}, {}, trk_ids[~nonempty].tolist()
        trk_binary = trk_masks > 0
        ious = mask_iou(det_masks > 0, trk_binary)
        ious = torch.where(det_prompt_ids.unsqueeze(1) == trk_prompt_ids.unsqueeze(0), ious, torch.zeros_like(ious))
        trk_matched = (ious >= cfg.trk_assoc_iou_thresh).any(dim=0)
        trk_nonempty = trk_binary.any(dim=(1, 2))
        unmatched = trk_ids[trk_nonempty & ~trk_matched].tolist()
        empty = trk_ids[~trk_nonempty].tolist()
        is_new = (det_scores >= cfg.new_det_thresh) & ~(ious >= cfg.assoc_iou_thresh).any(dim=1)
        new_inds = torch.where(is_new)[0].tolist()
        best_trk = ious.argmax(dim=1)
        high = (det_scores >= cfg.high_conf_thresh) & ~is_new & (ious.max(dim=1)[0] >= cfg.high_iou_thresh)
        det_to_trk, trk_to_high_conf_det = {}, {}
        for det_idx in range(det_masks.size(0)):
            det_to_trk[det_idx] = trk_ids[ious[det_idx] >= cfg.assoc_iou_thresh].tolist()
            if high[det_idx].item():
                trk_to_high_conf_det[trk_ids[best_trk[det_idx]].item()] = det_idx
        return new_inds, unmatched, det_to_trk, trk_to_high_conf_det, empty

    def _hotstart(self, state, frame_idx, det_to_trk, new_obj_ids, empty_ids, unmatched_ids) -> set[int]:
        cfg = self.config
        newly_removed: set[int] = set()
        suppressed = state.suppressed_obj_ids[frame_idx]
        hotstart_diff = frame_idx - cfg.hotstart_delay
        for obj_id in new_obj_ids:
            state.obj_first_frame_idx.setdefault(obj_id, frame_idx)
            state.trk_keep_alive[obj_id] = cfg.init_trk_keep_alive
        matched = {int(obj_id) for ids in det_to_trk.values() for obj_id in ids}
        for obj_id in matched:
            state.trk_keep_alive[obj_id] = min(cfg.max_trk_keep_alive, state.trk_keep_alive[obj_id] + 1)
        for obj_id in unmatched_ids:
            state.unmatched_frame_inds[obj_id].append(frame_idx)
            state.trk_keep_alive[obj_id] = max(cfg.min_trk_keep_alive, state.trk_keep_alive[obj_id] - 1)
        if cfg.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_ids:
                state.trk_keep_alive[obj_id] = max(cfg.min_trk_keep_alive, state.trk_keep_alive[obj_id] - 1)
        for obj_id, frames in state.unmatched_frame_inds.items():
            if obj_id in state.removed_obj_ids or obj_id in newly_removed:
                continue
            if len(frames) >= cfg.hotstart_unmatch_thresh and state.obj_first_frame_idx[obj_id] > hotstart_diff:
                newly_removed.add(obj_id)
            if (
                state.trk_keep_alive[obj_id] <= 0
                and not cfg.suppress_unmatched_only_within_hotstart
                and obj_id not in state.removed_obj_ids
                and obj_id not in newly_removed
            ):
                suppressed.add(obj_id)
        for ids in det_to_trk.values():
            if len(ids) < 2:
                continue
            first = min(ids, key=lambda obj_id: state.obj_first_frame_idx[obj_id])
            for obj_id in ids:
                if obj_id != first:
                    state.overlap_pair_to_frame_inds[(first, obj_id)].append(frame_idx)
        for (first, obj_id), frames in state.overlap_pair_to_frame_inds.items():
            if obj_id in state.removed_obj_ids or obj_id in newly_removed:
                continue
            if state.obj_first_frame_idx[obj_id] > hotstart_diff and len(frames) >= cfg.hotstart_dup_thresh:
                newly_removed.add(obj_id)
        state.removed_obj_ids.update(newly_removed)
        return newly_removed

    def _suppress_recently_occluded(self, state, frame_idx, trk_masks, newly_removed) -> tuple[torch.Tensor, dict]:
        binary = trk_masks > 0
        device = binary.device
        last_prev = torch.cat(
            [
                state.obj_id_to_last_occluded.get(
                    obj_id,
                    torch.full(
                        (1,),
                        NEVER_OCCLUDED if obj_id not in newly_removed else ALWAYS_OCCLUDED,
                        device=device,
                        dtype=torch.long,
                    ),
                )
                for obj_id in state.obj_ids
            ]
        )
        prompt_ids = torch.tensor([state.obj_id_to_prompt_id[i] for i in state.obj_ids], device=device)
        to_suppress = torch.zeros(len(state.obj_ids), device=device, dtype=torch.bool)
        threshold = self.config.suppress_overlapping_based_on_recent_occlusion_threshold
        for prompt_id in prompt_ids.unique(sorted=True):
            indices = torch.nonzero(prompt_ids == prompt_id, as_tuple=True)[0]
            if indices.numel() <= 1:
                continue
            masks = binary[indices]
            overlapping = torch.triu(mask_iou(masks, masks) >= threshold, diagonal=1)
            last_i, last_j = last_prev[indices].unsqueeze(1), last_prev[indices].unsqueeze(0)
            suppress_i = overlapping & (last_i > last_j) & (last_j > -1)
            suppress_j = overlapping & (last_j > last_i) & (last_i > -1)
            to_suppress[indices] = suppress_i.any(dim=1) | suppress_j.any(dim=0)
        occluded = ~binary.any(dim=(-1, -2)) | to_suppress
        last_new = last_prev.clone()
        last_new[occluded] = frame_idx
        last_occluded = {obj_id: last_new[i : i + 1] for i, obj_id in enumerate(state.obj_ids)}
        trk_masks[to_suppress] = NO_OBJ_LOGIT
        return trk_masks, last_occluded

    def _update_memories(self, state, frame_idx, low_res_masks, reconditioned: dict[int, torch.Tensor]) -> None:
        high_res = low_res_masks.unsqueeze(1)
        for index, mask in reconditioned.items():
            high_res[index] = mask.float()
            obj = state.objects[index]
            if frame_idx in obj.non_cond_frame_outputs:
                obj.cond_frame_outputs[frame_idx] = obj.non_cond_frame_outputs.pop(frame_idx)
        prompt_ids = [state.obj_id_to_prompt_id[obj_id] for obj_id in state.obj_ids]
        high_res = _per_prompt(_suppress_shrinkage, high_res, prompt_ids)
        scores = torch.where((high_res > 0).any(dim=(-1, -2)), 10.0, -10.0)
        features, positions = self.tracker.encode_memory(
            state.vision_feats[-1], high_res, scores.unsqueeze(-1), is_mask_from_pts=False
        )
        for index, obj in enumerate(state.objects):
            for outputs in (obj.cond_frame_outputs, obj.non_cond_frame_outputs):
                if frame_idx in outputs:
                    outputs[frame_idx]["maskmem_features"] = features[:, index : index + 1]
                    outputs[frame_idx]["maskmem_pos_enc"] = positions[:, index : index + 1]

    def _prune(self, state: Sam3VideoState, frame_idx: int) -> None:
        """Drop tensors that no later frame reads: decoded masks and memory outside the horizon."""
        stale = frame_idx - self.memory_horizon
        for obj in state.objects:
            obj.non_cond_frame_outputs.pop(stale, None)
            obj.mask_inputs.pop(frame_idx, None)
            for outputs in (obj.cond_frame_outputs, obj.non_cond_frame_outputs):
                out = outputs.get(frame_idx)
                if out is not None:
                    out.pop("high_res_masks", None)
                    if outputs is obj.non_cond_frame_outputs:
                        out.pop("pred_masks", None)

    # -- one frame ---------------------------------------------------------------

    def step(self, state: Sam3VideoState, frame_idx: int, pixel_values: torch.Tensor) -> Sam3VideoFrameOutput:
        cfg = self.config
        vision = self.detector.vision_encoder(pixel_values)
        det_out, det_idx_to_prompt_id = self._detect(state, vision)
        state.vision_feats, state.vision_pos_embeds = self._tracker_features(vision.last_hidden_state)
        trk_masks, trk_scores = self._propagate(state, frame_idx)
        prev_obj_ids = list(state.obj_ids)

        # Planning: associate, hot-start, recondition, occlusion suppression, memory update.
        det_scores = det_out["scores"].float()
        device = det_out["mask"].device
        det_prompt_ids = torch.tensor(
            [det_idx_to_prompt_id[i] for i in range(det_out["mask"].size(0))], device=device, dtype=torch.long
        )
        trk_prompt_ids = torch.tensor(
            [state.obj_id_to_prompt_id[obj_id] for obj_id in state.obj_ids], device=device, dtype=torch.long
        )
        new_inds, unmatched, det_to_trk, trk_to_high_conf_det, empty = self._associate(
            det_out["mask"], det_scores, trk_masks, state.obj_ids, det_prompt_ids, trk_prompt_ids
        )
        if len(state.obj_ids) + len(new_inds) > cfg.max_num_objects:
            keep = cfg.max_num_objects - len(state.obj_ids)
            top = torch.topk(det_scores[torch.tensor(new_inds, device=device)], k=keep, largest=True).indices
            new_inds = [new_inds[i] for i in top.tolist()]
        new_obj_ids = list(range(state.max_obj_id + 1, state.max_obj_id + 1 + len(new_inds)))
        for obj_id, det_idx in zip(new_obj_ids, new_inds):
            state.obj_id_to_prompt_id[obj_id] = det_idx_to_prompt_id[det_idx]
        newly_removed = self._hotstart(state, frame_idx, det_to_trk, new_obj_ids, empty, unmatched)

        reconditioned_masks: dict[int, torch.Tensor] = {}
        reconditioned_ids: set[int] = set()
        if (
            cfg.recondition_every_nth_frame > 0
            and frame_idx % cfg.recondition_every_nth_frame == 0
            and trk_to_high_conf_det
        ):
            for obj_id, det_idx in trk_to_high_conf_det.items():
                index = state.obj_ids.index(obj_id)
                if trk_scores[index] <= cfg.high_conf_thresh:
                    continue
                if cfg.recondition_on_trk_masks:
                    reconditioned_masks[index] = trk_masks[index : index + 1].unsqueeze(1)
                else:
                    reconditioned_masks[index] = det_out["mask"][det_idx : det_idx + 1].unsqueeze(1) >= 0.5
                reconditioned_ids.add(obj_id)

        last_occluded: dict[int, torch.Tensor] = {}
        if trk_masks.size(0) > 0:
            if cfg.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0:
                trk_masks, last_occluded = self._suppress_recently_occluded(state, frame_idx, trk_masks, newly_removed)
            self._update_memories(state, frame_idx, trk_masks, reconditioned_masks)

        # Execution: seed new objects from their detector masks, then drop removed ones.
        frame_scores = state.obj_id_to_tracker_score_frame_wise[frame_idx]
        if new_inds:
            new_scores = det_scores[torch.tensor(new_inds, dtype=torch.long, device=device)].tolist()
            state.obj_id_to_score.update(zip(new_obj_ids, new_scores))
            frame_scores.update(zip(new_obj_ids, new_scores))
            state.max_obj_id = max(state.max_obj_id, max(new_obj_ids))
            new_masks = det_out["mask"][torch.tensor(new_inds, dtype=torch.long, device=device)] >= 0.5
            new_indices = set()
            for obj_id, mask in zip(new_obj_ids, new_masks):
                index = state.add_object(obj_id)
                state.objects[index].mask_inputs[frame_idx] = mask[None, None].to(state.dtype)
                new_indices.add(index)
            state.new_input_objects = new_indices
            self.tracker.track_frame(
                state.objects,
                state.new_input_objects,
                frame_idx,
                state.vision_feats,
                state.vision_pos_embeds,
                state.num_frames,
                reverse=False,
                dtype=state.dtype,
                run_mem_encoder=True,
            )
        for obj_id in newly_removed:
            state.obj_id_to_score[obj_id] = REMOVED_SCORE
            frame_scores[obj_id] = REMOVED_SCORE
            last_occluded.pop(obj_id, None)
            state.remove_object(obj_id)
        state.obj_id_to_last_occluded = last_occluded

        # Outputs: tracked masklets (as tracked before this frame's removals), new detections, reconditions.
        obj_id_to_mask = {obj_id: mask.unsqueeze(0) for obj_id, mask in zip(prev_obj_ids, trk_masks)}
        if new_inds:
            new_det_masks = det_out["mask"][torch.tensor(new_inds, dtype=torch.long, device=device)]
            new_det_masks = fill_holes_in_mask_scores(new_det_masks.unsqueeze(1), cfg.fill_hole_area).squeeze(1)
            obj_id_to_mask.update((obj_id, mask.unsqueeze(0)) for obj_id, mask in zip(new_obj_ids, new_det_masks))
        for obj_id in reconditioned_ids:
            obj_id_to_mask[obj_id] = det_out["mask"][trk_to_high_conf_det[obj_id]].unsqueeze(0)
        if trk_scores.shape[0] > 0:
            frame_scores.update(zip(prev_obj_ids, trk_scores.sigmoid().tolist()))
        self._prune(state, frame_idx)
        return Sam3VideoFrameOutput(
            frame_idx=frame_idx,
            obj_id_to_mask=obj_id_to_mask,
            obj_id_to_score=dict(state.obj_id_to_score),
            obj_id_to_tracker_score=dict(frame_scores),
            removed_obj_ids=set(state.removed_obj_ids),
            suppressed_obj_ids=set(state.suppressed_obj_ids[frame_idx]),
        )

    def propagate(self, state: Sam3VideoState, frames: Iterator[torch.Tensor]) -> Iterator[Sam3VideoFrameOutput]:
        """Run every frame in order; outputs are held for ``hotstart_delay`` frames so late removals apply."""
        buffer: list[Sam3VideoFrameOutput] = []
        for frame_idx, pixel_values in enumerate(frames):
            out = self.step(state, frame_idx, pixel_values)
            if self.config.hotstart_delay <= 0:
                yield out
                continue
            buffer.append(out)
            state.hotstart_removed_obj_ids.update(out.removed_obj_ids)
            if len(buffer) >= self.config.hotstart_delay:
                yield buffer.pop(0)
        # The last frame releases everything still held.
        yield from buffer

    def postprocess(self, state: Sam3VideoState, out: Sam3VideoFrameOutput) -> dict[str, torch.Tensor]:
        """Upsample to the video size, hide suppressed/removed objects and resolve overlaps per prompt."""
        height, width = state.video_height, state.video_width
        obj_ids = sorted(out.obj_id_to_mask)
        device = out.obj_id_to_mask[obj_ids[0]].device if obj_ids else torch.device("cpu")
        if not obj_ids:
            return {
                "object_ids": torch.zeros(0, dtype=torch.int64),
                "scores": torch.zeros(0),
                "boxes": torch.zeros(0, 4),
                "masks": torch.zeros(0, height, width, dtype=torch.bool),
            }
        ids = torch.tensor(obj_ids, dtype=torch.int64)
        scores = torch.tensor([out.obj_id_to_score[obj_id] for obj_id in obj_ids])
        tracker_scores = torch.tensor([out.obj_id_to_tracker_score.get(obj_id, 0.0) for obj_id in obj_ids])
        low_res = torch.cat([out.obj_id_to_mask[obj_id] for obj_id in obj_ids], dim=0)
        masks = F.interpolate(low_res.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)
        masks = masks.squeeze(1) > 0
        keep = masks.any(dim=(1, 2)).cpu()
        hidden = list(out.suppressed_obj_ids) + list(state.hotstart_removed_obj_ids)
        if hidden:
            keep &= ~torch.isin(ids, torch.tensor(hidden, dtype=torch.int64))
        keep_idx = torch.nonzero(keep, as_tuple=True)[0]
        ids, scores, tracker_scores = ids[keep_idx], scores[keep_idx], tracker_scores[keep_idx]
        masks = masks[keep_idx.to(device)]
        boxes = masks_to_boxes(masks) if masks.shape[0] else torch.zeros(0, 4, device=device)
        if masks.shape[0] > 1:
            prompt_ids = [state.obj_id_to_prompt_id[int(i)] for i in ids]
            masks = (
                _per_prompt(
                    _object_wise_non_overlapping, masks.unsqueeze(1), prompt_ids, tracker_scores.unsqueeze(1).to(device)
                ).squeeze(1)
                > 0
            )
        return {"object_ids": ids, "scores": scores, "boxes": boxes, "masks": masks}
