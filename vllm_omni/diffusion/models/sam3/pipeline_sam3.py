# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SAM3 promptable concept segmentation pipeline.

One request segments one image or one video. For an image, the prompt text
names the concept ("red car"); optional exemplar boxes from ``extra_args`` add
positive or negative visual examples, and boxes alone segment "whatever looks
like these". The pipeline returns every instance whose score clears
``score_threshold`` as ``DiffusionOutput(output={"segmentation": ...})``:
bit-packed masks at the input image's resolution, ``xyxy`` pixel boxes and
scores.

For a video (checkpoints that carry the SAM3 tracker), the prompt text and any
``extra_args["prompts"]`` each name a concept; every instance is tracked through
the whole video with a stable object id and returned as
``DiffusionOutput(output={"video_segmentation": ...})`` with per-frame COCO RLE
masks, ``xyxy`` pixel boxes and scores.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Iterable, Iterator
from typing import Any

import av
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.v2 import functional as TF
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.sam3.sam3_model import (
    Sam3Detector,
    Sam3DetectorConfig,
    Sam3VisionNeck,
    load_sam3_weights,
)
from vllm_omni.diffusion.models.sam3.sam3_tracker import Sam3Tracker, Sam3TrackerConfig
from vllm_omni.diffusion.models.sam3.sam3_video import Sam3VideoConfig, Sam3VideoSegmenter, Sam3VideoState
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = init_logger(__name__)

IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
TEXT_MAX_LENGTH = 32
# The box-only prompt the reference processor substitutes for missing text.
VISUAL_PROMPT = "visual"
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_MASK_THRESHOLD = 0.5
MAX_PROMPT_BOXES = 64
MAX_VIDEO_PROMPTS = 16
# One video runs inside a single uninterruptible forward, and the tracker keeps every re-anchored
# frame's memory for the rest of the video, so cap the work (~68 s at 30 fps) a single request can take.
MAX_VIDEO_FRAMES = 2048
# The tracker's object-pointer window: a video shorter than this changes how pointers are weighted.
_POINTER_LOOKAHEAD = 16


def _number(extra: dict[str, Any], key: str, default: float) -> float:
    value = extra.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"SAM3 {key} must be a number in [0, 1], got {value!r}")
    return float(value)


def parse_prompt_boxes(extra: dict[str, Any], width: int, height: int) -> tuple[list[list[float]], list[int]]:
    """Validate ``boxes`` ([[x0, y0, x1, y1], ...] in pixels) and optional ``box_labels`` (1 positive, 0 negative)."""
    boxes = extra.get("boxes") or []
    labels = extra.get("box_labels")
    if not isinstance(boxes, list) or len(boxes) > MAX_PROMPT_BOXES:
        raise ValueError(f"SAM3 boxes must be a list of at most {MAX_PROMPT_BOXES} [x0, y0, x1, y1] boxes")
    parsed = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"SAM3 box must be [x0, y0, x1, y1], got {box!r}")
        x0, y0, x1, y1 = (float(value) for value in box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"SAM3 box {box!r} must lie inside the {width}x{height} image with x0 < x1 and y0 < y1")
        parsed.append([x0, y0, x1, y1])
    if labels is None:
        labels = [1] * len(parsed)
    if not isinstance(labels, list) or len(labels) != len(parsed) or any(label not in (0, 1) for label in labels):
        raise ValueError("SAM3 box_labels must list 1 (positive) or 0 (negative) for every box")
    return parsed, [int(label) for label in labels]


def parse_video_prompts(text: str, extra: dict[str, Any]) -> list[str]:
    """The request text plus ``extra_args["prompts"]``: distinct, non-empty concept names in order."""
    extra_prompts = extra.get("prompts") or []
    if not isinstance(extra_prompts, list) or not all(isinstance(p, str) for p in extra_prompts):
        raise ValueError("SAM3 prompts must be a list of strings")
    prompts = list(dict.fromkeys(p.strip() for p in [text, *extra_prompts] if p.strip()))
    if not prompts:
        raise ValueError("SAM3 video segmentation needs at least one text prompt")
    if len(prompts) > MAX_VIDEO_PROMPTS:
        raise ValueError(f"SAM3 video segmentation accepts at most {MAX_VIDEO_PROMPTS} prompts")
    return prompts


def _too_many_frames() -> ValueError:
    return ValueError(f"SAM3 video segmentation accepts at most {MAX_VIDEO_FRAMES} frames")


def _video_frames(video: Any) -> tuple[Iterator[torch.Tensor], float | None]:
    """RGB uint8 [3, H, W] frames from a video file path or already decoded frames, plus the frame rate.

    Raises ``ValueError`` for an input without a video stream or with more than ``MAX_VIDEO_FRAMES``
    frames: up front when the count is known, otherwise from the iterator once the cap is exceeded.
    """
    if isinstance(video, np.ndarray) and video.ndim == 4:
        if len(video) > MAX_VIDEO_FRAMES:
            raise _too_many_frames()
        return (torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1) for frame in video), None
    if isinstance(video, (list, tuple)):
        if len(video) > MAX_VIDEO_FRAMES:
            raise _too_many_frames()
        return (
            TF.pil_to_tensor(frame.convert("RGB"))
            if isinstance(frame, PIL.Image.Image)
            else torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
            for frame in video
        ), None
    if not isinstance(video, str):
        raise ValueError(f"SAM3 cannot read a video of type {type(video).__name__}")
    container = av.open(video)
    try:
        if not container.streams.video:
            raise ValueError("SAM3 video segmentation input has no video stream")
        stream = container.streams.video[0]
        # Containers such as MP4 report the frame count; Matroska/WebM report 0 and are capped while decoding.
        if stream.frames > MAX_VIDEO_FRAMES:
            raise _too_many_frames()
    except BaseException:
        container.close()
        raise
    fps = float(stream.average_rate) if stream.average_rate else None

    def decode() -> Iterator[torch.Tensor]:
        with container:
            for index, frame in enumerate(container.decode(stream)):
                if index >= MAX_VIDEO_FRAMES:
                    raise _too_many_frames()
                yield torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)

    return decode(), fps


def rle_counts(masks: torch.Tensor) -> list[list[int]]:
    """COCO uncompressed RLE of [N, H, W] boolean masks: column-major runs, the first counting background."""
    encoded = []
    for mask in masks:
        flat = mask.t().reshape(-1)
        change = torch.nonzero(flat[1:] != flat[:-1]).flatten() + 1
        bounds = torch.cat([change.new_zeros(1), change, change.new_tensor([flat.numel()])])
        counts = torch.diff(bounds).tolist()
        if bool(flat[0]):
            counts.insert(0, 0)
        encoded.append(counts)
    return encoded


class Sam3Pipeline(nn.Module, SupportImageInput):
    """Registered for both ``Sam3Model`` (image) and ``Sam3VideoModel`` (combined) checkpoints."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        model = od_config.model
        config = get_hf_file_to_dict("config.json", model)
        if not config:
            raise ValueError(f"SAM3 checkpoint {model!r} has no config.json")
        self.config = Sam3DetectorConfig.from_dict(config)
        self.image_size = self.config.vision_config.backbone_config.image_size
        self.detector = Sam3Detector(self.config)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=od_config.revision)
        # Combined image/video checkpoints also carry the memory tracker and its FPN neck.
        self.tracker: Sam3Tracker | None = None
        self.tracker_neck: Sam3VisionNeck | None = None
        self.video_config: Sam3VideoConfig | None = None
        if config.get("tracker_config"):
            self.tracker = Sam3Tracker(Sam3TrackerConfig.from_dict(config["tracker_config"]))
            self.tracker_neck = Sam3VisionNeck(self.config.vision_config)
            self.video_config = Sam3VideoConfig.from_dict(config)
            # Meta's native and SAM 3.1 multiplex layouts ship tracker tensors this module cannot load;
            # such checkpoints still serve images, so the tracker is allowed to stay empty.
            for module in (self.tracker, self.tracker_neck):
                for param in module.parameters():
                    param.is_checkpoint_optional = True
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder=None,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=False,
                # Transformers exports and Meta's native ``sam3*.safetensors`` checkpoints.
                allow_patterns_overrides=["model.safetensors", "model-*.safetensors", "sam3*.safetensors"],
            )
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tracker_tensors: list[tuple[str, torch.Tensor]] = []

        def detector_weights():
            for name, tensor in weights:
                if self.tracker is not None and name.startswith(("tracker_model.", "tracker_neck.")):
                    tracker_tensors.append((name, tensor))
                else:
                    yield name, tensor

        loaded = {"detector." + name for name in load_sam3_weights(self.detector, detector_weights())}
        if self.tracker is not None:
            loaded |= self._load_tracker_weights(tracker_tensors)
        return loaded

    def _load_tracker_weights(self, tensors: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Transformers-layout ``tracker_model.*`` / ``tracker_neck.*`` tensors, or disable video."""
        assert self.tracker is not None and self.tracker_neck is not None
        targets = {
            **{"tracker." + name: t for name, t in self.tracker.named_parameters()},
            **{"tracker." + name: t for name, t in self.tracker.named_buffers() if "positional_embedding" in name},
            **{"tracker_neck." + name: t for name, t in self.tracker_neck.named_parameters()},
        }
        coarsest = f"tracker_neck.fpn_layers.{len(self.tracker_neck.fpn_layers)}."
        loaded: set[str] = set()
        for name, tensor in tensors:
            if name.startswith(coarsest):
                continue  # The tracker drops the 0.5x level, like the detector neck.
            target_name = name.replace("tracker_model.", "tracker.", 1)
            target = targets.get(target_name)
            if target is None:
                raise ValueError(f"SAM3 checkpoint tensor maps to unknown tracker parameter {target_name!r}")
            default_weight_loader(target, tensor) if isinstance(target, nn.Parameter) else target.copy_(tensor)
            loaded.add(target_name)
        if not loaded:
            logger.info("SAM3 checkpoint has no Transformers-layout tracker weights; video segmentation is disabled.")
            self.tracker = self.tracker_neck = self.video_config = None
            return set()
        missing = set(targets) - loaded
        if missing:
            raise ValueError(f"SAM3 checkpoint is missing tracker weights: {sorted(missing)[:8]}")
        return {name for name in loaded if name in dict(self.named_parameters())}

    @property
    def supports_video(self) -> bool:
        return self.tracker is not None

    @property
    def _device(self) -> torch.device:
        return self.detector.text_projection.weight.device

    @property
    def _dtype(self) -> torch.dtype:
        return self.detector.text_projection.weight.dtype

    def _pixel_values(self, image: PIL.Image.Image | torch.Tensor) -> torch.Tensor:
        """Mirror the reference processor: uint8 bilinear antialiased resize, then normalize."""
        if isinstance(image, PIL.Image.Image):
            image = TF.pil_to_tensor(image.convert("RGB"))
        tensor = image.to(self._device)
        tensor = TF.resize(
            tensor, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR, antialias=True
        )
        tensor = TF.normalize(tensor.float(), [m * 255 for m in IMAGE_MEAN], [s * 255 for s in IMAGE_STD])
        return tensor.unsqueeze(0).to(self._dtype)

    def _prompt_boxes(self, boxes: list[list[float]], labels: list[int], width: int, height: int):
        if not boxes:
            return None, None
        tensor = torch.tensor(boxes, dtype=torch.float32, device=self._device)
        tensor = tensor / torch.tensor([width, height, width, height], dtype=torch.float32, device=self._device)
        x0, y0, x1, y1 = tensor.unbind(-1)
        cxcywh = torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)
        return cxcywh.unsqueeze(0).to(self._dtype), torch.tensor([labels], dtype=torch.long, device=self._device)

    @torch.inference_mode()
    def segment(
        self,
        image: PIL.Image.Image,
        text: str,
        boxes: list[list[float]],
        box_labels: list[int],
        score_threshold: float,
        mask_threshold: float,
    ) -> dict[str, Any]:
        width, height = image.size
        tokens = self.tokenizer(
            text or VISUAL_PROMPT,
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_boxes, input_labels = self._prompt_boxes(boxes, box_labels, width, height)
        outputs = self.detector(
            pixel_values=self._pixel_values(image),
            input_ids=tokens["input_ids"].to(self._device),
            attention_mask=tokens["attention_mask"].to(self._device),
            input_boxes=input_boxes,
            input_boxes_labels=input_labels,
        )
        scores = outputs.pred_logits[0].float().sigmoid() * outputs.presence_logits[0].float().sigmoid()
        keep = scores > score_threshold
        scores = scores[keep]
        pred_boxes = outputs.pred_boxes[0][keep].float()
        pred_boxes = pred_boxes * torch.tensor([width, height, width, height], device=pred_boxes.device)
        masks = outputs.pred_masks[0][keep].float().sigmoid()
        count = masks.shape[0]
        if count:
            masks = F.interpolate(masks[None], size=(height, width), mode="bilinear", align_corners=False)[0]
            masks = (masks > mask_threshold).cpu().numpy()
        else:
            masks = np.zeros((0, height, width), dtype=bool)
        return {
            # One bit per pixel keeps instance masks of large images small on the wire.
            "masks": np.packbits(masks.reshape(count, height * width), axis=1),
            "mask_shape": np.array([count, height, width], dtype=np.int64),
            "boxes": pred_boxes.cpu().numpy().astype(np.float32),
            "scores": scores.cpu().numpy().astype(np.float32),
        }

    @torch.inference_mode()
    def segment_video(self, video: Any, prompts: list[str]) -> dict[str, Any]:
        assert self.tracker is not None and self.tracker_neck is not None and self.video_config is not None
        frames, fps = _video_frames(video)
        head = list(itertools.islice(frames, _POINTER_LOOKAHEAD))
        if not head:
            raise ValueError("SAM3 video has no frames")
        height, width = head[0].shape[-2:]
        prompt_features = []
        for prompt in prompts:
            tokens = self.tokenizer(
                prompt, padding="max_length", max_length=TEXT_MAX_LENGTH, truncation=True, return_tensors="pt"
            )
            mask = tokens["attention_mask"].to(self._device)
            prompt_features.append((self.detector.encode_text(tokens["input_ids"].to(self._device), mask), mask.bool()))
        state = Sam3VideoState(
            # Only the first frames' count matters (object-pointer window); a long video is "unbounded".
            num_frames=len(head) if len(head) < _POINTER_LOOKAHEAD else sys.maxsize,
            video_height=int(height),
            video_width=int(width),
            dtype=self._dtype,
            prompts=prompts,
            prompt_features=prompt_features,
        )
        segmenter = Sam3VideoSegmenter(self.video_config, self.detector, self.tracker, self.tracker_neck)
        pixel_frames = (self._pixel_values(frame) for frame in itertools.chain(head, frames))
        results = []
        for out in segmenter.propagate(state, pixel_frames):
            post = segmenter.postprocess(state, out)
            results.append(
                {
                    "object_ids": post["object_ids"].tolist(),
                    "scores": post["scores"].float().tolist(),
                    "boxes": post["boxes"].float().cpu().tolist(),
                    "rle": rle_counts(post["masks"]),
                }
            )
        object_prompts = {}
        for frame in results:
            for obj_id in frame["object_ids"]:
                object_prompts.setdefault(obj_id, prompts[state.obj_id_to_prompt_id.get(obj_id, 0)])
        return {
            "width": int(width),
            "height": int(height),
            "num_frames": len(results),
            "fps": fps,
            "objects": object_prompts,
            "frames": results,
        }

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        if len(req.prompts) != 1:
            return DiffusionOutput(error="SAM3 segments exactly one image per request.", error_status_code=400)
        prompt = req.prompts[0]
        text = prompt if isinstance(prompt, str) else (prompt.get("prompt") or "")
        data = {} if isinstance(prompt, str) else (prompt.get("multi_modal_data") or {})
        if data.get("video") is not None:
            return self._forward_video(
                text, data["video"], dict(getattr(req.sampling_params, "extra_args", None) or {})
            )
        image = data.get("image")
        if isinstance(image, list):
            if len(image) != 1:
                return DiffusionOutput(error="SAM3 segments exactly one image per request.", error_status_code=400)
            image = image[0]
        if not isinstance(image, PIL.Image.Image):
            return DiffusionOutput(error="SAM3 requires an input image.", error_status_code=400)
        extra = dict(getattr(req.sampling_params, "extra_args", None) or {})
        try:
            boxes, labels = parse_prompt_boxes(extra, *image.size)
            if not text.strip() and not boxes:
                raise ValueError("SAM3 needs a text prompt, exemplar boxes, or both")
            score_threshold = _number(extra, "score_threshold", DEFAULT_SCORE_THRESHOLD)
            mask_threshold = _number(extra, "mask_threshold", DEFAULT_MASK_THRESHOLD)
        except (TypeError, ValueError) as error:
            return DiffusionOutput(error=str(error), error_status_code=400)
        result = self.segment(image, text.strip(), boxes, labels, score_threshold, mask_threshold)
        return DiffusionOutput(output={"segmentation": result})

    def _forward_video(self, text: str, video: Any, extra: dict[str, Any]) -> DiffusionOutput:
        if not self.supports_video:
            return DiffusionOutput(
                error="This SAM3 checkpoint has no tracker weights; serve a Transformers-format Sam3VideoModel "
                "checkpoint for video segmentation.",
                error_status_code=400,
            )
        try:
            prompts = parse_video_prompts(text, extra)
            result = self.segment_video(video, prompts)
        except (TypeError, ValueError, av.error.FFmpegError) as error:
            return DiffusionOutput(error=str(error), error_status_code=400)
        return DiffusionOutput(output={"video_segmentation": result})
