# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SAM3 promptable concept segmentation pipeline.

One request segments one image. The prompt text names the concept ("red car");
optional exemplar boxes from ``extra_args`` add positive or negative visual
examples, and boxes alone segment "whatever looks like these". The pipeline
returns every instance whose score clears ``score_threshold`` as
``DiffusionOutput(output={"segmentation": ...})``: bit-packed masks at the
input image's resolution, ``xyxy`` pixel boxes and scores.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.v2 import functional as TF
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.sam3.sam3_model import (
    Sam3Detector,
    Sam3DetectorConfig,
    load_sam3_weights,
)
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
        return {"detector." + name for name in load_sam3_weights(self.detector, weights)}

    @property
    def _device(self) -> torch.device:
        return self.detector.text_projection.weight.device

    @property
    def _dtype(self) -> torch.dtype:
        return self.detector.text_projection.weight.dtype

    def _pixel_values(self, image: PIL.Image.Image) -> torch.Tensor:
        """Mirror the reference processor: uint8 bilinear antialiased resize, then normalize."""
        tensor = TF.pil_to_tensor(image.convert("RGB")).to(self._device)
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
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        if len(req.prompts) != 1:
            return DiffusionOutput(error="SAM3 segments exactly one image per request.", error_status_code=400)
        prompt = req.prompts[0]
        text = prompt if isinstance(prompt, str) else (prompt.get("prompt") or "")
        data = {} if isinstance(prompt, str) else (prompt.get("multi_modal_data") or {})
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
