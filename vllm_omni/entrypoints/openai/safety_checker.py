# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from typing import Any

import PIL.Image
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_MODEL = "Falconsai/nsfw_image_detection"
_NSFW_THRESHOLD = 0.5


class SafetyChecker:
    """NSFW safety checker for generated images.

    Thread-safe lazy initialization. Runs on CPU to avoid
    GPU memory contention with diffusion models.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._pipeline: Any = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            from transformers import pipeline

            logger.info("Loading safety checker model: %s", self._model_name)
            self._pipeline = pipeline(
                "image-classification",
                model=self._model_name,
                device="cpu",
            )
            logger.info("Safety checker loaded successfully")

    def check_images(self, images: list[PIL.Image.Image]) -> list[tuple[bool, float]]:
        """Check multiple images for NSFW content.

        Returns list of (is_safe, nsfw_score) tuples.
        """
        if not images:
            return []
        self._ensure_loaded()
        all_preds = self._pipeline(images)
        if images and not isinstance(all_preds[0], list):
            all_preds = [all_preds]
        results = []
        for preds in all_preds:
            nsfw_score = next((r["score"] for r in preds if r["label"] == "nsfw"), 0.0)
            results.append((nsfw_score < _NSFW_THRESHOLD, nsfw_score))
        return results
