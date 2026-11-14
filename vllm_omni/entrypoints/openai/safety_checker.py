# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in NSFW check for generated images, enabled with ``--safety-checker``."""

import asyncio
import threading
from http import HTTPStatus
from typing import Any

import PIL.Image
from fastapi import HTTPException, Request
from vllm.logger import init_logger

from vllm_omni.errors import GuardrailViolationError

logger = init_logger(__name__)

DEFAULT_MODEL = "Falconsai/nsfw_image_detection"
NSFW_THRESHOLD = 0.5


class SafetyChecker:
    """Classify images as safe or NSFW with a Hugging Face image classifier.

    Runs on CPU to avoid contending with the diffusion model for GPU memory.
    The model is loaded once, on the first call to ``ensure_loaded``.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, threshold: float = NSFW_THRESHOLD):
        self._model_name = model_name
        self._threshold = threshold
        self._pipeline: Any = None
        self._lock = threading.Lock()

    def ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            from transformers import pipeline

            logger.info("Loading safety checker model: %s", self._model_name)
            self._pipeline = pipeline("image-classification", model=self._model_name, device="cpu")

    def check_images(self, images: list[PIL.Image.Image]) -> list[tuple[bool, float]]:
        """Return an ``(is_safe, nsfw_score)`` pair for each image."""
        if not images:
            return []
        self.ensure_loaded()
        predictions = self._pipeline(images)
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        results = []
        for labels in predictions:
            nsfw_score = next((item["score"] for item in labels if item["label"] == "nsfw"), 0.0)
            results.append((nsfw_score < self._threshold, nsfw_score))
        return results


async def check_images_safe(images: list[PIL.Image.Image], raw_request: Request) -> None:
    """Reject the request if any generated image is flagged.

    Does nothing unless the server was started with ``--safety-checker``.
    Raises ``GuardrailViolationError`` (HTTP 400) for a flagged image and
    ``HTTPException`` 503 if the checker itself fails.
    """
    checker: SafetyChecker | None = getattr(raw_request.app.state, "safety_checker", None)
    if checker is None:
        return
    try:
        results = await asyncio.get_running_loop().run_in_executor(None, checker.check_images, images)
    except Exception as e:
        logger.exception("Safety checker failed")
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail=f"Safety checker unavailable: {e}",
        ) from e
    flagged = [i for i, (is_safe, _) in enumerate(results) if not is_safe]
    if flagged:
        logger.warning("Safety checker flagged %d of %d generated images: %s", len(flagged), len(images), flagged)
        raise GuardrailViolationError("Generated image was flagged as potentially unsafe.")
