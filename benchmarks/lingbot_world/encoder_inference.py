# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bound VBench background CLIP activations without changing frame coverage."""

from __future__ import annotations

import importlib
import weakref
from types import SimpleNamespace
from typing import Any

BACKGROUND_CLIP_BATCH_SIZE = 32


def verify_background_coverage(stats: dict[str, Any], expected_frames: int) -> None:
    """Fail if the worker omitted frames, retained gradients or changed batching."""
    expected = {
        "batch_size": BACKGROUND_CLIP_BATCH_SIZE,
        "encoded_images": expected_frames,
        "forward_batches": (expected_frames + BACKGROUND_CLIP_BATCH_SIZE - 1) // BACKGROUND_CLIP_BATCH_SIZE,
        "max_forward_batch": min(expected_frames, BACKGROUND_CLIP_BATCH_SIZE),
    }
    if stats.get("gradients") is not False or any(
        type(stats.get(key)) is not int or stats[key] != value for key, value in expected.items()
    ):
        raise ValueError("VBench background CLIP must encode every frame in bounded inference batches")


def install_vbench_background_inference() -> dict[str, Any]:
    """Keep native transforms/normalization/scoring; batch only image encoding.

    The pinned background metric encodes the complete rollout with autograd
    enabled. A 600-chunk video exceeds even 140 GiB. Patch its local loader,
    leaving other metrics and the immutable evaluator checkout untouched.
    """
    background = importlib.import_module("vbench.background_consistency")
    existing = getattr(background, "_lingbot_clip_inference_stats", None)
    if existing is not None:
        return existing
    native_load = background.clip.load
    stats = {
        "batch_size": BACKGROUND_CLIP_BATCH_SIZE,
        "gradients": False,
        "encoded_images": 0,
        "forward_batches": 0,
        "max_forward_batch": 0,
    }

    def load(*args: Any, **kwargs: Any) -> Any:
        model, preprocess = native_load(*args, **kwargs)
        native_encode = weakref.WeakMethod(model.encode_image)

        def encode_image(images: Any) -> Any:
            import torch

            if images.ndim != 4 or images.shape[0] < 1:
                raise ValueError("CLIP background inference requires nonempty NCHW images")
            encoder = native_encode()
            if encoder is None:
                raise RuntimeError("CLIP encoder was released before inference")
            features = []
            with torch.inference_mode():
                for batch in images.split(BACKGROUND_CLIP_BATCH_SIZE):
                    encoded = encoder(batch)
                    if encoded.shape[0] != batch.shape[0]:
                        raise ValueError("CLIP encoder changed the image population")
                    features.append(encoded)
                    stats["encoded_images"] += batch.shape[0]
                    stats["forward_batches"] += 1
                    stats["max_forward_batch"] = max(stats["max_forward_batch"], batch.shape[0])
                return torch.cat(features, dim=0)

        model.encode_image = encode_image
        return model, preprocess

    # Only this native metric sees the loader adapter. Aesthetics/HPS and the
    # installed clip module retain their own native inference paths.
    background.clip = SimpleNamespace(load=load)
    background._lingbot_clip_inference_stats = stats
    return stats
