# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Policy image decoding and action transforms shared by local and remote serving."""

import pybase64 as base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_LIBERO_STATS = Path(__file__).resolve().parent / "libero_native_frame_wise_relative_rot6d.json"


def decode_policy_image(encoded: str) -> np.ndarray:
    """Decode PNG/base64, including data URLs, whitespace and missing padding."""
    if not isinstance(encoded, str):
        raise ValueError("'image' must be a base64 string")
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    encoded = "".join(encoded.split())
    try:
        raw = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_")
        with Image.open(io.BytesIO(raw)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid policy image: {exc}") from exc


def denormalize_range_actions(action: torch.Tensor, minimum: torch.Tensor, action_range: torch.Tensor) -> torch.Tensor:
    """Invert minmax/quantile scaling with the native operation order and precision."""
    return (action[..., : minimum.shape[0]] + 1.0) / 2.0 * action_range.to(action.device) + minimum.to(action.device)


class LiberoActionNormalizer:
    """Load the native physical 10D action statistics once, on CPU."""

    def __init__(self, stats_path: Path = DEFAULT_LIBERO_STATS) -> None:
        stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        metadata = stats["metadata"]
        expected = {
            "embodiment_type": "libero",
            "pose_convention": "frame_wise_relative",
            "rotation_format": "6d",
            "action_dim": 10,
            "chunk_length": 16,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Expected LIBERO frame_wise_relative rot6d statistics for 16x10 actions.")
        self.minimum = torch.tensor(stats["global_raw"]["q01"], dtype=torch.float32)
        maximum = torch.tensor(stats["global_raw"]["q99"], dtype=torch.float32)
        if (
            self.minimum.shape != (10,)
            or maximum.shape != (10,)
            or not torch.isfinite(self.minimum).all()
            or not torch.isfinite(maximum).all()
            or torch.any(maximum <= self.minimum)
        ):
            raise ValueError("LIBERO global_raw q01/q99 must contain 10 finite, ordered quantiles.")
        self.action_range = (maximum - self.minimum).clamp_min(1e-6)

    def denormalize(self, actions: np.ndarray) -> list[list[float]]:
        return denormalize_range_actions(torch.from_numpy(actions), self.minimum, self.action_range).tolist()
