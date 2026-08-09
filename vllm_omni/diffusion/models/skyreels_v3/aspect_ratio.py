# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolution buckets shared by the SkyReels-V3 pipelines.

SkyReels-V3 was trained on a fixed set of ``(height, width)`` buckets per resolution tier
instead of on free-form sizes, so an input image/video is mapped to the bucket whose aspect
ratio is closest to its own. Reproducing the bucket table is required for output parity with
the reference implementation.
"""

from __future__ import annotations

ASPECT_RATIO_CONFIG: dict[str, dict[str, tuple[int, int]]] = {
    "480P": {
        "0.38": (392, 1046),
        "0.43": (420, 980),
        "0.48": (444, 925),
        "0.50": (452, 904),
        "0.53": (466, 880),
        "0.54": (470, 870),
        "0.56": (480, 854),
        "0.62": (506, 810),
        "0.67": (522, 784),
        "0.75": (554, 738),
        "1.00": (640, 640),
        "1.33": (740, 555),
        "1.50": (784, 522),
        "1.78": (854, 480),
        "1.89": (880, 466),
        "2.00": (906, 454),
        "2.08": (924, 444),
    },
    "540P": {
        "0.34": (416, 1216),
        "0.38": (464, 1216),
        "0.40": (464, 1152),
        "0.43": (496, 1152),
        "0.48": (496, 1024),
        "0.50": (512, 1024),
        "0.53": (512, 960),
        "0.57": (544, 960),
        "0.60": (544, 912),
        "0.63": (576, 912),
        "0.67": (576, 864),
        "0.72": (624, 864),
        "0.75": (624, 832),
        "0.79": (656, 832),
        "0.84": (656, 784),
        "0.92": (720, 784),
        "1.00": (720, 720),
        "1.09": (784, 720),
        "1.26": (784, 624),
        "1.33": (832, 624),
        "1.40": (832, 592),
        "1.49": (880, 592),
        "1.62": (880, 544),
        "1.78": (960, 544),
        "1.88": (960, 512),
        "2.00": (1024, 512),
        "2.13": (1024, 480),
    },
    "720P": {
        "0.38": (588, 1568),
        "0.43": (628, 1466),
        "0.48": (666, 1388),
        "0.50": (678, 1356),
        "0.53": (698, 1318),
        "0.54": (706, 1306),
        "0.56": (720, 1280),
        "0.62": (758, 1212),
        "0.67": (784, 1176),
        "0.75": (832, 1110),
        "1.00": (960, 960),
        "1.33": (1108, 832),
        "1.50": (1176, 784),
        "1.78": (1280, 720),
        "1.89": (1320, 698),
        "2.00": (1358, 680),
        "2.08": (1386, 666),
    },
}

DEFAULT_SKYREELS_V3_RESOLUTION = "540P"

# The VAE downsamples spatially by 8 and the transformer patchifies by 2, so both sides must
# be a multiple of 16.
SKYREELS_V3_SIZE_ALIGNMENT = 16


def get_closest_aspect_ratio(height: float, width: float, ratios: dict[str, tuple[int, int]]) -> str:
    """Return the bucket key of ``ratios`` whose aspect ratio is closest to ``height / width``."""
    aspect_ratio = height / width
    return min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))


def resolve_bucket_size(height: float, width: float, resolution: str | None = None) -> tuple[int, int]:
    """Map a source ``(height, width)`` onto the closest SkyReels-V3 training bucket.

    The returned size is floored to :data:`SKYREELS_V3_SIZE_ALIGNMENT` because a few buckets
    (e.g. ``(466, 880)``) are not themselves aligned.
    """
    resolution = (resolution or DEFAULT_SKYREELS_V3_RESOLUTION).upper()
    if resolution not in ASPECT_RATIO_CONFIG:
        raise ValueError(
            f"Unsupported SkyReels-V3 resolution {resolution!r}, expected one of {sorted(ASPECT_RATIO_CONFIG)}."
        )
    ratios = ASPECT_RATIO_CONFIG[resolution]
    bucket_height, bucket_width = ratios[get_closest_aspect_ratio(height, width, ratios)]
    align = SKYREELS_V3_SIZE_ALIGNMENT
    return bucket_height // align * align, bucket_width // align * align
