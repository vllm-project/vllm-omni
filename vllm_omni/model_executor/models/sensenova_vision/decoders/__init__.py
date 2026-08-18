"""SenseNova-Vision decoding utilities.

This package contains pure-Python/NumPy/PIL decoders for the structured-text
and dense-image outputs produced by SenseNova-Vision-7B-MoT. It mirrors the
official SenseNova-Vision evaluation code (``utils/parsing_output.py``,
``utils/mask.py``, ``inference/utils_3d/camera_pose_parser.py`` and the
Marigold-style depth/normal evaluation scripts) while keeping the module free
of torch/GPU dependencies so it can be unit-tested on CPU without downloading
a model.
"""

from vllm_omni.model_executor.models.sensenova_vision.decoders.camera_pose import parse_camera_pose
from vllm_omni.model_executor.models.sensenova_vision.decoders.dense_decoders import (
    decode_depth,
    decode_normal,
    decode_point_map,
    decode_segmentation,
)
from vllm_omni.model_executor.models.sensenova_vision.decoders.text_parsers import (
    parse_bbox,
    parse_keypoints,
    parse_points,
)

__all__ = [
    "parse_bbox",
    "parse_points",
    "parse_keypoints",
    "parse_camera_pose",
    "decode_segmentation",
    "decode_depth",
    "decode_normal",
    "decode_point_map",
]
