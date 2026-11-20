# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Canonical deployment fixtures for Cosmos3-Nano-Transfer-Auto tests."""

from typing import Any


def multiview_lidar_contract(chunk: int = 9, context: int | None = 9) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "sample_posterior": False,
        "apply_validity_mask": True,
        "fps": 10.0,
        "latent_channels": 128,
        "temporal_compression_factor": 1,
        "spatial_compression": [16, 16],
        "network_config": {
            "resolution": [128, 1808],
            "patch_size": [2, 2],
            "depths": [1, 1, 1, 1],
            "z_dim": 128,
            "in_channels": 3,
            "temporal_downsample": [False, False, False],
        },
        "range_projection": {
            "semantic_width": 1800,
            "model_width": 1808,
            "native_height": 128,
            "model_width_transform": "circular_pad",
            "intensity_encoding": "unit",
            "min_range_m": 0.0,
            "max_range_m": 100.0,
        },
        "streaming_chunk_frames": chunk,
        "streaming_context_frames": context,
    }


def multiview_contract() -> dict[str, Any]:
    """The canonical exported deployment contract for Cosmos3-Nano-Transfer-Auto."""
    from vllm_omni.model_extras.cosmos3 import COSMOS3_MADS_CAMERAS

    return {
        "cameras": list(COSMOS3_MADS_CAMERAS),
        "cross_view_past_window_seconds": 0.4,
        "inference_defaults": {
            "resolution": "480",
            "fps": 30.0,
            "num_steps": 35,
            "guidance": 6.0,
            "shift": 10.0,
            "control_guidance": 1.0,
            "emphasize_control_in_prompt": True,
            "guidance_interval": None,
            "control_guidance_interval": None,
            "normalize_cfg": False,
        },
        "lidar": multiview_lidar_contract(chunk=20, context=21),
        "lidar_latent_patch_size_hw": [1, 1],
        "rig_view_embedding": {
            "num_embeddings": 12,
            "camera_ids": {camera: index for index, camera in enumerate(COSMOS3_MADS_CAMERAS)},
            "lidar_id": 11,
        },
    }
