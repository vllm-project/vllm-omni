# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Storage keys shared by the video generation path and the job store."""


def lidar_storage_key(video_id: str) -> str:
    """Storage key of the LiDAR sidecar written for a joint drive job."""
    return f"{video_id}.lidar.safetensors"
