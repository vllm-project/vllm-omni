# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility imports; realtime-video math lives in the installed package."""

from vllm_omni.benchmarks.realtime_video.workload import (
    CAMERA_PATTERNS as CAMERA_PATTERNS,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    DEFAULT_FLOW_SHIFT as DEFAULT_FLOW_SHIFT,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    DEFAULT_FPS as DEFAULT_FPS,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    DEFAULT_HEIGHT as DEFAULT_HEIGHT,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    DEFAULT_MODEL as DEFAULT_MODEL,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    DEFAULT_PROMPT as DEFAULT_PROMPT,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    DEFAULT_SEED as DEFAULT_SEED,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    DEFAULT_WARMUP_CHUNKS as DEFAULT_WARMUP_CHUNKS,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    DEFAULT_WIDTH as DEFAULT_WIDTH,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    FRAMES_PER_BLOCK as FRAMES_PER_BLOCK,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    TEMPORAL_COMPRESSION as TEMPORAL_COMPRESSION,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    ChunkRecord as ChunkRecord,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    PlaybackResult as PlaybackResult,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    PromptUpdate as PromptUpdate,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    Workload as Workload,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    aggregate_metrics as aggregate_metrics,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    build_camera_script as build_camera_script,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    camera_actions_for_chunk as camera_actions_for_chunk,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    chunks_for_num_frames as chunks_for_num_frames,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    compute_metrics as compute_metrics,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    default_workload as default_workload,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    load_workload as load_workload,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    num_frames_for_chunks as num_frames_for_chunks,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    percentile as percentile,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    pixel_frames_in_chunk as pixel_frames_in_chunk,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    simulate_playback as simulate_playback,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    steady_chunk_deadline_ms as steady_chunk_deadline_ms,
)
from vllm_omni.benchmarks.realtime_video.workload import (
    warmup_workload as warmup_workload,
)
