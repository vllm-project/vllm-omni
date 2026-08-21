# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from pathlib import Path

from benchmarks.ltx2.tiled_data_parallel_ab import (
    STAGE_2_SIGMAS,
    _generation_command,
    _overlap_bands,
    _parse_run_log,
)


def test_4k_overlap_bands_match_runtime_tile_geometry():
    bands = _overlap_bands(
        target_width=3840,
        target_height=2160,
        internal_width=3840,
        internal_height=2176,
        overlap=5,
    )

    assert bands == [
        {"axis": "horizontal", "x": 0, "y": 1024, "width": 3840, "height": 160},
        {"axis": "vertical", "x": 1856, "y": 0, "width": 160, "height": 2160},
    ]


def test_generation_commands_match_schedule_and_only_tdp_enables_tiles():
    args = Namespace(
        model="Lightricks/LTX-2.5-Diffusers",
        prompt="test prompt",
        width=3840,
        height=2160,
        num_frames=121,
        num_inference_steps=8,
        frame_rate=24.0,
        fps=24,
        seed=42,
        overlap=5,
        negative_prompt=None,
    )

    baseline = _generation_command(
        args,
        output=Path("baseline.mp4"),
        tiled=False,
        internal_width=3840,
        internal_height=2176,
    )
    tiled = _generation_command(
        args,
        output=Path("tdp.mp4"),
        tiled=True,
        internal_width=3840,
        internal_height=2176,
    )

    baseline_extra = baseline[baseline.index("--extra-body") + 1]
    tiled_extra = tiled[tiled.index("--extra-body") + 1]
    assert f'"stage_2_sigmas":[{",".join(str(value) for value in STAGE_2_SIGMAS)}]' in baseline_extra
    assert "ltx_tiled_data_parallel" not in baseline_extra
    assert '"ltx_tiled_data_parallel":true' in tiled_extra
    assert baseline[baseline.index("--height") + 1] == "2176"
    assert tiled[tiled.index("--height") + 1] == "2160"


def test_parse_run_log_summarizes_distributed_profiler_events():
    result = _parse_run_log(
        """
Total generation time: 12.5000 seconds (12500.00 ms)
Worker peak GPU memory (reserved): 4096.00 MiB (4.00 GiB)
[DiffusionPipelineProfiler] LTX2.forward took 10.000000s
[DiffusionPipelineProfiler] LTX2.forward took 12.000000s
"""
    )

    assert result["request_seconds"] == 12.5
    assert result["peak_memory_mib"] == 4096.0
    assert result["profiler_events"]["LTX2.forward"] == {
        "samples": 2,
        "min_seconds": 10.0,
        "mean_seconds": 11.0,
        "max_seconds": 12.0,
    }
