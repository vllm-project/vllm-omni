# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# Request params the AudioX pipeline reads from ``sampling_params.extra_args``
# (see ``pipeline_audiox.py`` ``forward``). Declaring them here routes
# ``extra_body`` (online) / ``--extra-body`` (offline) keys into ``extra_args``
# instead of silently dropping them.
AUDIOX_EXTRA_BODY_PARAMS = frozenset(
    {
        "audiox_task",  # t2a/t2m/v2a/v2m/tv2a/tv2m
        "seconds_start",
        "seconds_total",
        "sigma_min",
        "sigma_max",
        "cfg_rescale",
        "video_path",  # v2*/tv2* conditioning
        "audio_path",  # optional reference-audio conditioning
    }
)
# Echoed back in ``DiffusionOutput.custom_output`` (pipeline_audiox.py).
AUDIOX_EXTRA_OUTPUT_PARAMS = frozenset(
    {
        "audiox_task",
    }
)
