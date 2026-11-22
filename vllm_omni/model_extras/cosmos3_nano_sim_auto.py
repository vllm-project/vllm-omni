# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serving parameters exposed by the Cosmos3-Nano-Sim-Auto pipeline.

Multiview camera inputs and numeric LiDAR share the Cosmos3 multiview request
contract (``multiview.views[*]`` with per-camera prompts and WSM controls,
``lidar.control_path`` / ``condition_path``); the AR-Diffusion session fields
follow Cosmos3-Nano-Sim-Bimanual.
"""

from vllm_omni.model_extras.cosmos3 import COSMOS3_MULTIVIEW_EXTRA_BODY_PARAMS

COSMOS3_NANO_SIM_AUTO_EXTRA_BODY_PARAMS = frozenset(
    set(COSMOS3_MULTIVIEW_EXTRA_BODY_PARAMS)
    | {
        "ar_diffusion_tick",
        "close_session",
        "reset",
        "session_id",
    }
)

COSMOS3_NANO_SIM_AUTO_EXTRA_OUTPUT_PARAMS = frozenset()
