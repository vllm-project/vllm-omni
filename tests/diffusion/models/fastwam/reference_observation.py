# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A fixed LIBERO-style observation and the CAE-FastWAM reference chunk for it.

The reference rows come from the original CAE-FastWAM eager path
(AMD-AGI/faster-action-models ``FastWAM/src``, ``infer_action`` with
``compile_action_infer=False``) on CPU in bfloat16 for this observation,
``libero_spatial`` statistics, 10 steps, seed 42, after the LIBERO evaluator's
post-processing. vLLM-Omni's FastWAMPipeline reproduces that chunk bitwise on
CPU; on GPUs bfloat16 kernels round differently, hence the tolerance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
SUITE = "libero_spatial"
SEED = 42
REFERENCE_ROWS = (0, 9, 31)
# fmt: off
REFERENCE_CHUNK_ROWS = np.array([
    [-0.0842285081744194, 0.0289535503834486, -0.3387450873851776, -0.009933722205460072, 0.055613335222005844, -0.043820805847644806, -1.0],  # noqa: E501
    [-0.0533294640481472, -0.01230239775031805, -0.3515624701976776, -0.008803700096905231, 0.004399780184030533, -0.04956299066543579, -1.0],  # noqa: E501
    [-0.0604248009622097, -0.2508544921875, -0.3369140326976776, -0.004754453431814909, -0.017469095066189766, -0.0061376988887786865, -1.0],  # noqa: E501
], dtype=np.float32)
# fmt: on
# TODO(validation): confirm on MI355X / H100 once the GPU parity run is done.
GPU_ATOL = 2e-2


def build_libero_reference_observation(*, seed: int | None = SEED) -> dict[str, Any]:
    """OpenPI LIBERO observation: two 256x256 frames (already rotated upright, as the
    OpenPI LIBERO client sends them), the 8-D eef state, the task and the suite."""
    rng = np.random.RandomState(0)
    agentview = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    wrist = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    state = np.array([-0.05, 0.02, 1.10, 3.0, -0.1, -0.1, 0.039, -0.039], dtype=np.float32)
    observation: dict[str, Any] = {
        "observation/image": np.ascontiguousarray(agentview[::-1, ::-1]),
        "observation/wrist_image": np.ascontiguousarray(wrist[::-1, ::-1]),
        "observation/state": state,
        "prompt": TASK,
        "task_suite": SUITE,
    }
    if seed is not None:
        observation["seed"] = seed
    return observation
