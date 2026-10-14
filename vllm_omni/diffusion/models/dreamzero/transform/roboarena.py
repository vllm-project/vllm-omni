# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RoboArena dataset transform.

RoboArena uses 0-indexed exterior cameras, 3 views total (OXE_DROID embodiment).
Same stitching layout as DROID — both map to OXE_DROID in DreamZero.

"""

from __future__ import annotations

from vllm_omni.diffusion.models.dreamzero.transform.base import (
    register_transform,
)
from vllm_omni.diffusion.models.dreamzero.transform.droid import (
    DroidTransform,
)


class RoboArenaTransform(DroidTransform):
    """Transform for RoboArena dataset.

    Same embodiment as DROID (OXE_DROID), same stitching and template.
    Only difference: 0-indexed exterior camera keys.

    RoboArena observation keys (0-indexed):
        observation/exterior_image_0_left  → left exterior
        observation/exterior_image_1_left  → right exterior
        observation/wrist_image_left       → wrist

    """

    IMAGE_KEY_MAP = {
        "observation/exterior_image_0_left": "images/exterior_0",
        "observation/exterior_image_1_left": "images/exterior_1",
        "observation/wrist_image_left": "images/wrist",
    }


register_transform("roboarena", RoboArenaTransform())
