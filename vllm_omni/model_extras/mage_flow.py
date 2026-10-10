# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# Re-exported from the shared metadata module so the advertised key set and the
# pipeline's own validation stay one definition. ``model_metadata`` is import-safe
# here: ``vllm_omni.diffusion.__init__`` is empty and the module itself only
# depends on ``dataclasses``.
from vllm_omni.diffusion.model_metadata import MAGE_FLOW_EXTRA_BODY_PARAMS

__all__ = ["MAGE_FLOW_EXTRA_BODY_PARAMS"]
