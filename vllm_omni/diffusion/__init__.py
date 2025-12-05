# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion model support for vllm-omni."""

import os
import sys

# Optional: Add custom diffusers path for development
# Users can set VLLM_OMNI_DIFFUSERS_PATH environment variable
# to point to a local diffusers installation with unreleased features
_custom_diffusers = os.environ.get("VLLM_OMNI_DIFFUSERS_PATH")
if _custom_diffusers and os.path.exists(_custom_diffusers) and _custom_diffusers not in sys.path:
    sys.path.insert(0, _custom_diffusers)
