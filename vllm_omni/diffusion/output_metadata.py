# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility imports for diffusion output metadata.

The shared output payload and metadata schema lives in
``vllm_omni.outputs.output_metadata``.  Keep this module as a stable import
path for existing diffusion callers.
"""

from vllm_omni.outputs.output_metadata import *  # noqa: F403
