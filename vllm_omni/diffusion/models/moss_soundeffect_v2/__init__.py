# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Moss SoundEffect model support for vLLM-Omni."""

from .pipeline_moss_soundeffect_v2 import (
    MossSoundEffectPipeline,
    get_moss_soundeffect_post_process_func,
)

__all__ = [
    "MossSoundEffectPipeline",
    "get_moss_soundeffect_post_process_func",
]
