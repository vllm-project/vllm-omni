# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru_vllm/haru_vllm/duplex/__init__.py @ d3612b0.
from vllm_omni.model_executor.models.haru.duplex.plugin import HaruDuplexPlugin

__all__ = ["HaruDuplexPlugin"]
