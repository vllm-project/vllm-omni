# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
"""HuggingFace AutoConfig registration entry for minimind-o."""

from vllm_omni.model_executor.models.minimind_o.config import MiniMindOConfig

MiniMindOmniConfig = MiniMindOConfig

__all__ = ["MiniMindOmniConfig", "MiniMindOConfig"]
