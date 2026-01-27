"""
Configuration module for vLLM-Omni.
"""

from vllm_omni.config.lora import LoRAConfig
from vllm_omni.config.model import OmniModelConfig

__all__ = [
    "OmniModelConfig",
    "LoRAConfig",
]
