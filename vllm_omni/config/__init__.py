"""
Configuration management for vLLM-omni.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
from vllm_omni.dit_cache_interface import DiTCacheConfig

from vllm.config import VllmConfig


@dataclass
class OmniConfig:
    """
    The configuration for vLLM-omni.
    """

    """vllm config"""
    vllm_config: VllmConfig = field(default_factory=VllmConfig)
    """DiT cache config"""
    dit_cache_config: DiTCacheConfig = field(default_factory=DiTCacheConfig)    # DiT cache config