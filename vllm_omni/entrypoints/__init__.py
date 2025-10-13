"""
Entrypoints for vLLM-omni.
"""

from .omni_llm import OmniLLM
from .utils import load_stage_configs, load_stage_configs_from_yaml

__all__ = [
    "OmniLLM",
    "load_stage_configs",
    "load_stage_configs_from_yaml",
]

