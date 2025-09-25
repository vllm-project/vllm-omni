"""
Stage manager for orchestrating multiple engines in vLLM-omni.
"""

from typing import List, Optional
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.async_llm import AsyncLLM

from ..config import OmniStageConfig


class StageManager:
    """Manages multiple stage engines for multi-stage processing."""
    
    def __init__(self, stage_configs: List[OmniStageConfig], log_stats: bool = False):
        self.stage_configs = stage_configs
        self.log_stats = log_stats
        self.engine_list: List[LLMEngine] = []
        self.async_engine_list: List[AsyncLLM] = []
        self._initialized = False
        self._async_initialized = False
    
    def initialize_engines(self) -> None:
        """Initialize LLMEngine instances for each stage."""
        if self._initialized:
            return
        
        # For now, create placeholder engines
        # In a full implementation, this would create actual engines
        for stage_config in self.stage_configs:
            # Placeholder - would create actual engine here
            self.engine_list.append(None)
        
        self._initialized = True
    
    def initialize_async_engines(self) -> None:
        """Initialize AsyncLLM instances for each stage."""
        if self._async_initialized:
            return
        
        # For now, create placeholder engines
        # In a full implementation, this would create actual engines
        for stage_config in self.stage_configs:
            # Placeholder - would create actual engine here
            self.async_engine_list.append(None)
        
        self._async_initialized = True
    
    def get_engine(self, stage_id: int) -> LLMEngine:
        """Get the engine for a specific stage."""
        if not self._initialized:
            self.initialize_engines()
        
        if stage_id >= len(self.engine_list):
            raise IndexError(f"Stage {stage_id} not found. Available stages: 0-{len(self.engine_list)-1}")
        
        return self.engine_list[stage_id]
    
    def get_async_engine(self, stage_id: int) -> AsyncLLM:
        """Get the async engine for a specific stage."""
        if not self._async_initialized:
            self.initialize_async_engines()
        
        if stage_id >= len(self.async_engine_list):
            raise IndexError(f"Async stage {stage_id} not found. Available stages: 0-{len(self.async_engine_list)-1}")
        
        return self.async_engine_list[stage_id]
    
    def get_stage_config(self, stage_id: int) -> OmniStageConfig:
        """Get the configuration for a specific stage."""
        if stage_id >= len(self.stage_configs):
            raise IndexError(f"Stage config {stage_id} not found. Available stages: 0-{len(self.stage_configs)-1}")
        
        return self.stage_configs[stage_id]
    
    def get_num_stages(self) -> int:
        """Get the number of stages."""
        return len(self.stage_configs)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up engines if needed
        self.engine_list.clear()
        self.async_engine_list.clear()
        self._initialized = False
        self._async_initialized = False
