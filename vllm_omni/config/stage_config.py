"""
Stage configuration for vLLM-omni multi-stage processing.
"""

from dataclasses import dataclass
from typing import List, Optional, Type, Literal, Any
from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorBase as Executor


@dataclass
class DiTConfig:
    """Configuration for DiT (Diffusion Transformer) stages."""
    model_type: str
    scheduler_type: str
    num_inference_steps: int
    guidance_scale: float = 7.5
    use_diffusers: bool = False
    diffusers_pipeline: Optional[str] = None
    height: int = 512
    width: int = 512
    batch_size: int = 1


@dataclass
class DiTCacheTensor:
    """Configuration for DiT cache tensors."""
    name: str
    shape: List[int]
    dtype: str = "float32"
    persistent: bool = True


@dataclass
class DiTCacheConfig:
    """Configuration for DiT caching system."""
    cache_tensors: List[DiTCacheTensor]
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB
    cache_strategy: str = "fifo"  # fifo, lru, lfu
    enable_optimization: bool = True
    cache_compression: bool = False


@dataclass
class OmniStageConfig:
    """Configuration for a processing stage in vLLM-omni."""
    stage_id: int
    engine_type: Literal["AR", "DiT"]
    model_path: str
    input_modalities: List[str]
    output_modalities: List[str]
    vllm_config: Optional[VllmConfig] = None
    executor_class: Type[Executor] = None  # Will be set based on engine_type
    dit_config: Optional[DiTConfig] = None
    cache_config: Optional[DiTCacheConfig] = None
    stage_output: Optional[Any] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.engine_type not in ["AR", "DiT"]:
            raise ValueError(f"Invalid engine_type: {self.engine_type}. Must be 'AR' or 'DiT'")
        
        if self.engine_type == "DiT" and self.dit_config is None:
            raise ValueError("DiT engine requires dit_config")
        
        if not self.input_modalities:
            raise ValueError("input_modalities cannot be empty")
        
        if not self.output_modalities:
            raise ValueError("output_modalities cannot be empty")
    
    def get_executor_class(self) -> Type[Executor]:
        """Get the appropriate executor class for this stage."""
        if self.executor_class is not None:
            return self.executor_class
        
        if self.engine_type == "AR":
            from vllm.executor.uniproc_executor import UniProcExecutor
            return UniProcExecutor
        elif self.engine_type == "DiT":
            if self.dit_config and self.dit_config.use_diffusers:
                from vllm_omni.executor.diffusers_executor import DiffusersPipelineExecutor
                return DiffusersPipelineExecutor
            else:
                from vllm.executor.uniproc_executor import UniProcExecutor
                return UniProcExecutor
        
        raise ValueError(f"No executor class available for engine_type: {self.engine_type}")


def create_ar_stage_config(
    stage_id: int,
    model_path: str,
    input_modalities: List[str] = None,
    output_modalities: List[str] = None,
    vllm_config: Optional[VllmConfig] = None
) -> OmniStageConfig:
    """Create a configuration for an AR (Autoregressive) stage."""
    if input_modalities is None:
        input_modalities = ["text"]
    if output_modalities is None:
        output_modalities = ["text"]
    
    return OmniStageConfig(
        stage_id=stage_id,
        engine_type="AR",
        model_path=model_path,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        vllm_config=vllm_config
    )


def create_dit_stage_config(
    stage_id: int,
    model_path: str,
    input_modalities: List[str] = None,
    output_modalities: List[str] = None,
    dit_config: Optional[DiTConfig] = None,
    cache_config: Optional[DiTCacheConfig] = None,
    vllm_config: Optional[VllmConfig] = None
) -> OmniStageConfig:
    """Create a configuration for a DiT (Diffusion Transformer) stage."""
    if input_modalities is None:
        input_modalities = ["text"]
    if output_modalities is None:
        output_modalities = ["image"]
    
    if dit_config is None:
        dit_config = DiTConfig(
            model_type="dit",
            scheduler_type="ddpm",
            num_inference_steps=50
        )
    
    return OmniStageConfig(
        stage_id=stage_id,
        engine_type="DiT",
        model_path=model_path,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        vllm_config=vllm_config,
        dit_config=dit_config,
        cache_config=cache_config
    )
