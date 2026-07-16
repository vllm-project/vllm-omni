"""
Configuration module for vLLM-Omni.
"""

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.lora import LoRAConfig
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.config.omni_config import (
    BaseVllmOmniStageConfig,
    OmniStageCacheConfig,
    OmniStageConnectorConfig,
    OmniStageDiffusionParallelConfig,
    OmniStageLoadConfig,
    OmniStageModelConfig,
    OmniStageParallelConfig,
    OmniStageRuntimeConfig,
    OmniStageSchedulerConfig,
    StageConfigType,
    VllmOmniARStageConfig,
    VllmOmniConfig,
    VllmOmniDiffusionStageConfig,
    VllmOmniGenerationStageConfig,
    VllmOmniOrchestratorConfig,
)
from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import (
    PIPELINE_WIDE_ENGINE_FIELDS,
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.config.yaml_util import (
    create_config,
    load_yaml_config,
    merge_configs,
    to_dict,
)

__all__ = [
    # Legacy model-level configs.
    "LoRAConfig",
    "OmniModelConfig",
    # Structured Omni config entry points.
    "VllmOmniConfig",
    "BaseVllmOmniStageConfig",
    "VllmOmniARStageConfig",
    "VllmOmniGenerationStageConfig",
    "VllmOmniDiffusionStageConfig",
    "StageConfigType",
    # Structured Omni sub-configs.
    "OmniStageCacheConfig",
    "OmniStageConnectorConfig",
    "OmniStageDiffusionParallelConfig",
    "OmniStageLoadConfig",
    "OmniStageModelConfig",
    "VllmOmniOrchestratorConfig",
    "OmniStageParallelConfig",
    "OmniStageRuntimeConfig",
    "OmniStageSchedulerConfig",
    # Legacy pipeline/stage deploy config surface.
    "PIPELINE_WIDE_ENGINE_FIELDS",
    "DeployConfig",
    "PipelineConfig",
    "StageConfig",
    "StageConfigFactory",
    "StageDeployConfig",
    "StageType",
    "StageExecutionType",
    "StagePipelineConfig",
    "load_deploy_config",
    "merge_pipeline_deploy",
    "register_pipeline",
    # YAML utility helpers.
    "create_config",
    "load_yaml_config",
    "merge_configs",
    "to_dict",
]
