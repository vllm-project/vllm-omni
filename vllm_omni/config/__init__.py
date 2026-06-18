"""
Configuration module for vLLM-Omni.
"""

from vllm_omni.config.lora import LoRAConfig
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.config.omni_config import (
    CacheConfig,
    ConnectorConfig,
    DiffusionConfig,
    LoadConfig,
    ModelConfig,
    OrchestratorConfig,
    ParallelConfig,
    RuntimeConfig,
    SchedulerConfig,
    VllmOmniConfig,
    VllmOmniStageConfig,
)
from vllm_omni.config.stage_config import (
    PIPELINE_WIDE_ENGINE_FIELDS,
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageConfigFactory,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    load_deploy_config,
    merge_pipeline_deploy,
    register_pipeline,
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
    "VllmOmniStageConfig",
    # Structured Omni sub-configs.
    "CacheConfig",
    "ConnectorConfig",
    "DiffusionConfig",
    "LoadConfig",
    "ModelConfig",
    "OrchestratorConfig",
    "ParallelConfig",
    "RuntimeConfig",
    "SchedulerConfig",
    # Legacy pipeline/stage deploy config surface.
    "PIPELINE_WIDE_ENGINE_FIELDS",
    "DeployConfig",
    "PipelineConfig",
    "StageConfig",
    "StageConfigFactory",
    "StageDeployConfig",
    "StageExecutionType",
    "StagePipelineConfig",
    "StageType",
    "load_deploy_config",
    "merge_pipeline_deploy",
    "register_pipeline",
    # YAML utility helpers.
    "create_config",
    "load_yaml_config",
    "merge_configs",
    "to_dict",
]
