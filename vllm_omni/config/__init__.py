"""
Configuration module for vLLM-Omni.
"""

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

# StageConfigFactory / register_pipeline pull pipeline_registry, which eagerly
# imports PI0_PIPELINE → diffusion.data. Keep those lazy so
# `from vllm_omni.config.lora import LoRAConfig` (used while data.py is still
# loading) cannot close a circular import through DiffusionOutput.
_LAZY_ATTRS = {
    "StageConfigFactory": ("vllm_omni.config.config_factory", "StageConfigFactory"),
    "register_pipeline": ("vllm_omni.config.pipeline_registry", "register_pipeline"),
}

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


def __getattr__(name: str):
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
