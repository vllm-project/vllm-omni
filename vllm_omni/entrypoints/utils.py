from typing import List
import os
from omegaconf import OmegaConf
from vllm.transformers_utils.config import get_config
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.config import OmniStageConfig


def load_stage_configs(omni_args: OmniEngineArgs) -> List[OmniStageConfig]:
    """Load stage configs from model."""
    hf_config = get_config(**omni_args.__dict__)
    model_type = hf_config.model_type
    stage_config_file = f"vllm_omni/model_executor/stage_configs/{model_type}.yaml"
    if not os.path.exists(stage_config_file):
        raise FileNotFoundError(f"Stage config file {stage_config_file} not found")
    stage_configs = load_stage_configs_from_yaml(config_path=stage_config_file)
    return stage_configs


def load_stage_configs_from_model(model: str) -> List[OmniStageConfig]:
    """Load stage configs from model."""
    hf_config = get_config(model, trust_remote_code=True)
    model_type = hf_config.model_type
    stage_config_file = f"vllm_omni/model_executor/stage_configs/{model_type}.yaml"
    if not os.path.exists(stage_config_file):
        raise FileNotFoundError(f"Stage config file {stage_config_file} not found")
    stage_configs = load_stage_configs_from_yaml(config_path=stage_config_file)
    return stage_configs


def load_stage_configs_from_yaml(config_path: str) -> List[OmniStageConfig]:
    """Load stage configs from yaml file."""
    with open(config_path, "r") as f:
        config_data = OmegaConf.load(f)
    # Extract the list of stage configs from the yaml structure
    stage_configs = config_data.get('stage_engine_args', [])
    # Convert OmegaConf DictConfig objects to regular dicts
    stage_configs = [OmegaConf.to_container(cfg, resolve=True) for cfg in stage_configs]
    return stage_configs