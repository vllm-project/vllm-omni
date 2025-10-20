import os
from pathlib import Path
from omegaconf import OmegaConf
from vllm.transformers_utils.config import get_config

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_stage_configs_from_model(model: str):
    """Load stage configs from model."""
    hf_config = get_config(model, trust_remote_code=True)
    model_type = hf_config.model_type
    stage_config_file = f"vllm_omni/model_executor/stage_configs/{model_type}.yaml"
    stage_config_path = PROJECT_ROOT / stage_config_file
    if not os.path.exists(stage_config_path):
        raise FileNotFoundError(f"Stage config file {stage_config_path} not found")
    stage_configs = load_stage_configs_from_yaml(config_path=str(stage_config_path))
    return stage_configs


def load_stage_configs_from_yaml(config_path: str):
    """Load stage configs from yaml file."""
    config_data = OmegaConf.load(config_path)
    return config_data.stage_args