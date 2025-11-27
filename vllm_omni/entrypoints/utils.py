from __future__ import annotations

import logging
import os
from pathlib import Path

from omegaconf import OmegaConf
from vllm.transformers_utils.config import get_config

logger = logging.getLogger(__name__)

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_stage_configs_from_model(model: str) -> list:
    """Load stage configurations from model's default config file.

    Loads stage configurations based on the model type. Looks for a
    YAML configuration file in the stage_configs directory matching
    the model's model_type.

    Args:
        model: Model name or path (used to determine model_type)

    Returns:
        List of stage configuration dictionaries

    Raises:
        FileNotFoundError: If no stage config file exists for the model type
    """
    hf_config = get_config(model, trust_remote_code=True)
    model_type = hf_config.model_type
    stage_config_file = f"vllm_omni/model_executor/stage_configs/{model_type}.yaml"
    stage_config_path = PROJECT_ROOT / stage_config_file
    if not os.path.exists(stage_config_path):
        raise FileNotFoundError(f"Stage config file {stage_config_path} not found")
    stage_configs = load_stage_configs_from_yaml(config_path=str(stage_config_path))
    return stage_configs


def load_stage_configs_from_yaml(config_path: str) -> list:
    """Load stage configurations from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        List of stage configuration dictionaries from the file's stage_args
    """
    config_data = OmegaConf.load(config_path)
    return config_data.stage_args
