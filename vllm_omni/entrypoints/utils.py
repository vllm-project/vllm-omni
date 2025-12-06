import os
from pathlib import Path

from omegaconf import OmegaConf
from vllm.transformers_utils.config import get_config

from vllm_omni.utils import detect_device_type

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def resolve_model_config_path(model: str) -> str:
    """Resolve the stage config file path from the model name.

    Resolves stage configuration path based on the model type and device type.
    First tries to find a device-specific YAML file from stage_configs/{device_type}/
    directory. If not found, falls back to the default config file.

    Args:
        model: Model name or path (used to determine model_type)

    Returns:
        String path to the stage configuration file

    Raises:
        FileNotFoundError: If no stage config file exists for the model type
    """
    hf_config = get_config(model, trust_remote_code=True)
    model_type = hf_config.model_type
    device_type = detect_device_type()

    # Try device-specific config first
    if device_type != "cuda":
        device_config_file = f"vllm_omni/model_executor/stage_configs/{device_type}/{model_type}.yaml"
        device_config_path = PROJECT_ROOT / device_config_file
        if os.path.exists(device_config_path):
            return str(device_config_path)

    # Fall back to default config
    stage_config_file = f"vllm_omni/model_executor/stage_configs/{model_type}.yaml"
    stage_config_path = PROJECT_ROOT / stage_config_file
    if not os.path.exists(stage_config_path):
        raise FileNotFoundError(f"Stage config file {stage_config_path} not found")
    return str(stage_config_path)


def load_stage_configs_from_model(model: str) -> list:
    """Load stage configurations from model's default config file.

    Loads stage configurations based on the model type and device type.
    First tries to load a device-specific YAML file from stage_configs/{device_type}/
    directory. If not found, falls back to the default config file.

    Args:
        model: Model name or path (used to determine model_type)

    Returns:
        List of stage configuration dictionaries

    Raises:
        FileNotFoundError: If no stage config file exists for the model type
    """
    stage_config_path = resolve_model_config_path(model)
    stage_configs = load_stage_configs_from_yaml(config_path=stage_config_path)
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
