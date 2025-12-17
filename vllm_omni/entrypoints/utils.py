import os
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from vllm.transformers_utils.config import get_config

from vllm_omni.utils import detect_device_type

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _convert_dataclasses_to_dict(obj: Any) -> Any:
    """Recursively convert non-serializable objects to OmegaConf-compatible types.

    This is needed because OmegaConf cannot handle:
    - Dataclass objects with Literal type annotations (e.g., StructuredOutputsConfig)
    - Counter objects (from collections or vllm.utils)
    - Set objects
    - Other non-primitive types
    """
    # IMPORTANT: Check Counter BEFORE dict, since Counter is a subclass of dict
    # Handle Counter objects (convert to dict)
    # Check by class name first to catch both collections.Counter and vllm.utils.Counter
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "Counter":
        try:
            return dict(obj)
        except (TypeError, ValueError):
            # If Counter can't be converted to dict, return empty dict
            return {}
    # Also check isinstance for collections.Counter (must be before dict check)
    if isinstance(obj, Counter):
        return dict(obj)
    # Handle set objects (convert to list)
    if isinstance(obj, set):
        return list(obj)
    # Handle dataclass objects
    # Note: asdict() recursively converts nested dataclasses but not Counter objects,
    # so we need to recursively process the result
    if is_dataclass(obj):
        result = asdict(obj)
        # Recursively process the result to convert any Counter objects
        return _convert_dataclasses_to_dict(result)
    # Handle dictionaries (recurse into values)
    # Note: This must come AFTER Counter check since Counter is a dict subclass
    if isinstance(obj, dict):
        return {k: _convert_dataclasses_to_dict(v) for k, v in obj.items()}
    # Handle lists and tuples (recurse into items)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_convert_dataclasses_to_dict(item) for item in obj)
    # Try to convert any dict-like object (has keys/values methods) to dict
    if hasattr(obj, "keys") and hasattr(obj, "values") and not isinstance(obj, (str, bytes)):
        try:
            return {k: _convert_dataclasses_to_dict(v) for k, v in obj.items()}
        except (TypeError, ValueError, AttributeError):
            # If conversion fails, return as-is
            return obj
    # Primitive types and other objects that OmegaConf can handle
    return obj


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


def load_stage_configs_from_model(model: str, base_engine_args: dict | None = None) -> list:
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
    if base_engine_args is None:
        base_engine_args = {}
    stage_config_path = resolve_model_config_path(model)
    stage_configs = load_stage_configs_from_yaml(config_path=stage_config_path, base_engine_args=base_engine_args)
    return stage_configs


def load_stage_configs_from_yaml(config_path: str, base_engine_args: dict | None = None) -> list:
    """Load stage configurations from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        List of stage configuration dictionaries from the file's stage_args
    """
    if base_engine_args is None:
        base_engine_args = {}
    config_data = OmegaConf.load(config_path)
    stage_args = config_data.stage_args
    # Convert any nested dataclass objects to dicts before creating OmegaConf
    base_engine_args = _convert_dataclasses_to_dict(base_engine_args)
    base_engine_args = OmegaConf.create(base_engine_args)
    for stage_arg in stage_args:
        base_engine_args_tmp = base_engine_args.copy()
        # Update base_engine_args with stage-specific engine_args if they exist
        if hasattr(stage_arg, "engine_args") and stage_arg.engine_args is not None:
            base_engine_args_tmp = OmegaConf.merge(base_engine_args_tmp, stage_arg.engine_args)
        stage_arg.engine_args = base_engine_args_tmp
    return stage_args
