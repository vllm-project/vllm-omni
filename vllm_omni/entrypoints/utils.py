import os
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config, get_hf_file_to_dict
from vllm.transformers_utils.repo_utils import file_or_path_exists

from vllm_omni.utils import detect_device_type, is_rocm

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

logger = init_logger(__name__)


def _try_get_class_name_from_diffusers_config(model: str) -> str | None:
    """Try to get class name from diffusers model configuration files.

    Args:
        model: Model name or path

    Returns:
        Model type string if found, None otherwise
    """
    model_index = get_hf_file_to_dict("model_index.json", model, revision=None)
    if model_index and isinstance(model_index, dict) and "_class_name" in model_index:
        logger.debug(f"Found model_type '{model_index['_class_name']}' in model_index.json")
        return model_index["_class_name"]

    return None


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
        ValueError: If model_type cannot be determined
        FileNotFoundError: If no stage config file exists for the model type
    """
    # Try to get config from standard transformers format first
    try:
        hf_config = get_config(model, trust_remote_code=True)
        model_type = hf_config.model_type
    except (ValueError, Exception):
        # If standard transformers format fails, try diffusers format
        if file_or_path_exists(model, "model_index.json", revision=None):
            model_type = _try_get_class_name_from_diffusers_config(model)
            if model_type is None:
                raise ValueError(
                    f"Could not determine model_type for diffusers model: {model}. "
                    f"Please ensure the model has 'model_type' in transformer/config.json or model_index.json"
                )
        elif file_or_path_exists(model, "config.json", revision=None):
            # Try to read config.json manually for custom models like Bagel that fail get_config
            # but have a valid config.json with model_type
            try:
                config_dict = get_hf_file_to_dict("config.json", model, revision=None)
                if config_dict and "model_type" in config_dict:
                    model_type = config_dict["model_type"]
                else:
                    raise ValueError(f"config.json found but missing 'model_type' for model: {model}")
            except Exception as e:
                raise ValueError(f"Failed to read config.json for model: {model}. Error: {e}") from e
        else:
            raise ValueError(
                f"Could not determine model_type for model: {model}. "
                f"Model is not in standard transformers format and does not have model_index.json. "
                f"Please ensure the model has proper configuration files with 'model_type' field"
            )
    device_type = detect_device_type()

    # Try device-specific config first
    if device_type != "cuda" or is_rocm():
        device_config_file = f"vllm_omni/model_executor/stage_configs/{device_type}/{model_type}.yaml"
        if is_rocm():
            device_config_file = f"vllm_omni/model_executor/stage_configs/rocm/{model_type}.yaml"
        device_config_path = PROJECT_ROOT / device_config_file
        if os.path.exists(device_config_path):
            return str(device_config_path)

    # Fall back to default config
    stage_config_file = f"vllm_omni/model_executor/stage_configs/{model_type}.yaml"
    stage_config_path = PROJECT_ROOT / stage_config_file
    if not os.path.exists(stage_config_path):
        return None
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
    if stage_config_path is None:
        return []
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
        stage_type = getattr(stage_arg, "stage_type", "llm")
        if hasattr(stage_arg, "runtime") and stage_arg.runtime is not None and stage_type != "diffusion":
            runtime_cfg = stage_arg.runtime
            max_batch_size = int(runtime_cfg.get("max_batch_size", 1) or 1)
            base_engine_args_tmp["max_num_seqs"] = max_batch_size
        stage_arg.engine_args = base_engine_args_tmp
    return stage_args


def get_final_stage_id_for_e2e(
    output_modalities: list[str] | None, default_modalities: list[str], stage_list: list
) -> int:
    """Get the final stage id for e2e.

    Args:
        stage_list: List of stage configurations

    Returns:
        Final stage id for e2e
    """
    last_stage_id = len(stage_list) - 1
    if output_modalities is not None:
        prompt_modalities = []
        for modality in output_modalities:
            if modality not in default_modalities:
                logger.warning(f"Invalid output modality: {modality}, ignoring it")
                # TODO: if user specifies unsupported modalities, invalid it and raise an error
                continue
            prompt_modalities.append(modality)
        output_modalities = prompt_modalities
    else:
        output_modalities = default_modalities

    try:
        for _sid in range(last_stage_id, -1, -1):
            if (
                getattr(stage_list[_sid], "final_output", False)
                and stage_list[_sid].final_output_type in output_modalities
            ):
                final_stage_id_for_e2e = _sid
                break
        if final_stage_id_for_e2e < 0:
            final_stage_id_for_e2e = last_stage_id
    except Exception as e:
        logger.debug(
            "[Orchestrator] Failed to determine final stage for E2E; \
                falling back to last: %s",
            e,
            exc_info=True,
        )
        final_stage_id_for_e2e = last_stage_id

    return final_stage_id_for_e2e
