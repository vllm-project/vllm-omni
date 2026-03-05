import os
import types
from collections import Counter
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

from omegaconf import OmegaConf
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config, get_hf_file_to_dict
from vllm.transformers_utils.repo_utils import file_or_path_exists

from vllm_omni.entrypoints.stage_utils import _to_dict
from vllm_omni.platforms import current_omni_platform

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

logger = init_logger(__name__)


def inject_omni_kv_config(stage: Any, omni_conn_cfg: dict[str, Any], omni_from: str, omni_to: str) -> None:
    """Inject connector configuration into stage engine arguments."""
    # Prepare omni_kv_config dict
    omni_conf_dict = {}
    try:
        # Access engine_args safely (might be OmegaConf or dict)
        existing_args = stage.engine_args
        if hasattr(existing_args, "get"):
            _oc = existing_args.get("omni_kv_config", None)
            if _oc:
                if hasattr(_oc, "items"):  # dict-like
                    omni_conf_dict = dict(_oc)
                else:  # object?
                    omni_conf_dict = _to_dict(_oc)
    except Exception:
        omni_conf_dict = {}

    # Inject connector info
    omni_conf_dict["connector_config"] = omni_conn_cfg
    omni_conf_dict["omni_from_stage"] = omni_from
    omni_conf_dict["omni_to_stage"] = omni_to

    # Write back to engine_args
    try:
        if hasattr(stage.engine_args, "__setitem__"):
            stage.engine_args["omni_kv_config"] = omni_conf_dict
        else:
            setattr(stage.engine_args, "omni_kv_config", omni_conf_dict)
    except Exception as e:
        # Fallback for OmegaConf or similar if direct set fails?
        logger.error(f"Failed to inject omni connector config into stage: {e}")


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


def _filter_dict_like_object(obj: dict | Any) -> dict:
    """Filter dict-like object by removing callables and recursively converting values.

    Converts dict-like objects to regular dicts while filtering out callable values
    that are incompatible with OmegaConf. Recursively processes values through
    _convert_dataclasses_to_dict for nested object conversion.

    Args:
        obj: Dict or dict-like object to filter

    Returns:
        Regular dict with callables filtered out and values recursively converted

    Raises:
        TypeError: If obj doesn't support .items() method
        ValueError: If dict conversion fails unexpectedly
    """

    def _is_callable_value(value: Any) -> bool:
        if callable(value):
            return True
        return isinstance(
            value,
            (
                types.FunctionType,
                types.MethodType,
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
            ),
        )

    result = {}
    filtered_keys = []
    for k, v in obj.items():
        if _is_callable_value(v):
            filtered_keys.append(str(k))
        else:
            result[k] = _convert_dataclasses_to_dict(v)
    if filtered_keys:
        logger.warning(
            f"Filtered out {len(filtered_keys)} callable object(s) from base_engine_args "
            f"that are not compatible with OmegaConf: {filtered_keys}. "
        )
    return result


def _convert_dataclasses_to_dict(obj: Any) -> Any:
    """Recursively convert non-serializable objects to OmegaConf-compatible types.

    This is needed because OmegaConf cannot handle:
    - Dataclass objects with Literal type annotations (e.g., StructuredOutputsConfig)
    - Counter objects (from collections or vllm.utils)
    - Set objects
    - Callable objects (functions, methods, etc.)
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
    # Handle dictionaries (recurse into values) and filter out callables(cause error in OmegaConf.create)
    # Note: This must come AFTER Counter check since Counter is a dict subclass
    if isinstance(obj, dict):
        return _filter_dict_like_object(obj)
    # Handle callable objects (functions, methods, etc.) - skip them
    # Note: This comes after dict/list checks to avoid misclassifying dict-like objects
    if callable(obj):
        return None
    # Handle lists and tuples (recurse into items)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_convert_dataclasses_to_dict(item) for item in obj if not callable(item))
    # Try to convert any dict-like object (has keys/values methods) to dict
    if hasattr(obj, "keys") and hasattr(obj, "values") and not isinstance(obj, (str, bytes)):
        try:
            return _filter_dict_like_object(obj)
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

    default_config_path = current_omni_platform.get_default_stage_config_path()
    model_type_str = f"{model_type}.yaml"
    complete_config_path = PROJECT_ROOT / default_config_path / model_type_str
    if os.path.exists(complete_config_path):
        return str(complete_config_path)

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
    global_async_chunk = config_data.get("async_chunk", False)
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
            base_engine_args_tmp.async_chunk = global_async_chunk
        stage_arg.engine_args = base_engine_args_tmp
    return stage_args


def load_and_resolve_stage_configs(
    model: str,
    stage_configs_path: str | None,
    kwargs: dict | None,
    default_stage_cfg_factory: Any = None,
) -> tuple[str, list]:
    """Load stage configurations from model or YAML file with fallback to defaults.

    Args:
        model: Model name or path
        stage_configs_path: Optional path to YAML file containing stage configurations
        kwargs: Engine arguments to merge with stage configs
        default_stage_cfg_factory: Optional callable that takes no args and returns
            default stage config list when no configs are found

    Returns:
        Tuple of (config_path, stage_configs)
    """
    if stage_configs_path is None:
        config_path = resolve_model_config_path(model)
        stage_configs = load_stage_configs_from_model(model, base_engine_args=kwargs)
        if not stage_configs:
            if default_stage_cfg_factory is not None:
                default_stage_cfg = default_stage_cfg_factory()
                stage_configs = OmegaConf.create(default_stage_cfg)
            else:
                stage_configs = []
    else:
        config_path = stage_configs_path
        stage_configs = load_stage_configs_from_yaml(stage_configs_path, base_engine_args=kwargs)

    return config_path, stage_configs


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


def filter_dataclass_kwargs(cls: Any, kwargs: dict) -> dict:
    """Filter kwargs to only include fields defined in the dataclass.

    Args:
        cls: Dataclass type
        kwargs: Keyword arguments to filter

    Returns:
        Filtered keyword arguments containing only valid dataclass fields
    """
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dictionary")

    def _filter_value(value: Any, annotation: Any) -> Any:
        """Recursively filter nested dict/list values based on dataclass annotations."""
        if annotation is None:
            return value

        origin = get_origin(annotation)
        if origin is None:
            if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
                return filter_dataclass_kwargs(annotation, value)
            return value

        if origin in (list, tuple, set):
            args = get_args(annotation)
            inner = args[0] if args else None
            if isinstance(value, (list, tuple, set)):
                return type(value)(_filter_value(v, inner) for v in value)
            return value

        if origin is dict:
            args = get_args(annotation)
            val_type = args[1] if len(args) > 1 else None
            if isinstance(value, dict):
                return {k: _filter_value(v, val_type) for k, v in value.items()}
            return value

        if origin is types.UnionType or origin is getattr(types, "UnionType", None):
            for arg in get_args(annotation):
                if isinstance(arg, type) and is_dataclass(arg) and isinstance(value, dict):
                    return filter_dataclass_kwargs(arg, value)
                # Try container-style filtering for union members
                filtered = _filter_value(value, arg)
                if filtered is not value:
                    return filtered
            return value

        return value

    valid_fields = {f.name: f for f in fields(cls) if f.init}
    filtered_kwargs = {}
    for k, v in kwargs.items():
        if k not in valid_fields:
            continue
        field = valid_fields[k]
        filtered_kwargs[k] = _filter_value(v, field.type)

    return filtered_kwargs


# TODO(wuhang): Remove after PR #1115.
def build_base_engine_args(source: Any) -> dict[str, Any] | None:
    """Build base engine args with tokenizer and parallel configuration.

    Automatically detects whether source is a dict-like object or namespace object.

    Args:
        source: Source object (args namespace or kwargs dict) containing configuration.

    Returns:
        Dictionary containing tokenizer and parallel configuration overrides,
        or None if no configuration is present.
    """
    # Auto-detect source type: dict-like objects have 'get' method
    is_dict_like = hasattr(source, "get") and callable(getattr(source, "get"))

    # Extract tokenizer
    if is_dict_like:
        tokenizer = source.get("tokenizer", None)
    else:
        tokenizer = getattr(source, "tokenizer", None)

    base_engine_args = {"tokenizer": tokenizer} if tokenizer is not None else None

    # Extract parallel configuration
    parallel_keys = [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "data_parallel_size_local",
        "data_parallel_backend",
        "distributed_executor_backend",
    ]

    if is_dict_like:
        parallel_overrides = {k: source[k] for k in parallel_keys if k in source and source[k] is not None}
    else:
        parallel_overrides = {
            k: getattr(source, k) for k in parallel_keys if hasattr(source, k) and getattr(source, k) is not None
        }

    if parallel_overrides:
        base_engine_args = base_engine_args or {}
        base_engine_args.update(parallel_overrides)

    return base_engine_args


# The following code detects if the process is running in a container and if
# PID host is available. If so, we can use process-scoped memory tracking;
# otherwise we need sequential init locks.


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def in_container() -> bool:
    # Common Docker signal
    if os.path.exists("/.dockerenv"):
        return True

    # cgroup markers (works for Docker/containerd/K8s/Podman in many setups)
    cg = _read_text("/proc/1/cgroup") or ""
    markers = ("docker", "containerd", "kubepods", "libpod", "podman")
    return any(m in cg for m in markers)


def has_pid_host() -> bool | None:
    """
    Returns:
      True  -> very likely running with --pid=host (host PID namespace)
      False -> very likely isolated PID namespace (default)
      None  -> cannot determine
    """
    # Strong signal: in host pid namespace, PID 2 is usually kthreadd
    comm2 = _read_text("/proc/2/comm")
    if comm2 is not None:
        comm2 = comm2.strip()
        if comm2 == "kthreadd":
            return True
        # If PID 2 exists and is NOT kthreadd, we're almost certainly not in host pid ns
        return False

    # Fallback: check for other low-numbered kernel threads (best-effort)
    for pid, name in [(3, "rcu_gp"), (4, "rcu_par_gp"), (10, "ksoftirqd/0")]:
        comm = _read_text(f"/proc/{pid}/comm")
        if comm is not None:
            if comm.strip() == name:
                return True
            else:
                return False

    return False


def detect_pid_host() -> bool:
    ic = in_container()
    if not ic:
        return True

    return has_pid_host()
