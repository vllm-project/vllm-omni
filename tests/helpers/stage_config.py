"""Config/message construction helpers used by tests."""

import atexit
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def modify_stage_config(
    yaml_path: str,
    updates: dict[str, Any] = None,
    deletes: dict[str, Any] = None,
) -> str:
    """
    Modify configurations in a YAML file, supporting both top-level and stage-specific modifications,
    including addition, modification, and deletion of configurations.

    Args:
        yaml_path: Path to the YAML configuration file.
        updates: Dictionary containing both top-level and stage-specific modifications to add or update.
                Format: {
                    'async_chunk': True,
                    'stage_args': {
                        0: {'engine_args.max_model_len': 5800},
                        1: {'engine_args.max_num_seqs': 2}
                    }
                }
        deletes: Dictionary containing configurations to delete.
                Format: {
                    'old_config': None,  # Delete entire key
                    'stage_args': {
                        0: ['engine_args.old_param'],
                        1: ['runtime.unused_setting']
                    }
                }

    Returns:
        str: Path to the newly created modified YAML file with timestamp suffix.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"yaml does not exist: {path}")

    try:
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Cannot parse YAML file: {e}")

    # Helper function to apply update
    def apply_update(config_dict: dict, key_path: str, value: Any) -> None:
        """Apply update to dictionary using dot-separated path."""
        # Handle direct list assignment (e.g., engine_input_source: [1, 2])
        if "." not in key_path:
            # Simple key, set directly
            config_dict[key_path] = value
            return

        current = config_dict
        keys = key_path.split(".")

        for i in range(len(keys) - 1):
            key = keys[i]

            # Handle list indices
            if key.isdigit() and isinstance(current, list):
                index = int(key)
                if index < 0:
                    raise ValueError(f"Negative list index not allowed: {index}")
                if index >= len(current):
                    # Expand list if needed
                    while len(current) <= index:
                        # If we need to go deeper (more keys after this), create a dict
                        # Otherwise, create None placeholder
                        current.append({} if i < len(keys) - 2 else None)
                current = current[index]
            elif isinstance(current, dict):
                # Handle dictionary keys
                if key not in current:
                    # If there are more keys after this, create appropriate structure
                    if i < len(keys) - 1:
                        # Check if next key is a digit (list index) or string (dict key)
                        if keys[i + 1].isdigit():
                            current[key] = []
                        else:
                            current[key] = {}
                    else:
                        # This is the last key, create based on value type
                        current[key] = [] if isinstance(value, list) else {}
                elif not isinstance(current[key], (dict, list)) and i < len(keys) - 1:
                    # If current value is not dict/list but we need to go deeper, replace it
                    if keys[i + 1].isdigit():
                        current[key] = []
                    else:
                        current[key] = {}
                current = current[key]
            else:
                # Current is not a dict or list, cannot traverse further
                raise TypeError(
                    f"Cannot access {'.'.join(keys[: i + 1])} as a dict/list. It's a {type(current).__name__}"
                )

        # Set the final value
        last_key = keys[-1]
        if isinstance(current, list) and last_key.isdigit():
            # Setting a value in a list by index
            index = int(last_key)
            if index < 0:
                raise ValueError(f"Negative list index not allowed: {index}")
            if index >= len(current):
                # Expand list if needed
                while len(current) <= index:
                    current.append(None)
            current[index] = value
        elif isinstance(current, dict):
            # Special case: if the value is a list and we're setting a top-level key
            # Example: updating engine_input_source with [1, 2]
            current[last_key] = value
        else:
            # Current is not a dict, cannot set key
            raise TypeError(f"Cannot set value at {key_path}. Current type is {type(current).__name__}, expected dict.")

    # Helper function to delete by path
    def delete_by_path(config_dict: dict, path: str) -> None:
        """Delete configuration by dot-separated path."""
        if not path:
            return

        current = config_dict
        keys = path.split(".")

        # Traverse to the parent
        for i in range(len(keys) - 1):
            key = keys[i]

            # Handle list indices
            if key.isdigit() and isinstance(current, list):
                index = int(key)
                if index < 0 or index >= len(current):
                    raise KeyError(f"List index {index} out of bounds")
                current = current[index]
            elif isinstance(current, dict):
                if key not in current:
                    raise KeyError(f"Path {'.'.join(keys[: i + 1])} does not exist")
                current = current[key]
            else:
                raise TypeError(
                    f"Cannot access {'.'.join(keys[: i + 1])} as a dict/list. It's a {type(current).__name__}"
                )

        # Delete the item
        last_key = keys[-1]

        if isinstance(current, list) and last_key.isdigit():
            index = int(last_key)
            if index < 0 or index >= len(current):
                raise KeyError(f"List index {index} out of bounds")
            del current[index]
        elif isinstance(current, dict) and last_key in current:
            del current[last_key]
        else:
            print(f"Path {path} does not exist")

    _stage_key = "stages" if "stages" in config else "stage_args"

    # Apply deletions first
    if deletes:
        for key, value in deletes.items():
            if key in ("stage_args", "stages"):
                if value and isinstance(value, dict):
                    stage_args = config.get(_stage_key, [])
                    if not stage_args:
                        raise ValueError("stage_args does not exist in config")

                    for stage_id, delete_paths in value.items():
                        if not delete_paths:
                            continue

                        # Find stage by ID
                        target_stage = None
                        for stage in stage_args:
                            if stage.get("stage_id") == int(stage_id):
                                target_stage = stage
                                break

                        if target_stage is None:
                            continue

                        # Delete specified paths in this stage
                        # Avoid shadowing the original YAML Path used for the output filename below.
                        for delete_path in delete_paths:
                            if delete_path:  # Skip empty paths
                                delete_by_path(target_stage, delete_path)
            elif "." in key:
                # Delete using dot-separated path
                delete_by_path(config, key)
            elif value is None and key in config:
                # Delete entire key
                del config[key]

    # Apply updates
    if updates:
        for key, value in updates.items():
            if key in ("stage_args", "stages"):
                if value and isinstance(value, dict):
                    stage_args = config.get(_stage_key, [])
                    if not stage_args:
                        raise ValueError("stage_args does not exist in config")

                    for stage_id, stage_updates in value.items():
                        # Find stage by ID
                        target_stage = None
                        for stage in stage_args:
                            if stage.get("stage_id") == int(stage_id):
                                target_stage = stage
                                break

                        if target_stage is None:
                            available_ids = [s.get("stage_id") for s in stage_args if "stage_id" in s]
                            raise KeyError(f"Stage ID {stage_id} not found, available: {available_ids}")

                        # Apply updates to this stage
                        for update_path, val in stage_updates.items():
                            # Check if this is a simple key (not dot-separated)
                            # Example: 'engine_input_source' vs 'engine_args.max_model_len'
                            if "." not in update_path:
                                # Direct key assignment (e.g., updating a list value)
                                target_stage[update_path] = val
                            else:
                                # Dot-separated path (e.g., nested dict access)
                                apply_update(target_stage, update_path, val)
            elif "." in key:
                # Apply using dot-separated path
                apply_update(config, key, value)
            else:
                # Direct top-level key
                config[key] = value

    # Unique suffix: multiple modify_stage_config calls in one process often run
    # within the same second (e.g. test_qwen3_omni_expansion imports both
    # get_chunk_config and get_batch_token_config). int(time.time()) would collide
    # and the later write would overwrite the earlier YAML on disk.
    # Keep generated configs outside the repo and delete them when pytest exits.
    output_fd, output_path = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".yaml")
    atexit.register(Path(output_path).unlink, missing_ok=True)

    with os.fdopen(output_fd, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False, allow_unicode=True, indent=2)

    return str(output_path)


# ``stage_config.py`` lives under ``tests/helpers/``; repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEPLOY_DIR = _REPO_ROOT / "vllm_omni" / "deploy"
_CI_GENERATED_DIR = _REPO_ROOT / "tests" / ".ci_generated"


# CI overlays as Python dicts (LSP-friendly). Materialized on demand to
# tests/.ci_generated/<model>.yaml via get_deploy_config_path("ci/<name>.yaml").
_CI_OVERLAYS: dict[str, dict[str, Any]] = {
    "qwen2_5_omni": {
        "base_config": "qwen2_5_omni.yaml",
        "async_chunk": False,
        "stages": [
            {
                "stage_id": 0,
                "max_model_len": 16384,
                "max_num_batched_tokens": 16384,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.9,
                "skip_mm_profiling": True,
                "load_format": "dummy",
                "default_sampling_params": {"max_tokens": 128},
            },
            {
                "stage_id": 1,
                "max_model_len": 16384,
                "max_num_batched_tokens": 16384,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.4,
                "skip_mm_profiling": True,
                "load_format": "dummy",
                "default_sampling_params": {"max_tokens": 4096},
            },
            {
                "stage_id": 2,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.5,
                "max_num_batched_tokens": 8192,
                "max_model_len": 8192,
                "load_format": "dummy",
                "devices": "2",
                "default_sampling_params": {"max_tokens": 8192},
            },
        ],
        "platforms": {
            "xpu": {
                "stages": [
                    {
                        "stage_id": 0,
                        "gpu_memory_utilization": 0.9,
                        "max_num_batched_tokens": 16384,
                        "max_model_len": 16384,
                    },
                    {"stage_id": 1, "gpu_memory_utilization": 0.5},
                    {
                        "stage_id": 2,
                        "gpu_memory_utilization": 0.3,
                        "max_num_batched_tokens": 4096,
                        "max_model_len": 4096,
                        "devices": "2",
                    },
                ],
            },
        },
    },
    "qwen3_omni_moe": {
        "base_config": "qwen3_omni_moe.yaml",
        "async_chunk": False,
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 5,
                "max_model_len": 32768,
                "mm_processor_cache_gb": 0,
                "load_format": "dummy",
                "default_sampling_params": {"max_tokens": 150, "ignore_eos": False},
            },
            {
                "stage_id": 1,
                "gpu_memory_utilization": 0.5,
                "max_num_seqs": 5,
                "max_model_len": 32768,
                "load_format": "dummy",
                "default_sampling_params": {"max_tokens": 1000},
            },
            {
                "stage_id": 2,
                "max_num_seqs": 5,
                "max_num_batched_tokens": 100000,
                "load_format": "dummy",
                "default_sampling_params": {"max_tokens": 2000},
            },
        ],
        "platforms": {
            "xpu": {
                "stages": [
                    {
                        "stage_id": 0,
                        "gpu_memory_utilization": 0.85,
                        "max_num_seqs": 1,
                        "tensor_parallel_size": 4,
                        "enforce_eager": True,
                        "max_num_batched_tokens": 4096,
                        "max_model_len": 4096,
                        "max_cudagraph_capture_size": 0,
                        "skip_mm_profiling": True,
                        "devices": "0,1,2,3",
                        "default_sampling_params": {"max_tokens": 100, "ignore_eos": False},
                    },
                    {
                        "stage_id": 1,
                        "gpu_memory_utilization": 0.6,
                        "max_num_seqs": 1,
                        "enforce_eager": True,
                        "max_num_batched_tokens": 4096,
                        "max_model_len": 4096,
                        "max_cudagraph_capture_size": 0,
                        "skip_mm_profiling": True,
                        "devices": "4",
                    },
                    {
                        "stage_id": 2,
                        "gpu_memory_utilization": 0.3,
                        "max_num_seqs": 1,
                        "max_num_batched_tokens": 100000,
                        "max_cudagraph_capture_size": 0,
                        "skip_mm_profiling": True,
                        "devices": "5",
                        "default_sampling_params": {"max_tokens": 2000},
                    },
                ],
            },
        },
    },
    "bagel": {
        "base_config": "bagel.yaml",
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 3,
                "gpu_memory_utilization": 0.45,
                "load_format": "dummy",
            },
            {
                "stage_id": 1,
                "max_num_seqs": 1,
                "load_format": "dummy",
            },
        ],
    },
    "bagel_think": {
        "base_config": "bagel_think.yaml",
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 3,
                "gpu_memory_utilization": 0.45,
                "load_format": "dummy",
            },
            {
                "stage_id": 1,
                "max_num_seqs": 1,
                "load_format": "dummy",
            },
        ],
    },
    "bagel_single_stage": {
        "base_config": "bagel_single_stage.yaml",
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 1,
                "load_format": "dummy",
            },
        ],
    },
    "bagel_mooncake": {
        "base_config": "bagel.yaml",
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.45,
                "load_format": "dummy",
                "output_connectors": {"to_stage_1": "mooncake_connector"},
            },
            {
                "stage_id": 1,
                "max_num_seqs": 1,
                "load_format": "dummy",
                "input_connectors": {"from_stage_0": "mooncake_connector"},
            },
        ],
        "connectors": {
            "mooncake_connector": {
                "name": "MooncakeConnector",
                "extra": {
                    "host": "${MOONCAKE_HOST}",
                    "metadata_server": "http://${MOONCAKE_HOST}:${MOONCAKE_HTTP_PORT}/metadata",
                    "master": "${MOONCAKE_HOST}:${MOONCAKE_RPC_PORT}",
                    "segment": 64000000,
                    "localbuf": 64000000,
                    "proto": "tcp",
                },
            },
        },
    },
    "ming_flash_omni": {
        "base_config": "ming_flash_omni.yaml",
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.74,
                "max_model_len": 16384,
                "max_num_batched_tokens": 16384,
                "mm_processor_cache_gb": 0,
                "skip_mm_profiling": True,
                "enable_flashinfer_autotune": False,
                "load_format": "dummy",
                "default_sampling_params": {
                    "temperature": 0.0,
                    "max_tokens": 100,
                },
            },
            {
                "stage_id": 1,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.18,
                "load_format": "dummy",
            },
        ],
    },
    "ming_flash_omni_thinker_only": {
        "base_config": "ming_flash_omni_thinker_only.yaml",
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 16384,
                "max_num_batched_tokens": 16384,
                "mm_processor_cache_gb": 0,
                "skip_mm_profiling": True,
                "enable_flashinfer_autotune": False,
                "load_format": "dummy",
                "default_sampling_params": {
                    "temperature": 0.4,
                    "max_tokens": 100,
                },
            },
        ],
    },
    # Single-stage thinker-only topology for the abort test.
    "qwen2_5_omni_thinker_only": {
        "async_chunk": False,
        "pipeline": "qwen2_5_omni_thinker_only",
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
                "enable_prefix_caching": False,
                "max_num_batched_tokens": 16384,
                "max_model_len": 16384,
                "skip_mm_profiling": True,
                "mm_processor_cache_gb": 0,
                "load_format": "dummy",
                "devices": "0",
                "default_sampling_params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 128,
                    "seed": 42,
                    "repetition_penalty": 1.1,
                },
            },
        ],
    },
}


def _materialize_ci_overlay(model_type: str) -> Path:
    import yaml

    if model_type not in _CI_OVERLAYS:
        raise KeyError(f"No CI overlay registered for {model_type!r}. Available: {sorted(_CI_OVERLAYS)}")

    _CI_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out = _CI_GENERATED_DIR / f"{model_type}.yaml"

    overlay = {**_CI_OVERLAYS[model_type]}
    base = overlay.get("base_config")
    if base:
        overlay["base_config"] = str(_DEPLOY_DIR / base)

    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(overlay, f, sort_keys=False)
    return out


def get_deploy_config_path(rel_path: str) -> str:
    """Resolve a deploy yaml; ``ci/<model>.yaml`` materializes from ``_CI_OVERLAYS``."""
    if rel_path.startswith("ci/") and rel_path.endswith(".yaml"):
        model_type = rel_path[len("ci/") : -len(".yaml")]
        if model_type in _CI_OVERLAYS:
            return str(_materialize_ci_overlay(model_type))
    return str(_DEPLOY_DIR / rel_path)


__all__ = [
    "modify_stage_config",
    "get_deploy_config_path",
]
