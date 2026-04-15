"""Config/message construction helpers used by tests."""

import atexit
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def dummy_messages_from_mix_data(
    system_prompt: dict[str, Any] = None,
    video_data_url: Any = None,
    audio_data_url: Any = None,
    image_data_url: Any = None,
    content_text: str = None,
):
    """Create messages with video、image、audio data URL for OpenAI API."""
    if content_text is not None:
        content = [{"type": "text", "text": content_text}]
    else:
        content = []

    media_items = []
    if isinstance(video_data_url, list):
        for video_url in video_data_url:
            media_items.append((video_url, "video"))
    else:
        media_items.append((video_data_url, "video"))

    if isinstance(image_data_url, list):
        for url in image_data_url:
            media_items.append((url, "image"))
    else:
        media_items.append((image_data_url, "image"))

    if isinstance(audio_data_url, list):
        for url in audio_data_url:
            media_items.append((url, "audio"))
    else:
        media_items.append((audio_data_url, "audio"))

    content.extend(
        {"type": f"{media_type}_url", f"{media_type}_url": {"url": url}}
        for url, media_type in media_items
        if url is not None
    )
    messages = [{"role": "user", "content": content}]
    if system_prompt is not None:
        messages = [system_prompt] + messages
    return messages


def modify_stage_config(
    yaml_path: str,
    updates: dict[str, Any] = None,
    deletes: dict[str, Any] = None,
) -> str:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"yaml does not exist: {path}")

    try:
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Cannot parse YAML file: {e}")

    def apply_update(config_dict: dict, key_path: str, value: Any) -> None:
        if "." not in key_path:
            config_dict[key_path] = value
            return
        current = config_dict
        keys = key_path.split(".")
        for i in range(len(keys) - 1):
            key = keys[i]
            if key.isdigit() and isinstance(current, list):
                index = int(key)
                if index < 0:
                    raise ValueError(f"Negative list index not allowed: {index}")
                if index >= len(current):
                    while len(current) <= index:
                        current.append({} if i < len(keys) - 2 else None)
                current = current[index]
            elif isinstance(current, dict):
                if key not in current:
                    if keys[i + 1].isdigit():
                        current[key] = []
                    else:
                        current[key] = {}
                elif not isinstance(current[key], (dict, list)) and i < len(keys) - 1:
                    current[key] = [] if keys[i + 1].isdigit() else {}
                current = current[key]
            else:
                raise TypeError(
                    f"Cannot access {'.'.join(keys[: i + 1])} as a dict/list. It's a {type(current).__name__}"
                )

        last_key = keys[-1]
        if isinstance(current, list) and last_key.isdigit():
            index = int(last_key)
            if index < 0:
                raise ValueError(f"Negative list index not allowed: {index}")
            if index >= len(current):
                while len(current) <= index:
                    current.append(None)
            current[index] = value
        elif isinstance(current, dict):
            current[last_key] = value
        else:
            raise TypeError(f"Cannot set value at {key_path}.")

    def delete_by_path(config_dict: dict, path: str) -> None:
        if not path:
            return
        current = config_dict
        keys = path.split(".")
        for i in range(len(keys) - 1):
            key = keys[i]
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
                raise TypeError(f"Cannot access {'.'.join(keys[: i + 1])} as a dict/list.")
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

    if deletes:
        for key, value in deletes.items():
            if key == "stage_args":
                if value and isinstance(value, dict):
                    stage_args = config.get("stage_args", [])
                    if not stage_args:
                        raise ValueError("stage_args does not exist in config")

                    for stage_id, delete_paths in value.items():
                        if not delete_paths:
                            continue

                        target_stage = None
                        for stage in stage_args:
                            if stage.get("stage_id") == int(stage_id):
                                target_stage = stage
                                break

                        if target_stage is None:
                            continue

                        for delete_path in delete_paths:
                            if delete_path:
                                delete_by_path(target_stage, delete_path)
            elif "." in key:
                delete_by_path(config, key)
            elif value is None and key in config:
                del config[key]

    if updates:
        for key, value in updates.items():
            if key == "stage_args":
                if value and isinstance(value, dict):
                    stage_args = config.get("stage_args", [])
                    if not stage_args:
                        raise ValueError("stage_args does not exist in config")

                    for stage_id, stage_updates in value.items():
                        target_stage = None
                        for stage in stage_args:
                            if stage.get("stage_id") == int(stage_id):
                                target_stage = stage
                                break
                        if target_stage is None:
                            available_ids = [s.get("stage_id") for s in stage_args if "stage_id" in s]
                            raise KeyError(f"Stage ID {stage_id} not found, available: {available_ids}")
                        for p, val in stage_updates.items():
                            if "." not in p:
                                target_stage[p] = val
                            else:
                                apply_update(target_stage, p, val)
            elif "." in key:
                apply_update(config, key, value)
            else:
                config[key] = value

    # Unique paths: multiple modify_stage_config calls in one process would collide
    # if writing next to the source with a coarse timestamp. Use mkstemp and unlink at exit.
    output_fd, output_path_str = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".yaml")

    def _unlink_temp_cfg() -> None:
        try:
            Path(output_path_str).unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_unlink_temp_cfg)

    with os.fdopen(output_fd, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False, allow_unicode=True, indent=2)

    return str(output_path_str)


__all__ = [
    "dummy_messages_from_mix_data",
    "modify_stage_config",
]
