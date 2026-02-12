# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Helpers for resolving diffusion model configs from local paths or HF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig


def _load_local_json(model_path: Path, relative_path: str | Path) -> dict[str, Any] | None:
    """Load JSON from a local model directory if the file exists."""
    candidate = model_path / Path(relative_path)
    if not candidate.is_file():
        return None
    with candidate.open("r", encoding="utf-8") as file:
        return json.load(file)


def _is_bagel_config(config: dict[str, Any]) -> bool:
    model_type = config.get("model_type")
    architectures = config.get("architectures") or []
    return model_type == "bagel" or "BagelForConditionalGeneration" in architectures


def _populate_local_model_config(od_config: OmniDiffusionConfig, model_path: Path) -> None:
    config_json = _load_local_json(model_path, "config.json")
    config_dict = _load_local_json(model_path, "model_index.json") or config_json
    if config_dict is None:
        raise ValueError(f"Local model directory '{model_path}' is missing model_index.json or config.json")

    od_config.model_class_name = config_dict.get("_class_name", None)
    od_config.update_multimodal_support()

    tf_config_dict = _load_local_json(model_path, Path("transformer") / "config.json")
    if tf_config_dict is not None:
        od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
        return

    bagel_cfg = config_dict if _is_bagel_config(config_dict) else config_json
    if bagel_cfg is not None and _is_bagel_config(bagel_cfg):
        od_config.model_class_name = "BagelPipeline"
        od_config.tf_model_config = TransformerConfig()
        od_config.update_multimodal_support()
        return

    raise ValueError(f"Local model directory '{model_path}' is missing transformer/config.json")


def _populate_hf_model_config(od_config: OmniDiffusionConfig) -> None:
    try:
        config_dict = get_hf_file_to_dict("model_index.json", od_config.model)
        od_config.model_class_name = config_dict.get("_class_name", None)
        od_config.update_multimodal_support()
        tf_config_dict = get_hf_file_to_dict("transformer/config.json", od_config.model)
        od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
    except (AttributeError, OSError, ValueError):
        cfg = get_hf_file_to_dict("config.json", od_config.model)
        if cfg is None:
            raise ValueError(f"Could not find config.json or model_index.json for model {od_config.model}")

        if _is_bagel_config(cfg):
            od_config.model_class_name = "BagelPipeline"
            od_config.tf_model_config = TransformerConfig()
            od_config.update_multimodal_support()


def populate_diffusion_model_config(od_config: OmniDiffusionConfig) -> None:
    """Populate model config fields on ``od_config`` from local files or HF."""
    if od_config.model is None:
        raise ValueError("od_config.model cannot be None for diffusion model initialization")

    model_path = Path(od_config.model)
    if model_path.exists():
        _populate_local_model_config(od_config, model_path)
        return

    _populate_hf_model_config(od_config)
