# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SenseNova-Vision-7B-MoT config helpers.

SenseNova-Vision stores its real Qwen2/SigLIP parameters in sibling
``llm_config.json`` / ``vit_config.json`` (BAGEL-fork layout) while the
top-level ``config.json`` is metadata-only. ``BagelConfig`` builds proper
``Qwen2Config`` / ``SiglipVisionConfig`` from the nested ``llm_config`` /
``vit_config`` keys, so this helper merges the sibling configs into a patched
``config.json`` before vLLM parses it.
"""

from __future__ import annotations

import json
import os

from vllm.logger import init_logger

logger = init_logger(__name__)


def merge_sensenova_split_configs(config_root: str, hf_config_path: str) -> None:
    """Merge SenseNova-Vision-7B-MoT's split configs into a patched config dir.

    Reads ``llm_config.json`` / ``vit_config.json`` from the checkpoint root
    (local dir or HF cache) and injects them as nested ``llm_config`` /
    ``vit_config`` keys into ``hf_config_path/config.json`` so BagelConfig
    builds the real sub-configs instead of empty defaults.

    Args:
        config_root: The checkpoint root (local directory or HF repo id).
        hf_config_path: Directory whose ``config.json`` is parsed by vLLM.
    """
    if not hf_config_path:
        return
    config_path = os.path.join(hf_config_path, "config.json")
    if not os.path.isfile(config_path):
        return
    try:
        with open(config_path) as f:
            config_dict = json.load(f)
    except (OSError, ValueError):
        return

    # Resolve the checkpoint root (local dir or HF cache) for the sibling files.
    if not os.path.isdir(config_root):
        try:
            from huggingface_hub import hf_hub_download

            def _load_sibling(name: str) -> dict | None:
                try:
                    p = hf_hub_download(config_root, name)
                    with open(p) as f:
                        return json.load(f)
                except Exception:
                    return None
        except Exception:
            return
    else:

        def _load_sibling(name: str) -> dict | None:
            try:
                with open(os.path.join(config_root, name)) as f:
                    return json.load(f)
            except (OSError, ValueError):
                return None

    llm_config = _load_sibling("llm_config.json")
    vit_config = _load_sibling("vit_config.json")
    if not llm_config or not vit_config:
        logger.warning(
            "SenseNova-Vision: could not load llm_config.json/vit_config.json for %s",
            config_root,
        )
        return

    config_dict["llm_config"] = llm_config
    config_dict["vit_config"] = vit_config
    with open(config_path, "w") as f:
        json.dump(config_dict, f)
    logger.info("Merged SenseNova-Vision split configs (llm_config/vit_config) into patched config")
