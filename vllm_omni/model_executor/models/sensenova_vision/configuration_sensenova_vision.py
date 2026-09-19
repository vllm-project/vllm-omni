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
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# BAGEL-compatible image-processor defaults used to patch SenseNova-Vision
# checkpoints that do not ship a preprocessor_config.json.  Transcribed from
# ByteDance-Seed/BAGEL-7B-MoT's preprocessor_config.json so the patched file
# drives SiglipImageProcessor exactly like the upstream BAGEL checkpoint.
# This is the canonical copy shared by the AR stage
# (vllm_omni/engine/arg_utils.py) and the DiT stage
# (vllm_omni/diffusion/models/sensenova_vision/pipeline_sensenova_vision.py)
# so the two stages stay in lockstep.
SENSENOVA_VISION_PREPROCESSOR_CONFIG: dict[str, Any] = {
    "image_processor_type": "SiglipImageProcessor",
    "size": {"height": 980, "width": 980},
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "do_resize": True,
    "do_rescale": True,
    "do_normalize": True,
    "do_convert_rgb": True,
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
}


def ensure_sensenova_preprocessor_config(config_dir: str) -> bool:
    """Write ``preprocessor_config.json`` into ``config_dir`` if it is missing.

    Args:
        config_dir: A directory that otherwise mirrors the checkpoint (e.g. a
            temp dir containing symlinks to the checkpoint files).

    Returns:
        True if the file was written, False if it already existed.
    """
    config_path = os.path.join(config_dir, "preprocessor_config.json")
    if os.path.isfile(config_path):
        return False
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(SENSENOVA_VISION_PREPROCESSOR_CONFIG, f)
    logger.info(
        "SenseNova-Vision: wrote BAGEL-compatible preprocessor_config.json at %s",
        config_path,
    )
    return True


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
