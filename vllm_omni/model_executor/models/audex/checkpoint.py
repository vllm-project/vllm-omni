# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex checkpoint layout preparation.

The nvidia/Nemotron-Labs-Audex-2B repo deduplicates weights: the
``checkpoint_folder_audiogen`` (and ``checkpoint_folder_textonly``) index
files reference safetensors shards that physically live only under
``checkpoint_folder_full``. The official repo ships a prepare script that
symlinks them; this replicates that step automatically so users can pass the
repo root without any manual preparation.
"""

import json
import os
import shutil

from vllm.logger import init_logger

logger = init_logger(__name__)

_WEIGHT_SOURCE_FOLDER = "checkpoint_folder_full"


def ensure_audiogen_weights(model_dir: str) -> None:
    """Link index-referenced shards missing from ``model_dir`` from the full checkpoint.

    No-op when the shards are already present (or the layout is unexpected);
    raises only if a referenced shard exists nowhere.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return
    try:
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", index_path, exc)
        return

    source_dir = os.path.join(os.path.dirname(os.path.abspath(model_dir)), _WEIGHT_SOURCE_FOLDER)
    for shard in sorted(set(weight_map.values())):
        dst = os.path.join(model_dir, shard)
        if os.path.exists(dst):
            continue
        src = os.path.join(source_dir, shard)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Audex checkpoint shard {shard!r} is missing from {model_dir} and "
                f"{source_dir}. Download the repo's {_WEIGHT_SOURCE_FOLDER}/ folder "
                "(it holds the deduplicated weight shards)."
            )
        try:
            os.symlink(os.path.relpath(src, model_dir), dst)
            logger.info("Linked Audex weight shard %s -> %s", dst, src)
        except OSError:
            shutil.copy2(src, dst)
            logger.info("Copied Audex weight shard %s -> %s", src, dst)
