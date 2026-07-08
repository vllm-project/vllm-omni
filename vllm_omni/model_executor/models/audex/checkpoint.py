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

# The subset of the HF repo the TTS pipeline needs: thinker checkpoint +
# tokenizer, streaming decoder, the deduplicated weight shard (see
# ensure_audiogen_weights), and the root manifest.
_REQUIRED_SNAPSHOT_PATTERNS = [
    "config.json",
    "checkpoint_folder_audiogen/*",
    "audex_causal_speech_decoder/*",
    f"{_WEIGHT_SOURCE_FOLDER}/model-00001-of-00002.safetensors",
]


def ensure_audex_snapshot(model: str) -> str:
    """Resolve ``model`` to a local repo-root directory, downloading if needed.

    Local paths pass through untouched. For an HF repo id, download (or reuse
    from cache) exactly the subset the TTS pipeline needs, so per-stage
    ``model_subdir`` joins land on a real snapshot instead of the repo-id
    string on a fresh cache. Falls back to a cached snapshot when offline.
    """
    if os.path.isdir(model):
        return model

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model, allow_patterns=_REQUIRED_SNAPSHOT_PATTERNS)
    except Exception as download_exc:
        try:
            return snapshot_download(model, allow_patterns=_REQUIRED_SNAPSHOT_PATTERNS, local_files_only=True)
        except Exception:
            raise RuntimeError(
                f"Could not resolve the Audex repo {model!r}: the download failed and no "
                "cached snapshot with the required folders (checkpoint_folder_audiogen/, "
                "audex_causal_speech_decoder/) was found."
            ) from download_exc


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
