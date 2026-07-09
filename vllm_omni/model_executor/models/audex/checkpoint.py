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

# Download profiles: each deployment pulls only the repo subset its stages
# need. "tts" (thinker + streaming decoder + dedup shard) must stay
# byte-identical to the original single pattern list — a unit test pins it.
_SNAPSHOT_PROFILE_PATTERNS = {
    "tts": [
        "config.json",
        "checkpoint_folder_audiogen/*",
        "audex_causal_speech_decoder/*",
        f"{_WEIGHT_SOURCE_FOLDER}/model-00001-of-00002.safetensors",
    ],
    # TTA reuses the audiogen thinker; waveform decode happens in the
    # external XCodec1 checkpoint (see ensure_xcodec1_snapshot), so the
    # bundled speech decoder is not pulled.
    "tta": [
        "config.json",
        "checkpoint_folder_audiogen/*",
        f"{_WEIGHT_SOURCE_FOLDER}/model-00001-of-00002.safetensors",
    ],
    # The audio-capable full checkpoint: both weight shards, audio
    # preprocessor assets, chat template, and remote-code modeling files.
    "full": [
        "config.json",
        f"{_WEIGHT_SOURCE_FOLDER}/*",
    ],
}

XCODEC1_DEFAULT_REPO = "hf-audio/xcodec-hubert-general-balanced"


def ensure_audex_snapshot(model: str, profile: str = "tts") -> str:
    """Resolve ``model`` to a local repo-root directory, downloading if needed.

    Local paths pass through untouched. For an HF repo id, download (or reuse
    from cache) exactly the subset the requesting stage's ``profile`` needs,
    so per-stage ``model_subdir`` joins land on a real snapshot instead of
    the repo-id string on a fresh cache. Falls back to a cached snapshot
    when offline.
    """
    if os.path.isdir(model):
        return model

    patterns = _SNAPSHOT_PROFILE_PATTERNS.get(profile)
    if patterns is None:
        raise ValueError(
            f"Unknown Audex snapshot profile {profile!r}; expected one of {sorted(_SNAPSHOT_PROFILE_PATTERNS)}"
        )

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model, allow_patterns=patterns)
    except Exception as download_exc:
        try:
            return snapshot_download(model, allow_patterns=patterns, local_files_only=True)
        except Exception:
            raise RuntimeError(
                f"Could not resolve the Audex repo {model!r} (profile {profile!r}): the download "
                "failed and no cached snapshot with the required folders was found."
            ) from download_exc


def ensure_xcodec1_snapshot(model: str | None) -> str:
    """Resolve the external XCodec1 checkpoint (TTA waveform decoder).

    Accepts a local directory or an HF repo id; ``None``/empty falls back to
    the official ``hf-audio/xcodec-hubert-general-balanced``.
    """
    model = model or XCODEC1_DEFAULT_REPO
    if os.path.isdir(model):
        return model

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model)
    except Exception as download_exc:
        try:
            return snapshot_download(model, local_files_only=True)
        except Exception:
            raise RuntimeError(
                f"Could not resolve the XCodec1 checkpoint {model!r}: the download failed and "
                "no cached snapshot was found. Download it manually or point the TTA deploy "
                "yaml's stage-1 model at a local copy."
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
