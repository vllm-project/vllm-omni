# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru/haru/models/loaders.py @ d3612b0.
"""Checkpoint files of a Haru model: a local directory or a Hugging Face Hub repo."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


def checkpoint_file(path_or_repo: str, filename: str, revision: str | None = None) -> Path:
    """``filename`` of a checkpoint directory, or of a Hub repo (served from the local cache once downloaded)."""
    path = Path(path_or_repo).expanduser()
    if path.is_dir():
        return path / filename
    from vllm_omni.transformers_utils.repo_utils import hf_api

    return Path(hf_api().hf_hub_download(path_or_repo, filename, revision=revision))


def load_tokenizer(path_or_repo: str) -> Tokenizer:
    return Tokenizer.from_file(str(checkpoint_file(path_or_repo, "tokenizer.json")))
