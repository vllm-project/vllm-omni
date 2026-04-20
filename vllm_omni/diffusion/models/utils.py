# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os


def _load_json(model_path: str, filename: str, local_files_only: bool = True) -> dict:
    """Load a JSON config file from a local path or HuggingFace Hub repo."""
    if local_files_only:
        path = os.path.join(model_path, *filename.split("/"))
        with open(path) as f:
            return json.load(f)
    else:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=model_path, filename=filename)
        with open(cached) as f:
            return json.load(f)
