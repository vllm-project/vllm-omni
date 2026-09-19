# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixtures for Omni-DuplexEval CI guard."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def judge_server() -> str:
    """Return Judge model server base_url (does not start the server).

    Endpoint resolution order:
      1. Environment variable named by ``judge.base_url_env`` (e.g.
         ``VLLM_DUPLEX_EVAL_JUDGE_URL``).
      2. ``judge.base_url`` from the CI config JSON.

    Each CI queue (CUDA / NPU) can inject a different Judge endpoint via
    the environment variable — Q7 design.
    """
    config_path = Path(__file__).resolve().parent / "omni_duplex_eval_ci_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    judge_cfg = config["judge"]
    env_name = judge_cfg.get("base_url_env")
    base_url = os.environ.get(env_name) if env_name else None
    if base_url is not None:
        return base_url
    return str(judge_cfg.get("base_url", "http://127.0.0.1:8001"))
