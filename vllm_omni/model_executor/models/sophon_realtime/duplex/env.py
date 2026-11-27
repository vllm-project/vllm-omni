# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Find the served Sophon-Realtime directory / port and whether its thinker has a vision tower (no config plumbing
needed)."""

import json
import os
import sys


def _argv_after(flag):
    a = sys.argv
    for i, x in enumerate(a):
        if x == flag and i + 1 < len(a):
            return a[i + 1]
        if x.startswith(flag + "="):
            return x.split("=", 1)[1]
    return None


def model_dir():
    """SOPHON_MODEL, else the positional model of `vllm serve <model>` (local dir or HF id)."""
    if os.environ.get("SOPHON_MODEL"):
        return os.environ["SOPHON_MODEL"]
    a = sys.argv
    if "serve" in a:
        i = a.index("serve")
        if i + 1 < len(a) and not a[i + 1].startswith("-"):
            return a[i + 1]
    return _argv_after("--model")


def port():
    return os.environ.get("SOPHON_PORT") or _argv_after("--port") or "8000"


def api_key():
    """The key `vllm serve` checks (``--api-key``, else ``VLLM_API_KEY``), for calls to this server's own endpoints."""
    return _argv_after("--api-key") or os.environ.get("VLLM_API_KEY") or None


def vision():
    """SOPHON_VISION=0/1 overrides; otherwise look for a vision tower in <model>/thinker (HF ids: assume the released
    vision model)."""
    v = os.environ.get("SOPHON_VISION")
    if v is not None:
        return v == "1"
    d = model_dir()
    if d and os.path.isdir(d):
        cfg = os.path.join(d, "thinker", "config.json")
        try:
            return "vision_config" in json.load(open(cfg))
        except Exception:  # noqa: BLE001
            return os.path.exists(os.path.join(d, "thinker", "visual.safetensors"))
    return True
