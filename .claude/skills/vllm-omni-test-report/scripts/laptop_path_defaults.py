"""Default laptop paths for local vLLM-Omni and vllm-omni-kanban checkouts."""

from __future__ import annotations

import os
from pathlib import Path

# Documented defaults (tilde form for agent prompts and help text).
DEFAULT_LAPTOP_REPO_ROOT_DISPLAY = "~/vllm-omni"
DEFAULT_KANBAN_REPO_ROOT_DISPLAY = "~/vllm-omni-kanban"

DEFAULT_LAPTOP_REPO_ROOT = Path(DEFAULT_LAPTOP_REPO_ROOT_DISPLAY)
DEFAULT_KANBAN_REPO_ROOT = Path(DEFAULT_KANBAN_REPO_ROOT_DISPLAY)


def resolve_laptop_repo_root(env_value: str | None = None) -> Path:
    """Resolve REPO_ROOT from env or ``~/vllm-omni``."""
    raw = env_value if env_value is not None else (os.environ.get("REPO_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_LAPTOP_REPO_ROOT.expanduser().resolve()


def resolve_kanban_repo_root(env_value: str | None = None) -> Path:
    """Resolve KANBAN_REPO_ROOT from env or ``~/vllm-omni-kanban``."""
    raw = env_value if env_value is not None else (os.environ.get("KANBAN_REPO_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_KANBAN_REPO_ROOT.expanduser().resolve()
