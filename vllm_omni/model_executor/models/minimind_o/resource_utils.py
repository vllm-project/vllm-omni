from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=16)
def resolve_model_dir(path_or_repo: str, resource_name: str) -> str:
    """Return a local directory for either a local path or an HF repo ID."""
    if not path_or_repo:
        raise ValueError(f"MiniMind-O {resource_name} path or repo id is empty.")

    local_dir = Path(path_or_repo).expanduser()
    if local_dir.exists():
        return str(local_dir)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"huggingface_hub is required to download MiniMind-O {resource_name} "
            f"from '{path_or_repo}'."
        ) from exc

    logger.info("Downloading MiniMind-O %s from %s", resource_name, path_or_repo)
    try:
        return snapshot_download(repo_id=path_or_repo)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve MiniMind-O {resource_name} from '{path_or_repo}'."
        ) from exc
