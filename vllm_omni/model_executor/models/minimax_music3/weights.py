# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Component weight loading for MiniMax Music 3.

The checkpoint is a multi-component repo. The Qwen3 backbone is a normal
``Qwen3ForCausalLM`` under ``language_model/`` and loads through vLLM. The
audio embedding table and the depth decoder live in ``rvq_depth_decoder/``,
and the acoustic stage's three components live in their own folders. None of
those are things vLLM's loader knows how to map, so they are read directly.

Every loader here is strict. A silently partial load would produce audio that
sounds plausible but is wrong, which is far worse than a startup failure.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .constants import AUDIO_VOCAB_SIZE, NUM_CODEBOOKS

logger = init_logger(__name__)

_AUDIO_EMBEDDING_KEY = "audio_embeddings.weight"
_COMPONENT_WEIGHT_NAME = "diffusion_pytorch_model.safetensors"

# Component folders, relative to the repository root.
DEPTH_DECODER_DIR = "rvq_depth_decoder"
CONDITION_ENCODER_DIR = "condition_encoder"
TRANSFORMER_DIR = "transformer"
VOCODER_DIR = "vocoder"

# Folder names that identify a repository root. This is deliberately a
# name-only probe and NOT the set a stage must be able to load: see
# ``_REQUIRED_COMPONENTS``.
_ROOT_MARKERS = (DEPTH_DECODER_DIR, TRANSFORMER_DIR, VOCODER_DIR)

# Every component a stage actually reads. ``condition_encoder/`` is loaded by
# the acoustic stage, so a snapshot missing it is incomplete even though the
# root markers above are all present.
_REQUIRED_COMPONENTS = (DEPTH_DECODER_DIR, CONDITION_ENCODER_DIR, TRANSFORMER_DIR, VOCODER_DIR)

# The subset a stage reading the repo ROOT needs on disk. The AR backbone under
# ``language_model/`` is pulled by vLLM's own loader, and the repo's top-level
# ``.pth`` files and ``qwen_7B/`` folder belong to the reference implementation,
# so neither is downloaded here.
_COMPONENT_PATTERNS = (
    "*.json",
    f"{DEPTH_DECODER_DIR}/*",
    f"{CONDITION_ENCODER_DIR}/*",
    f"{TRANSFORMER_DIR}/*",
    f"{VOCODER_DIR}/*",
)


def _component_is_complete(folder: Path) -> bool:
    """True when *folder* holds a weight set ``load_component_state`` can read.

    A single shard satisfies a bare glob, so an interrupted multi-shard
    download would otherwise pass as complete. Hub shards are named
    ``<stem>-<index>-of-<total>.safetensors``, which states how many pieces
    to expect, so that run can be checked for gaps.

    Only that naming is provably incomplete. ``load_component_state`` loads
    whatever the same glob returns, so any other shard-named file is accepted
    here: judging it incomplete would force a needless re-download and fail
    closed offline on weights that are present and readable.
    """
    stem = Path(_COMPONENT_WEIGHT_NAME).stem
    if (folder / _COMPONENT_WEIGHT_NAME).is_file():
        return True

    shards = list(folder.glob(f"{stem}-*.safetensors"))
    if not shards:
        return False

    totals: set[int] = set()
    seen: set[int] = set()
    for shard in shards:
        marker = shard.name[len(stem) + 1 : -len(".safetensors")]
        index_text, separator, total_text = marker.partition("-of-")
        if not separator or not index_text.isdigit() or not total_text.isdigit():
            continue
        totals.add(int(total_text))
        seen.add(int(index_text))
    if not totals:
        # No piece announces a total, so there is nothing to be short of.
        return True
    if len(totals) != 1:
        return False
    return seen == set(range(1, totals.pop() + 1))


def _snapshot_has_components(root: Path) -> bool:
    """True when every required component folder holds its complete weights."""
    return all(_component_is_complete(root / component) for component in _REQUIRED_COMPONENTS)


def _download_component_snapshot(
    model: str,
    *,
    revision: str | None = None,
    download_dir: str | None = None,
) -> Path:
    """Resolve an HF repo id to a snapshot holding the component folders.

    Cache-first, so a warm cache needs no Hub round trip. Depending on the
    huggingface_hub version, ``local_files_only=True`` either raises on a
    partial cache or returns the partial snapshot root unchanged; both
    variants must fall through to the online call, or a cold cache fails
    startup without ever consulting the Hub.

    ``revision`` and ``download_dir`` must be threaded through: resolving the
    components against the default branch and the default cache while the AR
    backbone was pinned elsewhere would silently mix weights from two commits.
    """
    from vllm.transformers_utils.repo_utils import hf_api

    kwargs: dict[str, object] = {"allow_patterns": list(_COMPONENT_PATTERNS)}
    if revision is not None:
        kwargs["revision"] = revision
    if download_dir is not None:
        kwargs["cache_dir"] = download_dir

    try:
        cached = Path(hf_api().snapshot_download(model, local_files_only=True, **kwargs))
    except Exception:
        cached = None
    if cached is not None and _snapshot_has_components(cached):
        return cached
    return Path(hf_api().snapshot_download(model, **kwargs))


def _repo_id_and_revision_from_cache_path(path: Path) -> tuple[str | None, str | None]:
    """Recover the Hub repo id and commit when *path* is inside an HF cache.

    Stage init hands the talker the snapshot's ``language_model`` folder. On a
    cold cache the sibling component folders were never downloaded, so the
    marker walk fails even though the missing pieces are one snapshot_download
    away. The cache layout is ``models--{org}--{name}/snapshots/{revision}/...``,
    so the exact commit is recoverable too and must be reused: re-resolving the
    repo id alone would fetch the default branch instead of the snapshot the
    caller is already sitting in.
    """
    parts = path.parts
    for position, part in enumerate(parts):
        if not part.startswith("models--"):
            continue
        repo_id = part.removeprefix("models--").replace("--", "/")
        revision = None
        if position + 2 < len(parts) and parts[position + 1] == "snapshots":
            revision = parts[position + 2]
        return repo_id, revision
    return None, None


def resolve_repo_root(
    model_path: str | os.PathLike[str],
    *,
    revision: str | None = None,
    download_dir: str | None = None,
) -> Path:
    """Find the multi-component repository root from a stage's model path.

    A stage that declares ``model_subdir`` sees the subfolder as its model
    path, so the root is usually the parent. Both are checked, and the search
    walks up a couple of levels for snapshot layouts.

    The acoustic stage declares no ``model_subdir`` because it reads the root
    itself, so on a hub-id deployment its model path is the repo id rather
    than a directory. Resolve that to a local snapshot instead of treating it
    as a relative path.

    A cold cache is recoverable too: stage init pre-downloads only
    ``language_model/`` and ``tokenizer/``, so the talker's model path exists
    while the component folders do not. Fetch them here instead of failing.

    A candidate is accepted only when its components are actually loadable.
    Matching on folder names alone would accept a snapshot left weightless by
    an interrupted download and defer the failure to ``load_component_state``,
    past the point where the repair below could still run.

    Raises:
        FileNotFoundError: If no ancestor looks like the repository root.
    """
    path = Path(model_path).expanduser().resolve()
    candidates = [path, *path.parents[:3]]
    marker_roots = [
        candidate for candidate in candidates if all((candidate / marker).is_dir() for marker in _ROOT_MARKERS)
    ]
    for candidate in marker_roots:
        if _snapshot_has_components(candidate):
            return candidate

    # Either no ancestor carries the markers, or one does but is incomplete.
    # Three recoverable cases: the path is a Hub repo id (the acoustic stage
    # passes the id through), or it is a cache-snapshot subfolder whose sibling
    # component folders were never downloaded, or a cached snapshot lost part
    # of its weights. ``_download_component_snapshot`` is cache-first, so a
    # warm cache stays offline either way.
    cached_repo_id, cached_revision = _repo_id_and_revision_from_cache_path(path)
    reference = str(model_path) if not path.exists() else cached_repo_id
    if reference is not None:
        try:
            resolved = _download_component_snapshot(
                reference,
                revision=revision or cached_revision,
                download_dir=download_dir,
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"MiniMax Music 3 could not resolve {model_path!r} to a local repository root; "
                f"expected folders {_REQUIRED_COMPONENTS}"
            ) from exc
        if _snapshot_has_components(resolved):
            return resolved

    if marker_roots:
        raise FileNotFoundError(
            f"MiniMax Music 3 found a repository root at {marker_roots[0]} but its components are incomplete; "
            f"expected loadable weights under {_REQUIRED_COMPONENTS}"
        )
    raise FileNotFoundError(
        f"MiniMax Music 3 could not locate the repository root from {path}; "
        f"expected sibling folders {_REQUIRED_COMPONENTS}"
    )


def load_component_state(root: Path, component: str, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Read one component's safetensors, sharded or not.

    Raises:
        FileNotFoundError: If the component has no weight files.
    """
    from safetensors.torch import load_file

    folder = root / component
    single = folder / _COMPONENT_WEIGHT_NAME
    if single.exists():
        return load_file(str(single), device=device)

    shards = sorted(folder.glob(f"{Path(_COMPONENT_WEIGHT_NAME).stem}-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"MiniMax Music 3 component {component!r} has no weights under {folder}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(shard), device=device))
    return state


def load_depth_decoder_weights(
    vllm_config: VllmConfig,
    *,
    audio_embeddings: nn.Embedding,
    rvq_decoder: nn.Module,
) -> set[str]:
    """Load the audio embedding table and the RVQ depth decoder in place.

    Args:
        vllm_config: The stage's config, used to locate the checkpoint.
        audio_embeddings: The ``c1..c7`` embedding table to populate.
        rvq_decoder: The depth decoder to populate.

    Returns:
        The model-side parameter names that were loaded.

    Raises:
        ValueError: If the audio embedding is absent or the wrong shape.
    """
    root = resolve_repo_root(
        vllm_config.model_config.model,
        revision=vllm_config.model_config.revision,
        download_dir=vllm_config.load_config.download_dir,
    )
    state = load_component_state(root, DEPTH_DECODER_DIR)

    weight = state.pop(_AUDIO_EMBEDDING_KEY, None)
    expected = (AUDIO_VOCAB_SIZE * (NUM_CODEBOOKS - 1), audio_embeddings.embedding_dim)
    if weight is None or tuple(weight.shape) != expected:
        raise ValueError(
            f"MiniMax Music 3 audio embedding mismatch: expected {expected}, got "
            f"{None if weight is None else tuple(weight.shape)}"
        )
    target = audio_embeddings.weight
    target.data.copy_(weight.to(device=target.device, dtype=target.dtype))

    rvq_decoder.load_checkpoint_state(state)
    logger.info(
        "MiniMax Music 3 loaded audio embedding and depth decoder from %s",
        root / DEPTH_DECODER_DIR,
    )
    loaded = {"audio_embeddings.weight"}
    loaded.update(f"rvq_decoder.{name}" for name, _ in rvq_decoder.named_parameters())
    return loaded


__all__ = [
    "CONDITION_ENCODER_DIR",
    "DEPTH_DECODER_DIR",
    "TRANSFORMER_DIR",
    "VOCODER_DIR",
    "load_component_state",
    "load_depth_decoder_weights",
    "resolve_repo_root",
]
