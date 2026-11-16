# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Diffusion attention backend selector.

This module resolves diffusion attention backends from:
1. per-role AttentionConfig
2. platform default
"""

from __future__ import annotations

import importlib
from functools import cache
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

logger = init_logger(__name__)

# SequenceParallel auto-pad probes mask support before Attention layers exist,
# so it has no real head_dim. Callers must not treat this sentinel as geometry.
HEAD_SIZE_UNKNOWN = -1


def _load_backend_cls(cls_path: str) -> type[AttentionBackend]:
    """Load a backend class from its fully qualified path.

    Args:
        cls_path: Fully qualified class path (e.g.,
            "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend")

    Returns:
        The loaded backend class
    """
    module_path, class_name = cls_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        backend_class = getattr(module, class_name)
        return backend_class
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in module: {e}")


@cache
def _cached_get_backend_cls(
    backend_name: str | None,
    head_size: int,
    allow_trtllm_default: bool = True,
) -> type[AttentionBackend]:
    """Cache backend class resolution by (backend_name, head_size, allow_trtllm_default).

    This ensures platform validation (compute capability checks, package
    availability, etc.) runs only once per unique combination, avoiding
    repeated log messages.
    """
    from vllm_omni.platforms import current_omni_platform

    backend_cls_path = current_omni_platform.get_diffusion_attn_backend_cls(
        selected_backend=backend_name,
        head_size=head_size,
        allow_trtllm_default=allow_trtllm_default,
    )
    return _load_backend_cls(backend_cls_path)


@cache
def _log_backend_resolution(
    role: str,
    role_category: str | None,
    backend_name: str,
    source: str,
) -> None:
    if role_category is not None:
        logger.info(
            "Resolved diffusion attention backend '%s' for role=%r (role_category=%r) via %s",
            backend_name,
            role,
            role_category,
            source,
        )
        return

    logger.info(
        "Resolved diffusion attention backend '%s' for role=%r via %s",
        backend_name,
        role,
        source,
    )


def get_attn_backend_for_role(
    role: str,
    head_size: int,
    attention_config: AttentionConfig | None = None,
    role_category: str | None = None,
    allow_trtllm_default: bool = True,
) -> tuple[type[AttentionBackend], AttentionSpec | None]:
    """
    Get attention backend for a specific attention role.

    Lookup precedence:
      1. attention_config.per_role[role]           — exact match
      2. attention_config.per_role[role_category]   — category fallback
      3. attention_config.default                   — global default
      4. Platform default                           — hardware-specific

    Args:
        role: Attention role string (e.g. "self", "cross", "joint",
              "ltx2.audio_to_video")
        head_size: Head size for attention computation
        attention_config: The AttentionConfig from OmniDiffusionConfig.
            If None, falls back to platform default behavior.
        role_category: Optional category for fallback (e.g. "cross" for
            "ltx2.audio_to_video")

    Returns:
        Tuple of (backend_class, AttentionSpec or None).
        AttentionSpec is None when using platform default without explicit config.
    """
    # 1. Try config from OmniDiffusionConfig
    spec = None
    source = None
    if attention_config is not None:
        spec, source = attention_config.resolve_with_source(
            role=role,
            role_category=role_category,
        )

    if spec is not None:
        backend_cls = _cached_get_backend_cls(spec.backend, head_size)
        _log_backend_resolution(
            role=role,
            role_category=role_category,
            backend_name=backend_cls.get_name(),
            source=source or "attention_config",
        )
        return backend_cls, spec

    # 2. Platform default
    backend_cls = _cached_get_backend_cls(None, head_size, allow_trtllm_default)
    _log_backend_resolution(
        role=role,
        role_category=role_category,
        backend_name=backend_cls.get_name(),
        source="platform default",
    )
    return backend_cls, None


def get_attn_backend_for_capability(
    role: str,
    attention_config: AttentionConfig | None = None,
    role_category: str | None = None,
) -> type[AttentionBackend]:
    """Resolve the backend class for capability queries such as mask support.

    SequenceParallel auto-pad does not know ``head_size``. Explicit
    ``AttentionConfig`` selections are loaded from the registry without
    geometry validation, so CUDNN_ATTN is not rejected for the unknown
    sentinel. Platform defaults still consult the platform with
    ``HEAD_SIZE_UNKNOWN``.
    """
    spec = None
    if attention_config is not None:
        spec, _ = attention_config.resolve_with_source(
            role=role,
            role_category=role_category,
        )
    if spec is not None:
        from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum

        return DiffusionAttentionBackendEnum[spec.backend.upper()].get_class()
    return _cached_get_backend_cls(None, HEAD_SIZE_UNKNOWN)
