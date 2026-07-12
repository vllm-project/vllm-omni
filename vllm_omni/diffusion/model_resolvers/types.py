"""Shared types for diffusion model resolvers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

# Resolver inputs come from ``get_hf_file_to_dict(...)`` and similar helpers,
# which return plain dicts today but are consumed as read-only config views.
ModelConfigLike: TypeAlias = Mapping[str, Any]


__all__ = ["ModelConfigLike"]
