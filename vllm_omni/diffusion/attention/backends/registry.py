# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion attention backend registry.

This module provides an enum-based registry for diffusion attention backends,
similar to vLLM's AttentionBackendEnum. Each backend registers its class path,
and platforms can override or extend backends using register_backend().
"""

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


class _DiffusionBackendEnumMeta(EnumMeta):
    """Metaclass for DiffusionAttentionBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str) -> "DiffusionAttentionBackendEnum":
        """Get backend by name with helpful error messages."""
        try:
            return super().__getitem__(name)  # type: ignore[return-value]
        except KeyError:
            members = list(cls.__members__.keys())
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown diffusion attention backend: '{name}'. Valid options are: {valid_backends}"
            ) from None


class DiffusionAttentionBackendEnum(Enum, metaclass=_DiffusionBackendEnumMeta):
    """Enumeration of all supported diffusion attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()

    Example:
        # Get backend class
        backend = DiffusionAttentionBackendEnum.FLASH_ATTN
        backend_cls = backend.get_class()

        # Register custom backend
        @register_diffusion_backend(DiffusionAttentionBackendEnum.CUSTOM)
        class MyCustomBackend:
            ...
    """

    # Common backends (available on most platforms)
    FLASH_ATTN = "vllm_omni.diffusion.attention.backends.flash_attn.FlashAttentionBackend"
    TORCH_SDPA = "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"
    SAGE_ATTN = "vllm_omni.diffusion.attention.backends.sage_attn.SageAttentionBackend"

    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If backend has empty path and is not registered
        """
        path = _DIFFUSION_ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_diffusion_backend(DiffusionAttentionBackendEnum.{self.name}, "
                f"'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If backend has empty path and is not registered
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override
        """
        return self in _DIFFUSION_ATTN_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _DIFFUSION_ATTN_OVERRIDES.pop(self, None)


# Override registry
_DIFFUSION_ATTN_OVERRIDES: dict[DiffusionAttentionBackendEnum, str] = {}


def register_diffusion_backend(
    backend: DiffusionAttentionBackendEnum,
    class_path: str | None = None,
) -> Callable[[type], type]:
    """Register or override a diffusion backend implementation.

    Args:
        backend: The DiffusionAttentionBackendEnum member to register
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op

    Examples:
        # Override an existing backend
        @register_diffusion_backend(DiffusionAttentionBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn:
            ...

        # Override an existing backend (e.g., ASCEND_ATTN)
        @register_diffusion_backend(DiffusionAttentionBackendEnum.ASCEND_ATTN)
        class CustomAscendAttentionBackend:
            ...

        # Direct registration
        register_diffusion_backend(
            DiffusionAttentionBackendEnum.CUSTOM,
            "my.module.MyCustomBackend"
        )
    """

    def decorator(cls: type) -> type:
        _DIFFUSION_ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _DIFFUSION_ATTN_OVERRIDES[backend] = class_path
        return lambda x: x

    return decorator
