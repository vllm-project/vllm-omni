# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional FastWAM kernel backends, selected by ``VLLM_OMNI_FASTWAM_KERNELS``.

A kernel backend replaces the stock PyTorch action inference of a FastWAM checkpoint with an
optimised implementation (for example AMD's CAE-FastWAM kernel package on the Hugging Face
Kernel Hub, see :mod:`~vllm_omni.diffusion.models.fastwam.hub_kernels`). Backends are **off by
default**; the stock path is what the model PR validates.

* ``VLLM_OMNI_FASTWAM_KERNELS`` unset, empty or ``off``: the stock path (default).
* ``VLLM_OMNI_FASTWAM_KERNELS=<name>``: try the backend registered as ``<name>`` (built in: ``hub``).

A backend is a :class:`FastWAMKernelBackend` registered with
:func:`register_fastwam_kernel_backend`. At model load, :func:`load_fastwam_kernel_backend` asks the
selected backend whether it can run here and, if so, loads it. Like the VAE fast paths
(:mod:`vllm_omni.diffusion.vae_fast_path`), a backend never aborts model loading: when it is
unsupported (platform, GPU architecture, checkpoint) or fails to load (missing package, no Hub
access), one warning names the reason and the stock path serves the model.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

FASTWAM_KERNELS_ENV = "VLLM_OMNI_FASTWAM_KERNELS"
OFF = "off"
_OFF_VALUES = frozenset({"", "0", "off", "false", "no", "none", "stock"})


def parse_fastwam_kernels(value: str | None) -> str:
    """Normalise a ``VLLM_OMNI_FASTWAM_KERNELS`` value: ``"off"`` or a lower-case backend name.

    Whether the name is registered is checked when the backend is loaded, so that a backend
    registered by a plugin after import can still be selected.
    """
    if value is None:
        return OFF
    name = value.strip().lower()
    return OFF if name in _OFF_VALUES else name


def requested_fastwam_kernels(environ: Any = None) -> str:
    """The backend requested by the environment (``"off"`` unless ``VLLM_OMNI_FASTWAM_KERNELS`` names one)."""
    env = os.environ if environ is None else environ
    return parse_fastwam_kernels(env.get(FASTWAM_KERNELS_ENV))


@dataclass(frozen=True)
class FastWAMKernelContext:
    """What a backend may base its decisions on, taken from the pipeline at load time."""

    cfg: Any  # FastWAMConfig
    model_dir: str
    base_model_dir: str
    device: torch.device
    dtype: torch.dtype
    platform: Any = None


class FastWAMActionBackend(Protocol):
    """A loaded backend: owns its model and predicts normalised action chunks."""

    name: str
    # Components registered on the pipeline by role (``mot``, ``text_encoder``, ``vae_encoder``,
    # ``proprio_encoder``, ``action_codec``), so offloading and memory accounting see the backend's model.
    components: dict[str, torch.nn.Module | None]

    def infer_action(
        self,
        image: torch.Tensor,
        prompt: str,
        proprio: torch.Tensor,
        *,
        num_steps: int,
        generator: torch.Generator | None,
        suite_id: int | None,
    ) -> torch.Tensor:
        """Same contract as :meth:`FastWAMModel.infer_action`, but from the formatted prompt text.

        Returns a float32 CPU tensor [action_horizon, action_dim] in normalised action space.
        """
        ...


class FastWAMKernelBackend(Protocol):
    """One optional FastWAM kernel backend."""

    name: str

    def unsupported_reason(self, ctx: FastWAMKernelContext) -> str | None:
        """``None`` when the backend can run in ``ctx``; otherwise why it cannot (logged, stock path used)."""
        ...

    def load(self, ctx: FastWAMKernelContext) -> FastWAMActionBackend:
        """Load the backend and its model. May raise; the caller then falls back to the stock path."""
        ...


# Built-ins are imported on first use so that importing this module pulls in no backend code.
_BUILTIN_BACKENDS: tuple[str, ...] = ("vllm_omni.diffusion.models.fastwam.hub_kernels:HUB_KERNELS_BACKEND",)
_BACKENDS: dict[str, FastWAMKernelBackend] = {}
_BUILTINS_LOADED = False


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    for spec in _BUILTIN_BACKENDS:
        module_name, attr = spec.split(":")
        try:
            backend = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            logger.debug("FastWAM kernel backend %s unavailable: %s", spec, exc)
            continue
        # A user registration made before first use is newer and wins.
        _BACKENDS.setdefault(backend.name, backend)


def register_fastwam_kernel_backend(backend: FastWAMKernelBackend) -> None:
    """Make ``backend`` selectable by its name. Re-registering a name replaces it."""
    name = parse_fastwam_kernels(backend.name)
    if name == OFF:
        raise ValueError(f"{backend.name!r} is reserved for the stock path.")
    _load_builtins()
    _BACKENDS[name] = backend


def unregister_fastwam_kernel_backend(name: str) -> None:
    _load_builtins()
    _BACKENDS.pop(parse_fastwam_kernels(name), None)


def fastwam_kernel_backends() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_BACKENDS))


def _fallback(name: str, reason: str) -> None:
    logger.warning(
        "FastWAM kernel backend %r requested (%s=%s) but not used: %s. Serving with the stock PyTorch path.",
        name,
        FASTWAM_KERNELS_ENV,
        name,
        reason,
    )


def load_fastwam_kernel_backend(
    ctx: FastWAMKernelContext,
    requested: str | None = None,
    *,
    on_fallback: Callable[[], None] | None = None,
) -> FastWAMActionBackend | None:
    """Load the requested backend, or return ``None`` for the stock path.

    ``requested`` defaults to ``VLLM_OMNI_FASTWAM_KERNELS``. Never raises: every failure logs one
    warning with the reason and returns ``None``. ``on_fallback`` runs after a failed load (the
    pipeline uses it to free device memory a half-loaded backend may hold).
    """
    name = requested_fastwam_kernels() if requested is None else parse_fastwam_kernels(requested)
    if name == OFF:
        return None
    _load_builtins()
    backend = _BACKENDS.get(name)
    if backend is None:
        _fallback(name, f"no such backend (registered: {', '.join(sorted(_BACKENDS)) or 'none'})")
        return None
    try:
        reason = backend.unsupported_reason(ctx)
    except Exception as exc:  # a support probe must not abort model loading either
        reason = f"support check failed ({type(exc).__name__}: {exc})"
    if reason is not None:
        _fallback(name, reason)
        return None
    try:
        loaded = backend.load(ctx)
    except Exception as exc:
        logger.debug("FastWAM kernel backend %r failed to load", name, exc_info=True)
        _fallback(name, f"load failed ({type(exc).__name__}: {exc})")
        if on_fallback is not None:
            on_fallback()
        return None
    logger.info("FastWAM kernel backend %r active: %s", name, getattr(loaded, "description", name))
    return loaded


__all__ = [
    "FASTWAM_KERNELS_ENV",
    "FastWAMActionBackend",
    "FastWAMKernelBackend",
    "FastWAMKernelContext",
    "fastwam_kernel_backends",
    "load_fastwam_kernel_backend",
    "parse_fastwam_kernels",
    "register_fastwam_kernel_backend",
    "requested_fastwam_kernels",
    "unregister_fastwam_kernel_backend",
]
