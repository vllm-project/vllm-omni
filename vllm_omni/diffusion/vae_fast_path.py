# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pluggable VAE fast paths, selected by ``vae_fast_path`` / ``vae_encode_fast_path``.

A fast path is a :class:`VaeFastPath` registered with :func:`register_vae_fast_path`. At model load,
:func:`apply_vae_fast_paths` finds the pipeline's VAEs, picks for each one the most recently registered fast path
that matches it, and installs its decoder and its encoder at the configured levels.

Levels are a contract every fast path keeps:

* ``off``: the reference forwards.
* ``lossless`` (default): bit-identical to the reference forwards.
* ``channels_last``: may change values within the tolerance the fast path documents.

A fast path that cannot honour a level for a VAE reports it as not installed rather than approximating it.
Failures never abort model loading: the component keeps its reference forwards.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

COMPONENTS: tuple[str, ...] = ("decoder", "encoder")


def vae_compile_allowed(enforce_eager: bool, platform: Any) -> bool:
    """Whether a VAE fast path may use ``torch.compile``.

    Same precedence as the DiT compile decision in ``diffusion_model_runner.py``: ``--enforce-eager`` vetoes
    compilation, then the platform must support torch inductor. A platform that cannot answer (the base
    ``OmniPlatform`` raises ``NotImplementedError``) gets no compile.
    """
    if enforce_eager or platform is None:
        return False
    try:
        return bool(platform.supports_torch_inductor())
    except NotImplementedError:
        return False


@dataclass(frozen=True)
class VaeFastPathContext:
    """What a fast path may base its decisions on, taken from the stage config at load time."""

    decode_level: str = "lossless"
    encode_level: str = "lossless"
    platform: Any = None
    enforce_eager: bool = False

    @classmethod
    def from_config(cls, od_config: Any, platform: Any) -> VaeFastPathContext:
        return cls(
            decode_level=getattr(od_config, "vae_fast_path", "lossless"),
            encode_level=getattr(od_config, "vae_encode_fast_path", "lossless"),
            platform=platform,
            enforce_eager=bool(getattr(od_config, "enforce_eager", False)),
        )

    def level(self, component: str) -> str:
        if component not in COMPONENTS:
            raise ValueError(f"component must be one of {COMPONENTS}, got {component!r}")
        return self.decode_level if component == "decoder" else self.encode_level

    def compile_allowed(self) -> bool:
        return vae_compile_allowed(self.enforce_eager, self.platform)


class VaeFastPathResult(Protocol):
    installed: bool
    reason: str | None


class VaeFastPath(Protocol):
    """One family of VAE fast paths (for example every Wan-style causal VAE)."""

    name: str

    def matches(self, vae: nn.Module) -> bool:
        """Whether this fast path is the one responsible for ``vae``."""
        ...

    def supported(self, ctx: VaeFastPathContext) -> bool:
        """Whether it can run in ``ctx`` at all (platform, ...). Unsupported VAEs keep the reference forwards."""
        ...

    def install(self, vae: nn.Module, component: str, level: str, ctx: VaeFastPathContext) -> VaeFastPathResult:
        """Install ``component`` (``"decoder"`` / ``"encoder"``) at ``level`` (never ``"off"``). Idempotent."""
        ...


# Built-ins are imported on first use so that importing this module pulls in no model code.
_BUILTIN_FAST_PATHS: tuple[str, ...] = (
    "vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath.entry:WAN_FAMILY_FAST_PATH",
)
_FAST_PATHS: list[VaeFastPath] = []
_BUILTINS_LOADED = False


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    builtins: list[VaeFastPath] = []
    for spec in _BUILTIN_FAST_PATHS:
        module_name, attr = spec.split(":")
        try:
            builtins.append(getattr(importlib.import_module(module_name), attr))
        except (ImportError, AttributeError) as exc:
            logger.debug("VAE fast path %s unavailable: %s", spec, exc)
    # Registered before first use: keep the user's entry, which is newer and wins.
    registered = {fp.name for fp in _FAST_PATHS}
    _FAST_PATHS[:0] = [fp for fp in builtins if fp.name not in registered]


def register_vae_fast_path(fast_path: VaeFastPath) -> None:
    """Make ``fast_path`` eligible. Re-registering a name replaces it; the latest registration wins a VAE."""
    _load_builtins()
    _FAST_PATHS[:] = [fp for fp in _FAST_PATHS if fp.name != fast_path.name] + [fast_path]


def unregister_vae_fast_path(name: str) -> None:
    _load_builtins()
    _FAST_PATHS[:] = [fp for fp in _FAST_PATHS if fp.name != name]


def vae_fast_paths() -> tuple[VaeFastPath, ...]:
    _load_builtins()
    return tuple(_FAST_PATHS)


def find_vae_fast_path(vae: nn.Module) -> VaeFastPath | None:
    for fast_path in reversed(vae_fast_paths()):
        if fast_path.matches(vae):
            return fast_path
    return None


def _pipeline_vaes(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

    candidates: list[tuple[str, nn.Module]] = []
    try:
        discovered = ModuleDiscovery.discover(model)
        candidates = list(zip(discovered.vae_names, discovered.vaes, strict=True))
    except Exception:
        logger.debug("VAE fast path: component discovery failed; falling back to `vae`", exc_info=True)
    if not candidates and isinstance(getattr(model, "vae", None), nn.Module):
        candidates = [("vae", model.vae)]
    yield from candidates


def apply_vae_fast_paths(model: nn.Module, od_config: Any, *, platform: Any) -> None:
    """Install the registered fast paths on every VAE of ``model``; see the module docstring."""
    ctx = VaeFastPathContext.from_config(od_config, platform)
    if ctx.decode_level == "off" and ctx.encode_level == "off":
        return
    for name, vae in _pipeline_vaes(model):
        fast_path = find_vae_fast_path(vae)
        if fast_path is None:
            continue
        if not fast_path.supported(ctx):
            logger.debug("VAE fast path %s is not supported on this platform; skipping %s", fast_path.name, name)
            continue
        for component in COMPONENTS:
            level = ctx.level(component)
            if level == "off":
                continue
            try:
                report = fast_path.install(vae, component, level, ctx)
            except Exception:
                logger.warning(
                    "Failed to install the %s VAE %s fast path on %s; using the reference %s",
                    fast_path.name,
                    component,
                    name,
                    component,
                    exc_info=True,
                )
                continue
            if report.installed:
                logger.info("VAE %s fast path %s (%s) active on %s", component, fast_path.name, level, name)


__all__ = [
    "COMPONENTS",
    "VaeFastPath",
    "VaeFastPathContext",
    "VaeFastPathResult",
    "apply_vae_fast_paths",
    "find_vae_fast_path",
    "register_vae_fast_path",
    "unregister_vae_fast_path",
    "vae_compile_allowed",
    "vae_fast_paths",
]
