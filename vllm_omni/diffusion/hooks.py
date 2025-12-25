from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn


class BaseState:
    """Base class for hook state containers."""

    def reset(self) -> None:  # pragma: no cover - default is no-op
        pass


class StateManager:
    """Manage per-context hook state instances."""

    def __init__(self, state_cls: Callable[[], BaseState]):
        self._state_cls = state_cls
        self._states: dict[str, BaseState] = {}
        self._context: str = "default"

    def set_context(self, name: str) -> None:
        self._context = name or "default"

    def get_state(self) -> BaseState:
        if self._context not in self._states:
            self._states[self._context] = self._state_cls()
        return self._states[self._context]

    def reset(self) -> None:
        self._states.clear()


class ModelHook:
    """Base class for model hooks that can override a module's forward."""

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        raise NotImplementedError

    def new_forward(self, module: nn.Module, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def reset_state(self, module: nn.Module) -> nn.Module:
        raise NotImplementedError


@dataclass
class _WrappedForward:
    module: nn.Module

    def __call__(self, *args: Any, **kwargs: Any):
        registry: HookRegistry | None = getattr(self.module, "_hook_registry", None)
        if registry is None or not registry._hooks:
            return self.module._original_forward(*args, **kwargs)
        return registry.dispatch(*args, **kwargs)


class HookRegistry:
    """Registry of hooks attached to a module."""

    def __init__(self, module: nn.Module):
        self.module = module
        self._hooks: dict[str, ModelHook] = {}

    @classmethod
    def get_or_create(cls, module: nn.Module) -> HookRegistry:
        registry: HookRegistry | None = getattr(module, "_hook_registry", None)
        if registry is None:
            registry = cls(module)
            setattr(module, "_hook_registry", registry)

            # Wrap module.forward once so hooks can intercept calls.
            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward  # type: ignore[attr-defined]
                module.forward = _WrappedForward(module)  # type: ignore[assignment]

        return registry

    def register_hook(self, name: str, hook: ModelHook) -> None:
        hook.initialize_hook(self.module)
        self._hooks[name] = hook

    def get_hook(self, name: str) -> ModelHook | None:
        return self._hooks.get(name)

    def dispatch(self, *args: Any, **kwargs: Any):
        # For now we support a single active hook and call it directly.
        # This can be extended to a chain if needed.
        if not self._hooks:
            return self.module._original_forward(*args, **kwargs)  # type: ignore[attr-defined]
        # Deterministic order: sort by name.
        name = sorted(self._hooks.keys())[0]
        hook = self._hooks[name]
        return hook.new_forward(self.module, *args, **kwargs)

    def reset_hook(self, name: str) -> None:
        hook = self._hooks.get(name)
        if hook is not None:
            hook.reset_state(self.module)
