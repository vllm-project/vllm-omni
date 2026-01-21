# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base hook classes for model forward interception.

This module provides the foundational hook mechanism that allows intercepting
and modifying model forward passes without invasive changes to model code.
"""

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
    """Base class for model hooks that can override a module's forward.

    Hooks can intercept the forward pass at two points:
    - pre_forward: Called before the original forward, can modify args/kwargs
    - post_forward: Called after the original forward, can modify output

    Subclasses can override either or both methods. The default implementations
    pass through args/kwargs/output unchanged.

    For more complex behavior, override new_forward to completely replace
    the forward logic.
    """

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        """Initialize the hook when it's registered to a module.

        Args:
            module: The module this hook is being attached to.

        Returns:
            The module (possibly modified).
        """
        return module

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        """Called before the module's forward pass.

        Args:
            module: The module being called.
            *args: Positional arguments to forward.
            **kwargs: Keyword arguments to forward.

        Returns:
            Tuple of (args, kwargs) to pass to the forward method.
        """
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Called after the module's forward pass.

        Args:
            module: The module that was called.
            output: The output from the forward method.

        Returns:
            The (possibly modified) output.
        """
        return output

    def new_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        """Override the module's forward pass completely.

        The default implementation calls pre_forward, then the original forward,
        then post_forward. Override this method for more complex behavior.

        Args:
            module: The module being called.
            *args: Positional arguments to forward.
            **kwargs: Keyword arguments to forward.

        Returns:
            The output of the forward pass.
        """
        args, kwargs = self.pre_forward(module, *args, **kwargs)
        output = module._original_forward(*args, **kwargs)  # type: ignore[attr-defined]
        return self.post_forward(module, output)

    def reset_state(self, module: nn.Module) -> nn.Module:
        """Reset any state associated with this hook.

        Args:
            module: The module this hook is attached to.

        Returns:
            The module.
        """
        return module


@dataclass
class _WrappedForward:
    """Wrapper that intercepts forward calls and dispatches to hooks."""

    module: nn.Module

    def __call__(self, *args: Any, **kwargs: Any):
        registry: HookRegistry | None = getattr(self.module, "_hook_registry", None)
        if registry is None or not registry._hooks:
            return self.module._original_forward(*args, **kwargs)
        return registry.dispatch(*args, **kwargs)


class HookRegistry:
    """Registry of hooks attached to a module.

    Manages multiple hooks that can intercept a module's forward pass.
    Hooks are called in sorted order by name for determinism.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self._hooks: dict[str, ModelHook] = {}

    @classmethod
    def get_or_create(cls, module: nn.Module) -> HookRegistry:
        """Get existing registry or create a new one for the module.

        Args:
            module: The module to get/create a registry for.

        Returns:
            The HookRegistry for this module.
        """
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
        """Register a hook with the given name.

        Args:
            name: Unique name for this hook.
            hook: The hook instance to register.
        """
        hook.initialize_hook(self.module)
        self._hooks[name] = hook

    def remove_hook(self, name: str) -> None:
        """Remove a hook by name.

        Args:
            name: The name of the hook to remove.
        """
        if name in self._hooks:
            del self._hooks[name]

    def get_hook(self, name: str) -> ModelHook | None:
        """Get a hook by name.

        Args:
            name: The name of the hook.

        Returns:
            The hook if found, None otherwise.
        """
        return self._hooks.get(name)

    def dispatch(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch a forward call through registered hooks.

        Currently supports a single active hook. Multiple hooks are called
        in sorted order by name, with each hook's output passed to the next.

        Args:
            *args: Positional arguments to forward.
            **kwargs: Keyword arguments to forward.

        Returns:
            The output of the forward pass.
        """
        if not self._hooks:
            return self.module._original_forward(*args, **kwargs)  # type: ignore[attr-defined]

        # For single hook case, call directly
        if len(self._hooks) == 1:
            hook = next(iter(self._hooks.values()))
            return hook.new_forward(self.module, *args, **kwargs)

        # For multiple hooks, chain them in sorted order
        # Each hook can modify args/kwargs via pre_forward
        sorted_hooks = sorted(self._hooks.items(), key=lambda x: x[0])

        # Apply all pre_forward hooks
        for _, hook in sorted_hooks:
            args, kwargs = hook.pre_forward(self.module, *args, **kwargs)

        # Call original forward
        output = self.module._original_forward(*args, **kwargs)  # type: ignore[attr-defined]

        # Apply all post_forward hooks in reverse order
        for _, hook in reversed(sorted_hooks):
            output = hook.post_forward(self.module, output)

        return output

    def reset_hook(self, name: str) -> None:
        """Reset a hook's state by name.

        Args:
            name: The name of the hook to reset.
        """
        hook = self._hooks.get(name)
        if hook is not None:
            hook.reset_state(self.module)
