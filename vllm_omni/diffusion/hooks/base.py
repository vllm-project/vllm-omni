# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base hook classes for model forward interception.

This module provides the foundational hook mechanism that allows intercepting
and modifying model forward passes without invasive changes to model code.
"""

from __future__ import annotations

import functools
import inspect
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
        """Override the module's forward pass. This should be overridden for more complex
        cases, e.g., TeaCache. If this method is overridden in a subclass, it will be called
        instead of self.module._omni_original_forward when executing the hooks.

        Args:
            module: The module being called.
            *args: Positional arguments to forward.
            **kwargs: Keyword arguments to forward.

        Returns:
            The output of the replacement for the forward pass.
        """
        raise NotImplementedError("By default, hooks do not implement new_forward")

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
            return self.module._omni_original_forward(*args, **kwargs)
        return registry.dispatch(*args, **kwargs)


def sort_hooks_after_call(func):
    """Calls the method on the hook registry, then sorts the hooks.

    This should be added to methods that mutate add or remove hooks.
    """

    @functools.wraps(func)
    def wrapper(self: HookRegistry, *args, **kwargs):
        res = func(self, *args, **kwargs)
        self.update_sorted_hooks()
        return res

    return wrapper


@dataclass
class _WrappedMethod:
    """Wrapper that intercepts arbitrary method calls and runs hook pre_forward
    to trigger device swaps before the original method executes."""

    module: nn.Module
    original_method: Callable

    def __call__(self, *args: Any, **kwargs: Any):
        registry: HookRegistry | None = getattr(self.module, "_hook_registry", None)
        if registry is not None and registry._hooks:
            for _, hook in sorted(registry._hooks.items(), key=lambda x: x[0]):
                args, kwargs = hook.pre_forward(self.module, *args, **kwargs)
        return self.original_method(*args, **kwargs)


class HookRegistry:
    """Registry of hooks attached to a module.

    Manages multiple hooks that can intercept a module's forward pass.
    Hooks are called in sorted order by name for determinism.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self._hooks: dict[str, ModelHook] = {}
        # Hooks sorted by execution order
        self._sorted_hooks: list[ModelHook] = []
        # Hooks overriding new_forward (if any)
        self._new_fwd_impl_hook: ModelHook | None = None
        self._wrapped_methods: list[str] = []

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
            # NOTE: Use `_omni_original_forward` to avoid collision with cache-dit's
            # `_original_forward` attribute on the same module.
            if not hasattr(module, "_omni_original_forward"):
                module._omni_original_forward = module.forward  # type: ignore[attr-defined]
                wrapped = _WrappedForward(module)
                wrapped.__signature__ = inspect.signature(module.forward)  # type: ignore[attr-defined]
                module.forward = wrapped  # type: ignore[assignment]

        return registry

    def update_sorted_hooks(self):
        """Sort hooks by name, which dictates pre/post process order."""
        sorted_hooks = [self._hooks[k] for k in sorted(self._hooks) if self._hooks[k] != self._new_fwd_impl_hook]
        if self._new_fwd_impl_hook is not None:
            sorted_hooks.append(self._new_fwd_impl_hook)
        self._sorted_hooks = sorted_hooks

    @sort_hooks_after_call
    def register_hook(self, name: str, hook: ModelHook) -> None:
        """Register a hook with the given name.

        Args:
            name: Unique name for this hook.
            hook: The hook instance to register.
        """
        hook.initialize_hook(self.module)
        self._hooks[name] = hook
        # We can only have one hook that overrides new_forward,
        # since we don't currently have a mechanism for combining them.
        if type(hook).new_forward is not ModelHook.new_forward:
            if self._new_fwd_impl_hook is not None:
                raise RuntimeError("Cannot have multiple hooks that override forward active simultaneously")
            self._new_fwd_impl_hook = hook

    @sort_hooks_after_call
    def remove_hook(self, name: str) -> None:
        """Remove a hook by name.

        Args:
            name: The name of the hook to remove.
        """
        if name in self._hooks:
            # clear the forward hook if it's the one to delete
            if self._new_fwd_impl_hook is self._hooks[name]:
                self._new_fwd_impl_hook = None
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

        Multiple hooks may be used with the caveat that only one hook
        may override new_forward. While it is assumed that pre/post process
        on hooks are composable, the execution flow is as follows for determinism:

        - Run preprocess on all hooks in their sorted order; hooks are sorted alphabetically,
          except for the hook overriding forward (`self._new_fwd_impl_hook`), which is last
          if it exists.

        - If `self._new_fwd_impl_hook` isn't None, call its forward. Otherwise call the
          original model forward.

        - Run post process on all hooks in the reverse sorted order.

        Args:
            *args: Positional arguments to forward.
            **kwargs: Keyword arguments to forward.

        Returns:
            The output of the forward pass.
        """
        if not self._hooks:
            return self.module._omni_original_forward(*args, **kwargs)  # type: ignore[attr-defined]

        # Apply all pre_forward hooks; if _new_fwd_impl_hook is set, it's last
        for hook in self._sorted_hooks:
            args, kwargs = hook.pre_forward(self.module, *args, **kwargs)

        # If we have a hook that overrides new_forward, call it directly
        if self._new_fwd_impl_hook is not None:
            output = self._new_fwd_impl_hook.new_forward(self.module, *args, **kwargs)
        # Otherwise just call the original forward.
        else:
            output = self.module._omni_original_forward(*args, **kwargs)  # type: ignore[attr-defined]

        # Apply all post_forward hooks in reverse order; if _new_fwd_impl_hook is set, it's first
        for hook in reversed(self._sorted_hooks):
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

    def wrap_method(self, method_name: str) -> None:
        """Also intercept calls to the named method (e.g. 'decode', 'encode').

        The wrapper calls all registered hooks' pre_forward() before delegating
        to the original method. This is used to trigger CPU-offload device swaps
        for methods like vae.decode() that bypass module.forward().
        """
        original_attr = f"_omni_original_{method_name}"
        if hasattr(self.module, original_attr):
            return  # already wrapped
        original = getattr(self.module, method_name)
        setattr(self.module, original_attr, original)
        wrapped = _WrappedMethod(module=self.module, original_method=original)
        wrapped.__signature__ = inspect.signature(original)  # type: ignore[attr-defined]
        setattr(self.module, method_name, wrapped)
        self._wrapped_methods.append(method_name)

    def unwrap_method(self, method_name: str) -> None:
        """Restore the original method, removing the hook wrapper."""
        original_attr = f"_omni_original_{method_name}"
        if hasattr(self.module, original_attr):
            setattr(self.module, method_name, getattr(self.module, original_attr))
            delattr(self.module, original_attr)
            if method_name in self._wrapped_methods:
                self._wrapped_methods.remove(method_name)

    def unwrap_all_methods(self) -> None:
        """Restore all wrapped methods to their originals."""
        for method_name in list(self._wrapped_methods):
            self.unwrap_method(method_name)
