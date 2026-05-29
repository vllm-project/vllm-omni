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
from vllm.logger import init_logger

logger = init_logger(__name__)


class BaseState:
    """Base class for hook state containers."""

    def reset(self) -> None:  # pragma: no cover - default is no-op
        pass


class StateManager:
    """Manage per-context hook state instances."""

    def __init__(self, state_cls: Callable[[], BaseState], init_args: tuple = (), init_kwargs: dict | None = None):
        self._state_cls = state_cls
        self._init_args = init_args
        self._init_kwargs = init_kwargs or {}
        self._states: dict[str, BaseState] = {}
        self._context: str = "default"

    @property
    def _current_context(self) -> str | None:
        """Alias for _context for compatibility with diffusers hook code."""
        return self._context if self._context != "default" else None

    def set_context(self, name: str) -> None:
        self._context = name or "default"

    def get_state(self) -> BaseState:
        if self._context not in self._states:
            self._states[self._context] = self._state_cls(*self._init_args, **self._init_kwargs)
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

    @classmethod
    def check_if_exists_or_initialize(cls, module: nn.Module) -> HookRegistry:
        """Get existing registry or create a new one for the module.

        This method ensures a HookRegistry exists on the module and returns it.
        If a registry doesn't exist, it creates one and attaches it to the module.
        This is equivalent to get_or_create() for compatibility with diffusers API.

        Args:
            module: The module to get/create a registry for.

        Returns:
            The HookRegistry for this module.
        """
        return cls.get_or_create(module)

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
        if name in self._hooks:
            logger.warning(f"Hook with name '{name}' already exists. Overwriting existing hook.")
            self.remove_hook(name)

        hook.initialize_hook(self.module)

        if hasattr(hook, "fn_ref"):
            hook.fn_ref.original_forward = self.module._omni_original_forward
        else:
            original_forward = self.module._omni_original_forward  # type: ignore[attr-defined]

            class _FnRef:
                def __init__(self, orig_forward):
                    self.original_forward = orig_forward

            hook.fn_ref = _FnRef(original_forward)

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

    def reset(self) -> None:
        """Reset all hooks and clear the registry.

        This removes all hooks from the registry and resets each hook's state.
        Also restores module.forward to its original implementation.
        """
        for name, hook in list(self._hooks.items()):
            hook.reset_state(self.module)
        self._hooks.clear()

        if hasattr(self.module, "_omni_original_forward"):
            self.module.forward = self.module._omni_original_forward  # type: ignore[attr-defined]
            delattr(self.module, "_omni_original_forward")
