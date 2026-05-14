"""
Tests for hook registry.

NOTE: The hook registry is also tested indirectly through a lot of
other tests, e.g., tests/diffusion/distributed/test_sp_plan_hooks.py
"""

from typing import Any

import pytest
from torch import nn

from vllm_omni.diffusion.hooks.base import HookRegistry, ModelHook

DEFAULT_OUT = "ECHO"
OVERRIDE_OUT = "OVERRIDE"
INPUT_KWARG = "inp"


class EchoModule(nn.Module):
    """Just echo the input."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        input_val = kwargs[INPUT_KWARG]
        return input_val + DEFAULT_OUT


class AppendHook(ModelHook):
    """Append an echo value to the input string on pre / post forward."""

    def __init__(self, echo_val: str):
        self.echo_val = echo_val

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        input_val = kwargs[INPUT_KWARG]
        return (), {INPUT_KWARG: input_val + self.echo_val}

    def post_forward(self, module: nn.Module, output):
        return output + self.echo_val


class OverrideAppendHook(AppendHook):
    """Same as AppendHook, but replace the forward call with a different string."""

    def new_forward(self, module: nn.Module, *args, **kwargs):
        return kwargs[INPUT_KWARG] + OVERRIDE_OUT


def test_register_no_fwd_override_hooks():
    """Ensure registration is correct with no forward hooks."""
    mod = EchoModule()
    registry = HookRegistry.get_or_create(mod)
    first_hook = AppendHook("1")
    second_hook = AppendHook("2")
    sorted_no_fwd_hooks = [first_hook, second_hook]

    # Will add and sort the hook by key
    registry.register_hook(name="b", hook=second_hook)
    registry.register_hook(name="a", hook=first_hook)

    assert len(registry._hooks) == 2
    assert len(registry._sorted_hooks) == 2
    assert registry._new_fwd_impl_hook is None
    # Ensure registering a new hook sorting alphabetically
    for actual_hook, expected_hook in zip(registry._sorted_hooks, sorted_no_fwd_hooks):
        assert actual_hook is expected_hook


def test_register_with_forward_hooks():
    """Ensure registration is correct with a forward hooks."""
    mod = EchoModule()
    registry = HookRegistry.get_or_create(mod)
    first_hook = AppendHook("1")
    second_hook = AppendHook("2")
    exec_hook = OverrideAppendHook("3")
    sorted_no_fwd_hooks = [first_hook, second_hook]

    # Will add and sort the hook by key
    registry.register_hook(name="b", hook=second_hook)
    registry.register_hook(name="a", hook=first_hook)
    registry.register_hook(name="c", hook=exec_hook)

    assert len(registry._hooks) == 3
    assert len(registry._sorted_hooks) == 3
    assert registry._new_fwd_impl_hook is exec_hook
    # Ensure registering a new hook sorting alphabetically
    for actual_hook, expected_hook in zip(registry._sorted_hooks, sorted_no_fwd_hooks):
        assert actual_hook is expected_hook


def test_register_fails_with_multiple_forward_hooks():
    """Ensure registration only allows one hook overriding new_forward"""
    mod = EchoModule()
    registry = HookRegistry.get_or_create(mod)

    registry.register_hook(name="foo", hook=OverrideAppendHook("1"))
    with pytest.raises(RuntimeError):
        registry.register_hook(name="bar", hook=OverrideAppendHook("2"))


def test_remove_hooks():
    """Ensure removal sorts hooks."""
    mod = EchoModule()
    registry = HookRegistry.get_or_create(mod)

    first_hook = AppendHook("1")
    second_hook = AppendHook("2")
    exec_hook = OverrideAppendHook("3")

    registry.register_hook(name="b", hook=second_hook)
    registry.register_hook(name="a", hook=first_hook)
    registry.register_hook(name="c", hook=exec_hook)
    # Explicitly reorder our hooks to be in the wrong order, since register
    # forces them to be sorted too. Ensure that remove the hook will also
    # enforce the sorted order.
    registry._sorted_hooks = [second_hook, first_hook]

    assert registry._new_fwd_impl_hook is exec_hook
    registry.remove_hook("c")
    assert registry._new_fwd_impl_hook is None

    sorted_no_fwd_hooks = [first_hook, second_hook]
    for actual_hook, expected_hook in zip(registry._sorted_hooks, sorted_no_fwd_hooks):
        assert actual_hook is expected_hook


def test_dispatch_no_fwd_override_hooks():
    """Ensure dispatch runs hooks in deterministic sorted order."""
    mod = EchoModule()
    registry = HookRegistry.get_or_create(mod)

    first_hook = AppendHook("1")
    second_hook = AppendHook("2")

    # Register will sort the hooks, so hook 1 will run first
    # on preprocess and last in post process
    registry.register_hook(name="2", hook=second_hook)
    registry.register_hook(name="1", hook=first_hook)
    res = registry.dispatch(inp="")
    assert isinstance(res, str)
    assert res == f"12{DEFAULT_OUT}21"


def test_dispatch_with_fwd_hooks():
    """Ensure dispatch runs hooks in deterministic sorted order."""
    mod = EchoModule()
    registry = HookRegistry.get_or_create(mod)

    first_hook = AppendHook("1")
    second_hook = AppendHook("2")
    exec_hook = OverrideAppendHook("3")

    # Register will sort the hooks, so hook 1 will run first on preprocess and last in
    # post process. Since the override hook mutates forward, it will run last even
    # though the name of the exec_hook is alphabetically before the second hook.
    registry.register_hook(name="c", hook=second_hook)
    registry.register_hook(name="a", hook=first_hook)
    registry.register_hook(name="b", hook=exec_hook)
    res = registry.dispatch(inp="")
    assert isinstance(res, str)
    assert res == f"123{OVERRIDE_OUT}321"
