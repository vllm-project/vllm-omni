import inspect
from collections.abc import Callable
from typing import TypeVar

import torch
import torch.nn as nn

_T = TypeVar("_T", bound=type[nn.Module])


def dit_support_compile(
    cls: _T | None = None,
    dynamic_arg_dims: dict[str, int | list[int]] | None = None,
) -> Callable[[_T], _T] | _T:
    """
    A decorator to add support for compiling the forward method of a class.

    Usage 1: use directly as a decorator without arguments:

    ```python
    @dit_support_compile
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]): ...
    ```

    Usage 2: use as a decorator with arguments:

    ```python
    @dit_support_compile(dynamic_arg_dims={"x": 0, "y": 0})
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]): ...
    ```
    """

    def wrap(cls: _T) -> _T:
        original_forward = cls.forward
        sig = inspect.signature(original_forward)
        if dynamic_arg_dims is not None:
            compiled_forward = torch.compile(original_forward, fullgraph=True)
        else:
            compiled_forward = torch.compile(
                original_forward,
                fullgraph=True,
                dynamic=True,
            )

        dims_map: dict[str, list[int]] | None = None
        if dynamic_arg_dims is not None:
            dims_map = {}
            for arg_name, dims in dynamic_arg_dims.items():
                if isinstance(dims, int):
                    dims_map[arg_name] = [dims]

        def _mark_dynamic_once(instance, bound_args):
            if not dims_map:
                return

            if getattr(instance, "_dit_dynamic_marked", False):
                return

            for arg_name, dims in dims_map.items():
                arg_value = bound_args.arguments.get(arg_name)
                if arg_value is None:
                    continue
                if not isinstance(arg_value, torch.Tensor):
                    raise TypeError(
                        "dit_support_compile expects Tensor arguments (or None) for dynamic dims; "
                        f"got {type(arg_value)!r} for '{arg_name}'."
                    )

                for dim in dims:
                    resolved_dim = arg_value.ndim + dim if dim < 0 else dim
                    torch._dynamo.mark_dynamic(arg_value, resolved_dim)

            setattr(instance, "_dit_dynamic_marked", True)

        def wrapped_forward(self, *args, **kwargs):
            if dynamic_arg_dims is not None:
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
                _mark_dynamic_once(self, bound_args)
            return compiled_forward(self, *args, **kwargs)

        cls.forward = wrapped_forward
        return cls

    if cls is None:
        return wrap
    else:
        return wrap(cls)
