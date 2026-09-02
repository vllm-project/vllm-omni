# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)


def _matches_repeated_block(
    name: str,
    module: nn.Module,
    repeated_blocks: list[str],
    repeated_block_attrs: list[str],
) -> bool:
    class_name = module.__class__.__name__
    if class_name in repeated_blocks:
        return True

    for attr in ("_fsdp_wrapped_module", "module", "_orig_mod"):
        wrapped = getattr(module, attr, None)
        if wrapped is not None and wrapped.__class__.__name__ in repeated_blocks:
            return True

    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] in repeated_block_attrs and parts[-1].isdigit()


def regionally_compile(
    model: nn.Module,
    *compile_args: Any,
    **compile_kwargs: Any,
) -> nn.Module:
    """
    Apply regional compilation to a PyTorch model.

    Args:
        model: The PyTorch model instance to compile
        *compile_args: Positional arguments forwarded to torch.compile
        **compile_kwargs: Keyword arguments forwarded to torch.compile. With
            the Inductor backend and no non-default compile mode, values in an
            explicit ``options`` mapping override model defaults from
            ``_regional_compile_inductor_options``.

    Returns:
        The same model instance (modified in-place)
    """
    # Get the list of repeated blocks from the model
    repeated_blocks = getattr(model, "_repeated_blocks", None)

    if not repeated_blocks:
        logger.warning("Regional compilation skipped because the model does not define `_repeated_blocks`.")
        return model

    repeated_block_attrs = getattr(model, "_layerwise_offload_blocks_attrs", [])

    # Some repeated regions require model-specific Inductor options to retain
    # their eager numerical contract. Keep those defaults next to the model
    # definition while allowing an explicit caller override. Other backends
    # own their option contract. A non-default Inductor mode cannot be combined
    # with options in torch.compile, so fail before activating a compiled model
    # that would silently lose its eager numerical contract.
    model_compile_options = getattr(model, "_regional_compile_inductor_options", None)
    backend = compile_kwargs.get("backend", "inductor")
    mode = compile_kwargs.get("mode")
    if model_compile_options and backend == "inductor":
        if mode not in (None, "default"):
            raise ValueError(
                f"Regional Inductor options required by {model.__class__.__name__} "
                f"cannot be combined with torch.compile mode={mode!r}; pass "
                "equivalent explicit options instead"
            )
        if mode == "default":
            compile_kwargs = {key: value for key, value in compile_kwargs.items() if key != "mode"}
        caller_compile_options = compile_kwargs.get("options")
        if caller_compile_options is None:
            caller_compile_options = {}
        elif not isinstance(caller_compile_options, dict):
            raise TypeError("torch.compile options must be a dict or None")
        compile_kwargs = {
            **compile_kwargs,
            "options": {
                **model_compile_options,
                **caller_compile_options,
            },
        }

    # Build all compiled callables before mutating the model. This keeps setup
    # failures atomic: callers can safely continue with the uncompiled model if
    # torch.compile raises synchronously for any repeated block.
    # Keep the pre-hook callable separately.  DLO and the ordinary layerwise
    # offloader install a HookRegistry wrapper in ``module.forward``.  That
    # wrapper performs stream/event synchronization and storage rebinding, so
    # compiling it pulls offload control flow into the graph.  Compile the
    # original block compute instead and leave the wrapper outside the graph.
    compiled_forwards: list[tuple[nn.Module, Any | None, Any]] = []
    for name, submod in model.named_modules():
        if _matches_repeated_block(name, submod, repeated_blocks, repeated_block_attrs):
            # Compile the block compute while keeping nn.Module.__call__ hooks
            # outside the compiled graph. If a HookRegistry is already
            # installed, ``submod.forward`` is the hook dispatcher and the
            # original callable is stored in ``_omni_original_forward``.
            # NOTE: anything that wraps this callable must stay
            # signature-transparent — cache-dit's BlockAdapter matches blocks
            # by inspecting forward's parameter names and return annotation,
            # which torch.compile preserves.
            original_forward = getattr(submod, "_omni_original_forward", None)
            forward_to_compile = original_forward if original_forward is not None else submod.forward
            # torch.compile removes a few special entries from ``options`` in
            # place. Give every repeated block its own mapping so one compile
            # call cannot change the configuration of subsequent blocks.
            block_compile_kwargs = dict(compile_kwargs)
            block_compile_options = block_compile_kwargs.get("options")
            if isinstance(block_compile_options, dict):
                block_compile_kwargs["options"] = dict(block_compile_options)
            compiled_forwards.append(
                (
                    submod,
                    original_forward,
                    torch.compile(forward_to_compile, *compile_args, **block_compile_kwargs),
                )
            )

    if not compiled_forwards:
        logger.warning(f"Regional compilation skipped because {repeated_blocks} classes are not found in the model.")
    else:
        for submod, original_forward, compiled_forward in compiled_forwards:
            if original_forward is None:
                submod.forward = compiled_forward
                continue

            # Keep HookRegistry's dispatcher as ``forward`` and replace only
            # the callable it dispatches to.  Hooks such as MagCache retain a
            # reference to the original callable as well, so update those
            # references when they still point at the pre-compile function.
            submod._omni_original_forward = compiled_forward
            registry = getattr(submod, "_hook_registry", None)
            for hook in getattr(registry, "_hooks", {}).values():
                fn_ref = getattr(hook, "fn_ref", None)
                if fn_ref is not None and fn_ref.original_forward is original_forward:
                    fn_ref.original_forward = compiled_forward
        logger.info(
            "Regional compilation applied to %d module(s) for repeated blocks %s.",
            len(compiled_forwards),
            repeated_blocks,
        )

    return model
