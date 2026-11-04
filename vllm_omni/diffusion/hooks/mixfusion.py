# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion module-boundary hooks.

Mirrors :mod:`vllm_omni.diffusion.hooks.sequence_parallel`: the model declares
a ``_mixfusion_plan`` and the MixFusionDriver arms a
:class:`~vllm_omni.diffusion.forward_context.MixFusionStep` per packed denoise
step. The hooks execute the chunk-stacked layout at those boundaries, so
transformer and block ``forward`` code stays free of MixFusion branching.
Packing, boundary outputs and output restoration are delegated to the model's
adapter. All hooks are inert unless a ``mixfusion_step`` is armed.
"""

from __future__ import annotations

import inspect

from torch import nn

from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm_omni.diffusion.hooks.base import HookRegistry, ModelHook
from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name
from vllm_omni.diffusion.mixfusion.adapter import PackedInputs
from vllm_omni.diffusion.mixfusion.execution_plan import expand_per_chunk
from vllm_omni.diffusion.mixfusion.plan import (
    MixFusionExpandArg,
    MixFusionExpandRows,
    MixFusionReplaceOutput,
    MixFusionRoot,
)

# Fixed hook names, one per plan spec type (each target module owns its own
# registry, so the same name on ``transformer_blocks.*.img_mod`` instances
# never collides).
MIXFUSION_PACK_HOOK_NAME = "mixfusion_pack"
MIXFUSION_REPLACE_OUTPUT_HOOK_NAME = "mixfusion_replace_output"
MIXFUSION_EXPAND_ROWS_HOOK_NAME = "mixfusion_expand_rows"
MIXFUSION_EXPAND_ARG_HOOK_NAME = "mixfusion_expand_arg"

MIXFUSION_HOOK_NAMES = (
    MIXFUSION_PACK_HOOK_NAME,
    MIXFUSION_REPLACE_OUTPUT_HOOK_NAME,
    MIXFUSION_EXPAND_ROWS_HOOK_NAME,
    MIXFUSION_EXPAND_ARG_HOOK_NAME,
)


def _active_step():
    """Return the armed MixFusionStep, or None when this forward is not packed.

    A forward context may exist outside denoise (profiling, serial calls);
    only an armed ``mixfusion_step`` activates the hooks.
    """
    if not is_forward_context_available():
        return None
    return get_forward_context().mixfusion_step


class MixFusionRootHook(ModelHook):
    """Root (``""``) boundary: pack before forward, reassemble after it.

    The driver only arms a step after ``adapter.can_pack`` admitted the batch,
    so ``pre_forward`` packs unconditionally.
    """

    def __init__(self) -> None:
        # Previous bundle per CFG branch: sequential true-CFG interleaves the
        # positive and negative calls within one step. The adapter revalidates
        # the layout key, so a stale entry is a miss, never a wrong reuse.
        self._cached: dict[str | None, PackedInputs] = {}

    def pre_forward(self, module, *args, **kwargs):
        step = _active_step()
        if step is None:
            return args, kwargs
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        branch = getattr(module, "cfg_branch", None)
        packed = module.mixfusion_adapter.pack_inputs(module, bound, cache=self._cached.get(branch))
        self._cached[branch] = packed
        step.plan = packed.plan
        step.packed = packed
        return packed.args, packed.kwargs

    def post_forward(self, module, output):
        step = _active_step()
        if step is None or step.packed is None:
            return output
        return module.mixfusion_adapter.restore_output(module, output, step.plan)

    def reset_state(self, module):
        self._cached.clear()
        return module


class MixFusionReplaceOutputHook(ModelHook):
    """Declared boundary: return ``PackedInputs.boundary_outputs`` for this module.

    The adapter computes it once at the root; recomputing here would run on the
    wrong layout or repeat projection / ``pos_embed`` work.
    """

    def __init__(self, boundary: str) -> None:
        self._boundary = boundary

    def new_forward(self, module, *args, **kwargs):
        step = _active_step()
        if step is None or step.packed is None:
            return module._omni_original_forward(*args, **kwargs)
        try:
            return step.packed.boundary_outputs[self._boundary]
        except KeyError:
            raise KeyError(
                f"MixFusion boundary output {self._boundary!r} was not produced "
                f"by the adapter; declared on {type(module).__name__}."
            ) from None


class MixFusionExpandRowsHook(ModelHook):
    """Output side: expand a per-request ``[B, ...]`` tensor to per-chunk rows.

    E.g. AdaLN modulation parameters consumed by the chunk-stacked stream.
    """

    def post_forward(self, module, output):
        step = _active_step()
        if step is None or step.plan is None:
            return output
        return expand_per_chunk(output, step.plan.chunk_to_request)


class MixFusionExpandArgHook(ModelHook):
    """Input side: expand a named forward argument to per-chunk rows.

    E.g. ``norm_out(x, conditioning_embedding)`` takes request-level
    conditioning alongside chunk-stacked hidden states.
    """

    def __init__(self, arg_name: str, expected_dims: int = 2) -> None:
        self._arg_name = arg_name
        self._expected_dims = expected_dims

    def pre_forward(self, module, *args, **kwargs):
        step = _active_step()
        if step is None or step.plan is None:
            return args, kwargs
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        if self._arg_name not in bound.arguments:
            raise TypeError(
                f"MixFusionExpandArg on {type(module).__name__} expects a forward "
                f"argument named {self._arg_name!r}; signature is "
                f"{inspect.signature(module.forward)}."
            )
        tensor = bound.arguments[self._arg_name]
        if tensor.dim() != self._expected_dims:
            raise ValueError(
                f"{type(module).__name__} argument {self._arg_name!r} must be "
                f"{self._expected_dims}-D for MixFusion expansion, got shape "
                f"{tuple(tensor.shape)}."
            )
        bound.arguments[self._arg_name] = expand_per_chunk(tensor, step.plan.chunk_to_request)
        return bound.args, bound.kwargs


def _iter_module_path(root: nn.Module, name: str):
    """Yield the module(s) named by ``name`` plus their ancestors.

    Names use the plan's ``""`` / dotted / ``*`` syntax. Intermediate modules
    cover hooks on containers between a boundary and the root (e.g. a block
    offload hook whose ``img_mod`` is the boundary).
    """
    if not name:
        yield root
        return
    level = [root]
    for part in name.split("."):
        next_level = []
        for module in level:
            if part == "*":
                if isinstance(module, nn.ModuleList):
                    next_level.extend(module)
            else:
                child = getattr(module, part, None)
                if isinstance(child, nn.Module):
                    next_level.append(child)
        if not next_level:
            return
        yield from next_level
        level = next_level


def plan_foreign_hooks(module: nn.Module, plan: dict | None) -> list[str]:
    """Hooks, other than MixFusion's own, on modules the packed path drives.

    The packed forward bypasses some calls and swaps the batch layout for the
    rest, so hooks there would run with unintended semantics (layerwise
    offload step sizes, FSDP2 unshard all-gathers, cache keys). Returns
    ``"<plan name>:<hook name>"``; empty means the packed path can run. Both
    the Omni registry and PyTorch-native hook dicts are scanned.
    """
    foreign: set[str] = set()

    def scan(module: nn.Module, name: str) -> None:
        registry = getattr(module, "_hook_registry", None)
        for hook_name in getattr(registry, "_hooks", None) or {}:
            if hook_name not in MIXFUSION_HOOK_NAMES:
                foreign.add(f"{name or '<root>'}:{hook_name}")
        if getattr(module, "_forward_pre_hooks", None) or getattr(module, "_forward_hooks", None):
            foreign.add(f"{name or '<root>'}:torch-native")

    # The root always sees the swapped-in packed batch, whether or not the
    # plan declares it explicitly.
    scan(module, "")
    for name in plan or {}:
        for submodule in _iter_module_path(module, name):
            if submodule is module:
                continue
            scan(submodule, name)
    return sorted(foreign)


_SPEC_HOOKS = {
    MixFusionRoot: (MIXFUSION_PACK_HOOK_NAME, lambda spec: MixFusionRootHook()),
    MixFusionReplaceOutput: (
        MIXFUSION_REPLACE_OUTPUT_HOOK_NAME,
        lambda spec: MixFusionReplaceOutputHook(spec.boundary),
    ),
    MixFusionExpandRows: (MIXFUSION_EXPAND_ROWS_HOOK_NAME, lambda spec: MixFusionExpandRowsHook()),
    MixFusionExpandArg: (
        MIXFUSION_EXPAND_ARG_HOOK_NAME,
        lambda spec: MixFusionExpandArgHook(spec.arg_name, spec.expected_dims),
    ),
}


def _spec_entry(spec):
    try:
        return _SPEC_HOOKS[type(spec)]
    except KeyError:
        raise TypeError(f"Unsupported MixFusion plan entry: {type(spec).__name__} ({spec!r})") from None


def _plan_modules(module: nn.Module, name: str) -> list[nn.Module]:
    submodule = _get_submodule_by_name(module, name)
    return submodule if isinstance(submodule, list) else [submodule]


def apply_mixfusion(module: nn.Module, plan: dict) -> None:
    """Register MixFusion boundary hooks on ``module`` per its plan.

    Idempotent; stamps ``_mixfusion_hooks_applied`` on the model for pipeline
    gating.
    """
    if getattr(module, "_mixfusion_hooks_applied", False):
        return
    for name, spec in plan.items():
        hook_name, make_hook = _spec_entry(spec)
        hook = make_hook(spec)
        for m in _plan_modules(module, name):
            HookRegistry.get_or_create(m).register_hook(hook_name, hook)
    module._mixfusion_hooks_applied = True


def remove_mixfusion(module: nn.Module, plan: dict) -> None:
    """Remove the hooks applied by :func:`apply_mixfusion`.

    The root hook (and its cached bundle) is dropped, so a later
    ``apply_mixfusion`` starts from a clean cache.
    """
    for name, spec in plan.items():
        hook_name, _ = _spec_entry(spec)
        for m in _plan_modules(module, name):
            registry = getattr(m, "_hook_registry", None)
            if registry is not None:
                registry.remove_hook(hook_name)
    module._mixfusion_hooks_applied = False
