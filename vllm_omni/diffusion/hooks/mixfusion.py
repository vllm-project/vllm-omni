# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion module-boundary hooks.

Mirrors :mod:`vllm_omni.diffusion.hooks.sequence_parallel`: the model declares
a ``_mixfusion_plan`` (see
:mod:`vllm_omni.diffusion.mixfusion.plan`) and the pipeline arms
a :class:`~vllm_omni.diffusion.forward_context.MixFusionStep` on the forward
context for each packed denoise step. These hooks execute the packed
chunk-stacked layout at the declared module boundaries, keeping transformer
and block ``forward`` code free of MixFusion branching.

The hooks contain no model-specific knowledge: packing, boundary outputs and
output restoration are delegated to the model's
:class:`~vllm_omni.diffusion.mixfusion.adapter.MixFusionAdapter`.

All hooks are inert unless the forward context carries an armed
``mixfusion_step``, so the same model runs its serial path unchanged when
MixFusion is not active.
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
from vllm_omni.diffusion.mixfusion.adapter import MixFusionAdapter, PackedInputs
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


def _resolve_adapter(module: nn.Module, step) -> MixFusionAdapter:
    """Return the model's adapter, falling back to the step-carried one."""
    adapter = getattr(module, "mixfusion_adapter", None)
    if adapter is None:
        adapter = step.adapter
    if adapter is None:
        raise TypeError(
            f"MixFusion is armed but {type(module).__name__} declares no "
            "`mixfusion_adapter`. Add a MixFusionAdapter attribute to the model "
            "(see vllm_omni/diffusion/mixfusion/adapter.py) or leave MixFusion off."
        )
    return adapter


class MixFusionRootHook(ModelHook):
    """Root (``""``) boundary: pack before forward, reassemble after it.

    ``pre_forward`` binds the model's forward call and delegates to the
    adapter: it decides whether the call is packable and rewrites it into the
    packed layout. ``post_forward`` asks the adapter to turn the packed output
    back into one output per request.
    """

    def __init__(self) -> None:
        # Bundle from the previous denoise step: the layout (RoPE tables,
        # plan, row counts) depends only on request shapes, so the adapter
        # reuses it through the ``cache`` argument. Cleared by reset_state.
        self._cached: PackedInputs | None = None

    def pre_forward(self, module, *args, **kwargs):
        step = _active_step()
        if step is None:
            return args, kwargs
        adapter = _resolve_adapter(module, step)
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        can_pack, _ = adapter.can_pack(module, bound)
        if not can_pack:
            # A batched tensor while the step is armed (e.g. memory
            # profiling): stay serial.
            return args, kwargs
        packed = adapter.pack_inputs(module, bound, step=step, cache=self._cached)
        if packed is None:
            return args, kwargs
        self._cached = packed
        step.adapter = adapter
        step.plan = packed.plan
        step.layout = adapter.build_attention_layout(packed.plan)
        step.packed = packed
        return packed.args, packed.kwargs

    def post_forward(self, module, output):
        step = _active_step()
        if step is None or step.packed is None:
            return output
        # ``forward`` always returns Transformer2DModelOutput here (the model
        # ignores return_dict today); hand the pipeline the bare per-request
        # list the adapter restores.
        return step.adapter.restore_output(module, output, step.packed.plan)

    def reset_state(self, module):
        self._cached = None
        return module


class MixFusionReplaceOutputHook(ModelHook):
    """Declared boundary: return the adapter-computed value for this module.

    The value comes from ``PackedInputs.boundary_outputs``: the adapter
    computes it once at the root, where the request list and text stream are
    still visible. Recomputing it here would either run on the wrong layout or
    repeat projection / ``pos_embed`` work.
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
                f"by {type(step.adapter).__name__}; declared on {type(module).__name__}."
            ) from None


class MixFusionExpandRowsHook(ModelHook):
    """Output side: expand a per-request ``[B, ...]`` tensor to per-chunk rows.

    E.g. AdaLN modulation parameters computed per request but consumed by the
    chunk-stacked image stream.
    """

    def post_forward(self, module, output):
        step = _active_step()
        if step is None or step.plan is None:
            return output
        return expand_per_chunk(output, step.plan.chunk_to_request)


class MixFusionExpandArgHook(ModelHook):
    """Input side: expand a named forward argument to per-chunk rows.

    E.g. ``norm_out(x, conditioning_embedding)`` receives per-request
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

    Names use the plan's ``""`` / dotted / ``*`` syntax. Yielding every
    intermediate module covers hooks installed on containers between a plan
    boundary and the root (e.g. a layerwise-offload hook on a transformer
    block, whose ``img_mod`` is a boundary).
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

    The packed forward bypasses some module calls (boundaries, block forwards)
    and swaps the batch layout seen by the ones it keeps, so hooks installed
    there run with semantics they were not written for: layerwise offload
    would prefetch with a chunk-count step size, FSDP2's unshard pre-hook
    would all-gather against a batch dimension that no longer matches the
    request count, and cache hooks would build keys on the packed layout.
    Returns ``"<plan name>:<hook name>"`` descriptions; empty means the
    packed path can run. Both the Omni hook registry and PyTorch-native hook
    dicts are scanned.
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


def _spec_hook_name(spec) -> str:
    if isinstance(spec, MixFusionRoot):
        return MIXFUSION_PACK_HOOK_NAME
    if isinstance(spec, MixFusionReplaceOutput):
        return MIXFUSION_REPLACE_OUTPUT_HOOK_NAME
    if isinstance(spec, MixFusionExpandRows):
        return MIXFUSION_EXPAND_ROWS_HOOK_NAME
    if isinstance(spec, MixFusionExpandArg):
        return MIXFUSION_EXPAND_ARG_HOOK_NAME
    raise TypeError(f"Unsupported MixFusion plan entry: {type(spec).__name__} ({spec!r})")


def _spec_hook(spec) -> ModelHook:
    if isinstance(spec, MixFusionRoot):
        return MixFusionRootHook()
    if isinstance(spec, MixFusionReplaceOutput):
        return MixFusionReplaceOutputHook(spec.boundary)
    if isinstance(spec, MixFusionExpandRows):
        return MixFusionExpandRowsHook()
    if isinstance(spec, MixFusionExpandArg):
        return MixFusionExpandArgHook(spec.arg_name, spec.expected_dims)
    raise TypeError(f"Unsupported MixFusion plan entry: {type(spec).__name__} ({spec!r})")


def apply_mixfusion(module: nn.Module, plan: dict) -> None:
    """Register MixFusion boundary hooks on ``module`` per its plan.

    Mirrors :func:`~vllm_omni.diffusion.hooks.sequence_parallel.apply_sequence_parallel`:
    idempotent, resolves ``""`` / dotted / ``*`` wildcard module names, and
    stamps ``_mixfusion_hooks_applied`` on the model for pipeline gating.
    """
    if getattr(module, "_mixfusion_hooks_applied", False):
        return
    for name, spec in plan.items():
        hook_name = _spec_hook_name(spec)
        submodule = _get_submodule_by_name(module, name)
        if not isinstance(submodule, list):
            submodule = [submodule]
        hook = _spec_hook(spec)
        for m in submodule:
            registry = HookRegistry.get_or_create(m)
            registry.register_hook(hook_name, hook)
    module._mixfusion_hooks_applied = True


def remove_mixfusion(module: nn.Module, plan: dict) -> None:
    """Remove the MixFusion boundary hooks previously applied by ``apply_mixfusion``.

    The root hook instance (and its cached bundle) is dropped with the hook, so
    a later ``apply_mixfusion`` starts from a clean cache.
    """
    for name, spec in plan.items():
        hook_name = _spec_hook_name(spec)
        submodule = _get_submodule_by_name(module, name)
        if not isinstance(submodule, list):
            submodule = [submodule]
        for m in submodule:
            registry = getattr(m, "_hook_registry", None)
            if registry is None:
                continue
            registry.remove_hook(hook_name)
    module._mixfusion_hooks_applied = False
