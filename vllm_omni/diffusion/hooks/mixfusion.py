# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion module-boundary hooks.

Mirrors :mod:`vllm_omni.diffusion.hooks.sequence_parallel`: the model declares
a ``_mixfusion_plan`` (see
:mod:`vllm_omni.diffusion.distributed.mixfusion_plan`) and the pipeline arms
a :class:`~vllm_omni.diffusion.forward_context.MixFusionStep` on the forward
context for each packed denoise step. These hooks execute the packed
chunk-stacked layout at the declared module boundaries, keeping transformer
and block ``forward`` code free of MixFusion branching.

All hooks are inert unless the forward context carries an armed
``mixfusion_step``, so the same model runs its serial path unchanged when
MixFusion is not active.
"""

from __future__ import annotations

from torch import nn

from vllm_omni.diffusion.distributed.mixfusion_plan import (
    MixFusionExpandArg,
    MixFusionExpandRows,
    MixFusionRoot,
    MixFusionRopePrepare,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm_omni.diffusion.hooks.base import HookRegistry, ModelHook
from vllm_omni.diffusion.hooks.sequence_parallel import _get_submodule_by_name
from vllm_omni.diffusion.mixfusion.layout import (
    expand_per_chunk,
    reassemble_chunk_rows,
)

# Fixed hook names, one per plan spec type (each target module owns its own
# registry, so the same name on ``transformer_blocks.*.img_mod`` instances
# never collides).
MIXFUSION_PACK_HOOK_NAME = "mixfusion_pack"
MIXFUSION_ROPE_HOOK_NAME = "mixfusion_rope_prepare"
MIXFUSION_EXPAND_ROWS_HOOK_NAME = "mixfusion_expand_rows"
MIXFUSION_EXPAND_ARG_HOOK_NAME = "mixfusion_expand_arg"

MIXFUSION_HOOK_NAMES = (
    MIXFUSION_PACK_HOOK_NAME,
    MIXFUSION_ROPE_HOOK_NAME,
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

    ``pre_forward`` calls the transformer's pack method (default
    ``prepare_mixfusion_inputs``) on the per-request latent list and swaps the
    chunk-stacked batch into ``hidden_states``; ``post_forward`` reassembles
    the per-request outputs via :func:`reassemble_chunk_rows`.
    """

    def __init__(self, pack_method: str = "prepare_mixfusion_inputs") -> None:
        self._pack_method = pack_method
        # Bundle from the previous denoise step: the layout (RoPE tables,
        # plan, row counts) depends only on request shapes, so it is reused
        # via the pack method's ``cached`` fast path. Cleared by reset_state.
        self._cached = None

    def pre_forward(self, module, *args, **kwargs):
        step = _active_step()
        if step is None:
            return args, kwargs
        if "hidden_states" not in kwargs:
            # Contract: the packed call must pass hidden_states by keyword; the
            # pipeline already does. Keep the failure actionable rather than
            # silently packing nothing.
            raise TypeError(
                "MixFusion packing requires `hidden_states` to be passed as a "
                "keyword argument to the transformer forward."
            )
        if not isinstance(kwargs["hidden_states"], (list, tuple)):
            # A batched tensor means a non-packed forward while the step is
            # armed (e.g. memory profiling): stay serial.
            return args, kwargs
        packed = getattr(module, self._pack_method)(
            hidden_states=kwargs["hidden_states"],
            encoder_hidden_states=kwargs["encoder_hidden_states"],
            encoder_hidden_states_mask=kwargs.get("encoder_hidden_states_mask"),
            img_shapes=kwargs.get("img_shapes"),
            txt_seq_lens=kwargs.get("txt_seq_lens"),
            attention_kwargs=kwargs.get("attention_kwargs"),
            cached=self._cached,
        )
        self._cached = packed
        step.plan = packed.plan
        step.image_row_counts = packed.image_row_counts
        # Forward unpacks ``image_rope_prepare`` as
        # ``hidden_states, vid_freqs, txt_freqs``; hand it the root-computed
        # packed triple so that boundary becomes a no-op.
        step.packed = (packed.hidden_states, *packed.image_rotary_emb)
        kwargs["hidden_states"] = packed.hidden_states
        kwargs["encoder_hidden_states_mask"] = packed.encoder_hidden_states_mask
        return args, kwargs

    def post_forward(self, module, output):
        step = _active_step()
        if step is None or step.plan is None:
            return output
        # ``forward`` always returns Transformer2DModelOutput here (the model
        # ignores return_dict today); hand the pipeline the bare per-request
        # list the packed path already produced pre-hook.
        return reassemble_chunk_rows(output.sample, step.plan.request_chunk_ranges, step.image_row_counts)

    def reset_state(self, module):
        self._cached = None
        return module


class MixFusionRopePrepareHook(ModelHook):
    """``image_rope_prepare`` boundary: return the root-computed packed tensors.

    The packed hidden states (already projected by ``img_in``) and the RoPE
    tables are produced by the root hook, because only the root boundary also
    sees the text stream. Recomputing them here would either run on the wrong
    layout (``img_in`` on a list) or repeat the projection/pos_embed work.
    """

    def new_forward(self, module, *args, **kwargs):
        step = _active_step()
        if step is None or step.packed is None:
            return module._omni_original_forward(*args, **kwargs)
        return step.packed


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
    """Input side: expand a positional argument to per-chunk rows.

    E.g. ``norm_out(x, temb)`` receives per-request ``temb`` alongside
    chunk-stacked hidden states.
    """

    def __init__(self, arg_index: int, expected_dims: int = 2) -> None:
        self._arg_index = arg_index
        self._expected_dims = expected_dims

    def pre_forward(self, module, *args, **kwargs):
        step = _active_step()
        if step is None or step.plan is None:
            return args, kwargs
        args = list(args)
        if self._arg_index >= len(args):
            raise IndexError(
                f"MixFusionExpandArg on {type(module).__name__} expects "
                f"positional argument {self._arg_index}, got {len(args)} "
                "positional arguments."
            )
        tensor = args[self._arg_index]
        if tensor.dim() != self._expected_dims:
            raise ValueError(
                f"{type(module).__name__} argument {self._arg_index} must be "
                f"{self._expected_dims}-D for MixFusion expansion, got shape "
                f"{tuple(tensor.shape)}."
            )
        args[self._arg_index] = expand_per_chunk(tensor, step.plan.chunk_to_request)
        return tuple(args), kwargs


def _spec_hook_name(spec) -> str:
    if isinstance(spec, MixFusionRoot):
        return MIXFUSION_PACK_HOOK_NAME
    if isinstance(spec, MixFusionRopePrepare):
        return MIXFUSION_ROPE_HOOK_NAME
    if isinstance(spec, MixFusionExpandRows):
        return MIXFUSION_EXPAND_ROWS_HOOK_NAME
    if isinstance(spec, MixFusionExpandArg):
        return MIXFUSION_EXPAND_ARG_HOOK_NAME
    raise TypeError(f"Unsupported MixFusion plan entry: {type(spec).__name__} ({spec!r})")


def _spec_hook(spec) -> ModelHook:
    if isinstance(spec, MixFusionRoot):
        return MixFusionRootHook(spec.pack_method)
    if isinstance(spec, MixFusionRopePrepare):
        return MixFusionRopePrepareHook()
    if isinstance(spec, MixFusionExpandRows):
        return MixFusionExpandRowsHook()
    if isinstance(spec, MixFusionExpandArg):
        return MixFusionExpandArgHook(spec.arg_index, spec.expected_dims)
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
