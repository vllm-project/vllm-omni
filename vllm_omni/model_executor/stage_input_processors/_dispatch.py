# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Consumer-side (orchestrator) input-dispatch contract layer.

Collapses the two legacy runtime probes (arity in ``process_engine_inputs``,
sampling_params-name in ``_forward_to_next_stage``) into one normalization:

* ``OrchestratorInputContext`` — the fixed transition context passed to every
  consumer-side builder; ``PlaceholderPromptBuilder`` / ``DiffusionInputBuilder``
  are the two C1-compatible callable roles ``(source_outputs, ctx)``.
* ``wrap_orchestrator_processor`` passes through ``ctx``-native processors and
  adapts legacy positional shapes (C0 3-arg; C2 ``streaming_context`` placeholder
  incl. ``_streaming_context``/four-positional; C3 ``sampling_params`` diffusion;
  C4 ``moss_tts.talker2codec``) with a one-time ``DeprecationWarning``.

Producer roles are out of scope: their ``pooling_output``/``multimodal_output``
kwargs are load-bearing and stay as-is (validated in ``_registry``).
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

from vllm_omni.inputs.data import OmniTokensPrompt

__all__ = [
    "OrchestratorInputContext",
    "PlaceholderPromptBuilder",
    "DiffusionInputBuilder",
    "wrap_orchestrator_processor",
    "invoke_orchestrator_processor",
]


@dataclass(frozen=True)
class OrchestratorInputContext:
    """Fixed transition context passed to orchestrator-facing processors.

    ``target_model_config`` / ``next_stage_hf_config`` are populated by the
    stage client from the stage's own vllm config so that legacy processors
    (e.g. JoyAI ``joyai_action_to_tts``, Nemotron ``thinker2talker_token_only``)
    can size next-stage prompts from the target model configuration.
    """

    prompt: Any | None = None
    requires_multimodal_data: bool = False
    streaming_context: Any | None = None
    sampling_params: Any | None = None
    target_model_config: Any | None = None
    next_stage_hf_config: Any | None = None


class PlaceholderPromptBuilder(Protocol):
    """C2-style sync builder: upstream outputs -> next-stage token prompts."""

    def __call__(
        self,
        source_outputs: list[Any],
        ctx: OrchestratorInputContext,
    ) -> list[OmniTokensPrompt]: ...


class DiffusionInputBuilder(Protocol):
    """Diffusion-stage input builder: upstream outputs -> diffusion payload(s)."""

    def __call__(
        self,
        source_outputs: list[Any],
        ctx: OrchestratorInputContext,
    ) -> dict | list[dict] | None: ...


# ---------------------------------------------------------------------------
# Signature inspection helpers
# ---------------------------------------------------------------------------


def _is_orchestrator_context_annotation(annotation: Any) -> bool:
    """Whether ``annotation`` names ``OrchestratorInputContext`` (class, string, or dotted)."""
    if annotation is inspect.Parameter.empty:
        return False
    if annotation is OrchestratorInputContext:
        return True
    if isinstance(annotation, str):
        text = annotation.strip().strip("'\"")
        return text == "OrchestratorInputContext" or text.endswith("OrchestratorInputContext")
    return False


def _accepts_orchestrator_input_context(fn: Any) -> bool:
    """Whether ``fn`` already speaks the ``ctx`` contract (param name or annotation)."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for name, param in signature.parameters.items():
        if name == "ctx":
            return True
        if _is_orchestrator_context_annotation(param.annotation):
            return True
    return False


def _legacy_signature(fn: Any) -> inspect.Signature | None:
    """Best-effort ``inspect.signature`` probe for a callable."""
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _positional_names(signature: inspect.Signature) -> list[str]:
    """Names of positional (or positional-or-keyword) parameters."""
    return [
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]


def _legacy_shape(signature: inspect.Signature | None) -> str:
    """Classify a legacy positional shape (full signature, keyword-only included).

    ``sampling_params`` -> C3 (shells too); ``>= 4`` positional -> C2 streaming.
    """
    if signature is None:
        return "c0"
    params = signature.parameters
    pos_names = _positional_names(signature)
    if len(pos_names) >= 2 and pos_names[0] == "stage_list" and pos_names[1] == "engine_input_source":
        return "moss"
    if "sampling_params" in pos_names:
        return "c3"
    if len(pos_names) >= 4:
        return "c2pos"
    if "sampling_params" in params:
        return "c3"
    if "streaming_context" in params or "_streaming_context" in params:
        return "c2kw"
    return "c0"


def _ctx_kwarg_names(signature: inspect.Signature | None) -> tuple[str, ...]:
    """Keyword-only ctx config params a legacy signature accepts.

    Excludes ``sampling_params``/``streaming_context`` (injected by the C3/C2
    adapters — forwarding again would raise a duplicate-kwarg ``TypeError``).
    """
    if signature is None:
        return ()
    return tuple(
        name
        for name, param in signature.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY and name in ("target_model_config", "next_stage_hf_config")
    )


def _warn_legacy_contract(fn: Any) -> None:
    """Emit a ``DeprecationWarning`` for a legacy positional processor shape."""
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
    warnings.warn(
        f"stage-input processor {name!r} uses the legacy positional contract; "
        "migrate it to the OrchestratorInputContext contract.",
        DeprecationWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Legacy adapters
# ---------------------------------------------------------------------------


def _legacy_adapter(fn: Any, signature: inspect.Signature | None, shape: str) -> Any:
    """Build the C1 adapter for a legacy *shape*; resolve and validate once.

    Call shape fixed at wrap time; dry-run ``bind`` fails fast on broken
    legacy calls at configuration time.
    """
    pos_names = _positional_names(signature) if signature is not None else []
    ctx_kwarg_names = _ctx_kwarg_names(signature)
    c2pos_fourth = (
        "next_stage_hf_config" if len(pos_names) > 3 and pos_names[3] == "next_stage_hf_config" else "streaming_context"
    )
    if signature is not None:
        if shape == "c0":
            signature.bind(None, None, None, **{name: None for name in ctx_kwarg_names})
        elif shape == "c2pos":
            signature.bind(None, None, None, None, **{name: None for name in ctx_kwarg_names})
        elif shape == "c3":
            signature.bind(None, None, None, sampling_params=None)
        elif shape == "c2kw":
            signature.bind(None, None, None, streaming_context=None)
        elif shape == "moss":
            signature.bind(None, None)

    def _adapted(source_outputs: list[Any], ctx: OrchestratorInputContext) -> Any:
        if shape == "moss":
            return fn(source_outputs, ctx.prompt)
        if shape == "c3":
            return fn(
                source_outputs,
                ctx.prompt,
                ctx.requires_multimodal_data,
                sampling_params=ctx.sampling_params,
            )
        if shape == "c2pos":
            return fn(
                source_outputs,
                ctx.prompt,
                ctx.requires_multimodal_data,
                getattr(ctx, c2pos_fourth),
                **{name: getattr(ctx, name) for name in ctx_kwarg_names},
            )
        if shape == "c2kw":
            return fn(
                source_outputs,
                ctx.prompt,
                ctx.requires_multimodal_data,
                streaming_context=ctx.streaming_context,
            )
        # C0 3-arg legacy (and the uninspectable / builtin fallback).
        return fn(
            source_outputs,
            ctx.prompt,
            ctx.requires_multimodal_data,
            **{name: getattr(ctx, name) for name in ctx_kwarg_names},
        )

    _warn_legacy_contract(fn)
    return _adapted


#: Wrap memo: the signature probe, the call-shape resolution and the
#: ``DeprecationWarning`` happen once per processor, not per forward.
_WRAP_CACHE: dict[Any, Any] = {}


def wrap_orchestrator_processor(fn: Any) -> PlaceholderPromptBuilder | DiffusionInputBuilder:
    """Return a C1-compatible ``(source_outputs, ctx)`` callable for ``fn``.

    ``ctx``-native processors pass through unchanged; legacy shapes are adapted
    per ``_legacy_shape`` with a one-time ``DeprecationWarning``.  Result is
    memoized in ``_WRAP_CACHE`` so the wrap (and warning) happens once per
    processor, not per forward.
    """
    if not callable(fn):
        raise TypeError(f"stage-input processor must be callable, got {fn!r}")

    try:
        cached = _WRAP_CACHE.get(fn)
    except TypeError:  # pragma: no cover - unhashable callable (rare)
        cached = None
    if cached is not None:
        return cached  # type: ignore[return-value]

    if _accepts_orchestrator_input_context(fn):
        wrapped: Any = fn
    else:
        signature = _legacy_signature(fn)
        wrapped = _legacy_adapter(fn, signature, _legacy_shape(signature))

    try:
        _WRAP_CACHE[fn] = wrapped
    except TypeError:  # pragma: no cover - unhashable callable (rare)
        pass
    return wrapped  # type: ignore[return-value]


def invoke_orchestrator_processor(
    fn: Any,
    source_outputs: list[Any],
    ctx: OrchestratorInputContext,
) -> Any:
    """Invoke an orchestrator-facing processor under ``(source_outputs, ctx)``."""
    return wrap_orchestrator_processor(fn)(source_outputs, ctx)
