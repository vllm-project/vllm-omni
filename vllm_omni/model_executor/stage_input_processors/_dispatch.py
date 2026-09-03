# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Orchestrator input-dispatch contract layer (RFC #4872).

This module defines the **consumer-side (orchestrator / stage-client) dispatch
contract** for ``stage_input_processors``.  It replaces the two inconsistent
``inspect.signature`` probes in the orchestrator runtime with one shared
contract:

* ``stage_engine_core_client.process_engine_inputs`` used an **arity** probe
  (``len(signature.parameters) >= 4``) to pick a 3-arg vs 4-arg positional
  call.
* ``orchestrator._forward_to_next_stage`` (diffusion branch) used a **name**
  probe (``"sampling_params" in signature.parameters``) to decide whether to
  inject the diffusion-stage sampling params as a keyword.

Both are collapsed into a single normalization layer:

* ``OrchestratorInputContext`` — the fixed transition context the runtime
  passes to every consumer-side builder.
* ``PlaceholderPromptBuilder`` / ``DiffusionInputBuilder`` — the two
  orchestrator-facing callable roles (C1-compatible ``(source_outputs, ctx)``).
* ``wrap_orchestrator_processor`` / ``invoke_orchestrator_processor`` —
  processors that already speak the ``ctx`` contract are passed through
  unchanged; legacy positional shapes (C0/C2/C3/C4 below) are adapted with a
  ``DeprecationWarning``.

**Producer roles are intentionally out of scope here.**  ``FullPayloadProducer``
(``*, transfer_manager, pooling_output, request, is_finished=...``) and
``AsyncChunkProducer`` (``*, transfer_manager, multimodal_output, request,
is_finished=False``) run inside workers and never receive an
``OrchestratorInputContext``; their ``pooling_output`` / ``multimodal_output``
keyword names are load-bearing parts of the producer contract and are kept
as-is (the producer kwargs contract stays unchanged).

Legacy positional shapes normalized here (detection reads the **full**
signature, keyword-only parameters included; the final invocation is validated
with ``inspect.Signature.bind``):

* C0 3-arg: ``(source_outputs, prompt, requires_multimodal_data)``.
* C2 placeholder: ``(source_outputs, prompt, requires_multimodal_data,
  streaming_context=None)`` (the ``_streaming_context`` variant used by
  ``forced_aligner.code2wav2aligner`` / ``minicpmo_4_5_omni.llm2tts`` is also
  recognized; a keyword-only ``*, streaming_context=None`` shell receives the
  value as a keyword, and any four-positional callable falls back to this
  shape regardless of the 4th parameter's name).
* C3 diffusion: ``(source_outputs, prompt, requires_multimodal_data,
  sampling_params=None)`` — a keyword-only ``*, sampling_params=None`` shell
  also receives ``ctx.sampling_params``.
* C4 legacy multi-source: ``(stage_list, engine_input_source, ...)`` — only
  ``moss_tts.talker2codec`` remains today; normalized by
  ``_adapt_moss_processor``, which binds ``stage_list=source_outputs`` and
  ``engine_input_source=ctx.prompt`` and drops the trailing
  ``prompt``/``requires_multimodal_data`` arguments (unused by that
  implementation).
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from vllm_omni.inputs.data import OmniTokensPrompt

__all__ = [
    "OrchestratorInputContext",
    "PlaceholderPromptBuilder",
    "DiffusionInputBuilder",
    "FullPayloadProducer",
    "AsyncChunkProducer",
    "wrap_orchestrator_processor",
    "invoke_orchestrator_processor",
]


@dataclass(frozen=True)
class OrchestratorInputContext:
    """Fixed transition context passed to orchestrator-facing processors.

    There is deliberately **no** ``model_config`` field: a processor that needs
    a model config reads it through the upstream stage closure, never through
    this context.
    """

    prompt: Any | None = None
    requires_multimodal_data: bool = False
    streaming_context: Any | None = None
    sampling_params: Any | None = None


@runtime_checkable
class PlaceholderPromptBuilder(Protocol):
    """C2-style sync builder: upstream outputs -> next-stage token prompts."""

    def __call__(
        self,
        source_outputs: list[Any],
        ctx: OrchestratorInputContext,
    ) -> list[OmniTokensPrompt]: ...


@runtime_checkable
class DiffusionInputBuilder(Protocol):
    """Diffusion-stage input builder: upstream outputs -> diffusion payload(s)."""

    def __call__(
        self,
        source_outputs: list[Any],
        ctx: OrchestratorInputContext,
    ) -> dict | list[dict] | None: ...


@runtime_checkable
class FullPayloadProducer(Protocol):
    """Producer-side (worker) full-payload builder.

    ``pooling_output`` is a **load-bearing** keyword name: it is the connector
    data-plane contract and must not be renamed or made positional.  These
    builders never receive an ``OrchestratorInputContext``.
    """

    def __call__(
        self,
        *,
        transfer_manager: Any,
        pooling_output: Any,
        request: Any,
        is_finished: bool = ...,
    ) -> Any: ...


@runtime_checkable
class AsyncChunkProducer(Protocol):
    """Producer-side (worker) async-chunk builder.

    ``multimodal_output`` is a **load-bearing** keyword name: it is the
    connector data-plane contract and must not be renamed or made positional.
    These builders never receive an ``OrchestratorInputContext``.
    """

    def __call__(
        self,
        *,
        transfer_manager: Any,
        multimodal_output: Any,
        request: Any,
        is_finished: bool = False,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Signature inspection helpers
# ---------------------------------------------------------------------------


def _is_orchestrator_context_annotation(annotation: Any) -> bool:
    """Whether ``annotation`` names ``OrchestratorInputContext``.

    Handles the class object, plain string annotations (``from __future__
    import annotations``) and fully-qualified dotted strings.
    """
    if annotation is inspect.Parameter.empty:
        return False
    if annotation is OrchestratorInputContext:
        return True
    if isinstance(annotation, str):
        text = annotation.strip().strip("'\"")
        return text == "OrchestratorInputContext" or text.endswith("OrchestratorInputContext")
    return False


def _accepts_orchestrator_input_context(fn: Any) -> bool:
    """Whether ``fn`` already speaks the new ``ctx`` contract.

    True when the signature contains a parameter named ``ctx``, or any
    parameter whose type annotation is ``OrchestratorInputContext``.
    """
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
    """Classify a legacy positional processor shape.

    The probe reads the **full** signature — keyword-only parameters included —
    so a ``def f(..., *, sampling_params=None)`` shell is still recognized as a
    C3 diffusion builder (and receives ``ctx.sampling_params``), and any
    four-positional callable falls back to the legacy C2 streaming shape
    regardless of the 4th parameter's name (the legacy arity probe treated every
    ``>= 4``-parameter callable that way).
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


def _bind_invoke(fn: Any, signature: inspect.Signature | None, *args: Any, **kwargs: Any) -> Any:
    """Invoke *fn* with a legacy call shape bound via ``inspect.Signature.bind``.

    Binding at the final invocation guarantees a legacy adapter never silently
    drops context fields (e.g. forwarding ``sampling_params`` to a
    positional-only parameter raises instead of being ignored) and never passes
    an argument the processor cannot accept.
    """
    if signature is None:
        # Uninspectable callable (e.g. builtin): pass the shape through.
        return fn(*args, **kwargs)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return fn(*bound.args, **bound.kwargs)


def _warn_legacy_contract(fn: Any) -> None:
    """Emit a ``DeprecationWarning`` for a legacy positional processor shape."""
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
    warnings.warn(
        f"stage-input processor {name!r} uses the legacy positional contract; "
        "migrate it to the OrchestratorInputContext contract "
        "(RFC #4872 Phase 2).",
        DeprecationWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Legacy adapters
# ---------------------------------------------------------------------------


def _adapt_moss_processor(fn: Any, signature: inspect.Signature | None) -> Any:
    """C4 legacy multi-source adapter (e.g. ``moss_tts.talker2codec``).

    The legacy multi-source sync processors take
    ``(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False)``.
    The orchestrator contract dispatches ``(source_outputs, ctx)``; we bind
    ``stage_list=source_outputs`` and ``engine_input_source=ctx.prompt`` and
    drop the trailing ``prompt``/``requires_multimodal_data`` arguments (the
    ``moss_tts.talker2codec`` implementation does not use them, so this is
    behaviour-preserving w.r.t. the previous 4-arg positional call).
    """

    def _adapted(source_outputs: list[Any], ctx: OrchestratorInputContext) -> Any:
        return _bind_invoke(fn, signature, source_outputs, ctx.prompt)

    _warn_legacy_contract(fn)
    return _adapted


#: Wrap memo: the signature probe + ``DeprecationWarning`` must happen **once per
#: processor**, not on every forward.  ``invoke_orchestrator_processor`` runs per
#: request, and re-wrapping a legacy C2 shape (e.g. ``thinker2talker_token_only``)
#: would otherwise re-emit the warning on every forward.  Keyed by the callable
#: object itself (module-level processors are few and live for the process).
_WRAP_CACHE: dict[Any, Any] = {}


def wrap_orchestrator_processor(fn: Any) -> PlaceholderPromptBuilder | DiffusionInputBuilder:
    """Return a C1-compatible callable ``(source_outputs, ctx)`` for ``fn``.

    - If ``fn`` already accepts ``ctx`` (``_accepts_orchestrator_input_context``),
      it is returned unchanged.
    - Otherwise the legacy positional shape is adapted:
      * C4 ``(stage_list, engine_input_source, ...)`` -> ``_adapt_moss_processor``;
      * C3 ``(source_outputs, prompt, requires_multimodal_data, sampling_params=...)``
        forwards ``ctx.sampling_params`` when the parameter exists;
      * C2 ``(source_outputs, prompt, requires_multimodal_data,
        streaming_context=...)`` (or ``_streaming_context``) forwards
        ``ctx.streaming_context``;
      * C0 3-arg ``(source_outputs, prompt, requires_multimodal_data)``.
    Every legacy shape emits a ``DeprecationWarning``.

    The result is memoized per callable (see ``_WRAP_CACHE``): the wrap — and its
    ``DeprecationWarning`` — happens exactly once per processor, at first
    resolution/invocation, so ``invoke_orchestrator_processor`` does not re-wrap
    (or re-warn) on every forward.
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
        shape = _legacy_shape(signature)

        if shape == "moss":
            wrapped = _adapt_moss_processor(fn, signature)
        else:

            def _adapted(source_outputs: list[Any], ctx: OrchestratorInputContext) -> Any:
                if shape == "c3":
                    # C3 diffusion: forwarding sampling_params (diffusion-stage
                    # params).  Read from the full signature so a keyword-only
                    # ``*, sampling_params=None`` shell receives the value too.
                    return _bind_invoke(
                        fn,
                        signature,
                        source_outputs,
                        ctx.prompt,
                        ctx.requires_multimodal_data,
                        sampling_params=ctx.sampling_params,
                    )
                if shape == "c2pos":
                    # C2 four-positional legacy fallback (any 4th parameter
                    # name): the streaming context is the 4th positional arg.
                    return _bind_invoke(
                        fn,
                        signature,
                        source_outputs,
                        ctx.prompt,
                        ctx.requires_multimodal_data,
                        ctx.streaming_context,
                    )
                if shape == "c2kw":
                    # Keyword-only streaming context shell.
                    return _bind_invoke(
                        fn,
                        signature,
                        source_outputs,
                        ctx.prompt,
                        ctx.requires_multimodal_data,
                        streaming_context=ctx.streaming_context,
                    )
                # C0 3-arg legacy (and the uninspectable / builtin fallback).
                return _bind_invoke(fn, signature, source_outputs, ctx.prompt, ctx.requires_multimodal_data)

            _warn_legacy_contract(fn)
            wrapped = _adapted

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
    """Invoke an orchestrator-facing processor under the fixed ``(source_outputs, ctx)`` contract."""
    return wrap_orchestrator_processor(fn)(source_outputs, ctx)
