# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage-input processor registry (RFC #4872).

This module is the **structural validation** layer behind the three ``importlib``
resolution points that are validated as independent symbol lookups:

* ``engine/stage_init_utils.extract_legacy_stage_metadata`` — consumer-side
  (orchestrator) input processors (``custom_process_input_func`` /
  ``sync_process_input_func``).
* ``worker/omni_connector_model_runner_mixin._load_custom_func`` — the
  producer-side full-payload builder (``FullPayloadProducer``).
* ``distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.__init__``
  — the producer-side async-chunk builder (``AsyncChunkProducer``).

Scope / responsibility split
----------------------------
The registry **only performs signature-level structural checks**
(``inspect.signature``); it never executes processor logic and never loads
model weights.  Behavioural guarantees (payload shape, flush semantics, golden
outputs) belong to the golden / parity tests, not here — do not conflate the
two responsibilities.

Producer kwargs are contract.  ``FullPayloadProducer`` keeps ``pooling_output``
(and ``is_finished`` is **optional** — the worker retries without it when the
processor rejects the kwarg); ``AsyncChunkProducer`` keeps ``multimodal_output``
and **must** accept ``is_finished`` — the scheduler always passes it on the
async-chunk path (see ``chunk_transfer_adapter._send_single_request``).

Kind inference is name-driven (suffix-based), following the naming convention
documented in :mod:`vllm_omni.model_executor.stage_input_processors`:

* ``_token_only``            -> ``placeholder_prompt_builder``
* ``_full_payload`` / ``_batch`` -> ``producer_full_payload``
* ``_async_chunk``           -> ``producer_async_chunk``
* ``ar2diffusion`` / ``ar2dit`` / ``thinker2imagegen`` (and friends) ->
  ``diffusion_input_builder``
* no suffix                  -> ``legacy_orchestrator_builder``
* ``moss_tts.talker2codec`` (legacy multi-source shape) ->
  ``legacy_multi_source``

``resolve_processor`` is a drop-in replacement for the legacy
``getattr(importlib.import_module(mod_path), fn_name)`` lookups: it returns a
:class:`ProcessorSpec` whose ``fn`` is the **same callable object** the legacy
code produced, so callers keep identical runtime behaviour while gaining
validated kind metadata.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

ProcessorKind = Literal[
    "placeholder_prompt_builder",
    "diffusion_input_builder",
    "legacy_orchestrator_builder",
    "producer_full_payload",
    "producer_async_chunk",
    "legacy_multi_source",
]

#: Suffixes that identify a producer-side full-payload builder.
_FULL_PAYLOAD_SUFFIXES: tuple[str, ...] = ("_full_payload", "_batch")
#: Suffix that identifies a producer-side async-chunk builder.
_ASYNC_CHUNK_SUFFIX: str = "_async_chunk"
#: Suffix that identifies a consumer-side placeholder builder.
_TOKEN_ONLY_SUFFIX: str = "_token_only"

#: Diffusion-stage input-builder name stems (suffix match).
_DIFFUSION_SUFFIXES: tuple[str, ...] = (
    "ar2diffusion",
    "ar2dit",
    "thinker2imagegen",
    "ar2acoustic",
    "ar_to_dit",
    "llm2audio_vae",
    "thinker2token2wav",
)

#: Module hint for the only remaining legacy multi-source processor family.
_MOSS_MULTI_SOURCE_MODULE_HINT: str = "moss_tts"
#: Exact function names in ``moss_tts`` with the multi-source shape.
_MOSS_MULTI_SOURCE_NAMES: frozenset[str] = frozenset({"talker2codec"})

#: Producer kinds are never orchestrator-side, so the dead-processor hint never
#: applies to them.
_PRODUCER_KINDS: frozenset[str] = frozenset({"producer_full_payload", "producer_async_chunk"})

#: All valid :class:`ProcessorKind` values (used to validate manual overrides).
_PROCESSOR_KINDS: frozenset[str] = frozenset(
    {
        "placeholder_prompt_builder",
        "diffusion_input_builder",
        "legacy_orchestrator_builder",
        "producer_full_payload",
        "producer_async_chunk",
        "legacy_multi_source",
    }
)

#: Soft-check scalar return annotations that contradict a stage-input role.
_SCALAR_RETURN_ANNOTATIONS: frozenset[Any] = frozenset({int, str, bool, float})

__all__ = [
    "ProcessorKind",
    "ProcessorSpec",
    "ProcessorValidationError",
    "import_symbol",
    "infer_kind",
    "validate_processor",
    "resolve_processor",
    "dead_processor_hint",
    "register_processor",
]


@dataclass(frozen=True)
class ProcessorSpec:
    """Resolved and validated stage-input processor.

    Attributes:
        path: The dotted ``module.attr`` path the processor was resolved from.
        kind: The inferred (or overridden) processor kind.
        fn: The callable itself — identical to what the legacy
            ``getattr(importlib.import_module(...), ...)`` lookup returned.
    """

    path: str
    kind: ProcessorKind
    fn: Any


class ProcessorValidationError(Exception):
    """A stage-input processor failed structural validation.

    Carries enough context to locate the offending stage / config line:
    ``stage_id`` (when a ``stage_config`` was supplied), ``path``, ``kind`` and
    the specific ``rule`` that was violated.
    """

    def __init__(
        self,
        *,
        stage_id: int | None = None,
        path: str | None = None,
        kind: ProcessorKind | None = None,
        rule: str | None = None,
        message: str | None = None,
    ) -> None:
        self.stage_id = stage_id
        self.path = path
        self.kind = kind
        self.rule = rule
        self.message = message
        super().__init__(self._format())

    def _format(self) -> str:
        parts: list[str] = []
        if self.stage_id is not None:
            parts.append(f"stage_id={self.stage_id}")
        if self.path:
            parts.append(f"path={self.path!r}")
        if self.kind:
            parts.append(f"kind={self.kind!r}")
        if self.rule:
            parts.append(f"rule={self.rule}")
        detail = ", ".join(parts) or "processor validation failed"
        if self.message:
            return f"{detail}: {self.message}"
        return detail


# ---------------------------------------------------------------------------
# Small path / config helpers
# ---------------------------------------------------------------------------


def _fn_name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _module_part(path: str) -> str:
    return path.rsplit(".", 1)[0]


def _stage_id_of(stage_config: Any) -> int | None:
    """Best-effort stage id extraction from an arbitrary config object."""
    if stage_config is None:
        return None
    sid = getattr(stage_config, "stage_id", None)
    if sid is None:
        return None
    try:
        return int(sid)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------


def import_symbol(path: str) -> Any:
    """Import a dotted ``module.attr`` symbol (importlib + ``getattr``).

    This mirrors the two-step shape used by every legacy resolution point:
    ``importlib.import_module(module_path)`` then ``getattr(module, attr)``.
    A missing attribute or module raises the underlying
    ``ImportError``/``AttributeError``; callers that need a best-effort lookup
    (e.g. the worker candidate chain) catch these themselves.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"processor path must be a non-empty dotted string, got {path!r}")
    module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


# ---------------------------------------------------------------------------
# Kind inference (name / suffix driven)
# ---------------------------------------------------------------------------

#: Manual kind overrides registered via :func:`register_processor`.
_KIND_OVERRIDES: dict[str, ProcessorKind] = {}


def register_processor(path: str, kind: ProcessorKind) -> None:
    """Register a manual kind override for a processor *path*.

    This is an escape hatch for models whose function names do not follow the
    suffix convention.  ``infer_kind`` consults these overrides before
    applying the suffix rules.  Overrides are validated eagerly so typos fail
    fast.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"processor path must be a non-empty dotted string, got {path!r}")
    if kind not in _PROCESSOR_KINDS:
        raise ValueError(f"unknown processor kind {kind!r}")
    _KIND_OVERRIDES[path] = kind


def infer_kind(fn: Any, *, path: str) -> ProcessorKind:
    """Infer the processor kind from the *suffix* of ``path``.

    ``fn`` is accepted for API symmetry with :func:`validate_processor` but the
    inference is purely name-driven (suffix-based convention); the callable
    object itself is not inspected here.
    """
    del fn  # inference is name-based; kept for a symmetric API
    if not isinstance(path, str) or not path or "." not in path:
        raise ValueError(f"processor path must be a non-empty dotted string, got {path!r}")

    override = _KIND_OVERRIDES.get(path)
    if override is not None:
        return override

    fn_name = _fn_name(path)

    if _MOSS_MULTI_SOURCE_MODULE_HINT in _module_part(path) and fn_name in _MOSS_MULTI_SOURCE_NAMES:
        return "legacy_multi_source"

    if fn_name.endswith(_TOKEN_ONLY_SUFFIX):
        return "placeholder_prompt_builder"
    if fn_name.endswith(_FULL_PAYLOAD_SUFFIXES):
        return "producer_full_payload"
    if fn_name.endswith(_ASYNC_CHUNK_SUFFIX):
        return "producer_async_chunk"
    for suffix in _DIFFUSION_SUFFIXES:
        if fn_name.endswith(suffix):
            return "diffusion_input_builder"
    return "legacy_orchestrator_builder"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _warn(path: str, kind: ProcessorKind | None, rule: str, message: str) -> None:
    """Emit a soft ``RuntimeWarning`` for a non-fatal structural mismatch."""
    warnings.warn(
        f"stage-input processor {path!r} (kind={kind!r}, rule={rule}): {message}",
        RuntimeWarning,
        stacklevel=3,
    )


def _fail(
    *,
    path: str,
    kind: ProcessorKind | None,
    rule: str,
    stage_config: Any,
    message: str,
) -> None:
    raise ProcessorValidationError(
        stage_id=_stage_id_of(stage_config),
        path=path,
        kind=kind,
        rule=rule,
        message=message,
    )


def _check_orchestrator_first_param(
    params: Mapping[str, inspect.Parameter],
    *,
    path: str,
    kind: ProcessorKind,
    stage_config: Any,
) -> None:
    """First param must be ``source_outputs`` or the signature must carry ``ctx``."""
    names = [
        name
        for name, param in params.items()
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    first = names[0] if names else None
    if first != "source_outputs" and "ctx" not in params:
        _warn(
            path,
            kind,
            "orchestrator_first_param",
            f"expected first parameter 'source_outputs' or a 'ctx' parameter, got first={first!r}",
        )


def _check_suffix(
    path: str,
    kind: ProcessorKind,
    *,
    suffix_pool: tuple[str, ...],
    stage_config: Any,
    what: str,
) -> None:
    fn_name = _fn_name(path)
    if not any(fn_name.endswith(suffix) for suffix in suffix_pool):
        _warn(path, kind, "suffix_kind_mismatch", f"name {fn_name!r} does not match expected {what}")


def _has_var_keyword(params: Mapping[str, inspect.Parameter]) -> bool:
    """Whether the signature accepts arbitrary keywords (``**kwargs``)."""
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def _check_full_payload_is_finished(
    params: Mapping[str, inspect.Parameter],
    *,
    path: str,
    kind: ProcessorKind,
    stage_config: Any,
) -> None:
    """``is_finished`` is optional for the full-payload producer.

    The worker passes it best-effort and retries without it on
    ``TypeError`` (see ``omni_connector_model_runner_mixin._build_custom_process_payload``).
    A ``**kwargs`` producer accepts it through ``VAR_KEYWORD``.
    """
    if _has_var_keyword(params):
        return  # **kwargs accepts is_finished
    if "is_finished" not in params:
        _warn(
            path,
            kind,
            "is_finished_optional",
            "full-payload producer does not declare is_finished; worker retries without it (best-effort)",
        )
        return
    if params["is_finished"].kind == inspect.Parameter.POSITIONAL_ONLY:
        _warn(
            path,
            kind,
            "is_finished_positional_only",
            "full-payload producer declares is_finished positional-only; worker retries without it",
        )


def _check_async_chunk_is_finished(
    params: Mapping[str, inspect.Parameter],
    *,
    path: str,
    kind: ProcessorKind,
    stage_config: Any,
) -> None:
    """``is_finished`` is **required** for the async-chunk producer.

    The scheduler always passes ``is_finished`` (see
    ``chunk_transfer_adapter._send_single_request``), so a processor that cannot
    accept it is a hard configuration error.  A ``**kwargs`` producer accepts it
    through ``VAR_KEYWORD``.
    """
    if _has_var_keyword(params):
        return  # **kwargs accepts is_finished
    if "is_finished" not in params:
        _fail(
            path=path,
            kind=kind,
            rule="is_finished_required",
            stage_config=stage_config,
            message="async-chunk producer must declare 'is_finished' (the scheduler always passes it)",
        )
        return  # unreachable; kept for type-checkers
    is_finished_param = params["is_finished"]
    if is_finished_param.kind == inspect.Parameter.POSITIONAL_ONLY:
        _fail(
            path=path,
            kind=kind,
            rule="is_finished_positional_only",
            stage_config=stage_config,
            message="async-chunk producer declares 'is_finished' positional-only; "
            "it must be keyword-only or have a default",
        )
        return  # unreachable
    if (
        is_finished_param.kind != inspect.Parameter.KEYWORD_ONLY
        and is_finished_param.default is inspect.Parameter.empty
    ):
        _warn(
            path,
            kind,
            "is_finished_default",
            "async-chunk producer declares a required positional-or-keyword 'is_finished'; "
            "prefer keyword-only or a default",
        )


#: Sentinel value for :func:`_check_producer_keyword_bind` probes (the value
#: itself is never inspected or passed to the processor).
_BIND_PLACEHOLDER: Any = object()


def _check_producer_keyword_bind(
    signature: inspect.Signature,
    *,
    path: str,
    kind: ProcessorKind,
    stage_config: Any,
    required: tuple[str, ...],
    what: str,
) -> None:
    """Bind the exact worker keyword call shape to *signature*.

    The worker invokes producers with a fixed keyword call (see
    ``omni_connector_model_runner_mixin._build_custom_process_payload`` for the
    full-payload producer and ``chunk_transfer_adapter._send_single_request``
    for the async-chunk producer).  We probe that call with
    ``inspect.Signature.bind`` so a producer that *declares* the right parameter
    names but cannot actually accept the keyword call — wrong parameter names,
    positional-only payload parameters, or missing required parameters — is
    rejected at configuration time instead of failing later when the worker
    passes ``pooling_output=`` / ``multimodal_output=``.  A ``**kwargs``
    (``VAR_KEYWORD``) producer accepts any keyword and is therefore compatible.
    """
    try:
        signature.bind(**{name: _BIND_PLACEHOLDER for name in required})
    except TypeError as exc:
        _fail(
            path=path,
            kind=kind,
            rule="producer_kwargs",
            stage_config=stage_config,
            message=(f"{what} producer does not accept the worker keyword call ({', '.join(required)}): {exc}"),
        )


def _check_return_annotation(
    signature: inspect.Signature,
    *,
    path: str,
    kind: ProcessorKind,
) -> None:
    """Soft return-annotation check: mismatches warn, never fail."""
    ann = signature.return_annotation
    if ann is inspect.Signature.empty or isinstance(ann, str):
        # Unannotated or postponed-string annotations are not verifiable here.
        return
    if ann in _SCALAR_RETURN_ANNOTATIONS:
        _warn(
            path,
            kind,
            "return_annotation",
            f"return annotation {ann!r} is a scalar, unlikely a stage-input processor result",
        )


def validate_processor(fn: Any, *, kind: ProcessorKind, path: str, stage_config: Any = None) -> None:
    """Structurally validate *fn* against the expected *kind*.

    Only ``inspect.signature`` metadata is examined; the processor body is never
    executed.  Hard contract violations raise :class:`ProcessorValidationError`;
    soft mismatches emit a ``RuntimeWarning``.

    Args:
        fn: The processor callable.
        kind: The expected :class:`ProcessorKind`.
        path: The dotted ``module.attr`` path (used for reporting).
        stage_config: Optional config object; its ``stage_id`` is attached to
            any raised :class:`ProcessorValidationError`.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"processor path must be a non-empty dotted string, got {path!r}")

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        _fail(
            path=path,
            kind=kind,
            rule="inspectable_signature",
            stage_config=stage_config,
            message=f"cannot inspect signature: {exc}",
        )
        return  # unreachable

    params = signature.parameters

    if kind == "placeholder_prompt_builder":
        _check_orchestrator_first_param(params, path=path, kind=kind, stage_config=stage_config)
        _check_suffix(
            path,
            kind,
            suffix_pool=(_TOKEN_ONLY_SUFFIX,),
            stage_config=stage_config,
            what="a *_token_only suffix",
        )

    elif kind == "diffusion_input_builder":
        _check_orchestrator_first_param(params, path=path, kind=kind, stage_config=stage_config)
        _check_suffix(
            path,
            kind,
            suffix_pool=_DIFFUSION_SUFFIXES,
            stage_config=stage_config,
            what="a diffusion-builder suffix (ar2diffusion/ar2dit/thinker2imagegen/...)",
        )

    elif kind == "producer_full_payload":
        # Bind the exact worker keyword call (transfer_manager, pooling_output,
        # request) so a producer that only *declares* the right names but cannot
        # accept the keyword call (wrong names / positional-only payload params)
        # is rejected at config time; **kwargs producers are compatible.
        _check_producer_keyword_bind(
            signature,
            path=path,
            kind=kind,
            stage_config=stage_config,
            required=("transfer_manager", "pooling_output", "request"),
            what="full-payload",
        )
        _check_full_payload_is_finished(params, path=path, kind=kind, stage_config=stage_config)

    elif kind == "producer_async_chunk":
        # ``is_finished`` is required and always passed on the async-chunk path;
        # validate it first so its dedicated rules (is_finished_required /
        # is_finished_positional_only) fire before the generic keyword bind.
        _check_async_chunk_is_finished(params, path=path, kind=kind, stage_config=stage_config)
        _check_producer_keyword_bind(
            signature,
            path=path,
            kind=kind,
            stage_config=stage_config,
            required=("transfer_manager", "multimodal_output", "request", "is_finished"),
            what="async-chunk",
        )

    elif kind == "legacy_multi_source":
        if _MOSS_MULTI_SOURCE_MODULE_HINT not in _module_part(path):
            _fail(
                path=path,
                kind=kind,
                rule="legacy_multi_source_allowlist",
                stage_config=stage_config,
                message=f"legacy multi-source processors are allowlisted to moss_tts.*, got {path!r}",
            )

    elif kind == "legacy_orchestrator_builder":
        # No structural contract: the orchestrator normalizes legacy positional
        # shapes (C0/C2/C3/C4) through wrap_orchestrator_processor, and the
        # first-parameter name is not reliable for legacy builders (configs may
        # legitimately point at C-callables / operator builtins as test doubles).
        # Deliberately no first-param check here.
        pass

    else:  # pragma: no cover - defensive against future literal additions
        raise ValueError(f"unknown processor kind {kind!r}")

    _check_return_annotation(signature, path=path, kind=kind)


# ---------------------------------------------------------------------------
# Resolution entry point
# ---------------------------------------------------------------------------


def resolve_processor(
    path: str,
    *,
    expected_kind: ProcessorKind | None = None,
    stage_config: Any = None,
) -> ProcessorSpec:
    """Resolve a dotted *path* to a validated :class:`ProcessorSpec`.

    Flow: ``import_symbol`` -> ``infer_kind`` -> ``validate_processor`` ->
    (optional) ``expected_kind`` check.  The returned ``spec.fn`` is the same
    callable the legacy ``getattr(importlib.import_module(...), ...)`` lookup
    produced, so callers keep identical runtime behaviour.

    Args:
        path: Dotted ``module.attr`` path of the processor.
        expected_kind: When given, ``infer_kind`` must match; a mismatch raises
            :class:`ProcessorValidationError` (rule ``expected_kind``).
        stage_config: Optional config whose ``stage_id`` is attached to errors.
    """
    fn = import_symbol(path)
    if not callable(fn):
        _fail(
            path=path,
            kind=expected_kind,
            rule="callable",
            stage_config=stage_config,
            message=f"resolved symbol is not callable: {fn!r}",
        )
        raise AssertionError("unreachable")  # pragma: no cover

    kind = infer_kind(fn, path=path)
    validate_processor(fn, kind=kind, path=path, stage_config=stage_config)

    if expected_kind is not None and kind != expected_kind:
        _fail(
            path=path,
            kind=kind,
            rule="expected_kind",
            stage_config=stage_config,
            message=f"inferred kind {kind!r} does not match expected_kind {expected_kind!r}",
        )
        raise AssertionError("unreachable")  # pragma: no cover

    return ProcessorSpec(path=path, kind=kind, fn=fn)


# ---------------------------------------------------------------------------
# Dead-processor hint (gate three-state)
# ---------------------------------------------------------------------------


def dead_processor_hint(
    kind: ProcessorKind,
    *,
    async_chunk: bool,
    downstream_receives_async_chunks: bool,
    has_sync: bool,
    custom_is_sync: bool = False,
) -> bool:
    """Whether an orchestrator-side ``custom_process_input_func`` is never invoked.

    This mirrors the orchestrator gate in ``engine/orchestrator._route_output``
    (``(not self.async_chunk or not self._stage_receives_async_chunks(stage_id + 1))``):

    * ``async_chunk=True`` and the downstream stage **receives** async chunks ->
      the orchestrator skips ``process_engine_inputs`` for that transition, so
      the input processor may be **dead** (returns ``True``).
    * ``async_chunk=True`` and the downstream stage does **not** receive async
      chunks -> the transition still runs forward, so the processor is alive
      (returns ``False``).
    * ``async_chunk=False`` -> forward path is active; the processor is only
      dead when a *different* ``sync_process_input_func`` overrides it
      (``has_sync=True`` and ``custom_is_sync=False``).  When the custom hook
      *is* the selected sync hook (``custom_is_sync=True``) it is the active
      processor and is **not** dead.

    Producer kinds are never orchestrator-side, so they are never "dead" under
    this definition.

    This is a **pure decision helper** for warnings/tests only; the orchestrator
    does not enforce it at runtime (M0 phase is warn-first).
    """
    if kind in _PRODUCER_KINDS:
        return False
    if async_chunk:
        return bool(downstream_receives_async_chunks)
    return bool(has_sync) and not custom_is_sync
