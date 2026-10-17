# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage-input processor registry: signature-level structural validation.

Serves the three importlib resolution points (orchestrator-side
``custom_process_input_func``/``sync_process_input_func``, producer-side
``*_full_payload``, producer-side ``*_async_chunk``).  Checks are purely
signature-based (``inspect.signature``); behaviour is covered by the golden /
parity tests.

Producer kwargs are contract: full-payload keeps ``pooling_output``
(``is_finished`` optional — the worker retries without it); async-chunk keeps
``multimodal_output`` and **must** accept ``is_finished`` (the scheduler always
passes it).

Kind inference is name-driven (suffix-based); ``resolve_processor`` returns a
:class:`ProcessorSpec` whose ``fn`` matches the legacy importlib lookup.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, get_args

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
_PROCESSOR_KINDS: frozenset[str] = frozenset(get_args(ProcessorKind))

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
    """Resolved and validated stage-input processor (path, kind, fn)."""

    path: str
    kind: ProcessorKind
    fn: Any


class ProcessorValidationError(Exception):
    """A stage-input processor failed structural validation."""

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
        detail = (
            ", ".join(
                f"{name}={value}"
                for name, value in (
                    ("stage_id", self.stage_id),
                    ("path", repr(self.path) if self.path else None),
                    ("kind", repr(self.kind) if self.kind else None),
                    ("rule", self.rule),
                )
                if value is not None
            )
            or "processor validation failed"
        )
        return f"{detail}: {self.message}" if self.message else detail


# ---------------------------------------------------------------------------
# Small path / config helpers
# ---------------------------------------------------------------------------


def _fn_name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _module_part(path: str) -> str:
    return path.rsplit(".", 1)[0]


def _stage_id_of(stage_config: Any) -> int | None:
    """Best-effort stage id extraction from an arbitrary config object."""
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
    """Import a dotted ``module.attr`` symbol; missing module/attr propagate."""
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
    """Register a manual kind override (escape hatch for non-suffix names)."""
    if not isinstance(path, str) or not path:
        raise ValueError(f"processor path must be a non-empty dotted string, got {path!r}")
    if kind not in _PROCESSOR_KINDS:
        raise ValueError(f"unknown processor kind {kind!r}")
    _KIND_OVERRIDES[path] = kind


def infer_kind(fn: Any, *, path: str) -> ProcessorKind:
    """Infer the processor kind from the *suffix* of ``path`` (name-driven)."""
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
    warnings.warn(
        f"stage-input processor {path!r} (kind={kind!r}, rule={rule}): {message}", RuntimeWarning, stacklevel=3
    )


def _fail(
    *,
    path: str,
    kind: ProcessorKind | None,
    rule: str,
    stage_config: Any,
    message: str,
) -> NoReturn:
    raise ProcessorValidationError(
        stage_id=_stage_id_of(stage_config),
        path=path,
        kind=kind,
        rule=rule,
        message=message,
    )


def _check_orchestrator_first_param(
    signature: inspect.Signature,
    *,
    path: str,
    kind: ProcessorKind,
    stage_config: Any,
) -> None:
    """First param must be ``source_outputs`` or the signature must carry ``ctx``."""
    params = signature.parameters
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


def _has_var_keyword(params: Mapping[str, inspect.Parameter]) -> bool:
    """Whether the signature accepts arbitrary keywords (``**kwargs``)."""
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def _check_is_finished(
    params: Mapping[str, inspect.Parameter],
    *,
    path: str,
    kind: ProcessorKind,
    stage_config: Any,
    required: bool,
) -> None:
    """Validate the ``is_finished`` kwarg contract (hard only for async-chunk)."""
    if _has_var_keyword(params):
        return  # **kwargs accepts is_finished
    if "is_finished" not in params:
        if required:
            _fail(
                path=path,
                kind=kind,
                rule="is_finished_required",
                stage_config=stage_config,
                message="async-chunk producer must declare 'is_finished' (the scheduler always passes it)",
            )
        _warn(
            path,
            kind,
            "is_finished_optional",
            "full-payload producer does not declare is_finished; worker retries without it (best-effort)",
        )
        return
    is_finished_param = params["is_finished"]
    if is_finished_param.kind == inspect.Parameter.POSITIONAL_ONLY:
        if required:
            _fail(
                path=path,
                kind=kind,
                rule="is_finished_positional_only",
                stage_config=stage_config,
                message="async-chunk producer declares 'is_finished' positional-only; "
                "it must be keyword-only or have a default",
            )
        _warn(
            path,
            kind,
            "is_finished_positional_only",
            "full-payload producer declares is_finished positional-only; worker retries without it",
        )
        return
    if required and (
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
    """Bind the worker keyword call to *signature* via ``inspect.Signature.bind``.

    Producers that declare the right names but reject the keyword call shape
    fail at configuration time instead of when the worker passes
    ``pooling_output=`` / ``multimodal_output=``.
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


def _check_full_payload(
    signature: inspect.Signature,
    *,
    path: str,
    kind: ProcessorKind,
    stage_config: Any,
) -> None:
    """Full-payload producer: keyword bind, then soft is_finished."""
    params = signature.parameters
    required: tuple[str, ...] = ("transfer_manager", "pooling_output", "request")
    is_finished_param = params.get("is_finished")
    if is_finished_param is not None and is_finished_param.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        required = required + ("is_finished",)
    _check_producer_keyword_bind(
        signature,
        path=path,
        kind=kind,
        stage_config=stage_config,
        required=required,
        what="full-payload",
    )
    _check_is_finished(params, path=path, kind=kind, stage_config=stage_config, required=False)


def _check_async_chunk(
    signature: inspect.Signature,
    *,
    path: str,
    kind: ProcessorKind,
    stage_config: Any,
) -> None:
    """Async-chunk producer: hard is_finished, then keyword bind."""
    params = signature.parameters
    _check_is_finished(params, path=path, kind=kind, stage_config=stage_config, required=True)
    _check_producer_keyword_bind(
        signature,
        path=path,
        kind=kind,
        stage_config=stage_config,
        required=("transfer_manager", "multimodal_output", "request", "is_finished"),
        what="async-chunk",
    )


def _check_legacy_multi_source(
    _signature: inspect.Signature,
    *,
    path: str,
    kind: ProcessorKind,
    stage_config: Any,
) -> None:
    """Legacy multi-source processors are allowlisted to ``moss_tts.*``."""
    if _MOSS_MULTI_SOURCE_MODULE_HINT not in _module_part(path):
        _fail(
            path=path,
            kind=kind,
            rule="legacy_multi_source_allowlist",
            stage_config=stage_config,
            message=f"legacy multi-source processors are allowlisted to moss_tts.*, got {path!r}",
        )


#: Per-kind structural checks run in order by :func:`validate_processor`.
#: ``legacy_orchestrator_builder`` is deliberately check-free: the orchestrator
#: normalizes legacy shapes through ``wrap_orchestrator_processor``.
_VALIDATORS: dict[ProcessorKind, tuple[Callable[..., None], ...]] = {
    "placeholder_prompt_builder": (_check_orchestrator_first_param,),
    "diffusion_input_builder": (_check_orchestrator_first_param,),
    "producer_full_payload": (_check_full_payload,),
    "producer_async_chunk": (_check_async_chunk,),
    "legacy_multi_source": (_check_legacy_multi_source,),
    "legacy_orchestrator_builder": (),
}


def validate_processor(fn: Any, *, kind: ProcessorKind, path: str, stage_config: Any = None) -> None:
    """Structurally validate *fn* against the expected *kind* (signature only).

    Hard violations raise; soft mismatches warn; checks come from ``_VALIDATORS``.
    """
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

    validators = _VALIDATORS.get(kind)
    if validators is None:  # pragma: no cover - defensive against future literal additions
        raise ValueError(f"unknown processor kind {kind!r}")
    for check in validators:
        check(signature, path=path, kind=kind, stage_config=stage_config)

    # Soft return-annotation check: mismatches warn, never fail.
    ann = signature.return_annotation
    if ann is not inspect.Signature.empty and not isinstance(ann, str) and ann in _SCALAR_RETURN_ANNOTATIONS:
        _warn(
            path,
            kind,
            "return_annotation",
            f"return annotation {ann!r} is a scalar, unlikely a stage-input processor result",
        )


# ---------------------------------------------------------------------------
# Resolution entry point
# ---------------------------------------------------------------------------


def resolve_processor(
    path: str,
    *,
    expected_kind: ProcessorKind | None = None,
    stage_config: Any = None,
) -> ProcessorSpec:
    """Resolve *path* to a validated :class:`ProcessorSpec`.

    Flow: ``import_symbol`` -> ``infer_kind`` -> ``validate_processor`` ->
    optional ``expected_kind`` check; ``spec.fn`` matches the legacy lookup.
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
    """Whether an orchestrator-side custom hook is never invoked (gate three-state).

    Mirrors ``engine/orchestrator._route_output``: async-chunk transitions skip
    forward processing when the downstream stage receives chunks; otherwise a
    differing ``sync_process_input_func`` overrides the custom hook.  Pure
    decision helper (warn-first; the orchestrator does not enforce it).
    """
    if kind in _PRODUCER_KINDS:
        return False
    if async_chunk:
        return bool(downstream_receives_async_chunks)
    return bool(has_sync) and not custom_is_sync
