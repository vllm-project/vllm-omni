# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse-audio output protocol: marker classification and payload routing.

A step's per-request `multimodal_outputs` lists may cover only a SUBSET of
the batch (requests that produced audio this step). The producer declares
this with two metadata keys (nested ``meta`` dict or flattened ``meta.*``):

* ``meta.req_id`` — the request ids that DO have output, in list order;
* ``meta.sparse_audio`` — the marker saying "per-request lists are aligned
  to ``meta.req_id`` order, not batch order".

The marker's carrier form is ENFORCED, not normalized. The design contract
is a list of strings (e.g. ``["1"]`` — what every in-tree producer emits); a
bare string is also accepted. Interpreting deviant carriers would need a
second truth predicate, and predicate divergence is exactly how
``np.array(["0"])`` came to route as sparse while ``["0"]`` routed as dense;
per the refactor series' Policy A the runner<->model interface owes no
tolerance shim.

A present-but-invalid declaration (illegal marker carrier, or marker set
with an unusable ``meta.req_id``) FAILS CLOSED: it most likely belongs to a
producer that INTENDED sparse output, and falling back to dense would
batch-index-assign its payloads to the WRONG requests. `InvalidSparseDeclarationError`
is raised by the classifiers and caught INSIDE this module by
`resolve_sparse_mm_routing`, which routes no payload for the affected step —
the exception never crosses the module boundary, so it can never escape a
v1 runner (which does not catch per-request exceptions).

This module is the single home of the contract until the model-state
refactor series (G9-C2.6) turns the marker into a typed bool resolved at
one adapter site and unifies the nested/flattened encodings. The NPU runner
still carries a tolerant pre-enforcement copy — NPU-track handoff item.
"""

import threading
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# The DESIGNED marker carrier is a list of strings (bare str also accepted);
# truthiness of a string item:
_SPARSE_MARKER_TRUTHY = ("1", "true", "yes", "on")

# Per-process dedup so a broken producer logs once per offending shape, not
# once per step:
_logged_offenders: set[str] = set()
_error_state_lock = threading.Lock()

# A persistent invalid producer must remain observable without flooding the
# per-step output path. Classifier errors above stay deduplicated per carrier
# type; the routing consequence is reported at most once per interval, with a
# count of repeats suppressed since the previous report.
_FAIL_CLOSED_LOG_INTERVAL_S = 60.0
_fail_closed_last_log_at: float | None = None
_fail_closed_suppressed = 0


class InvalidSparseDeclarationError(ValueError):
    """Sparse-audio metadata is PRESENT but out of contract.

    Internal control signal: raised by `is_sparse_audio_marker` /
    `resolve_sparse_mm_req_ids` and caught by `resolve_sparse_mm_routing`
    in this same module, which fails closed. Callers outside this module
    only ever see the routing result, never the exception.
    """


def _error_once(key: str, msg: str, *args: Any) -> None:
    with _error_state_lock:
        if key in _logged_offenders:
            return
        _logged_offenders.add(key)
    logger.error(msg, *args)


def _error_fail_closed_rate_limited(cause: str) -> None:
    global _fail_closed_last_log_at, _fail_closed_suppressed

    now = time.monotonic()
    with _error_state_lock:
        if _fail_closed_last_log_at is not None and now - _fail_closed_last_log_at < _FAIL_CLOSED_LOG_INTERVAL_S:
            _fail_closed_suppressed += 1
            return
        suppressed = _fail_closed_suppressed
        _fail_closed_last_log_at = now
        _fail_closed_suppressed = 0

    logger.error(
        "Invalid sparse-audio declaration: failing closed - no multimodal payload is routed "
        "for the affected step(s). Fix the emitting model. Suppressed repeats before this "
        "report: %d. Cause: %s",
        suppressed,
        cause,
    )


def is_sparse_audio_marker(value: Any) -> bool:
    """Classify a ``meta.sparse_audio`` marker.

    Legal carriers (list of strings, bare string) evaluate against the
    literal truthy set; ``None`` means "marker absent" and returns ``False``.
    An ILLEGAL carrier (tensors, arrays, numbers, lists with non-string
    items) is a producer bug: logged at error level once per offending type,
    then raised as `InvalidSparseDeclarationError`. Deviant carriers are never
    read element-wise, so a device tensor can never introduce a D2H sync
    here.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in _SPARSE_MARKER_TRUTHY
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return any(item.lower() in _SPARSE_MARKER_TRUTHY for item in value)
    vtype = type(value).__name__
    _error_once(
        vtype,
        'Illegal sparse-audio marker of type %s: the marker MUST be a list of strings (e.g. ["1"]). '
        "The emitting model is out of contract; routing fails closed for the affected step(s).",
        vtype,
    )
    raise InvalidSparseDeclarationError(f"illegal sparse-audio marker carrier: {vtype}")


def resolve_sparse_mm_req_ids(multimodal_outputs: Any) -> list[str] | None:
    """Resolve the sparse request-id list.

    Returns the request-id list when a legal sparse declaration is present,
    and ``None`` when there is no sparse declaration (dense). Raises
    `InvalidSparseDeclarationError` when the declaration is present but out of
    contract (illegal marker carrier, or marker set with an unusable
    ``meta.req_id``).
    """
    if not isinstance(multimodal_outputs, dict):
        return None
    meta = multimodal_outputs.get("meta")
    nested_req_ids = meta.get("req_id") if isinstance(meta, dict) else None
    flat_req_ids = multimodal_outputs.get("meta.req_id")

    # Resolve both marker encodings independently, even when the request ids
    # came from the other encoding. Previously a nested `req_id` prevented the
    # flattened marker from being read, while the mirror combination worked.
    # Evaluating both also preserves the strict contract: an illegal carrier
    # on either side wins over an otherwise-valid declaration and fails closed.
    nested_marker_value = meta.get("sparse_audio") if isinstance(meta, dict) else None
    flat_marker_value = multimodal_outputs.get("meta.sparse_audio")
    nested_marker = is_sparse_audio_marker(nested_marker_value)
    flat_marker = is_sparse_audio_marker(flat_marker_value)
    if nested_marker_value is not None and flat_marker_value is not None and nested_marker != flat_marker:
        _error_once(
            "marker:conflict",
            "Nested and flattened sparse-audio markers disagree. Failing closed.",
        )
        raise InvalidSparseDeclarationError("nested and flattened sparse markers disagree")
    sparse_audio = nested_marker or flat_marker
    if not sparse_audio:
        return None

    req_id_carriers = [
        (name, value) for name, value in (("nested", nested_req_ids), ("flattened", flat_req_ids)) if value is not None
    ]
    if not req_id_carriers:
        req_id_carriers = [("missing", None)]

    validated_req_ids: list[list[str]] = []
    for carrier_name, req_ids in req_id_carriers:
        if isinstance(req_ids, list) and all(isinstance(rid, str) for rid in req_ids):
            validated_req_ids.append(req_ids)
            continue
        rtype = type(req_ids).__name__
        _error_once(
            f"req_id:{carrier_name}:{rtype}",
            "Sparse-audio marker is set but the %s meta.req_id carrier is unusable (%s; "
            "must be a list of request-id strings). Failing closed.",
            carrier_name,
            rtype,
        )
        raise InvalidSparseDeclarationError(f"sparse marker set but {carrier_name} meta.req_id is unusable: {rtype}")

    req_ids = validated_req_ids[0]
    if len(validated_req_ids) == 2 and validated_req_ids[1] != req_ids:
        _error_once(
            "req_id:conflict",
            "Nested and flattened sparse-audio request-id lists disagree. Failing closed.",
        )
        raise InvalidSparseDeclarationError("nested and flattened sparse request-id lists disagree")
    if len(req_ids) != len(set(req_ids)):
        _error_once(
            "req_id:duplicate",
            "Sparse-audio request-id list contains duplicates. Failing closed.",
        )
        raise InvalidSparseDeclarationError("sparse request-id list contains duplicates")
    return list(req_ids)


def resolve_sparse_mm_routing(
    *,
    engine_output_type: str,
    req_ids_output_copy: list[str],
    downstream_req_ids: list[str],
    multimodal_outputs: Any,
) -> tuple[list[str], dict[str, int], bool]:
    """Resolve `(downstream_req_ids, sparse_mm_index, audio_sparse_output)`.

    The only entry point runners use; all protocol errors are absorbed here.
    """
    try:
        sparse_mm_req_ids = resolve_sparse_mm_req_ids(multimodal_outputs)
    except InvalidSparseDeclarationError as exc:
        if engine_output_type != "audio":
            # Non-audio pipelines never consult sparse routing; the carrier
            # error is already logged by the classifier.
            return downstream_req_ids, {}, False
        # FAIL CLOSED: a present-but-invalid sparse declaration most likely
        # belongs to a producer that INTENDED sparse output; routing it as
        # dense would batch-index-assign its payloads to the WRONG requests.
        # Route no payload for this step instead - a visible, bounded gap
        # rather than silent cross-request corruption. (Empty-sparse is an
        # already-exercised path: the zero-audio sparse step ships
        # req_id=[] through the same shape.)
        _error_fail_closed_rate_limited(str(exc))
        return [], {}, True
    sparse_mm_index = {rid: i for i, rid in enumerate(sparse_mm_req_ids or [])}
    if engine_output_type != "audio" or sparse_mm_req_ids is None:
        return downstream_req_ids, sparse_mm_index, False

    sparse_req_id_set = set(sparse_mm_req_ids)
    sparse_downstream_req_ids = [rid for rid in req_ids_output_copy if rid in sparse_req_id_set]
    return sparse_downstream_req_ids, sparse_mm_index, True
