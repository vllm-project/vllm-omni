"""OmniPrometheusStatLogger — wrap upstream PrometheusStatLogger.

Goal (RFC §3.2.7): rewrite the upstream `engine` single-label scheme into a
`stage` + `replica` two-label scheme so that the ~37 `vllm:*` metric families
automatically gain per-(stage, replica) visibility for multi-replica deployments.

Phase 2.1 ships only the three wrapper metric classes + the process-level
engine→(stage, replica) map. The OmniPrometheusStatLogger subclass that wires
everything together lands in Phase 2.2.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Process-wide translation table written by OmniPrometheusStatLogger at init.
# Keys are flat engine_idx values (as upstream PrometheusStatLogger sees them);
# values are the (stage_name, replica_id_str) tuple we expose as labels.
#
# Module-level rather than per-instance because the wrapper metric classes are
# constructed by upstream's __init__ and never get a back-reference to the
# StatLogger that owns them. vLLM runs a single Orchestrator/StatLogger per
# process, so a module global is safe; tests isolate by .clear()ing first.
_ENGINE_INDEX_MAP: dict[int, tuple[str, str]] = {}


def _rewrite_labelnames(labelnames):
    """Replace `engine` in ``labelnames`` with (`stage`, `replica`) in place.

    Preserves ordering (so ``["model_name", "engine", "reason"]`` becomes
    ``["model_name", "stage", "replica", "reason"]``) and the original
    container type (list vs tuple).
    """
    if labelnames is None:
        return labelnames
    seq = list(labelnames)
    if "engine" not in seq:
        return labelnames
    out: list[str] = []
    for name in seq:
        if name == "engine":
            out.extend(("stage", "replica"))
        else:
            out.append(name)
    return type(labelnames)(out) if not isinstance(labelnames, list) else out


def _rewrite_label_kwargs(kwargs: dict) -> dict:
    """Translate ``.labels(engine=idx, ...)`` kwargs into ``stage``/``replica``.

    Raises ``KeyError`` when ``engine_idx`` is missing from the map — fail-fast
    is preferable to silently emitting series under a wrong (stage, replica).
    """
    if "engine" not in kwargs:
        return kwargs
    engine_idx = kwargs.pop("engine")
    stage, replica = _ENGINE_INDEX_MAP[engine_idx]
    kwargs["stage"] = stage
    kwargs["replica"] = replica
    return kwargs


class _RelabelMixin:
    """Mixin: rewrite ``labelnames`` at family creation and ``.labels()`` kwargs.

    Used to derive ``_RelabelGauge`` / ``_RelabelCounter`` / ``_RelabelHistogram``
    that drop into upstream ``PrometheusStatLogger._gauge_cls`` / ``_counter_cls`` /
    ``_histogram_cls`` slots.
    """

    def __init__(self, *args, **kwargs):
        if "labelnames" in kwargs:
            kwargs["labelnames"] = _rewrite_labelnames(kwargs["labelnames"])
        super().__init__(*args, **kwargs)

    def labels(self, *args, **kwargs):
        if kwargs:
            kwargs = _rewrite_label_kwargs(kwargs)
        return super().labels(*args, **kwargs)


class _RelabelGauge(_RelabelMixin, Gauge):
    pass


class _RelabelCounter(_RelabelMixin, Counter):
    pass


class _RelabelHistogram(_RelabelMixin, Histogram):
    pass
