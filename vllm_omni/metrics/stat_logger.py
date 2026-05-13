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
from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import PrometheusStatLogger

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


def _engine_to_stage_replica(engine_value) -> tuple[str, str]:
    """Look up (stage, replica) for an engine_idx, accepting int or str input.

    Upstream emits engine values in two flavors:
    - int form, e.g. ``gauge_engine_sleep_state.labels(engine=idx, ...)`` (loggers.py:510)
    - str form, e.g. ``info_gauge.labels(**metrics_info)`` where ``metrics_info["engine"] = str(idx)`` (loggers.py:1055)

    Raises ``KeyError`` when the value is missing from the map — fail-fast is
    preferable to silently emitting series under a wrong (stage, replica).
    """
    key = int(engine_value) if isinstance(engine_value, str) else engine_value
    return _ENGINE_INDEX_MAP[key]


class _RelabelMixin:
    """Mixin: rewrite ``labelnames`` at family creation and ``.labels()`` calls.

    Handles all four upstream forms encountered in
    ``vllm.v1.metrics.loggers.PrometheusStatLogger``:

    1. ``.labels(engine=idx, ...)`` kwarg with int engine (loggers.py:510)
    2. ``.labels(model_name, str(idx), source)`` positional with str engine
       (loggers.py:646, 679)
    3. ``.labels(**metrics_info)`` kwarg with str engine (loggers.py:1056)
    4. Families without an ``engine`` label — passthrough (e.g. lora_info)

    Drops into upstream's ``_gauge_cls`` / ``_counter_cls`` / ``_histogram_cls``
    class slots.
    """

    def __init__(self, *args, **kwargs):
        # Remember where `engine` sat in the original labelnames so positional
        # `.labels()` calls can splice (stage, replica) at the right offset.
        labelnames = kwargs.get("labelnames")
        if labelnames is not None:
            original = list(labelnames)
            self._engine_label_index = (
                original.index("engine") if "engine" in original else -1
            )
            kwargs["labelnames"] = _rewrite_labelnames(labelnames)
        else:
            self._engine_label_index = -1
        super().__init__(*args, **kwargs)

    def labels(self, *args, **kwargs):
        if self._engine_label_index >= 0:
            if args:
                # Positional form: replace args[engine_idx] with (stage, replica).
                idx = self._engine_label_index
                if idx < len(args):
                    stage, replica = _engine_to_stage_replica(args[idx])
                    args = (*args[:idx], stage, replica, *args[idx + 1 :])
            elif "engine" in kwargs:
                stage, replica = _engine_to_stage_replica(kwargs.pop("engine"))
                kwargs["stage"] = stage
                kwargs["replica"] = replica
        return super().labels(*args, **kwargs)


class _RelabelGauge(_RelabelMixin, Gauge):
    pass


class _RelabelCounter(_RelabelMixin, Counter):
    pass


class _RelabelHistogram(_RelabelMixin, Histogram):
    pass


class OmniPrometheusStatLogger(PrometheusStatLogger):
    """Wrap upstream PrometheusStatLogger to expose per-(stage, replica) labels.

    Replaces the upstream single ``engine`` label with two labels ``stage`` and
    ``replica`` so that the ~37 ``vllm:*`` metric families gain per-replica
    visibility for multi-replica deployments. See RFC §3.2.7.

    The orchestrator builds ``stage_replica_map`` from the static stage_pools
    config; flat engine_idx values map 1:1 to (stage_name, replica_id) tuples.
    Dynamic add/remove of replicas at runtime is intentionally not supported
    in this iteration — see RFC §3.4 risks.
    """

    # Inject our wrapper metric classes into upstream's class-level slots so
    # every ~37 family is created with `engine` rewritten to `stage`+`replica`.
    _gauge_cls = _RelabelGauge
    _counter_cls = _RelabelCounter
    _histogram_cls = _RelabelHistogram

    def __init__(
        self,
        vllm_config: VllmConfig,
        stage_replica_map: dict[int, tuple[str, str]],
    ) -> None:
        self._stage_replica_map = stage_replica_map
        # Populate the process-level translation table that wrapper metric
        # classes consult on every `.labels()` call. Cleared first so a
        # second OmniPrometheusStatLogger in the same process (e.g. tests,
        # orchestrator restart) starts from a clean slate.
        _ENGINE_INDEX_MAP.clear()
        _ENGINE_INDEX_MAP.update(stage_replica_map)
        super().__init__(
            vllm_config=vllm_config,
            engine_indexes=list(stage_replica_map.keys()),
        )

    @property
    def stage_replica_map(self) -> dict[int, tuple[str, str]]:
        return self._stage_replica_map

    @property
    def per_engine_labelvalues(self) -> dict[int, list[object]]:
        return self._omni_per_engine_labelvalues

    @per_engine_labelvalues.setter
    def per_engine_labelvalues(self, value: dict[int, list[object]]) -> None:
        # Upstream sets {idx: [model_name, str(idx)]} (loggers.py:433); we drop
        # the engine str and append (stage, replica) so labelvalues match the
        # 3-element labelnames our wrapper classes produce.
        rewritten: dict[int, list[object]] = {}
        for idx, vals in value.items():
            model_name = vals[0]
            stage, replica = self._stage_replica_map[idx]
            rewritten[idx] = [model_name, stage, replica]
        self._omni_per_engine_labelvalues = rewritten
