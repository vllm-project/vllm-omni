"""OmniPrometheusStatLogger — wrap upstream PrometheusStatLogger.

Rewrites the upstream ``engine`` single-label scheme into a ``stage`` +
``replica`` two-label scheme so the ~37 ``vllm:*`` metric families gain
per-(stage, replica) visibility for multi-replica deployments.

Contents:
- ``_ENGINE_INDEX_MAP``: process-wide engine_idx → (stage_name, replica_id)
  lookup, populated by ``OmniPrometheusStatLogger.__init__``.
- ``_RelabelMixin``: rewrites ``labelnames`` at family creation and translates
  ``.labels()`` calls; applied via ``_RelabelGauge`` / ``_RelabelCounter`` /
  ``_RelabelHistogram``.
- ``_OmniPerfMetricsProm`` / ``_OmniSpecDecodingProm`` / ``_OmniKVConnectorProm``:
  helper-class wraps so the upstream sub-collectors construct their internal
  families through the relabel mixin too.
- ``OmniPrometheusStatLogger``: the subclass that wires everything together
  and rewrites ``per_engine_labelvalues`` from 2-tuple to 3-tuple at setter
  time.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorProm
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.perf import PerfMetricsProm
from vllm.v1.spec_decode.metrics import SpecDecodingProm

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
            self._engine_label_index = original.index("engine") if "engine" in original else -1
            kwargs["labelnames"] = _rewrite_labelnames(labelnames)
        else:
            self._engine_label_index = -1
        super().__init__(*args, **kwargs)

    def labels(self, *args, **kwargs):
        if self._engine_label_index >= 0:
            if args:
                # Positional form. There are TWO upstream patterns:
                #
                # (a) Pre-rewritten path: create_metric_per_engine fans
                #     `per_engine_labelvalues` (already a 3-tuple
                #     [model_name, stage, replica] thanks to the property-
                #     descriptor setter on OmniPrometheusStatLogger) into
                #     `metric.labels(*values)`. len(args) matches the
                #     rewritten label set already, so just pass through.
                #
                # (b) Legacy 2-tuple path: upstream sites like
                #     `counter_request_success.labels(model_name, str(idx),
                #     str(reason))` pass values shaped to the *original*
                #     labelnames (engine still present at idx). Here
                #     len(args) is short by 1 — splice (stage, replica)
                #     in place of the engine value at engine_label_index.
                if len(args) == len(self._labelnames):
                    return super().labels(*args, **kwargs)
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


# ----------------------------------------------------------------------------
# Helper-class wraps for the three sub-metric collectors that upstream
# PrometheusStatLogger constructs in its __init__ (loggers.py:438-446):
#
#     self.spec_decoding_prom = self._spec_decoding_cls(...)
#     self.kv_connector_prom = self._kv_connector_cls(...)
#     self.perf_metrics_prom = self._perf_metrics_cls(...)
#
# Each helper receives raw `labelnames` as a constructor argument and uses
# its own class-level `_counter_cls` / `_gauge_cls` / `_histogram_cls` slots
# to build internal Counter/Gauge/Histogram families. The slot overrides on
# OmniPrometheusStatLogger only reach families created via *its* slots, so
# the helpers would otherwise still construct 2-label families and then hit
# `Incorrect label count` when create_metric_per_engine feeds the rewritten
# 3-element per_engine_labelvalues. Subclassing each helper and overriding
# its slots routes the relabel mixin through to the helper-internal families
# too. The helper kept seeing the OLD 2-element labelnames param, but that
# is fine because the wrapper rewrites it at family-creation time.
# ----------------------------------------------------------------------------


class _OmniPerfMetricsProm(PerfMetricsProm):
    _counter_cls = _RelabelCounter


class _OmniSpecDecodingProm(SpecDecodingProm):
    _counter_cls = _RelabelCounter


class _OmniKVConnectorProm(KVConnectorProm):
    _gauge_cls = _RelabelGauge
    _counter_cls = _RelabelCounter
    _histogram_cls = _RelabelHistogram


class OmniPrometheusStatLogger(PrometheusStatLogger):
    """Wrap upstream PrometheusStatLogger to expose per-(stage, replica) labels.

    Replaces the upstream single ``engine`` label with two labels ``stage`` and
    ``replica`` so the ~37 ``vllm:*`` metric families gain per-replica
    visibility for multi-replica deployments.

    The orchestrator builds ``stage_replica_map`` from the static stage_pools
    config; flat engine_idx values map 1:1 to (stage_name, replica_id) tuples.
    Dynamic add/remove of replicas at runtime is intentionally not supported —
    the map is built once at construction and never mutated afterward.
    """

    # Inject our wrapper metric classes into upstream's class-level slots so
    # every ~37 family is created with `engine` rewritten to `stage`+`replica`.
    _gauge_cls = _RelabelGauge
    _counter_cls = _RelabelCounter
    _histogram_cls = _RelabelHistogram
    # Inject helper-class wraps too so the perf / spec-decoding / kv-connector
    # sub-collectors get the same labelname rewrite and don't crash with
    # `Incorrect label count` when create_metric_per_engine fans out the
    # rewritten 3-element per_engine_labelvalues over their internal families.
    _perf_metrics_cls = _OmniPerfMetricsProm
    _spec_decoding_cls = _OmniSpecDecodingProm
    _kv_connector_cls = _OmniKVConnectorProm

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
