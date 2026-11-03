# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import multiprocessing
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest, multiprocess
from vllm.config import ModelConfig, ObservabilityConfig, SpeculativeConfig, VllmConfig
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.stat_logger import (
    _ENGINE_INDEX_MAP,
    OmniPrometheusStatLogger,
    _engine_to_stage_replica,
    _OmniKVConnectorProm,
    _OmniPerfMetricsProm,
    _OmniSpecDecodingProm,
    _RelabelCounter,
    _RelabelGauge,
    _RelabelHistogram,
    _rewrite_labelnames,
)
from vllm_omni.outputs import StagePostWarmupMemoryStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _publish_memory_snapshots(connection):
    # Spawn imports Prometheus after inheriting PROMETHEUS_MULTIPROC_DIR,
    # matching API frontend startup. Only publication time is controlled.
    logger = OmniPrometheusStatLogger.__new__(OmniPrometheusStatLogger)
    logger._omni_per_engine_labelvalues = {index: ["m", "1", str(index)] for index in (0, 1)}
    logger._init_stage_post_warmup_memory_families()
    while (command := connection.recv()) is not None:
        engine_idx, allocated, reserved, timestamp = command
        with patch("prometheus_client.metrics.time.time", return_value=timestamp):
            logger.record_stage_post_warmup_memory(
                StagePostWarmupMemoryStats(allocated_bytes=allocated, reserved_bytes=reserved),
                engine_idx=engine_idx,
            )
        connection.send("recorded")
    connection.close()


def test_memory_snapshots_aggregate_shared_frontends(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    context = multiprocessing.get_context("spawn")
    frontends = []
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(tmp_path))

    def start_frontend():
        parent, child = context.Pipe()
        process = context.Process(target=_publish_memory_snapshots, args=(child,))
        process.start()
        child.close()
        frontends.append((process, parent))
        return process, parent

    def publish(frontend, replica, allocated, reserved, timestamp):
        process, connection = frontend
        connection.send((replica, allocated, reserved, timestamp))
        assert connection.poll(45), f"Snapshot writer did not reply (exitcode={process.exitcode})"
        assert connection.recv() == "recorded"

    def assert_snapshots(expected, families=None):
        names = (defs.STAGE_POST_WARMUP_MEMORY_ALLOCATED_BYTES, defs.STAGE_POST_WARMUP_MEMORY_RESERVED_BYTES)
        families = registry.collect() if families is None else families
        samples = [sample for family in families for sample in family.samples if sample.name in names]
        assert len(samples) == 2 * len(expected)
        for sample in samples:
            replica = int(sample.labels["replica"])
            assert sample.labels == {"model_name": "m", "stage": "1", "replica": str(replica)}
            assert sample.value == expected[replica][names.index(sample.name)]

    try:
        first, second = start_frontend(), start_frontend()
        assert first[0].pid != second[0].pid
        publish(first, 0, 101, 103, 1)
        publish(second, 0, 101, 103, 2)
        assert_snapshots({0: (101, 103)})

        # The latest observation may be smaller; neither sum nor max is valid.
        publish(second, 0, 11, 13, 3)
        publish(first, 1, 17, 19, 4)
        assert_snapshots({0: (11, 13), 1: (17, 19)})

        # Collection order must not replace a newer observation with an older one.
        publish(first, 0, 101, 103, 2)
        files = sorted(str(path) for path in tmp_path.glob("gauge_*.db"))
        for order in (files, files[::-1]):
            assert_snapshots(
                {0: (11, 13), 1: (17, 19)},
                multiprocess.MultiProcessCollector.merge(order),
            )

        # API process exit does not retire the shared GPU worker's snapshot.
        second[1].send(None)
        second[0].join(timeout=10)
        assert second[0].exitcode == 0
        multiprocess.mark_process_dead(second[0].pid, path=str(tmp_path))
        assert_snapshots({0: (11, 13), 1: (17, 19)})

        replacement = start_frontend()
        publish(replacement, 0, 23, 29, 5)
        assert_snapshots({0: (23, 29), 1: (17, 19)})
    finally:
        for process, connection in frontends:
            if process.is_alive():
                with suppress(BrokenPipeError):
                    connection.send(None)
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
            connection.close()
            process.close()


@pytest.fixture(autouse=True)
def _isolate_engine_map():
    """Each test gets a clean _ENGINE_INDEX_MAP."""
    _ENGINE_INDEX_MAP.clear()
    yield
    _ENGINE_INDEX_MAP.clear()


@pytest.fixture
def registry() -> CollectorRegistry:
    return CollectorRegistry()


# ---------------------------------------------------------------------------
# _rewrite_labelnames
# ---------------------------------------------------------------------------


class TestRewriteLabelnames:
    def test_engine_at_end(self):
        assert _rewrite_labelnames(["model_name", "engine"]) == [
            "model_name",
            "stage",
            "replica",
        ]

    def test_engine_in_middle(self):
        # Upstream uses `labelnames + ["reason"]` etc., putting engine in middle.
        assert _rewrite_labelnames(["model_name", "engine", "reason"]) == [
            "model_name",
            "stage",
            "replica",
            "reason",
        ]

    def test_no_engine_label(self):
        # Unaffected (e.g. omni's own families that don't use engine).
        assert _rewrite_labelnames(["model_name"]) == ["model_name"]

    def test_tuple_input_returns_tuple(self):
        out = _rewrite_labelnames(("model_name", "engine"))
        assert isinstance(out, tuple)
        assert out == ("model_name", "stage", "replica")

    def test_none_passthrough(self):
        assert _rewrite_labelnames(None) is None


# ---------------------------------------------------------------------------
# _engine_to_stage_replica
# ---------------------------------------------------------------------------


class TestEngineToStageReplica:
    def test_int_engine_value(self):
        # Mirrors upstream `.labels(engine=idx, ...)` with int (loggers.py:510).
        _ENGINE_INDEX_MAP[7] = ("talker", "1")
        assert _engine_to_stage_replica(7) == ("talker", "1")

    def test_str_engine_value(self):
        # Mirrors upstream `metrics_info["engine"] = str(idx)` (loggers.py:1055).
        _ENGINE_INDEX_MAP[2] = ("thinker", "0")
        assert _engine_to_stage_replica("2") == ("thinker", "0")

    def test_missing_engine_idx_raises(self):
        # Empty map → fail-fast rather than emit a wrong (stage, replica).
        with pytest.raises(KeyError):
            _engine_to_stage_replica(999)


# ---------------------------------------------------------------------------
# Wrapper class behavior
# ---------------------------------------------------------------------------


class TestRelabelGauge:
    def test_labelnames_rewritten_at_creation(self, registry):
        g = _RelabelGauge(
            name="omni_test_gauge",
            documentation="test",
            labelnames=["model_name", "engine"],
            registry=registry,
        )
        assert g._labelnames == ("model_name", "stage", "replica")

    def test_labels_kwarg_translated(self, registry):
        _ENGINE_INDEX_MAP[5] = ("diffusion", "0")
        g = _RelabelGauge(
            name="omni_test_gauge_kwarg",
            documentation="test",
            labelnames=["model_name", "engine"],
            registry=registry,
        )
        g.labels(engine=5, model_name="qwen-omni").set(42.0)

        out = generate_latest(registry).decode()
        assert 'omni_test_gauge_kwarg{model_name="qwen-omni",replica="0",stage="diffusion"} 42.0' in out

    def test_labels_positional_passthrough(self, registry):
        # Double-rewrite guard: the per_engine_labelvalues setter rewrites
        # the values to 3-tuple [model_name, stage, replica] BEFORE
        # create_metric_per_engine fans them into .labels(*values). The mixin
        # must detect that args length already matches the rewritten
        # labelnames and pass through, otherwise it would re-interpret
        # args[engine_label_index] as an engine_idx and splice (stage, replica)
        # again, blowing label count to 4.
        g = _RelabelGauge(
            name="omni_test_gauge_pos",
            documentation="test",
            labelnames=["model_name", "engine"],
            registry=registry,
        )
        g.labels("qwen-omni", "thinker", "0").set(7.0)

        out = generate_latest(registry).decode()
        assert 'omni_test_gauge_pos{model_name="qwen-omni",replica="0",stage="thinker"} 7.0' in out

    def test_multiprocess_mode_kwarg_passthrough(self, registry):
        # Upstream creates Gauges with multiprocess_mode="mostrecent" — must not
        # be eaten by our mixin.
        g = _RelabelGauge(
            name="omni_test_gauge_mp",
            documentation="test",
            labelnames=["model_name", "engine"],
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        assert g._multiprocess_mode == "mostrecent"


class TestRelabelCounter:
    def test_labelnames_rewritten(self, registry):
        c = _RelabelCounter(
            name="omni_test_counter",
            documentation="test",
            labelnames=["model_name", "engine", "finished_reason"],
            registry=registry,
        )
        assert c._labelnames == (
            "model_name",
            "stage",
            "replica",
            "finished_reason",
        )

    def test_labels_kwarg_translated(self, registry):
        _ENGINE_INDEX_MAP[2] = ("thinker", "0")
        c = _RelabelCounter(
            name="omni_test_counter_kwarg",
            documentation="test",
            labelnames=["model_name", "engine", "finished_reason"],
            registry=registry,
        )
        c.labels(engine=2, model_name="m", finished_reason="stop").inc(3)

        out = generate_latest(registry).decode()
        assert (
            'omni_test_counter_kwarg_total{finished_reason="stop",model_name="m",replica="0",stage="thinker"} 3.0'
            in out
        )


class TestRelabelHistogram:
    def test_labelnames_rewritten(self, registry):
        h = _RelabelHistogram(
            name="omni_test_histo",
            documentation="test",
            labelnames=["model_name", "engine"],
            registry=registry,
        )
        assert h._labelnames == ("model_name", "stage", "replica")

    def test_labels_kwarg_translated_and_observe(self, registry):
        _ENGINE_INDEX_MAP[0] = ("talker", "0")
        h = _RelabelHistogram(
            name="omni_test_histo_obs",
            documentation="test",
            labelnames=["model_name", "engine"],
            registry=registry,
        )
        h.labels(engine=0, model_name="m").observe(0.5)

        out = generate_latest(registry).decode()
        assert 'omni_test_histo_obs_count{model_name="m",replica="0",stage="talker"} 1.0' in out

    def test_no_engine_label_unaffected(self, registry):
        # Families without engine label (e.g. omni-side own metrics) pass through.
        h = _RelabelHistogram(
            name="omni_test_no_engine",
            documentation="test",
            labelnames=["model_name"],
            registry=registry,
        )
        assert h._labelnames == ("model_name",)
        h.labels(model_name="m").observe(1.0)


# ---------------------------------------------------------------------------
# Positional .labels() with engine value (loggers.py:646, 679)
# ---------------------------------------------------------------------------


class TestPositionalEngine:
    def test_positional_engine_at_middle_index(self, registry):
        # Mirrors `counter_prompt_tokens_by_source.labels(model_name, str(idx), source)`.
        # Family original labelnames = ["model_name", "engine", "source"].
        _ENGINE_INDEX_MAP[5] = ("talker", "0")
        c = _RelabelCounter(
            name="omni_test_pos_mid",
            documentation="test",
            labelnames=["model_name", "engine", "source"],
            registry=registry,
        )
        c.labels("m", "5", "decoder").inc(2)

        out = generate_latest(registry).decode()
        assert 'omni_test_pos_mid_total{model_name="m",replica="0",source="decoder",stage="talker"} 2.0' in out

    def test_positional_engine_with_int_value(self, registry):
        # Defensive: positional form may also receive an int (we accept both).
        _ENGINE_INDEX_MAP[3] = ("thinker", "1")
        c = _RelabelCounter(
            name="omni_test_pos_int",
            documentation="test",
            labelnames=["model_name", "engine", "reason"],
            registry=registry,
        )
        c.labels("m", 3, "stop").inc()

        out = generate_latest(registry).decode()
        assert 'omni_test_pos_int_total{model_name="m",reason="stop",replica="1",stage="thinker"} 1.0' in out


# ---------------------------------------------------------------------------
# String-form engine kwarg (loggers.py:1056 info_gauge)
# ---------------------------------------------------------------------------


class TestStrEngineKwarg:
    def test_engine_kwarg_str_form(self, registry):
        # Mirrors `info_gauge.labels(**metrics_info)` where metrics_info["engine"]="0".
        _ENGINE_INDEX_MAP[0] = ("thinker", "0")
        g = _RelabelGauge(
            name="omni_test_info",
            documentation="test",
            labelnames=["cache_size", "engine"],
            multiprocess_mode="mostrecent",
            registry=registry,
        )
        # Upstream pattern: pass everything as kwargs from the metrics_info dict.
        g.labels(cache_size="big", engine="0").set(1)

        out = generate_latest(registry).decode()
        assert 'omni_test_info{cache_size="big",replica="0",stage="thinker"} 1.0' in out


# ---------------------------------------------------------------------------
# Child metric does not re-trigger relabel logic
# ---------------------------------------------------------------------------


class TestChildNoRecursion:
    def test_child_set_does_not_relookup(self, registry):
        # Once .labels() returns a child, subsequent .set()/.inc() must not
        # consult _ENGINE_INDEX_MAP again. We verify by clearing the map
        # AFTER labels() and proving .set() still works.
        _ENGINE_INDEX_MAP[4] = ("diffusion", "0")
        g = _RelabelGauge(
            name="omni_test_child",
            documentation="test",
            labelnames=["model_name", "engine"],
            registry=registry,
        )
        child = g.labels(engine=4, model_name="m")
        _ENGINE_INDEX_MAP.clear()  # would break a second .labels() lookup
        child.set(99.0)  # but set() is on the bound child — no map needed

        # Re-populate so generate_latest doesn't trip on anything else.
        _ENGINE_INDEX_MAP[4] = ("diffusion", "0")
        out = generate_latest(registry).decode()
        assert 'omni_test_child{model_name="m",replica="0",stage="diffusion"} 99.0' in out


# ---------------------------------------------------------------------------
# OmniPrometheusStatLogger — focused on the wrap mechanics (full
# PrometheusStatLogger init requires a real VllmConfig and is exercised by
# the orchestrator integration tests).
# ---------------------------------------------------------------------------


class TestOmniPrometheusStatLogger:
    def test_class_slots_point_to_wrappers(self):
        # Upstream's __init__ uses self._gauge_cls(...) etc. when constructing
        # families; class-level slot override is how we inject the relabel logic.
        assert OmniPrometheusStatLogger._gauge_cls is _RelabelGauge
        assert OmniPrometheusStatLogger._counter_cls is _RelabelCounter
        assert OmniPrometheusStatLogger._histogram_cls is _RelabelHistogram

    def test_per_engine_labelvalues_setter_rewrites_to_3tuple(self):
        # Construct via __new__ to skip the upstream PrometheusStatLogger __init__
        # (which needs a real VllmConfig). We only verify the property descriptor.
        sl = OmniPrometheusStatLogger.__new__(OmniPrometheusStatLogger)
        sl._stage_replica_map = {
            0: ("thinker", "0"),
            1: ("talker", "0"),
            2: ("talker", "1"),
        }

        # Mirror upstream's loggers.py:433 assignment shape.
        sl.per_engine_labelvalues = {
            0: ["my-model", "0"],
            1: ["my-model", "1"],
            2: ["my-model", "2"],
        }

        # Getter should return the 3-tuple form for downstream
        # create_metric_per_engine consumers.
        assert sl.per_engine_labelvalues == {
            0: ["my-model", "thinker", "0"],
            1: ["my-model", "talker", "0"],
            2: ["my-model", "talker", "1"],
        }

    def test_per_engine_labelvalues_getter_returns_internal_dict(self):
        sl = OmniPrometheusStatLogger.__new__(OmniPrometheusStatLogger)
        sl._stage_replica_map = {0: ("thinker", "0")}
        sl._omni_per_engine_labelvalues = {0: ["m", "thinker", "0"]}
        assert sl.per_engine_labelvalues == {0: ["m", "thinker", "0"]}

    def test_stage_replica_map_property_exposed(self):
        sl = OmniPrometheusStatLogger.__new__(OmniPrometheusStatLogger)
        srm = {0: ("thinker", "0"), 1: ("diffusion", "0")}
        sl._stage_replica_map = srm
        assert sl.stage_replica_map is srm

    def test_init_populates_engine_index_map(self):
        # Simulate the bookkeeping portion of __init__ (clear + update) without
        # calling super, since super needs a real VllmConfig.
        _ENGINE_INDEX_MAP[99] = ("stale", "stale")  # leftover from prior
        srm = {0: ("thinker", "0"), 1: ("talker", "0")}

        # Manually invoke the bookkeeping the way __init__ does it.
        _ENGINE_INDEX_MAP.clear()
        _ENGINE_INDEX_MAP.update(srm)

        assert dict(_ENGINE_INDEX_MAP) == srm
        assert 99 not in _ENGINE_INDEX_MAP  # old entry was cleared

    def test_record_stage_post_warmup_memory_updates_values(self, registry):
        sl = OmniPrometheusStatLogger.__new__(OmniPrometheusStatLogger)
        sl._omni_per_engine_labelvalues = {3: ["m", "1", "0"]}
        sl._init_stage_post_warmup_memory_families(registry)

        sl.record_stage_post_warmup_memory(
            StagePostWarmupMemoryStats(
                allocated_bytes=11,
                reserved_bytes=13,
            ),
            engine_idx=3,
        )

        labels = {"model_name": "m", "stage": "1", "replica": "0"}
        assert registry.get_sample_value(defs.STAGE_POST_WARMUP_MEMORY_ALLOCATED_BYTES, labels) == 11
        assert registry.get_sample_value(defs.STAGE_POST_WARMUP_MEMORY_RESERVED_BYTES, labels) == 13

    def test_stage_post_warmup_memory_families_register_only_with_logger(self, registry):
        metric_name = defs.STAGE_POST_WARMUP_MEMORY_ALLOCATED_BYTES
        assert metric_name not in generate_latest(registry).decode()

        sl = OmniPrometheusStatLogger.__new__(OmniPrometheusStatLogger)
        sl._init_stage_post_warmup_memory_families(registry)

        assert f"# HELP {metric_name}" in generate_latest(registry).decode()

    def test_stage_post_warmup_memory_families_reused_on_logger_recreation(self, monkeypatch, registry):
        init_families = OmniPrometheusStatLogger._init_stage_post_warmup_memory_families
        monkeypatch.setattr(
            OmniPrometheusStatLogger,
            "_init_stage_post_warmup_memory_families",
            lambda self: init_families(self, registry),
        )

        def init_upstream(self, vllm_config, engine_indexes):
            self.per_engine_labelvalues = {idx: ["m", str(idx)] for idx in engine_indexes}

        monkeypatch.setattr(PrometheusStatLogger, "__init__", init_upstream)
        stage_replica_map = {0: ("1", "0")}

        first = OmniPrometheusStatLogger(SimpleNamespace(), stage_replica_map)
        second = OmniPrometheusStatLogger(SimpleNamespace(), stage_replica_map)

        assert first._stage_post_warmup_memory_allocated_family is second._stage_post_warmup_memory_allocated_family

    def test_record_stage_post_warmup_memory_preserves_replica_values(self, registry):
        sl = OmniPrometheusStatLogger.__new__(OmniPrometheusStatLogger)
        sl._omni_per_engine_labelvalues = {
            0: ["m", "1", "0"],
            1: ["m", "1", "1"],
        }
        sl._init_stage_post_warmup_memory_families(registry)

        sl.record_stage_post_warmup_memory(
            StagePostWarmupMemoryStats(allocated_bytes=11, reserved_bytes=12),
            engine_idx=0,
        )
        sl.record_stage_post_warmup_memory(
            StagePostWarmupMemoryStats(allocated_bytes=13, reserved_bytes=14),
            engine_idx=0,
        )
        sl.record_stage_post_warmup_memory(
            StagePostWarmupMemoryStats(allocated_bytes=17, reserved_bytes=18),
            engine_idx=1,
        )

        replica_0 = {"model_name": "m", "stage": "1", "replica": "0"}
        replica_1 = {"model_name": "m", "stage": "1", "replica": "1"}
        assert registry.get_sample_value(defs.STAGE_POST_WARMUP_MEMORY_ALLOCATED_BYTES, replica_0) == 13
        assert registry.get_sample_value(defs.STAGE_POST_WARMUP_MEMORY_ALLOCATED_BYTES, replica_1) == 17

    def test_remove_stage_post_warmup_memory_preserves_slots_and_other_replicas(self, stats_config):
        sl = OmniPrometheusStatLogger(stats_config, {0: ("0", "0"), 1: ("0", "1")})
        iteration = IterationStats()
        iteration.num_generation_tokens = 7
        sl.record(None, iteration, engine_idx=0)
        for engine_idx in (0, 1):
            sl.record_stage_post_warmup_memory(
                StagePostWarmupMemoryStats(allocated_bytes=11, reserved_bytes=13),
                engine_idx=engine_idx,
            )

        sl.remove_stage_post_warmup_memory(engine_idx=0)
        sl.remove_stage_post_warmup_memory(engine_idx=0)
        sl.remove_stage_post_warmup_memory(engine_idx=99)

        removed = {"model_name": "test-model", "stage": "0", "replica": "0"}
        retained = removed | {"replica": "1"}
        for metric, value in (
            (defs.STAGE_POST_WARMUP_MEMORY_ALLOCATED_BYTES, 11),
            (defs.STAGE_POST_WARMUP_MEMORY_RESERVED_BYTES, 13),
        ):
            assert REGISTRY.get_sample_value(metric, removed) is None
            assert REGISTRY.get_sample_value(metric, retained) == value
        assert sl.stage_replica_map == {0: ("0", "0"), 1: ("0", "1")}
        assert sl.per_engine_labelvalues[0] == ["test-model", "0", "0"]
        assert REGISTRY.get_sample_value("vllm:generation_tokens_total", removed) == 7

        sl.remove_stage_post_warmup_memory(engine_idx=1)
        exposition = generate_latest(REGISTRY).decode()
        assert f"# HELP {defs.STAGE_POST_WARMUP_MEMORY_ALLOCATED_BYTES}" in exposition
        assert f"# HELP {defs.STAGE_POST_WARMUP_MEMORY_RESERVED_BYTES}" in exposition
        sl.record_stage_post_warmup_memory(
            StagePostWarmupMemoryStats(allocated_bytes=17, reserved_bytes=19),
            engine_idx=0,
        )
        assert REGISTRY.get_sample_value(defs.STAGE_POST_WARMUP_MEMORY_ALLOCATED_BYTES, removed) == 17
        assert REGISTRY.get_sample_value(defs.STAGE_POST_WARMUP_MEMORY_RESERVED_BYTES, removed) == 19
        sl.remove_stage_post_warmup_memory(engine_idx=0)


# ---------------------------------------------------------------------------
# Helper-class wraps for upstream's spec_decoding / kv_connector /
# perf_metrics sub-collectors. Without these, OmniPrometheusStatLogger
# crashes at startup with `Incorrect label count` because each helper builds
# its internal Counter/Gauge/Histogram families with raw 2-element labelnames
# (passed via constructor arg) while consuming the rewritten 3-element
# per_engine_labelvalues from the property descriptor.
# ---------------------------------------------------------------------------


class TestHelperClassWraps:
    def test_perf_metrics_wrap_routes_through_relabel_counter(self):
        assert _OmniPerfMetricsProm._counter_cls is _RelabelCounter

    def test_spec_decoding_wrap_routes_through_relabel_counter(self):
        assert _OmniSpecDecodingProm._counter_cls is _RelabelCounter

    def test_kv_connector_wrap_routes_through_all_three_relabel_classes(self):
        # KVConnector lets each connector build any of Gauge/Counter/Histogram,
        # so all three slots must be intercepted.
        assert _OmniKVConnectorProm._gauge_cls is _RelabelGauge
        assert _OmniKVConnectorProm._counter_cls is _RelabelCounter
        assert _OmniKVConnectorProm._histogram_cls is _RelabelHistogram

    def test_omni_logger_slots_point_to_helper_subclasses(self):
        # Upstream's PrometheusStatLogger.__init__ instantiates each sub-helper
        # via `self._<name>_cls(...)`, so the slot overrides on the omni
        # subclass are what routes through to the relabel mixin.
        assert OmniPrometheusStatLogger._perf_metrics_cls is _OmniPerfMetricsProm
        assert OmniPrometheusStatLogger._spec_decoding_cls is _OmniSpecDecodingProm
        assert OmniPrometheusStatLogger._kv_connector_cls is _OmniKVConnectorProm


# ---------------------------------------------------------------------------
# Double-rewrite guard. The mixin's positional-args path used to
# unconditionally splice (stage, replica) at engine_label_index. Because
# per_engine_labelvalues is rewritten to 3-tuples *before* feeding into
# create_metric_per_engine, that splice would run a second time on the
# already-rewritten values and blow the label count to 4. The guard now
# detects len(args) == len(self._labelnames) and short-circuits to passthrough.
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_registry():
    return CollectorRegistry()


class TestDoubleRewriteGuard:
    def test_pre_rewritten_3tuple_passes_through(self, fresh_registry):
        # 2-label original → 3-label rewritten family. Caller passes 3 values
        # (the rewritten shape) and they should land verbatim, not get
        # re-spliced.
        _ENGINE_INDEX_MAP.clear()
        _ENGINE_INDEX_MAP[0] = ("0", "0")
        _ENGINE_INDEX_MAP[1] = ("1", "0")
        g = _RelabelGauge(
            name="dr_pre_rewritten",
            documentation="t",
            labelnames=["model_name", "engine"],
            registry=fresh_registry,
        )
        # 3 positional args matching the rewritten 3-label family.
        g.labels("m", "1", "0").set(42)
        out = generate_latest(fresh_registry).decode()
        assert 'dr_pre_rewritten{model_name="m",replica="0",stage="1"} 42.0' in out

    def test_legacy_2tuple_with_extra_label_still_splices(self, fresh_registry):
        # 3-label original (engine in middle) → 4-label rewritten family.
        # Caller passes 3 values matching the ORIGINAL labelnames (the
        # gauge_waiting_by_reason / counter_request_success pattern from
        # upstream loggers.py:646, 679). The mixin must splice
        # (stage, replica) at engine's position to reach the 4-label family.
        _ENGINE_INDEX_MAP.clear()
        _ENGINE_INDEX_MAP[1] = ("1", "0")
        c = _RelabelCounter(
            name="dr_legacy_with_extra",
            documentation="t",
            labelnames=["model_name", "engine", "reason"],
            registry=fresh_registry,
        )
        # 3 positional args matching ORIGINAL labelnames (model_name,
        # engine_str, reason).
        c.labels("m", "1", "stop").inc(3)
        out = generate_latest(fresh_registry).decode()
        assert 'dr_legacy_with_extra_total{model_name="m",reason="stop",replica="0",stage="1"} 3.0' in out


@pytest.fixture
def stats_config():
    config = Mock(
        spec=VllmConfig,
        model_config=Mock(spec=ModelConfig, served_model_name="test-model", max_model_len=2048, is_diffusion=False),
        observability_config=ObservabilityConfig(kv_cache_metrics=True),
        speculative_config=Mock(spec=SpeculativeConfig, num_speculative_tokens=3),
        kv_transfer_config=None,
        ec_transfer_config=None,
        lora_config=None,
    )
    yield config
    unregister_vllm_metrics()


@pytest.mark.parametrize("initial_map", [{}, {0: ("0", "0"), 1: ("1", "0")}])
def test_register_replica_preserves_existing_metrics(stats_config, initial_map):
    stat_logger = OmniPrometheusStatLogger(stats_config, dict(initial_map))
    iteration = IterationStats()
    iteration.num_generation_tokens = 7
    if initial_map:
        stat_logger.record(None, iteration, engine_idx=0)
        stat_logger.record_sleep_state(sleep=1, level=1)
    new_idx = len(initial_map)
    stat_logger.register_replica(new_idx, "0", "4")
    scheduler = SchedulerStats(num_running_reqs=2, num_waiting_reqs=3)
    scheduler.spec_decoding_stats = SpecDecodingStats(
        num_spec_tokens=3,
        num_drafts=1,
        num_draft_tokens=3,
        num_accepted_tokens=2,
        num_accepted_tokens_per_pos=[1, 1, 0],
    )
    stat_logger.record(scheduler, iteration, engine_idx=new_idx)
    stat_logger.register_replica(new_idx, "0", "4")  # Rejoin keeps the same counters.
    stat_logger.record(None, iteration, engine_idx=new_idx)
    labels = {"model_name": "test-model", "stage": "0", "replica": "4"}
    assert REGISTRY.get_sample_value("vllm:generation_tokens_total", labels) == 14
    assert REGISTRY.get_sample_value("vllm:num_requests_running", labels) == 2
    assert REGISTRY.get_sample_value("vllm:engine_sleep_state", labels | {"sleep_state": "awake"}) == 1
    assert REGISTRY.get_sample_value("vllm:iteration_tokens_total_count", labels) == 2
    assert (
        REGISTRY.get_sample_value("vllm:spec_decode_num_accepted_tokens_per_pos_total", labels | {"position": "1"}) == 1
    )
    if initial_map:
        assert (
            REGISTRY.get_sample_value("vllm:engine_sleep_state", labels | {"replica": "0", "sleep_state": "awake"}) == 0
        )
        assert REGISTRY.get_sample_value("vllm:generation_tokens_total", labels | {"replica": "0"}) == 7
    with pytest.raises(ValueError, match="already belongs"):
        stat_logger.register_replica(new_idx, "2", "0")


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_replicas", [0, 1])
async def test_orchestrator_records_stats_from_late_replica(stats_config, initial_replicas):
    pool = Mock(spec=StagePool, num_replicas=initial_replicas, stage_vllm_config=stats_config)
    pool.available_replica_ids.return_value = list(range(initial_replicas))
    pool.process_llm_raw_outputs = AsyncMock(return_value=[])
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=[pool],
        log_stats=True,
    )
    raw = EngineCoreOutputs(outputs=[], scheduler_stats=SchedulerStats(num_running_reqs=5))
    await orchestrator._process_llm_stage_outputs(0, 3, raw, set())
    assert (
        REGISTRY.get_sample_value(
            "vllm:num_requests_running", {"model_name": "test-model", "stage": "0", "replica": "3"}
        )
        == 5
    )
    await orchestrator._process_llm_stage_outputs(0, 3, raw, set())
    assert len(orchestrator._stage_replica_to_engine_idx) == initial_replicas + 1
