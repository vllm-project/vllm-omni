from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from vllm_omni.metrics.stat_logger import (
    _ENGINE_INDEX_MAP,
    _RelabelCounter,
    _RelabelGauge,
    _RelabelHistogram,
    _rewrite_label_kwargs,
    _rewrite_labelnames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
# _rewrite_label_kwargs
# ---------------------------------------------------------------------------


class TestRewriteLabelKwargs:
    def test_engine_kwarg_translated(self):
        _ENGINE_INDEX_MAP[7] = ("talker", "1")
        out = _rewrite_label_kwargs({"engine": 7, "model_name": "m"})
        assert out == {"stage": "talker", "replica": "1", "model_name": "m"}

    def test_engine_with_extra_kwargs(self):
        # Mirrors upstream's `.labels(engine=idx, model_name=m, sleep_state=s)`.
        _ENGINE_INDEX_MAP[3] = ("thinker", "0")
        out = _rewrite_label_kwargs(
            {"engine": 3, "model_name": "m", "sleep_state": "awake"}
        )
        assert out == {
            "stage": "thinker",
            "replica": "0",
            "model_name": "m",
            "sleep_state": "awake",
        }

    def test_no_engine_kwarg_passthrough(self):
        out = _rewrite_label_kwargs({"model_name": "m", "stage": "talker"})
        assert out == {"model_name": "m", "stage": "talker"}

    def test_missing_engine_idx_raises(self):
        # Empty map → fail-fast rather than emit a wrong (stage, replica).
        with pytest.raises(KeyError):
            _rewrite_label_kwargs({"engine": 999, "model_name": "m"})


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
        assert (
            'omni_test_gauge_kwarg{model_name="qwen-omni",replica="0",stage="diffusion"} 42.0'
            in out
        )

    def test_labels_positional_passthrough(self, registry):
        # Phase 2.2's per_engine_labelvalues setter feeds positional 3-tuples;
        # our mixin must not mangle positional .labels() calls.
        g = _RelabelGauge(
            name="omni_test_gauge_pos",
            documentation="test",
            labelnames=["model_name", "engine"],
            registry=registry,
        )
        g.labels("qwen-omni", "thinker", "0").set(7.0)

        out = generate_latest(registry).decode()
        assert (
            'omni_test_gauge_pos{model_name="qwen-omni",replica="0",stage="thinker"} 7.0'
            in out
        )

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
        assert (
            'omni_test_histo_obs_count{model_name="m",replica="0",stage="talker"} 1.0'
            in out
        )

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
