from __future__ import annotations

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.transfer import OmniTransferMetrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_MODEL = "test-transfer-model"


@pytest.fixture(scope="module")
def tx() -> OmniTransferMetrics:
    return OmniTransferMetrics(model_name=_MODEL)


def _sample_value(output: str, line_prefix: str) -> float | None:
    for line in output.splitlines():
        if line.startswith(line_prefix):
            return float(line.split()[-1])
    return None


_EXPECTED_FAMILIES = [
    defs.TRANSFER_SIZE_BYTES,
    defs.TRANSFER_TX_TIME_MS,
    defs.TRANSFER_RX_DECODE_TIME_MS,
    defs.TRANSFER_IN_FLIGHT_TIME_MS,
]


# ---------------------------------------------------------------------------
# Family registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_all_four_families_present(self, tx: OmniTransferMetrics) -> None:
        # Trigger one observation per family so the registry exposes them.
        tx.observe_size(0, 0, 1, 0, 1024)
        tx.observe_tx_time(0, 0, 1, 0, 5.0)
        tx.observe_rx_decode_time(0, 0, 1, 0, 8.0)
        tx.observe_in_flight_time(0, 0, 1, 0, 2.0)

        out = generate_latest(REGISTRY).decode()
        for name in _EXPECTED_FAMILIES:
            assert f"# HELP {name}" in out, f"missing family: {name}"


# ---------------------------------------------------------------------------
# Observe APIs
# ---------------------------------------------------------------------------


class TestObserveSize:
    def test_size_observed_with_correct_labels(self, tx: OmniTransferMetrics) -> None:
        # Distinct (from, to) so test isolation holds across cases.
        tx.observe_size(2, 0, 3, 1, 65536)
        out = generate_latest(REGISTRY).decode()
        prefix = (
            f'{defs.TRANSFER_SIZE_BYTES}_sum'
            f'{{from_replica="0",from_stage="2",model_name="{_MODEL}",'
            f'to_replica="1",to_stage="3"}}'
        )
        assert _sample_value(out, prefix) == 65536.0


class TestObserveTxTime:
    def test_tx_time_observed(self, tx: OmniTransferMetrics) -> None:
        tx.observe_tx_time(2, 1, 3, 0, 12.5)
        out = generate_latest(REGISTRY).decode()
        prefix = (
            f'{defs.TRANSFER_TX_TIME_MS}_sum'
            f'{{from_replica="1",from_stage="2",model_name="{_MODEL}",'
            f'to_replica="0",to_stage="3"}}'
        )
        assert _sample_value(out, prefix) == 12.5


class TestObserveRxDecodeTime:
    def test_rx_decode_time_observed(self, tx: OmniTransferMetrics) -> None:
        tx.observe_rx_decode_time(0, 0, 1, 0, 4.2)
        out = generate_latest(REGISTRY).decode()
        prefix = (
            f'{defs.TRANSFER_RX_DECODE_TIME_MS}_sum'
            f'{{from_replica="0",from_stage="0",model_name="{_MODEL}",'
            f'to_replica="0",to_stage="1"}}'
        )
        assert _sample_value(out, prefix) == 4.2


class TestObserveInFlightTime:
    def test_in_flight_time_observed(self, tx: OmniTransferMetrics) -> None:
        tx.observe_in_flight_time(0, 0, 1, 0, 1.7)
        out = generate_latest(REGISTRY).decode()
        prefix = (
            f'{defs.TRANSFER_IN_FLIGHT_TIME_MS}_sum'
            f'{{from_replica="0",from_stage="0",model_name="{_MODEL}",'
            f'to_replica="0",to_stage="1"}}'
        )
        assert _sample_value(out, prefix) == 1.7


# ---------------------------------------------------------------------------
# Multi (from, to) cardinality
# ---------------------------------------------------------------------------


class TestCardinality:
    def test_multiple_edges_produce_independent_series(
        self, tx: OmniTransferMetrics
    ) -> None:
        # Same family, different (from_replica, to_replica) → distinct series.
        tx.observe_size(5, 0, 6, 0, 100)
        tx.observe_size(5, 0, 6, 1, 200)
        tx.observe_size(5, 1, 6, 0, 300)

        out = generate_latest(REGISTRY).decode()

        prefix_a = (
            f'{defs.TRANSFER_SIZE_BYTES}_sum'
            f'{{from_replica="0",from_stage="5",model_name="{_MODEL}",'
            f'to_replica="0",to_stage="6"}}'
        )
        prefix_b = (
            f'{defs.TRANSFER_SIZE_BYTES}_sum'
            f'{{from_replica="0",from_stage="5",model_name="{_MODEL}",'
            f'to_replica="1",to_stage="6"}}'
        )
        prefix_c = (
            f'{defs.TRANSFER_SIZE_BYTES}_sum'
            f'{{from_replica="1",from_stage="5",model_name="{_MODEL}",'
            f'to_replica="0",to_stage="6"}}'
        )
        assert _sample_value(out, prefix_a) == 100.0
        assert _sample_value(out, prefix_b) == 200.0
        assert _sample_value(out, prefix_c) == 300.0


# ---------------------------------------------------------------------------
# Bucket selection
# ---------------------------------------------------------------------------


class TestBucketSelection:
    def test_size_uses_bytes_buckets(self, tx: OmniTransferMetrics) -> None:
        tx.observe_size(7, 0, 8, 0, 4096)
        out = generate_latest(REGISTRY).decode()
        # BYTES_BUCKETS contains 1024 — distinctive vs MS / SECONDS buckets.
        marker = f'{defs.TRANSFER_SIZE_BYTES}_bucket{{from_replica="0",from_stage="7",le="1024"'
        assert marker in out

    def test_time_families_use_ms_buckets(self, tx: OmniTransferMetrics) -> None:
        tx.observe_tx_time(7, 0, 8, 0, 1.0)
        out = generate_latest(REGISTRY).decode()
        # MS_BUCKETS contains 1.0 — present in tx_time histogram.
        marker = f'{defs.TRANSFER_TX_TIME_MS}_bucket{{from_replica="0",from_stage="7",le="1.0"'
        assert marker in out


# ---------------------------------------------------------------------------
# OrchestratorAggregator emit hook (Phase 4.2)
# ---------------------------------------------------------------------------


from vllm_omni.metrics.stats import OrchestratorAggregator, StageRequestStats, StageStats


class _StubTransferEmitter:
    """Records every observe_* call so the hook routing can be asserted
    without standing up a Prometheus registry."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def observe_size(self, fs, fr, ts, tr, n):
        self.calls.append(("observe_size", fs, fr, ts, tr, n))

    def observe_tx_time(self, fs, fr, ts, tr, t):
        self.calls.append(("observe_tx_time", fs, fr, ts, tr, t))

    def observe_rx_decode_time(self, fs, fr, ts, tr, t):
        self.calls.append(("observe_rx_decode_time", fs, fr, ts, tr, t))

    def observe_in_flight_time(self, fs, fr, ts, tr, t):
        self.calls.append(("observe_in_flight_time", fs, fr, ts, tr, t))


def _make_stats(stage_id, request_id, *, rx_decode=0.0, rx_in_flight=0.0, rx_bytes=0):
    """Minimal StageRequestStats for record_transfer_rx input."""
    return StageRequestStats(
        batch_id=1,
        batch_size=1,
        num_tokens_in=0,
        num_tokens_out=0,
        stage_gen_time_ms=0.0,
        rx_transfer_bytes=rx_bytes,
        rx_decode_time_ms=rx_decode,
        rx_in_flight_time_ms=rx_in_flight,
        stage_stats=StageStats(),
        stage_id=stage_id,
        request_id=request_id,
    )


class TestEmitHookTx:
    def test_record_transfer_tx_emits_size_and_tx_time(self):
        emitter = _StubTransferEmitter()
        agg = OrchestratorAggregator(
            num_stages=3,
            log_stats=False,
            wall_start_ts=0.0,
            final_stage_id_for_e2e=2,
            transfer_emitter=emitter,
            replica_resolver=lambda s, rid: {0: 1, 1: 0}.get(s),
        )
        agg.record_transfer_tx(
            from_stage=0,
            to_stage=1,
            request_id="r-tx-1",
            size_bytes=2048,
            tx_time_ms=7.5,
            used_shm=False,
        )
        assert emitter.calls == [
            ("observe_size", 0, 1, 1, 0, 2048),
            ("observe_tx_time", 0, 1, 1, 0, 7.5),
        ]

    def test_record_transfer_tx_no_emit_when_emitter_none(self):
        agg = OrchestratorAggregator(
            num_stages=3,
            log_stats=False,
            wall_start_ts=0.0,
            final_stage_id_for_e2e=2,
            transfer_emitter=None,
            replica_resolver=lambda s, rid: 0,
        )
        # Should not raise; just no-op the emit.
        evt = agg.record_transfer_tx(
            from_stage=0,
            to_stage=1,
            request_id="r-tx-2",
            size_bytes=128,
            tx_time_ms=1.0,
            used_shm=True,
        )
        # Accumulation still happens — only Prometheus emit is skipped.
        assert evt is not None
        assert evt.size_bytes == 128
        assert evt.tx_time_ms == 1.0

    def test_record_transfer_tx_no_emit_when_resolver_returns_none(self):
        emitter = _StubTransferEmitter()
        agg = OrchestratorAggregator(
            num_stages=3,
            log_stats=False,
            wall_start_ts=0.0,
            final_stage_id_for_e2e=2,
            transfer_emitter=emitter,
            replica_resolver=lambda s, rid: None,  # always-None resolver
        )
        agg.record_transfer_tx(
            from_stage=0,
            to_stage=1,
            request_id="r-tx-3",
            size_bytes=512,
            tx_time_ms=2.0,
            used_shm=False,
        )
        assert emitter.calls == []

    def test_record_transfer_tx_no_emit_when_one_side_resolves_to_none(self):
        emitter = _StubTransferEmitter()
        agg = OrchestratorAggregator(
            num_stages=3,
            log_stats=False,
            wall_start_ts=0.0,
            final_stage_id_for_e2e=2,
            transfer_emitter=emitter,
            replica_resolver=lambda s, rid: 0 if s == 0 else None,  # to-side fails
        )
        agg.record_transfer_tx(
            from_stage=0,
            to_stage=1,
            request_id="r-tx-4",
            size_bytes=64,
            tx_time_ms=0.5,
            used_shm=False,
        )
        assert emitter.calls == []


class TestEmitHookRx:
    def test_record_transfer_rx_emits_decode_and_in_flight(self):
        emitter = _StubTransferEmitter()
        agg = OrchestratorAggregator(
            num_stages=3,
            log_stats=False,
            wall_start_ts=0.0,
            final_stage_id_for_e2e=2,
            transfer_emitter=emitter,
            replica_resolver=lambda s, rid: {0: 1, 1: 0}.get(s),
        )
        stats = _make_stats(
            stage_id=1, request_id="r-rx-1", rx_decode=4.2, rx_in_flight=1.7
        )
        agg.record_transfer_rx(stats)
        assert emitter.calls == [
            ("observe_rx_decode_time", 0, 1, 1, 0, 4.2),
            ("observe_in_flight_time", 0, 1, 1, 0, 1.7),
        ]

    def test_record_transfer_rx_skips_stage_zero(self):
        emitter = _StubTransferEmitter()
        agg = OrchestratorAggregator(
            num_stages=3,
            log_stats=False,
            wall_start_ts=0.0,
            final_stage_id_for_e2e=2,
            transfer_emitter=emitter,
            replica_resolver=lambda s, rid: 0,
        )
        # stage_id=0 has no upstream, so record_transfer_rx returns early.
        stats = _make_stats(stage_id=0, request_id="r-rx-2", rx_decode=4.2)
        agg.record_transfer_rx(stats)
        assert emitter.calls == []

    def test_record_transfer_rx_no_emit_when_emitter_none(self):
        agg = OrchestratorAggregator(
            num_stages=3,
            log_stats=False,
            wall_start_ts=0.0,
            final_stage_id_for_e2e=2,
        )
        stats = _make_stats(stage_id=1, request_id="r-rx-3", rx_decode=1.0, rx_in_flight=0.5)
        evt = agg.record_transfer_rx(stats)
        # Accumulation still happens — only Prometheus emit is skipped.
        assert evt is not None
        assert evt.rx_decode_time_ms == 1.0
        assert evt.in_flight_time_ms == 0.5
