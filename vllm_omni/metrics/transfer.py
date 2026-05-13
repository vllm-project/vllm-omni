"""OmniTransferMetrics — cross-stage transfer Prometheus families (RFC G3).

Four Histogram families with ``{model_name, from_stage, from_replica,
to_stage, to_replica}`` labels. Each ``.observe_*()`` call corresponds to one
physical transfer event (one chunk hop from a sender replica to a receiver
replica), not the per-request accumulated total — so the Histogram tracks
the distribution of physical transfers, not request-aggregated sums.

Data source: ``vllm_omni.metrics.stats.TransferEdgeStats`` accumulators in
``OrchestratorAggregator.record_transfer_tx`` / ``record_transfer_rx``. The
emit hook lives in stats.py; this module only registers the families and
exposes the typed observe API.

Note on label deviation from RFC §3.2.6: the RFC lists only the four
stage/replica labels. We add ``model_name`` so transfer aligns with the rest
of the omni_* family naming and PromQL can join on model_name uniformly.
"""

from __future__ import annotations

from prometheus_client import Histogram

from vllm_omni.metrics import definitions as defs

_labelnames = list(defs.TRANSFER_LABELS)


# ----------------------------------------------------------------------------
# TX-side families (observed when record_transfer_tx fires)
# ----------------------------------------------------------------------------
_transfer_size_bytes_family = Histogram(
    defs.TRANSFER_SIZE_BYTES,
    "Per-transfer payload size in bytes (one observation per physical hop).",
    labelnames=_labelnames,
    buckets=defs.BYTES_BUCKETS,
)
_transfer_tx_time_ms_family = Histogram(
    defs.TRANSFER_TX_TIME_MS,
    "Sender-side time in milliseconds (serialize + submit to connector).",
    labelnames=_labelnames,
    buckets=defs.MS_BUCKETS,
)


# ----------------------------------------------------------------------------
# RX-side families (observed when record_transfer_rx fires)
# ----------------------------------------------------------------------------
_transfer_rx_decode_time_ms_family = Histogram(
    defs.TRANSFER_RX_DECODE_TIME_MS,
    "Receiver-side time in milliseconds (recv + deserialize).",
    labelnames=_labelnames,
    buckets=defs.MS_BUCKETS,
)
_transfer_in_flight_time_ms_family = Histogram(
    defs.TRANSFER_IN_FLIGHT_TIME_MS,
    "Network in-flight time in milliseconds (TX done -> RX recv start).",
    labelnames=_labelnames,
    buckets=defs.MS_BUCKETS,
)


class OmniTransferMetrics:
    """Per-(from, to) replica observe API for cross-stage transfers.

    A single instance per pipeline; ``model_name`` is bound at init and
    every observe call carries it in the label set. Stage/replica are
    passed at observe time because the same instance serves all
    (from_stage, from_replica) -> (to_stage, to_replica) edges.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    # ---- TX side (record_transfer_tx hook) -------------------------------

    def observe_size(
        self,
        from_stage: int,
        from_replica: int,
        to_stage: int,
        to_replica: int,
        size_bytes: int,
    ) -> None:
        _transfer_size_bytes_family.labels(
            model_name=self._model_name,
            from_stage=str(from_stage),
            from_replica=str(from_replica),
            to_stage=str(to_stage),
            to_replica=str(to_replica),
        ).observe(size_bytes)

    def observe_tx_time(
        self,
        from_stage: int,
        from_replica: int,
        to_stage: int,
        to_replica: int,
        tx_time_ms: float,
    ) -> None:
        _transfer_tx_time_ms_family.labels(
            model_name=self._model_name,
            from_stage=str(from_stage),
            from_replica=str(from_replica),
            to_stage=str(to_stage),
            to_replica=str(to_replica),
        ).observe(tx_time_ms)

    # ---- RX side (record_transfer_rx hook) -------------------------------

    def observe_rx_decode_time(
        self,
        from_stage: int,
        from_replica: int,
        to_stage: int,
        to_replica: int,
        rx_decode_time_ms: float,
    ) -> None:
        _transfer_rx_decode_time_ms_family.labels(
            model_name=self._model_name,
            from_stage=str(from_stage),
            from_replica=str(from_replica),
            to_stage=str(to_stage),
            to_replica=str(to_replica),
        ).observe(rx_decode_time_ms)

    def observe_in_flight_time(
        self,
        from_stage: int,
        from_replica: int,
        to_stage: int,
        to_replica: int,
        in_flight_time_ms: float,
    ) -> None:
        _transfer_in_flight_time_ms_family.labels(
            model_name=self._model_name,
            from_stage=str(from_stage),
            from_replica=str(from_replica),
            to_stage=str(to_stage),
            to_replica=str(to_replica),
        ).observe(in_flight_time_ms)
