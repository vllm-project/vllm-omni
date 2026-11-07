# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

from vllm_omni.metrics import OrchestratorAggregator
from vllm_omni.metrics.stream_edge import RequestStreamEdgeEvents


class ClientRequestState:
    """Tracks one entrypoint request and its output queue."""

    def __init__(
        self,
        request_id: str,
        external_request_id: str | None = None,
        queue: asyncio.Queue | None = None,
        final_stage_id: int | None = None,
    ):
        self.request_id = request_id
        self.external_request_id = external_request_id
        self.stage_id: int | None = None
        self.final_stage_id: int | None = final_stage_id
        self.queue = queue if queue is not None else asyncio.Queue()
        self.metrics: OrchestratorAggregator | None = None
        self.input_stream_task: asyncio.Task | None = None
        # Request-scoped idempotency guard for Prometheus failure counters.
        self.failure_recorded = False
        # Wall-clock time at which the user's request arrived in the engine
        # entrypoint. Set in async_omni.generate() before the orchestrator
        # accepts the request. Used as the t0 anchor for audio_ttfp.
        self.request_arrival_ts: float = 0.0
        # Wall-clock time at which the first audio packet was observed for
        # this request. None means the streaming hook hasn't fired yet.
        # Used as the once-per-request guard for audio_ttfp_s emit.
        self.first_audio_ts: float | None = None
        # Per-chunk timeline (seconds since request_arrival_ts) and PCM byte
        # counts for the audio streaming response. Populated by the streaming
        # endpoint on every audio.chunk emit; consumed at request finalize to
        # compute audio_underrun_s and audio_continuity_ok_total.
        self.audio_chunk_arrivals_s: list[float] = []
        self.audio_chunk_bytes: list[int] = []
        self.audio_sample_rate: int | None = None
        # Stage / replica that produced the audio packets — captured at the
        # first-packet hook so the finalize-time emit can label correctly
        # without re-querying stage_pools.
        self.audio_emit_stage_id: int | None = None
        self.audio_emit_replica_id: int | None = None
        # De-dup set for metric messages: OmniBase populates this in
        # ``_handle_output_message`` / ``_process_single_result`` so the same
        # ``id(msg)`` isn't counted twice into per-request metrics. Kept on
        # the request state (not a class-level dict) so it is released with
        # the state — see #6462 / #6561.
        self.consumed_metric_message_ids: set[int] = set()
        self.stream_edge_events: RequestStreamEdgeEvents | None = None
        self._stream_edge_metrics_closed = False

    def enable_stream_edge_metrics(self) -> RequestStreamEdgeEvents:
        """Opt in at an instrumented producer boundary for this request instance."""
        if self._stream_edge_metrics_closed:
            raise RuntimeError("Request stream-edge metrics have been released")
        if self.stream_edge_events is None:
            self.stream_edge_events = RequestStreamEdgeEvents()
        return self.stream_edge_events

    def release_stream_edge_metrics(self) -> None:
        self._stream_edge_metrics_closed = True
        if self.stream_edge_events is not None:
            self.stream_edge_events.close()
