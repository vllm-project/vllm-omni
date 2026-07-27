import asyncio

from vllm_omni.metrics import OrchestratorAggregator


class ClientRequestState:
    """Tracks one entrypoint request and its output queue."""

    def __init__(
        self,
        request_id: str,
        external_request_id: str | None = None,
        queue: asyncio.Queue | None = None,
    ):
        self.request_id = request_id
        self.external_request_id = external_request_id
        self.stage_id: int | None = None
        self.queue = queue if queue is not None else asyncio.Queue()
        self.metrics: OrchestratorAggregator | None = None
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
