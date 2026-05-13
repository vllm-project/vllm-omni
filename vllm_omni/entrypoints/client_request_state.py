import asyncio

from vllm_omni.metrics import OrchestratorAggregator


class ClientRequestState:
    """Tracks the state of an individual request in the orchestrator."""

    def __init__(self, request_id: str, queue: asyncio.Queue | None = None):
        self.request_id = request_id
        self.stage_id: int | None = None
        self.queue = queue if queue is not None else asyncio.Queue()
        self.metrics: OrchestratorAggregator | None = None
        # Wall-clock time at which the user's request arrived in the engine
        # entrypoint. Set in async_omni.generate() before the orchestrator
        # accepts the request. Used as the "起算" anchor for audio_ttfp.
        self.request_arrival_ts: float = 0.0
        # Wall-clock time at which the first audio packet was observed for
        # this request. None means the streaming hook hasn't fired yet.
        # Used as the once-per-request guard for audio_ttfp_seconds emit.
        self.first_audio_ts: float | None = None
