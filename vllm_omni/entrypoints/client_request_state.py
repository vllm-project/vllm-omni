import asyncio

from vllm_omni.metrics import OrchestratorAggregator


class ClientRequestState:
    """Tracks the state of an individual request in the orchestrator."""

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
