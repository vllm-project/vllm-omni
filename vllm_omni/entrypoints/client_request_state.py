import asyncio

from vllm_omni.metrics import OrchestratorAggregator


class ClientRequestState:
    """Tracks the state of an individual request in the orchestrator."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.stage_id: int | None = None
        self.metrics: OrchestratorAggregator | None = None


class AsyncClientRequestState(ClientRequestState):
    """Per-request state for AsyncOmni; includes a queue for output routing."""

    def __init__(self, request_id: str, queue: asyncio.Queue | None = None):
        super().__init__(request_id)
        self.queue = queue if queue is not None else asyncio.Queue()
