import asyncio
from typing import Optional


class ClientRequestState:
    """Tracks the state of an individual request in the orchestrator."""

    def __init__(self, request_id: str, queue: Optional[asyncio.Queue] = None):
        self.request_id = request_id
        self.stage_id: Optional[int] = None
        self.queue = queue if queue is not None else asyncio.Queue()
