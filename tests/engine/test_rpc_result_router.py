import queue
from typing import Literal

import pytest

from vllm_omni.engine.messages import (
    CollectiveRPCResultMessage,
    EngineQueueMessage,
    ErrorMessage,
)
from vllm_omni.engine.rpc_result_router import RpcResultRouter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class GenericCorrelatedResultMessage(EngineQueueMessage, kw_only=True):
    type: Literal["generic_correlated_result"] = "generic_correlated_result"
    namespace: str
    correlation_id: str

    @property
    def rpc_correlation_key(self) -> tuple[str, str]:
        return (self.namespace, self.correlation_id)


def _generic_result(correlation_id: str) -> GenericCorrelatedResultMessage:
    return GenericCorrelatedResultMessage(
        namespace="plugin",
        correlation_id=correlation_id,
    )


def test_rpc_result_router_routes_out_of_order_results_by_correlation_id():
    source: queue.Queue = queue.Queue()
    router = RpcResultRouter(source)
    first = router.register(("plugin", "first"))
    second = router.register(("plugin", "second"))

    source.put(_generic_result("second"))
    source.put(_generic_result("first"))

    assert first.get(timeout=1).correlation_id == "first"
    assert second.get(timeout=1).correlation_id == "second"
    router.close()


def test_rpc_result_router_drops_only_the_late_result_after_unregister():
    source: queue.Queue = queue.Queue()
    router = RpcResultRouter(source)
    expired = router.register(("plugin", "expired"))
    active = router.register(("collective", "active"))
    router.unregister(("plugin", "expired"), expired)

    source.put(_generic_result("expired"))
    source.put(
        CollectiveRPCResultMessage(
            rpc_id="active",
            method="health",
            stage_ids=[0],
            results=["ok"],
        )
    )

    assert active.get(timeout=1).rpc_id == "active"
    assert expired.empty()
    router.close()


def test_rpc_result_router_broadcasts_fatal_errors_to_pending_waiters():
    source: queue.Queue = queue.Queue()
    router = RpcResultRouter(source)
    plugin = router.register(("plugin", "one"))
    collective = router.register(("collective", "two"))

    source.put(ErrorMessage(error="orchestrator failed", fatal=True))

    assert plugin.get(timeout=1).error == "orchestrator failed"
    assert collective.get(timeout=1).error == "orchestrator failed"
    with pytest.raises(RuntimeError, match="orchestrator failed"):
        router.register(("plugin", "after-failure"))
    router.close()


def test_rpc_result_router_does_not_broadcast_uncorrelated_nonfatal_errors():
    source: queue.Queue = queue.Queue()
    router = RpcResultRouter(source)
    waiter = router.register(("plugin", "active"))

    source.put(ErrorMessage(error="request failed", fatal=False, request_id="other"))
    source.put(_generic_result("active"))

    assert waiter.get(timeout=1).correlation_id == "active"
    router.close()


def test_rpc_result_router_close_unblocks_waiters_and_stops_consumer():
    source: queue.Queue = queue.Queue()
    router = RpcResultRouter(source)
    waiter = router.register(("plugin", "pending"))

    router.close()

    result = waiter.get(timeout=1)
    assert isinstance(result, ErrorMessage)
    assert result.fatal is True
    assert result.error == "RPC result router closed"
    assert not router._thread.is_alive()
    with pytest.raises(RuntimeError, match="router is closed"):
        router.register(("plugin", "after-close"))
