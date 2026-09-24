"""Regression tests for vllm-project/vllm-omni#5349 (P1): normal completion
goes through _free_request(), not finish_requests() (the external
abort/cancel entry point). Without a cleanup_receiver() call there,
_active_streams never releases on ordinary completion, so the bounded-K
window fills with stale entries after K completions.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / RequestStatus are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.v1.request import RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler(*, chunk_transfer_adapter=None) -> OmniARScheduler:
    """Minimal OmniARScheduler exercising _free_request()'s no-KV-transfer
    happy path."""
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._omits_kv_transfer_cache = {}
    sched._connector_finished = lambda request: (False, None)
    sched.encoder_cache_manager = MagicMock()
    sched.finished_req_ids = set()
    sched._new_prompt_len_snapshot = {}
    sched.finished_req_ids_dict = None
    sched._should_transfer_kv_for_request = lambda req_id: False
    sched._free_blocks = MagicMock()
    sched._free_input_coordinator_request = MagicMock()
    sched.chunk_transfer_adapter = chunk_transfer_adapter
    return sched


class _FakeFinishedRequest:
    """Not a SimpleNamespace: SimpleNamespace defines __eq__, which makes it
    unhashable, and _free_request() puts the request in a set."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id

    def is_finished(self) -> bool:
        return True


def _make_finished_request(request_id: str = "req-1"):
    return _FakeFinishedRequest(request_id)


def test_free_request_releases_chunk_transfer_adapter_receiver_state():
    adapter = MagicMock()
    sched = _make_scheduler(chunk_transfer_adapter=adapter)
    request = _make_finished_request("req-1")

    sched._free_request(request)

    adapter.cleanup_receiver.assert_called_once_with("req-1")


def test_free_request_is_safe_without_a_chunk_transfer_adapter():
    """Most stages have no chunk transfer adapter configured."""
    sched = _make_scheduler(chunk_transfer_adapter=None)
    request = _make_finished_request("req-1")

    sched._free_request(request)  # must not raise

    sched._free_input_coordinator_request.assert_called_once_with("req-1")


def test_native_mooncake_finish_uses_confirmed_token_boundary() -> None:
    sched = _make_scheduler(chunk_transfer_adapter=None)
    sched.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnector"))
    observed = {}

    def connector_finished(request):
        observed["status"] = request.status
        observed["num_computed_tokens"] = request.num_computed_tokens
        return True, None

    sched._connector_finished = connector_finished
    request = _make_finished_request("req-native")
    request.client_index = 0
    request.status = RequestStatus.FINISHED_STOPPED
    request.kv_transfer_params = {
        "do_remote_decode": True,
        "transfer_id": "xfer-req-native",
    }
    request.num_computed_tokens = 13
    request.num_output_placeholders = 1
    request.num_prompt_tokens = 10

    kv_params, _ = sched._free_request(request)

    assert observed == {
        "status": RequestStatus.FINISHED_LENGTH_CAPPED,
        "num_computed_tokens": 12,
    }
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.num_computed_tokens == 13
    assert kv_params == {
        "transfer_id": "xfer-req-native",
        "num_transfer_tokens": 12,
    }


def test_native_mooncake_recycles_ar_pages_in_physical_order() -> None:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnector"))
    blocks = [SimpleNamespace(block_id=7), SimpleNamespace(block_id=2), SimpleNamespace(block_id=4)]
    sched.kv_cache_manager = SimpleNamespace(
        enable_caching=False,
        pop_blocks_for_free=MagicMock(return_value=blocks),
        block_pool=SimpleNamespace(free_blocks=MagicMock()),
    )
    sched.defer_block_free = False
    sched.processed_step_seq = 3
    request = SimpleNamespace(last_sched_seq=3)

    sched._free_request_blocks(request)

    sched.kv_cache_manager.block_pool.free_blocks.assert_called_once()
    freed = sched.kv_cache_manager.block_pool.free_blocks.call_args.args[0]
    assert [block.block_id for block in freed] == [2, 4, 7]


def test_native_mooncake_deferred_ar_pages_drain_in_physical_order() -> None:
    from collections import deque

    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnector"))
    blocks = [SimpleNamespace(block_id=7), SimpleNamespace(block_id=2), SimpleNamespace(block_id=4)]
    sched.kv_cache_manager = SimpleNamespace(
        enable_caching=False,
        pop_blocks_for_free=MagicMock(return_value=blocks),
        block_pool=SimpleNamespace(free_blocks=MagicMock()),
    )
    sched.defer_block_free = True
    sched.processed_step_seq = 2
    sched.sched_step_seq = 3
    sched.deferred_frees = deque()
    request = SimpleNamespace(last_sched_seq=3)

    sched._free_request_blocks(request)
    assert [block.block_id for block in sched.deferred_frees[0][1]] == [7, 4, 2]

    sched.processed_step_seq = 3
    sched._drain_deferred_frees()
    freed = list(sched.kv_cache_manager.block_pool.free_blocks.call_args.args[0])
    assert [block.block_id for block in freed] == [2, 4, 7]
