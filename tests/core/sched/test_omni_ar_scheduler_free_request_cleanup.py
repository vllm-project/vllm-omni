"""Regression tests for vllm-project/vllm-omni#5349 (P1): normal completion
goes through _free_request(), not finish_requests() (the external
abort/cancel entry point). Without a cleanup_receiver() call there,
_active_streams never releases on ordinary completion, so the bounded-K
window fills with stale entries after K completions.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / RequestStatus are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
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
