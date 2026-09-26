# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Dispatch for advancing a request's grammar with just-sampled tokens.

``accept_structured_output_tokens`` must work against both the released
``should_advance`` + per-request ``grammar`` layout and the newer
``StructuredOutputManager.accept_tokens`` layout.

Two properties are load-bearing and pinned here:

* it is a module-level function, so scheduler stubs that call
  ``update_from_output`` unbound do not have to grow an attribute for it;
* it dispatches on the manager *class*, because an ``getattr`` on a ``MagicMock``
  instance is never ``None`` and would make the older branch unreachable.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.core.sched.omni_scheduler_mixin import (
    OmniSchedulerMixin,
    accept_structured_output_tokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request():
    grammar = MagicMock()
    grammar.accept_tokens.return_value = True
    return SimpleNamespace(
        request_id="req-0",
        structured_output_request=SimpleNamespace(grammar=grammar),
    )


class _NewLayoutManager:
    """Manager whose class advertises ``accept_tokens``."""

    def __init__(self, accepted: bool = True):
        self.accepted = accepted
        self.calls: list[tuple[str, list[int]]] = []

    def accept_tokens(self, request, token_ids) -> bool:
        self.calls.append((request.request_id, list(token_ids)))
        return self.accepted

    def should_advance(self, request) -> bool:  # pragma: no cover - must not be used
        raise AssertionError("the newer layout must not gate on should_advance")


def test_helper_is_module_level_not_a_mixin_method():
    # A mixin method disappears when tests call ``update_from_output`` unbound
    # on a stub namespace; a module-level helper is reachable from both callers.
    assert not hasattr(OmniSchedulerMixin, "_accept_structured_output_tokens")
    assert callable(accept_structured_output_tokens)


def test_manager_without_accept_tokens_gates_on_should_advance():
    request = _request()
    manager = MagicMock()
    manager.should_advance.return_value = False

    assert accept_structured_output_tokens(manager, request, [7]) is True
    manager.should_advance.assert_called_once_with(request, new_token_ids=[7])
    request.structured_output_request.grammar.accept_tokens.assert_not_called()


def test_manager_without_accept_tokens_accepts_through_the_grammar():
    request = _request()
    manager = MagicMock()
    manager.should_advance.return_value = True
    manager.trim_reasoning_for_advance.side_effect = lambda request, tokens: tokens

    assert accept_structured_output_tokens(manager, request, [7, 8]) is True
    request.structured_output_request.grammar.accept_tokens.assert_called_once_with("req-0", [7, 8])


def test_manager_without_accept_tokens_reports_a_grammar_rejection():
    request = _request()
    request.structured_output_request.grammar.accept_tokens.return_value = False
    manager = MagicMock()
    manager.should_advance.return_value = True
    manager.trim_reasoning_for_advance.side_effect = lambda request, tokens: tokens

    assert accept_structured_output_tokens(manager, request, [7]) is False


def test_manager_class_with_accept_tokens_is_preferred():
    request = _request()
    manager = _NewLayoutManager()

    assert accept_structured_output_tokens(manager, request, [1, 2]) is True
    assert manager.calls == [("req-0", [1, 2])]
    request.structured_output_request.grammar.accept_tokens.assert_not_called()


def test_manager_class_with_accept_tokens_reports_rejection():
    request = _request()
    manager = _NewLayoutManager(accepted=False)

    assert accept_structured_output_tokens(manager, request, [3]) is False


@pytest.mark.parametrize("tokens,suffix", [([10, 99, 123], [123]), ([10, 99], []), ([10], None)])
def test_029_reasoning_boundary_through_ar_scheduler(tokens, suffix, monkeypatch):
    from vllm.v1.request import RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager

    from tests.core.sched.test_omni_ar_scheduler_logprobs import (
        _bind_request_lifecycle,
        _make_scheduler_stub,
        _Request,
    )
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    if hasattr(StructuredOutputManager, "accept_tokens"):
        pytest.skip("Exercises the released vLLM 0.29 manager")

    class Reasoner:
        def is_reasoning_end_streaming(self, all_token_ids, delta_ids):
            return 99 in delta_ids

    class Grammar:
        def __init__(self):
            self.accepted: list[list[int]] = []

        def accept_tokens(self, request_id, token_ids):
            self.accepted.append(list(token_ids))
            return token_ids == [123]

    manager = StructuredOutputManager.__new__(StructuredOutputManager)
    manager.enable_in_reasoning = False
    monkeypatch.setattr(manager, "_get_reasoner", lambda request: Reasoner())

    class ReasoningRequest(_Request):
        use_structured_output: bool
        all_token_ids: list[int]
        structured_output_request: SimpleNamespace

    request = ReasoningRequest("req")
    request.sampling_params.num_logprobs = None
    request.use_structured_output = True
    request.all_token_ids = [1, 2]
    request.num_computed_tokens = 8
    request.num_output_placeholders = 2
    grammar = Grammar()
    request.structured_output_request = SimpleNamespace(
        grammar=grammar, reasoning_ended=False, reasoning_end_token_index=None
    )
    scheduler = _make_scheduler_stub([request])
    scheduler.structured_output_manager = manager

    def append_tokens(request, new_token_ids):
        request.all_token_ids.extend(new_token_ids)
        return new_token_ids, False

    _bind_request_lifecycle(scheduler, update_request=append_tokens)
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"req": len(tokens)}, scheduled_spec_decode_tokens={}, num_invalid_spec_tokens=0
    )
    sampled = SimpleNamespace(
        sampled_token_ids=[tokens],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={"req": 0},
        routed_experts=None,
    )
    outputs = OmniARScheduler.update_from_output(scheduler, scheduled, sampled)
    assert request.status is RequestStatus.RUNNING
    assert outputs[0].outputs[0].new_token_ids == tokens
    assert grammar.accepted == ([suffix] if suffix else [])
    assert request.structured_output_request.reasoning_ended is (suffix is not None)
