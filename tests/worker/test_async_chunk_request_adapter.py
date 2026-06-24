"""Unit coverage for the runner-side async-chunk request shim.

Covers the pure, GPU-free helpers that the AR runner uses to feed the
async-chunk stage-input processors from a worker-side ``CachedRequestState``:
``_strip_trailing_placeholder_tokens`` and ``_AsyncChunkRequestAdapter``.

The actual ``send_chunk`` call is gated by the async-chunk model-runner
transport predicate and exercised end-to-end by Qwen3 async-chunk tests; that
gate is covered separately in test_omni_scheduling_coordinator.py.
"""

from types import SimpleNamespace

from vllm_omni.worker.omni_connector_model_runner_mixin import (
    _AsyncChunkRequestAdapter,
    _strip_trailing_placeholder_tokens,
)


def test_strip_trailing_placeholder_tokens():
    assert _strip_trailing_placeholder_tokens(None) == []
    assert _strip_trailing_placeholder_tokens([]) == []
    assert _strip_trailing_placeholder_tokens([1, 2, 3]) == [1, 2, 3]
    # Only trailing -1 placeholders are dropped; interior values are kept.
    assert _strip_trailing_placeholder_tokens([1, 2, -1, -1]) == [1, 2]
    assert _strip_trailing_placeholder_tokens([-1, -1]) == []
    assert _strip_trailing_placeholder_tokens([5, -1, 6]) == [5, -1, 6]


def _cached_state():
    return SimpleNamespace(
        req_id="internal-1",
        prompt_token_ids=[10, 11, 12],
        output_token_ids=[20, 21, -1],
        additional_information="add-info-sentinel",
    )


def test_adapter_exposes_request_identity_fields():
    inner = _cached_state()
    adapter = _AsyncChunkRequestAdapter(inner, external_req_id="ext-99", finished=False)

    # external id is the passed cross-stage id; request_id/req_id mirror the inner req_id.
    assert adapter.external_req_id == "ext-99"
    assert adapter.request_id == "internal-1"
    assert adapter.req_id == "internal-1"


def test_adapter_all_token_ids_strips_placeholder_output():
    inner = _cached_state()
    adapter = _AsyncChunkRequestAdapter(inner, external_req_id="ext-99", finished=False)

    # output_token_ids drops the trailing -1; all_token_ids = prompt + stripped output.
    assert adapter.output_token_ids == [20, 21]
    assert adapter.prompt_token_ids == [10, 11, 12]
    assert adapter.all_token_ids == [10, 11, 12, 20, 21]


def test_adapter_is_finished_reflects_constructor_flag():
    inner = _cached_state()
    assert _AsyncChunkRequestAdapter(inner, external_req_id="e", finished=False).is_finished() is False
    assert _AsyncChunkRequestAdapter(inner, external_req_id="e", finished=True).is_finished() is True


def test_adapter_delegates_unknown_attrs_to_inner():
    inner = _cached_state()
    adapter = _AsyncChunkRequestAdapter(inner, external_req_id="e", finished=False)
    # additional_information (used by speaker/language extractors) is not a
    # declared property -> must delegate to the wrapped CachedRequestState.
    assert adapter.additional_information == "add-info-sentinel"


def test_adapter_handles_empty_prompt():
    inner = SimpleNamespace(req_id="r", prompt_token_ids=None, output_token_ids=[7])
    adapter = _AsyncChunkRequestAdapter(inner, external_req_id="e", finished=False)
    assert adapter.all_token_ids == [7]
