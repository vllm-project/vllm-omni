# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_omni.core.sched.omni_ar_scheduler import (
    _get_request_num_cached_tokens,
    _get_request_num_external_computed_tokens,
    _make_engine_core_output,
)


def test_get_request_num_cached_tokens_defaults_to_zero_when_field_is_missing():
    request = SimpleNamespace(request_id="rid", num_computed_tokens=5)

    assert _get_request_num_cached_tokens(request) == 0


def test_get_request_num_external_computed_tokens_defaults_to_zero_when_field_is_missing():
    request = SimpleNamespace(request_id="rid")

    assert _get_request_num_external_computed_tokens(request) == 0


def test_make_engine_core_output_ignores_fields_missing_from_current_vllm():
    output = _make_engine_core_output(
        request_id="rid",
        new_token_ids=[1],
        finish_reason=None,
        new_logprobs=None,
        new_prompt_logprobs_tensors=None,
        pooling_output=None,
        stop_reason=None,
        events=[],
        kv_transfer_params=None,
        trace_headers={},
        num_cached_tokens=3,
        num_external_computed_tokens=4,
        routed_experts=None,
        num_nans_in_logits=0,
        is_segment_finished=None,
        new_prompt_len_snapshot=None,
    )

    assert output.request_id == "rid"
    assert output.new_token_ids == [1]


if __name__ == "__main__":
    test_get_request_num_cached_tokens_defaults_to_zero_when_field_is_missing()
    test_get_request_num_external_computed_tokens_defaults_to_zero_when_field_is_missing()
    test_make_engine_core_output_ignores_fields_missing_from_current_vllm()
