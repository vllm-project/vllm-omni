# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused tests for the per-request key contract in minicpmo_4_5_omni_tts.

The MRv2 engine side addresses every per-request mapping by ``req_id`` (the
engine request id) while V1 buffers carry ``request_id``. Keying any map on
something else silently degrades to the batch row index, which is reused
whenever a slot is recycled, so the release path can never match the entry.
"""

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _request_key,
)


def test_request_key_prefers_engine_req_id():
    # MRv2 model_intermediate_buffer rows carry ``req_id`` (engine id).
    info = {"req_id": "7", "request_id": "stale"}
    assert _request_key(info, 0) == "7"


def test_request_key_falls_back_to_request_id():
    # V1 runtime_additional_information rows carry ``request_id`` only.
    info = {"request_id": "engine-9"}
    assert _request_key(info, 3) == "engine-9"


def test_request_key_falls_back_to_row_index():
    # No identifier at all: degrade to the batch row index (legacy behavior).
    assert _request_key({}, 5) == "5"
    assert _request_key(None, 2) == "2"


def test_request_key_stringifies_int_ids():
    # Engine ids may arrive as ints; keys must be strings so every dict
    # lookup agrees on one representation.
    assert _request_key({"req_id": 11}, 0) == "11"
