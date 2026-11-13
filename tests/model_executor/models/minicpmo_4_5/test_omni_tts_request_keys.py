# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused tests for the per-request key contract in minicpmo_4_5_omni_tts.

The MRv2 engine side addresses every per-request mapping by ``req_id`` (the
engine request id) while V1 buffers carry ``request_id``. Keying any map on
something else silently degrades to the batch row index, which is reused
whenever a slot is recycled, so the release path can never match the entry.
"""

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _request_key,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


def _decode_preprocess_is_identity(k_frames):
    """Evaluate the flag the way the wrapper and the runner read it.

    ``minicpmo_4_5_omni.py`` copies it with ``bool(getattr(talker, ...))`` and
    the model state resolves it with ``bool(getattr(model, ...))``; both go
    through normal attribute access, i.e. through the property's ``fget``.
    """
    import types

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    prop = MiniCPMO45OmniTTSForConditionalGeneration.__dict__["mrv2_decode_preprocess_is_identity"]
    fake = types.SimpleNamespace(_k_step_frames=k_frames)
    return bool(prop.__get__(fake, MiniCPMO45OmniTTSForConditionalGeneration))


def test_decode_preprocess_is_identity_for_single_frame():
    # K=1: the runner's embedding already wrote the real codec token and
    # preprocess_decode_batch_mrv2 early-returns, so decode rows can be
    # skipped entirely by the model state.
    assert _decode_preprocess_is_identity(1) is True
    assert _decode_preprocess_is_identity(0) is True


def test_decode_preprocess_is_not_identity_for_kstep():
    # K>1: decode rows carry K-step placeholder codes that must be
    # re-embedded from the previous real codec frame every step.
    assert _decode_preprocess_is_identity(8) is False
