# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCore IPC coverage for runner-owned stage payload tensors."""

from __future__ import annotations

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.serialization import (
    deserialize_model_intermediate_buffer,
    serialize_model_intermediate_buffer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(buffer: dict) -> OmniEngineCoreRequest:
    return OmniEngineCoreRequest(
        request_id="req-tensor-handoff",
        prompt_token_ids=[0, 0],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        model_intermediate_buffer=serialize_model_intermediate_buffer(buffer),
    )


@pytest.mark.parametrize(
    "value",
    [
        torch.arange(24, dtype=torch.float32).reshape(4, 6),
        torch.arange(24, dtype=torch.float32).reshape(4, 6).transpose(0, 1),
        torch.empty((0, 8), dtype=torch.float32),
        torch.tensor(3.0, dtype=torch.float32),
    ],
    ids=["contiguous-2d", "non-contiguous-2d", "empty-2d", "scalar"],
)
def test_typed_engine_request_round_trip_restores_tensor(value: torch.Tensor) -> None:
    request = _request(
        {
            "ids": {"tts": [11, 12]},
            "hidden_states": {"tts": value},
            "legacy": [[1.0, 2.0]],
        }
    )

    encoded = MsgpackEncoder().encode(request)
    decoded = MsgpackDecoder(OmniEngineCoreRequest).decode(encoded)
    restored = deserialize_model_intermediate_buffer(decoded.model_intermediate_buffer)

    assert restored is not None
    assert restored["ids"]["tts"] == [11, 12]
    assert restored["legacy"] == [[1.0, 2.0]]
    hidden = restored["hidden_states"]["tts"]
    assert isinstance(hidden, torch.Tensor)
    assert hidden.dtype == value.dtype
    assert hidden.shape == value.shape
    assert hidden.is_contiguous()
    assert torch.equal(hidden, value)


def test_ordinary_marker_like_dictionary_is_left_unchanged() -> None:
    marker_like = {
        "__vllm_omni_model_buffer_tensor__": 1,
        "dtype": "float32",
        "shape": [1],
        "data": b"\x00\x00\x00\x00",
        "application_field": True,
    }

    assert deserialize_model_intermediate_buffer(marker_like) == marker_like


def test_corrupt_tensor_envelope_fails_instead_of_falling_back() -> None:
    corrupt = {
        "__vllm_omni_model_buffer_tensor__": 1,
        "dtype": "float32",
        "shape": [2],
        "data": b"\x00\x00\x00\x00",
    }

    with pytest.raises(ValueError, match="byte-size mismatch"):
        deserialize_model_intermediate_buffer(corrupt)


def test_sparse_tensor_fails_instead_of_changing_layout_semantics() -> None:
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]]),
        torch.tensor([1.0]),
        size=(2, 2),
    )

    with pytest.raises(TypeError, match="strided layout"):
        serialize_model_intermediate_buffer({"hidden": sparse})


def test_source_tensor_storage_is_not_shared_with_restored_request_value() -> None:
    source = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    restored = deserialize_model_intermediate_buffer(
        serialize_model_intermediate_buffer({"hidden": source})
    )

    assert restored is not None
    restored["hidden"][0, 0] = -1
    assert source[0, 0].item() == 0


def test_non_dictionary_top_level_fails_clearly() -> None:
    with pytest.raises(TypeError, match="must be a dictionary"):
        deserialize_model_intermediate_buffer([])  # type: ignore[arg-type]
