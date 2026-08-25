# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.engine.tensor_envelope import (
    build_inline_tensor_envelope,
    install_tensor_envelope,
    validate_inline_tensor_envelope,
)
from vllm_omni.experimental.fullduplex.engine.intermediate import get_tts_handoff

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _buffer(tensor: torch.Tensor) -> dict[str, object]:
    buffer: dict[str, object] = {
        "global_request_id": ["request-a"],
        "duplex": {"session_id": "session-a", "epoch": 3, "turn_id": 7},
        "ids": {"tts": [10, 11]},
        "hidden_states": {"tts": tensor},
    }
    install_tensor_envelope(
        buffer,
        name="hidden_states.tts",
        envelope=build_inline_tensor_envelope(
            tensor,
            request_id="request-a",
            payload_path="hidden_states.tts",
            session_id="session-a",
            epoch=3,
            chunk_id=7,
        ),
    )
    return buffer


def test_inline_tensor_envelope_validates_without_copy() -> None:
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    buffer = _buffer(tensor)

    resolved = validate_inline_tensor_envelope(
        buffer,
        name="hidden_states.tts",
        payload=tensor,
    )

    assert resolved is tensor
    envelope = buffer["meta"]["tensor_envelopes"]["hidden_states.tts"]
    assert envelope["session_id"] == "session-a"
    assert envelope["epoch"] == 3
    assert envelope["chunk_id"] == 7
    assert envelope["shape"] == [2, 4]
    assert envelope["dtype"] == "torch.float32"
    assert envelope["device"] == "cpu"
    token_ids, hidden_states = get_tts_handoff(buffer)
    assert token_ids == [10, 11]
    assert hidden_states is tensor


def test_inline_tensor_envelope_rejects_cross_request_or_shape_mismatch() -> None:
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    buffer = _buffer(tensor)

    buffer["global_request_id"] = ["request-b"]
    with pytest.raises(ValueError, match="request mismatch"):
        validate_inline_tensor_envelope(buffer, name="hidden_states.tts", payload=tensor)

    buffer["global_request_id"] = ["request-a"]
    buffer["duplex"]["epoch"] = 4
    with pytest.raises(ValueError, match="epoch mismatch"):
        validate_inline_tensor_envelope(buffer, name="hidden_states.tts", payload=tensor)

    buffer["duplex"]["epoch"] = 3
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_inline_tensor_envelope(
            buffer,
            name="hidden_states.tts",
            payload=torch.arange(4, dtype=torch.float32),
        )

    with pytest.raises(ValueError, match="dtype mismatch"):
        validate_inline_tensor_envelope(
            buffer,
            name="hidden_states.tts",
            payload=tensor.to(torch.float16),
        )


def test_legacy_list_and_unknown_future_handle_keep_fallback_payload() -> None:
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    buffer = _buffer(tensor)
    legacy = tensor.tolist()

    assert validate_inline_tensor_envelope(buffer, name="hidden_states.tts", payload=legacy) is legacy

    envelope = buffer["meta"]["tensor_envelopes"]["hidden_states.tts"]
    envelope["handle"]["kind"] = "npu_native_handle"
    assert validate_inline_tensor_envelope(buffer, name="hidden_states.tts", payload=tensor) is tensor


def test_inline_tensor_envelope_rejects_unknown_version_or_payload_path() -> None:
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    buffer = _buffer(tensor)
    envelope = buffer["meta"]["tensor_envelopes"]["hidden_states.tts"]

    envelope["version"] = 2
    with pytest.raises(ValueError, match="unsupported tensor envelope version"):
        validate_inline_tensor_envelope(buffer, name="hidden_states.tts", payload=tensor)

    envelope["version"] = 1
    envelope["handle"]["payload_path"] = "hidden_states.other"
    with pytest.raises(ValueError, match="payload path mismatch"):
        validate_inline_tensor_envelope(buffer, name="hidden_states.tts", payload=tensor)
