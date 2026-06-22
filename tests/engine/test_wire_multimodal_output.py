# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Msgpack roundtrip tests for OmniEngineCoreOutputs with multimodal_output.

Validates that tensor-only payloads survive msgspec encode/decode and that
non-tensor values fail decoding (enforcing the wire invariant).
"""

import pytest
import torch
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _roundtrip(outputs: OmniEngineCoreOutputs) -> OmniEngineCoreOutputs:
    encoder = MsgpackEncoder()
    encoded = encoder.encode(outputs)
    decoder = MsgpackDecoder(OmniEngineCoreOutputs)
    return decoder.decode(encoded)


def test_tensor_only_roundtrip():
    """Tensor-only multimodal_output survives msgpack roundtrip."""
    audio = torch.randn(16000)
    sr = torch.tensor(24000)
    eco = OmniEngineCoreOutput(
        request_id="req-1",
        new_token_ids=[1, 2, 3],
        finish_reason=None,
        multimodal_output={"audio": audio, "sr": sr},
    )
    decoded = _roundtrip(OmniEngineCoreOutputs(outputs=[eco]))
    assert len(decoded.outputs) == 1
    out = decoded.outputs[0]
    assert out.request_id == "req-1"
    assert isinstance(out.multimodal_output, dict)
    assert torch.allclose(out.multimodal_output["audio"], audio)
    assert out.multimodal_output["sr"].item() == 24000


def test_empty_multimodal_roundtrip():
    """None multimodal_output roundtrips correctly."""
    eco = OmniEngineCoreOutput(
        request_id="req-2",
        new_token_ids=[4],
        finish_reason=None,
        multimodal_output=None,
    )
    decoded = _roundtrip(OmniEngineCoreOutputs(outputs=[eco]))
    assert decoded.outputs[0].multimodal_output is None


def test_multiple_tensor_keys_roundtrip():
    """Multiple tensor keys survive roundtrip."""
    hidden = torch.randn(10, 4096)
    codes = torch.randint(0, 1024, (1, 100))
    eco = OmniEngineCoreOutput(
        request_id="req-3",
        new_token_ids=[],
        finish_reason=None,
        multimodal_output={"hidden": hidden, "codes": codes},
    )
    decoded = _roundtrip(OmniEngineCoreOutputs(outputs=[eco]))
    out = decoded.outputs[0]
    assert torch.allclose(out.multimodal_output["hidden"], hidden)
    assert torch.equal(out.multimodal_output["codes"], codes)


def test_non_tensor_value_fails_decode():
    """Non-tensor values in multimodal_output cause decode failure.

    This validates that the wire field must be tensor-only and that
    the _ensure_tensor_values boundary in model runners is necessary.
    """
    eco = OmniEngineCoreOutput(
        request_id="req-bad",
        new_token_ids=[],
        finish_reason=None,
        multimodal_output={"audio": torch.randn(10), "sr": 24000},  # scalar!
    )
    encoder = MsgpackEncoder()
    encoded = encoder.encode(OmniEngineCoreOutputs(outputs=[eco]))
    decoder = MsgpackDecoder(OmniEngineCoreOutputs)
    with pytest.raises(Exception):
        decoder.decode(encoded)
