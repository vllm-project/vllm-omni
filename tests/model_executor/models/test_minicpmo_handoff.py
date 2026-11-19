# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
from vllm_omni.model_executor.models.minicpmo_4_5.handoff import pack_handoff_tensor, unpack_handoff_tensor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("shape", [(), (0, 4), (3, 4), (144000,)])
def test_packed_handoff_matches_legacy_values_across_untyped_wire(dtype, shape):
    source = (
        torch.linspace(-1.23456789, 1.987654321, int(torch.tensor(shape).prod().item()), dtype=torch.float64)
        .reshape(shape)
        .to(dtype)
    )
    if source.ndim == 2:
        source = source.t()
    expected = torch.tensor(source.tolist(), dtype=torch.float32).reshape(source.shape)
    packed = pack_handoff_tensor(source)
    envelope = {"model_intermediate_buffer": {"hidden_states": {"tts": packed}}}
    # The real request field is dict[str, Any]: raw tensors cannot rely on a
    # typed decoder hook here. Only standard bytes cross this boundary.
    received = MsgpackDecoder(dict[str, Any]).decode(MsgpackEncoder().encode(envelope))
    actual = unpack_handoff_tensor(received["model_intermediate_buffer"]["hidden_states"]["tts"])
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == source.shape
    assert torch.equal(actual, expected)
    source.fill_(-7)
    assert torch.equal(actual, expected)
    other = unpack_handoff_tensor(packed)
    actual.fill_(9)
    assert torch.equal(other, expected)


def test_legacy_handoff_values_remain_supported():
    for value in (None, [1.0, 2.0], torch.ones(2)):
        assert unpack_handoff_tensor(value) is value


@pytest.mark.parametrize("payload", ["not a tensor", [1.0], torch.ones(2, dtype=torch.int64)])
def test_invalid_packed_type_rejected(payload):
    with pytest.raises(ValueError, match="CPU float32 tensor"):
        unpack_handoff_tensor(OmniSerializer.serialize(payload))
