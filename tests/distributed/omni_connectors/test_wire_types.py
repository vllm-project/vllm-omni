# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct

import msgspec
import pytest
import torch

from vllm_omni.distributed.omni_connectors.utils.wire_types import (
    TENSOR_DICT_COMPRESSION_NONE,
    deserialize_tensor_dict,
    is_tensor_dict,
    serialize_tensor_dict,
)

pytestmark = [pytest.mark.cpu, pytest.mark.parallel, pytest.mark.core_model]


def test_is_tensor_dict_requires_string_tensor_items():
    assert is_tensor_dict({"tokens": torch.arange(4)})
    assert not is_tensor_dict({"tokens": [1, 2, 3]})
    assert not is_tensor_dict({1: torch.arange(4)})
    assert not is_tensor_dict([torch.arange(4)])


def test_tensor_dict_wire_format_round_trip():
    payload = {
        "tokens": torch.arange(12, dtype=torch.int64).reshape(3, 4),
        "codes": torch.arange(8, dtype=torch.int32)[::2].contiguous(),
        "scores": torch.tensor([[1.25, 2.5]], dtype=torch.float32),
        "scalar": torch.tensor(7, dtype=torch.int64),
        "empty": torch.empty((0, 3), dtype=torch.float16),
    }

    restored = deserialize_tensor_dict(serialize_tensor_dict(payload))

    assert restored.keys() == payload.keys()
    for key, expected in payload.items():
        assert restored[key].dtype == expected.dtype
        assert restored[key].shape == expected.shape
        assert torch.equal(restored[key], expected.cpu())


def test_tensor_dict_wire_format_rejects_unknown_compression():
    encoded = bytearray(serialize_tensor_dict({"tokens": torch.arange(4, dtype=torch.int64)}))

    prefix = struct.Struct("<4sII")
    _magic, _version, header_len = prefix.unpack(encoded[: prefix.size])
    header_start = prefix.size
    header_end = header_start + header_len
    header = msgspec.msgpack.decode(encoded[header_start:header_end])
    header["entries"][0]["compression"] = TENSOR_DICT_COMPRESSION_NONE + 1
    new_header = msgspec.msgpack.encode(header)
    encoded[: prefix.size] = prefix.pack(b"OMTD", 1, len(new_header))
    encoded[header_start:header_end] = new_header

    with pytest.raises(ValueError, match="Unsupported tensor-dict compression"):
        deserialize_tensor_dict(encoded)
