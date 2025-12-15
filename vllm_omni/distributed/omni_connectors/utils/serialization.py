# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
from typing import Any

import numpy as np
import torch

from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

# Type codes for serialization
_TYPE_GENERIC = 0
_TYPE_TENSOR = 1
_TYPE_NDARRAY = 2


class OmniSerializer:
    """
    Centralized serialization handler for OmniConnectors.

    Wraps vLLM's MsgpackEncoder/MsgpackDecoder for safe serialization.

    Wire format:
        [type: u8][num_bufs: u32][len_0: u32][len_1: u32]...[buf_0][buf_1]...
    """

    _encoder = MsgpackEncoder()
    _decoder = MsgpackDecoder()
    _tensor_decoder = MsgpackDecoder(torch.Tensor)
    _ndarray_decoder = MsgpackDecoder(np.ndarray)

    @staticmethod
    def serialize(obj: Any) -> bytes:
        """Serialize an object to bytes."""
        # Determine type code
        if isinstance(obj, torch.Tensor):
            type_code = _TYPE_TENSOR
        elif isinstance(obj, np.ndarray):
            type_code = _TYPE_NDARRAY
        else:
            type_code = _TYPE_GENERIC

        bufs = OmniSerializer._encoder.encode(obj)

        # Header: type_code + num_bufs + length of each buffer
        header = struct.pack("<BI", type_code, len(bufs))
        for buf in bufs:
            header += struct.pack("<I", len(buf))

        data = b"".join(
            buf if isinstance(buf, bytes) else bytes(memoryview(buf))  # type: ignore[arg-type]
            for buf in bufs
        )
        return header + data

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize bytes to an object."""
        # Parse header
        type_code, num_bufs = struct.unpack("<BI", data[:5])
        header_size = 5 + num_bufs * 4
        len_arr = struct.unpack(f"<{num_bufs}I", data[5:header_size])

        # Select decoder based on type
        if type_code == _TYPE_TENSOR:
            decoder = OmniSerializer._tensor_decoder
        elif type_code == _TYPE_NDARRAY:
            decoder = OmniSerializer._ndarray_decoder
        else:
            decoder = OmniSerializer._decoder

        if num_bufs == 1:
            return decoder.decode(data[header_size:])

        # Multi-buffer: split by lengths
        bufs = []
        offset = header_size
        for length in len_arr:
            bufs.append(data[offset : offset + length])
            offset += length
        return decoder.decode(bufs)
