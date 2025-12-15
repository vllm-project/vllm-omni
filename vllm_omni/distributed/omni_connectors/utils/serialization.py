# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
from typing import Any

from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


class OmniSerializer:
    """
    Centralized serialization handler for OmniConnectors.

    Wraps vLLM's MsgpackEncoder/MsgpackDecoder for safe serialization.

    Wire format:
        [num_bufs: u32][len_0: u32][len_1: u32]...[buf_0][buf_1]...
    """

    _encoder = MsgpackEncoder()
    _decoder = MsgpackDecoder()

    @staticmethod
    def serialize(obj: Any) -> bytes:
        """Serialize an object to bytes."""
        bufs = OmniSerializer._encoder.encode(obj)

        # Header: num_bufs + length of each buffer
        header = struct.pack("<I", len(bufs))
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
        (num_bufs,) = struct.unpack("<I", data[:4])
        header_size = 4 + num_bufs * 4
        len_arr = struct.unpack(f"<{num_bufs}I", data[4:header_size])

        if num_bufs == 1:
            return OmniSerializer._decoder.decode(data[header_size:])

        # Multi-buffer: split by lengths
        bufs = []
        offset = header_size
        for length in len_arr:
            bufs.append(data[offset : offset + length])
            offset += length
        return OmniSerializer._decoder.decode(bufs)
