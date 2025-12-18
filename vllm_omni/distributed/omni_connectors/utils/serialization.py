# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
from typing import Any

import numpy as np
import torch
from PIL import Image
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

# Type codes for serialization
_TYPE_GENERIC = 0
_TYPE_TENSOR = 1
_TYPE_NDARRAY = 2

# Marker for PIL.Image serialization
_PIL_IMAGE_MARKER = "__pil_image__"


class OmniMsgpackEncoder(MsgpackEncoder):
    """Extended MsgpackEncoder with PIL.Image support."""

    def enc_hook(self, obj: Any) -> Any:
        # ---- PIL.Image support ----
        if isinstance(obj, Image.Image):
            # Force safe, contiguous uint8 ndarray
            arr = np.asarray(obj, dtype=np.uint8)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)

            return {
                _PIL_IMAGE_MARKER: True,
                "mode": obj.mode,
                "size": obj.size,   # (W, H)
                "data": arr,
            }

        # ---- fallback to vLLM default ----
        return super().enc_hook(obj)


class OmniMsgpackDecoder(MsgpackDecoder):
    """Extended MsgpackDecoder with PIL.Image support."""

    def dec_hook(self, t: type, obj: Any) -> Any:
        # ---- PIL.Image reconstruction ----
        if isinstance(obj, dict) and obj.get(_PIL_IMAGE_MARKER):
            data = obj["data"]

            # Safety: force uint8 ndarray
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.uint8)
            elif data.dtype != np.uint8:
                data = data.astype(np.uint8, copy=False)

            try:
                return Image.fromarray(data, mode=obj["mode"])
            except Exception:
                # Fallback: let PIL infer mode
                return Image.fromarray(data)

        # ---- fallback ----
        return super().dec_hook(t, obj)


class OmniSerde:
    """
    Serialization/deserialization handler for OmniConnectors.
    Wraps vLLM's MsgpackEncoder/MsgpackDecoder for safe serialization.

    Similar to vLLM's MsgpackSerde but uses struct instead of pickle for metadata.
    """

    def __init__(self):
        self.encoder = OmniMsgpackEncoder()
        self.decoder = OmniMsgpackDecoder()
        self.tensor_decoder = OmniMsgpackDecoder(torch.Tensor)
        self.ndarray_decoder = OmniMsgpackDecoder(np.ndarray)

    def serialize(self, obj: Any) -> tuple[bytes | list[bytes], int, bytes, int]:
        """
        Serialize an object to bytes.

        Args:
            obj: The object to serialize.

        Returns:
            tuple: (data, nbytes, metadata, metadata_len)
                - data: Serialized data (bytes or list of bytes for multi-buffer)
                - nbytes: Total size of serialized data
                - metadata: Header containing type and length info
                - metadata_len: Size of metadata
        """

        # Determine type code
        if isinstance(obj, torch.Tensor):
            type_code = _TYPE_TENSOR
        elif isinstance(obj, np.ndarray):
            type_code = _TYPE_NDARRAY
        else:
            type_code = _TYPE_GENERIC

        bufs = self.encoder.encode(obj)
        len_arr = [len(buf) for buf in bufs]  # type: ignore[arg-type]
        nbytes = sum(len_arr)

        # Convert bufs to bytes
        data: bytes | list[bytes]
        if len(bufs) == 1:
            buf = bufs[0]
            data = buf if isinstance(buf, bytes) else bytes(memoryview(buf))  # type: ignore[arg-type]
        else:
            data = [
                buf if isinstance(buf, bytes) else bytes(memoryview(buf))  # type: ignore[arg-type]
                for buf in bufs
            ]

        # Metadata: type_code + num_bufs + length of each buffer
        metadata = struct.pack("<BI", type_code, len(bufs))
        for length in len_arr:
            metadata += struct.pack("<I", length)

        return data, nbytes, metadata, len(metadata)

    def deserialize(self, data: bytes | memoryview) -> Any:
        """
        Deserialize bytes to an object.

        Args:
            data: The bytes to deserialize (can be memoryview for zero-copy).

        Returns:
            Deserialized object.
        """
        # Parse header
        type_code, num_bufs = struct.unpack("<BI", data[:5])
        header_size = 5 + num_bufs * 4
        len_arr = struct.unpack(f"<{num_bufs}I", data[5:header_size])

        # Select decoder based on type
        if type_code == _TYPE_TENSOR:
            decoder = self.tensor_decoder
        elif type_code == _TYPE_NDARRAY:
            decoder = self.ndarray_decoder
        else:
            decoder = self.decoder

        if num_bufs == 1:
            return decoder.decode(data[header_size:])

        # Multi-buffer: split by lengths
        bufs = []
        offset = header_size
        for length in len_arr:
            bufs.append(data[offset : offset + length])
            offset += length
        return decoder.decode(bufs)


class OmniSerializer:
    """
    Simple serialization interface for OmniConnectors.
    Wraps OmniSerde with a simpler bytes-in/bytes-out interface.
    """

    _serde = OmniSerde()

    @staticmethod
    def serialize(obj: Any) -> bytes:
        """Serialize an object to bytes."""
        data, _, metadata, _ = OmniSerializer._serde.serialize(obj)

        if isinstance(data, bytes):
            return metadata + data

        # Multi-buffer: concatenate all buffers
        return metadata + b"".join(data)  # type: ignore[arg-type]

    @staticmethod
    def deserialize(data: bytes | memoryview) -> Any:
        """Deserialize bytes to an object."""
        return OmniSerializer._serde.deserialize(data)
