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

            # Store raw bytes instead of ndarray to avoid deserialization issues
            # msgspec doesn't know to convert list back to ndarray without type hint
            return {
                _PIL_IMAGE_MARKER: True,
                "mode": obj.mode,
                "size": obj.size,   # (W, H)
                "shape": list(arr.shape),  # Store shape separately
                "data": arr.tobytes(),  # Store as raw bytes
            }

        # ---- fallback to vLLM default ----
        return super().enc_hook(obj)


class OmniMsgpackDecoder(MsgpackDecoder):
    """Extended MsgpackDecoder with PIL.Image support."""

    def dec_hook(self, t: type, obj: Any) -> Any:
        # ---- PIL.Image reconstruction ----
        if isinstance(obj, dict) and obj.get(_PIL_IMAGE_MARKER):
            data = obj["data"]
            shape = obj.get("shape")
            mode = obj["mode"]

            # Reconstruct ndarray from raw bytes
            if isinstance(data, (bytes, memoryview)):
                arr = np.frombuffer(data, dtype=np.uint8)
                if shape:
                    arr = arr.reshape(shape)
            elif isinstance(data, np.ndarray):
                arr = data if data.dtype == np.uint8 else data.astype(np.uint8, copy=False)
            else:
                arr = np.array(data, dtype=np.uint8)
                if shape:
                    arr = arr.reshape(shape)

            try:
                return Image.fromarray(arr, mode=mode)
            except Exception:
                return Image.fromarray(arr)

        # ---- fallback ----
        return super().dec_hook(t, obj)


def _is_encoded_tensor(obj: Any) -> bool:
    """Check if obj looks like an encoded tensor: (dtype_str, shape_tuple, data)."""
    if not isinstance(obj, (list, tuple)) or len(obj) != 3:
        return False
    dtype_str, shape, data = obj
    if not isinstance(dtype_str, str):
        return False
    if not isinstance(shape, (list, tuple)):
        return False
    # Check shape contains only integers
    if not all(isinstance(s, int) for s in shape):
        return False
    # dtype_str should be a valid torch dtype name (without 'torch.' prefix)
    # Common dtypes: float32, float16, bfloat16, int64, int32, etc.
    valid_dtypes = {
        "float32", "float16", "bfloat16", "float64",
        "int64", "int32", "int16", "int8", "uint8", "bool",
    }
    if dtype_str not in valid_dtypes:
        return False
    # data should be bytes-like or memoryview or buffer index (int)
    if isinstance(data, (bytes, memoryview, bytearray, int)):
        return True
    return False


def _decode_tensor_from_encoded(obj: Any, aux_buffers: Any = None) -> torch.Tensor:
    """Decode an encoded tensor (dtype_str, shape_tuple, data) back to torch.Tensor.

    Args:
        obj: Encoded tensor as (dtype_str, shape_tuple, data)
        aux_buffers: Optional list of auxiliary buffers for multi-buffer decoding
    """
    dtype_str, shape, data = obj

    # Get the raw bytes
    if isinstance(data, int):
        # Buffer index - need aux_buffers
        if aux_buffers is None:
            raise ValueError("Buffer index requires aux_buffers")
        buffer = aux_buffers[data]
        if isinstance(buffer, memoryview):
            buffer = bytearray(buffer)
    elif isinstance(data, (bytes, memoryview, bytearray)):
        buffer = bytearray(data) if isinstance(data, memoryview) else data
    else:
        buffer = bytearray(data)

    torch_dtype = getattr(torch, dtype_str)
    if not buffer:
        return torch.empty(shape, dtype=torch_dtype)
    arr = torch.frombuffer(buffer, dtype=torch.uint8)
    return arr.view(torch_dtype).reshape(shape)


def _post_process_decoded(obj: Any, aux_buffers: Any = None) -> Any:
    """Recursively post-process decoded objects to restore PIL Images and tensors.

    msgspec's dec_hook is only called when a specific type is expected.
    When deserializing generic objects (dicts, lists), special encoded formats
    need to be manually converted after decoding.

    This handles:
    1. PIL Image marker dicts -> PIL.Image objects
    2. Encoded tensors (dtype, shape, data) -> torch.Tensor objects

    Args:
        obj: The decoded object to post-process
        aux_buffers: Optional list of auxiliary buffers for multi-buffer tensor decoding
    """
    if isinstance(obj, dict):
        # Check for PIL Image marker
        if obj.get(_PIL_IMAGE_MARKER):
            data = obj["data"]
            shape = obj.get("shape")
            mode = obj["mode"]

            # Reconstruct ndarray from raw bytes
            if isinstance(data, (bytes, memoryview)):
                arr = np.frombuffer(data, dtype=np.uint8)
                if shape:
                    arr = arr.reshape(shape)
            elif isinstance(data, np.ndarray):
                arr = data if data.dtype == np.uint8 else data.astype(np.uint8, copy=False)
            else:
                # Fallback for list (shouldn't happen with new format)
                arr = np.array(data, dtype=np.uint8)
                if shape:
                    arr = arr.reshape(shape)

            try:
                return Image.fromarray(arr, mode=mode)
            except Exception:
                return Image.fromarray(arr)
        # Recursively process dict values
        return {k: _post_process_decoded(v, aux_buffers) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        # Check if this looks like an encoded tensor
        if _is_encoded_tensor(obj):
            try:
                return _decode_tensor_from_encoded(obj, aux_buffers)
            except Exception:
                # If decoding fails, it might not be a tensor after all
                pass
        # Recursively process list/tuple elements
        processed = [_post_process_decoded(item, aux_buffers) for item in obj]
        return type(obj)(processed) if isinstance(obj, tuple) else processed

    return obj


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

        # Split data into buffers
        bufs: list[Any] = []
        offset = header_size
        for length in len_arr:
            bufs.append(data[offset : offset + length])
            offset += length

        if num_bufs == 1:
            result = decoder.decode(bufs[0])
        else:
            result = decoder.decode(bufs)

        # Post-process to restore PIL Images and tensors from their encoded forms
        # This is needed because dec_hook is only called for typed decoding,
        # but generic decoding returns raw dicts/lists
        if type_code == _TYPE_GENERIC:
            # Pass aux_buffers for multi-buffer tensor decoding
            result = _post_process_decoded(result, bufs if num_bufs > 1 else None)

        return result


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
