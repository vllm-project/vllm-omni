# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
from typing import Any, Sequence, Union

import numpy as np
import torch
from PIL import Image
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

# Marker for PIL.Image serialization
_PIL_IMAGE_MARKER = "__pil_image__"

# Valid torch dtype names (without 'torch.' prefix)
_VALID_TORCH_DTYPES = frozenset({
    "float32", "float16", "bfloat16", "float64",
    "int64", "int32", "int16", "int8", "uint8", "bool",
})


class OmniMsgpackEncoder(MsgpackEncoder):
    """Extended MsgpackEncoder with PIL.Image support."""

    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, Image.Image):
            arr = np.asarray(obj, dtype=np.uint8)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            return {
                _PIL_IMAGE_MARKER: True,
                "mode": obj.mode,
                "shape": list(arr.shape),
                "data": arr.tobytes(),
            }
        return super().enc_hook(obj)


class OmniMsgpackDecoder(MsgpackDecoder):
    """Extended MsgpackDecoder with post-processing for nested tensors and PIL Images.

    The base MsgpackDecoder only calls dec_hook when a specific type is expected.
    For generic objects (dicts containing tensors), we need post-processing.
    """

    def decode(self, bufs: Union[bytes, bytearray, memoryview, Sequence]) -> Any:
        """Decode with post-processing for nested structures."""
        # For multi-buffer case, save aux_buffers before parent clears them
        if isinstance(bufs, (list, tuple)) and len(bufs) > 1:
            saved_bufs = bufs
            result = super().decode(bufs)
            # Post-process with saved buffers (parent's finally block clears aux_buffers)
            return self._post_process(result, saved_bufs)
        else:
            result = super().decode(bufs)
            return self._post_process(result, None)

    def _post_process(self, obj: Any, aux_buffers: Sequence | None) -> Any:
        """Recursively restore tensors and PIL Images from their encoded forms."""
        if isinstance(obj, dict):
            if obj.get(_PIL_IMAGE_MARKER):
                return self._decode_pil_image(obj)
            return {k: self._post_process(v, aux_buffers) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            if self._is_encoded_tensor(obj):
                try:
                    return self._decode_tensor_from_list(obj, aux_buffers)
                except Exception:
                    pass  # Not a tensor, process as list
            result = [self._post_process(item, aux_buffers) for item in obj]
            return tuple(result) if isinstance(obj, tuple) else result

        return obj

    def _is_encoded_tensor(self, obj: Any) -> bool:
        """Check if obj is an encoded tensor: [dtype_str, shape_list, data]."""
        if not isinstance(obj, (list, tuple)) or len(obj) != 3:
            return False
        dtype_str, shape, data = obj
        return (
            isinstance(dtype_str, str)
            and dtype_str in _VALID_TORCH_DTYPES
            and isinstance(shape, (list, tuple))
            and all(isinstance(s, int) for s in shape)
            and isinstance(data, (bytes, memoryview, bytearray, int))
        )

    def _decode_tensor_from_list(self, arr: Any, aux_buffers: Sequence | None) -> torch.Tensor:
        """Decode [dtype_str, shape, data] to torch.Tensor.

        Mirrors vLLM's MsgpackDecoder._decode_tensor logic.
        """
        dtype, shape, data = arr

        # Get buffer - use passed aux_buffers for multi-buffer case
        if isinstance(data, int):
            if aux_buffers is None:
                raise ValueError("Buffer index requires aux_buffers")
            buffer = aux_buffers[data]
            # Copy to bytearray to make it writable
            buffer = bytearray(buffer)
        else:
            buffer = bytearray(data)

        torch_dtype = getattr(torch, dtype)
        if not buffer:
            return torch.empty(shape, dtype=torch_dtype)

        # Create uint8 array, then view as target dtype and shape
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        return arr.view(torch_dtype).view(shape)

    def _decode_pil_image(self, obj: dict) -> Image.Image:
        """Decode PIL Image marker dict to PIL.Image."""
        data, shape, mode = obj["data"], obj["shape"], obj["mode"]
        arr = np.frombuffer(data, dtype=np.uint8).reshape(shape)
        return Image.fromarray(arr, mode=mode)


class OmniSerde:
    """
    Serialization/deserialization handler for OmniConnectors.

    Similar to vLLM's MsgpackSerde but:
    - Uses struct instead of pickle for metadata
    - Supports PIL.Image
    - Post-processes nested structures to restore tensors
    """

    def __init__(self):
        self.encoder = OmniMsgpackEncoder()
        self.decoder = OmniMsgpackDecoder()

    def serialize(self, obj: Any) -> bytes:
        """
        Serialize an object to bytes.

        Args:
            obj: The object to serialize.

        Returns:
            bytes: Serialized data with header.
        """
        bufs = self.encoder.encode(obj)

        # Build header: num_bufs + lengths
        num_bufs = len(bufs)
        header = struct.pack("<I", num_bufs)
        for buf in bufs:
            header += struct.pack("<I", len(buf))

        # Concatenate header + all buffers
        if num_bufs == 1:
            buf = bufs[0]
            data = buf if isinstance(buf, bytes) else bytes(memoryview(buf))
            return header + data

        return header + b"".join(
            buf if isinstance(buf, bytes) else bytes(memoryview(buf))
            for buf in bufs
        )

    def serialize_with_metadata(self, obj: Any) -> tuple[Union[bytes, list[bytes]], int, bytes, int]:
        """
        Serialize an object to bytes with separate metadata.

        Args:
            obj: The object to serialize.

        Returns:
            tuple: (data, nbytes, metadata, metadata_len)
        """
        bufs = self.encoder.encode(obj)
        len_arr = [len(buf) for buf in bufs]
        nbytes = sum(len_arr)

        # Convert bufs to bytes
        if len(bufs) == 1:
            buf = bufs[0]
            data: Union[bytes, list[bytes]] = buf if isinstance(buf, bytes) else bytes(memoryview(buf))
        else:
            data = [buf if isinstance(buf, bytes) else bytes(memoryview(buf)) for buf in bufs]

        # Metadata: num_bufs + length of each buffer (using struct, not pickle)
        metadata = struct.pack("<I", len(bufs))
        for length in len_arr:
            metadata += struct.pack("<I", length)

        return data, nbytes, metadata, len(metadata)

    def deserialize(self, data: Union[bytes, memoryview]) -> Any:
        """
        Deserialize bytes to an object.

        Args:
            data: The bytes to deserialize.

        Returns:
            Deserialized object with tensors and PIL Images restored.
        """
        # Parse header
        num_bufs = struct.unpack("<I", data[:4])[0]
        header_size = 4 + num_bufs * 4
        lengths = struct.unpack(f"<{num_bufs}I", data[4:header_size])

        # Split into buffers
        bufs: list[Any] = []
        offset = header_size
        for length in lengths:
            bufs.append(data[offset:offset + length])
            offset += length

        # Decode (post-processing happens inside OmniMsgpackDecoder.decode)
        return self.decoder.decode(bufs[0] if num_bufs == 1 else bufs)


# Global instance for simple interface
OmniSerializer = OmniSerde()
