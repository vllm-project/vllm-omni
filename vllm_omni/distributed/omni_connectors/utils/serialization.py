# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict, is_dataclass
from typing import Any

import msgspec
import numpy as np
import torch
from msgspec import msgpack
from PIL import Image
from vllm.outputs import CompletionOutput, RequestOutput

# Type markers for custom serialization
_TENSOR_MARKER = "__tensor__"
_NDARRAY_MARKER = "__ndarray__"
_PIL_IMAGE_MARKER = "__pil_image__"

# Keys that identify a RequestOutput dict (for reconstruction)
_REQUEST_OUTPUT_KEYS = frozenset({"request_id", "prompt", "prompt_token_ids", "outputs", "finished"})

# Keys that identify a CompletionOutput dict (for reconstruction)
_COMPLETION_OUTPUT_KEYS = frozenset({"index", "text", "token_ids", "finish_reason"})

# Keys that identify an OmniRequestOutput dict (for reconstruction)
# OmniRequestOutput has 'final_output_type' which is unique, or can be identified by
# having 'finished' and ('images' or 'final_output_type')
_OMNI_REQUEST_OUTPUT_KEYS = frozenset({"finished", "final_output_type"})


class OmniMsgpackEncoder:
    """
    This implementation is adapted from vLLM’s MsgpackEncoder.
    However, zero-copy support has not been implemented yet.
    Handles torch.Tensor, numpy.ndarray, PIL.Image, RequestOutput and
    CompletionOutput by converting them to serializable dict representations.
    TODO: Enable zero-copy support.
    """

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self._enc_hook)

    def encode(self, obj: Any) -> bytes:
        """Encode an object to bytes."""
        return self.encoder.encode(obj)

    def _enc_hook(self, obj: Any) -> Any:
        """Custom encoding hook for non-standard types."""
        # torch.Tensor
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # numpy.ndarray (exclude object/void dtypes)
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        # PIL.Image
        if isinstance(obj, Image.Image):
            return self._encode_pil_image(obj)

        # RequestOutput (not a dataclass, needs special handling)
        if isinstance(obj, RequestOutput):
            return self._encode_request_output(obj)

        # CompletionOutput (dataclass)
        if isinstance(obj, CompletionOutput):
            return self._encode_completion_output(obj)

        # Other dataclasses
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)

        # slice
        if isinstance(obj, slice):
            return (obj.start, obj.stop, obj.step)

        raise TypeError(
            f"Object of type {type(obj).__name__} is not serializable. "
            "Supported types: torch.Tensor, np.ndarray, PIL.Image, dataclass, "
            "RequestOutput, and standard Python types (dict, list, str, int, float, bool, None, bytes)."
        )

    def _encode_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        """Encode torch.Tensor to dict."""
        t = tensor.detach().contiguous().cpu()
        # Handle 0-dimensional (scalar) tensors by reshaping to 1D first
        if t.dim() == 0:
            t = t.reshape(1)
        t = t.view(torch.uint8)
        return {
            _TENSOR_MARKER: True,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": t.numpy().tobytes(),
        }

    def _encode_ndarray(self, arr: np.ndarray) -> dict[str, Any]:
        """Encode numpy.ndarray to dict."""
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return {
            _NDARRAY_MARKER: True,
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "data": arr.tobytes(),
        }

    def _encode_pil_image(self, img: Image.Image) -> dict[str, Any]:
        """Encode PIL.Image to dict."""
        arr = np.asarray(img, dtype=np.uint8)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return {
            _PIL_IMAGE_MARKER: True,
            "mode": img.mode,
            "shape": list(arr.shape),
            "data": arr.tobytes(),
        }

    def _encode_request_output(self, obj: RequestOutput) -> dict[str, Any]:
        """Encode RequestOutput to dict.

        RequestOutput is not a dataclass, so we manually extract its attributes.
        Also handles dynamically added 'multimodal_output' attribute.
        """
        # msgspec can serialize CompletionOutput dataclasses directly, but it
        # drops dynamic fields such as multimodal_output. Encode them manually
        # to preserve multimodal payloads across IPC.
        encoded_outputs = []
        for o in obj.outputs:
            if isinstance(o, CompletionOutput):
                encoded_outputs.append(self._encode_completion_output(o))
            else:
                encoded_outputs.append(o)

        result = {
            "request_id": obj.request_id,
            "prompt": obj.prompt,
            "prompt_token_ids": obj.prompt_token_ids,
            "prompt_logprobs": obj.prompt_logprobs,
            "outputs": encoded_outputs,
            "finished": obj.finished,
            "metrics": obj.metrics,
            "lora_request": obj.lora_request,
            "encoder_prompt": obj.encoder_prompt,
            "encoder_prompt_token_ids": obj.encoder_prompt_token_ids,
            "num_cached_tokens": obj.num_cached_tokens,
            "multi_modal_placeholders": obj.multi_modal_placeholders,
            "kv_transfer_params": obj.kv_transfer_params,
        }
        # Handle dynamically added multimodal_output attribute
        mm_output = getattr(obj, "multimodal_output", None)
        if mm_output is not None:
            result["multimodal_output"] = mm_output
        return result

    def _encode_completion_output(self, obj: CompletionOutput) -> dict[str, Any]:
        """Encode CompletionOutput to dict, preserving multimodal payloads."""
        result = asdict(obj)
        mm_output = getattr(obj, "multimodal_output", None)
        if mm_output is not None:
            result["multimodal_output"] = mm_output
        return result


class OmniMsgpackDecoder:
    """
    This implementation is adapted from vLLM’s MsgpackDecoder.
    However, zero-copy support has not been implemented yet.

    Automatically reconstructs torch.Tensor, numpy.ndarray, PIL.Image,
    RequestOutput and CompletionOutput from their dict representations.
    TODO: Enable zero-copy support.
    """

    def __init__(self):
        self.decoder = msgpack.Decoder()

    def decode(self, data: bytes | bytearray | memoryview) -> Any:
        """Decode bytes to object."""
        result = self.decoder.decode(data)
        return self._post_process(result)

    def _post_process(self, obj: Any) -> Any:
        """Recursively restore tensor/ndarray/image/RequestOutput/OmniRequestOutput from their dict representations."""
        if isinstance(obj, dict):
            # Check for type markers first
            if obj.get(_TENSOR_MARKER):
                return self._decode_tensor(obj)
            if obj.get(_NDARRAY_MARKER):
                return self._decode_ndarray(obj)
            if obj.get(_PIL_IMAGE_MARKER):
                return self._decode_pil_image(obj)

            # Process values recursively first
            processed = {k: self._post_process(v) for k, v in obj.items()}

            # Check if this looks like an OmniRequestOutput (check before RequestOutput
            # since OmniRequestOutput may also have some RequestOutput-like fields)
            if self._is_omni_request_output(processed):
                return self._decode_omni_request_output(processed)

            # Check if this looks like a RequestOutput
            if _REQUEST_OUTPUT_KEYS.issubset(processed.keys()):
                return self._decode_request_output(processed)

            # Check if this looks like a CompletionOutput
            if _COMPLETION_OUTPUT_KEYS.issubset(processed.keys()):
                return self._decode_completion_output(processed)

            return processed

        if isinstance(obj, list):
            return [self._post_process(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(self._post_process(item) for item in obj)

        return obj

    def _is_omni_request_output(self, obj: dict[str, Any]) -> bool:
        """Check if a dict looks like an OmniRequestOutput.

        OmniRequestOutput can be identified by:
        - Having 'finished' and 'final_output_type' fields (unique to OmniRequestOutput)
        - OR having 'finished' and 'images' fields (diffusion mode)
        """
        # Must have 'finished' field
        if "finished" not in obj:
            return False

        # Check for unique identifier: 'final_output_type'
        if "final_output_type" in obj:
            return True

        # Alternative: check for 'images' field (diffusion mode)
        if "images" in obj:
            return True

        return False

    def _decode_omni_request_output(self, obj: dict[str, Any]) -> Any:
        """Decode dict to OmniRequestOutput.

        OmniRequestOutput is a dataclass, so we can use msgspec.convert
        or construct it directly.
        """
        from vllm_omni.outputs import OmniRequestOutput

        try:
            # Use msgspec.convert for dataclass reconstruction
            return msgspec.convert(obj, OmniRequestOutput)
        except Exception:
            try:
                # Fallback: construct directly if msgspec.convert fails
                # (e.g., if some fields are missing or have wrong types)
                return OmniRequestOutput(**obj)
            except Exception:
                # If both attempts fail, return dict as-is (defensive fallback)
                # This should rarely happen if _is_omni_request_output is correct
                return obj

    def _decode_tensor(self, obj: dict[str, Any]) -> torch.Tensor:
        """Decode dict to torch.Tensor."""
        dtype_str = obj["dtype"]
        shape = obj["shape"]
        data = obj["data"]

        torch_dtype = getattr(torch, dtype_str)
        if not data:
            return torch.empty(shape, dtype=torch_dtype)

        buffer = bytearray(data) if isinstance(data, (bytes, memoryview)) else data
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        return arr.view(torch_dtype).reshape(shape)

    def _decode_ndarray(self, obj: dict[str, Any]) -> np.ndarray:
        """Decode dict to numpy.ndarray."""
        dtype = obj["dtype"]
        shape = obj["shape"]
        data = obj["data"]
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def _decode_pil_image(self, obj: dict[str, Any]) -> Image.Image:
        """Decode dict to PIL.Image."""
        mode = obj["mode"]
        shape = obj["shape"]
        data = obj["data"]
        arr = np.frombuffer(data, dtype=np.uint8).reshape(shape)
        return Image.fromarray(arr, mode=mode)

    def _decode_completion_output(self, obj: dict[str, Any]) -> CompletionOutput:
        """Decode dict to CompletionOutput using msgspec.convert."""
        mm_output = obj.pop("multimodal_output", None)
        co = msgspec.convert(obj, CompletionOutput)
        if mm_output is not None:
            setattr(co, "multimodal_output", mm_output)
        return co

    def _decode_request_output(self, obj: dict[str, Any]) -> RequestOutput:
        """Decode dict to RequestOutput.

        RequestOutput is not a dataclass, so msgspec.convert doesn't work.
        We construct it manually, passing all known fields via **kwargs.
        """
        # Extract multimodal_output before constructing (it's dynamically added)
        mm_output = obj.pop("multimodal_output", None)

        # RequestOutput.__init__ accepts **kwargs for forward compatibility
        ro = RequestOutput(**obj)

        # Restore dynamically added multimodal_output attribute
        if mm_output is not None:
            setattr(ro, "multimodal_output", mm_output)
        return ro


class OmniSerde:
    """Serialization/deserialization handler for Omni IPC."""

    def __init__(self):
        self.encoder = OmniMsgpackEncoder()
        self.decoder = OmniMsgpackDecoder()

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes."""
        return self.encoder.encode(obj)

    def deserialize(self, data: bytes | bytearray | memoryview) -> Any:
        """Deserialize bytes to an object."""
        return self.decoder.decode(data)


# Global instance for simple interface
OmniSerializer = OmniSerde()
