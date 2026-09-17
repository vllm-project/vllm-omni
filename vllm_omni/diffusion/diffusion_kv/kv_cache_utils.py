# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small diffusion identity helpers alongside vLLM's native KV cache utilities.

Content hashing runs before Scheduler admission, only with prefix caching on.
Requests carry native MultiModalFeatureSpec / PlaceholderRange values; native
generate_block_hash_extra_keys handles block intersections, with no per-token
extra-key expansion or intermediate dependency objects.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.multimodal.hasher import MultiModalHasher

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def _normalize_identity(value: object) -> object:
    # MultiModalHasher handles tensor bytes (including BF16). Its generic
    # recursive serialization alone does not canonicalize nested dict order
    # or distinguish every container/scalar boundary. A typed JSON envelope
    # supplies those guarantees and rejects the upstream pickle fallback.
    if value is None:
        return ["none"]
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, np.generic):
            # Extended-precision scalars have no lossless Python scalar;
            # item() returns another NumPy scalar instead of making progress.
            raise TypeError(f"Unsupported prefix-cache identity scalar dtype: {value.dtype}")
        return _normalize_identity(scalar)
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError("Prefix-cache identity requires a dense, non-quantized tensor")
        tensor = value.detach().cpu().contiguous()
        # A byte view also covers dtypes NumPy cannot represent directly.
        data = tensor.reshape(-1).view(torch.uint8).numpy()
        return ["tensor", str(tensor.dtype), list(tensor.shape), MultiModalHasher.hash_kwargs("sha256", data=data)]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise TypeError("Prefix-cache identity does not support object or structured arrays")
        return [
            "array",
            value.dtype.str,
            list(value.shape),
            MultiModalHasher.hash_kwargs("sha256", data=np.ascontiguousarray(value)),
        ]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Prefix-cache identity dictionaries require string keys")
        return ["dict", [[key, _normalize_identity(value[key])] for key in sorted(value)]]
    if isinstance(value, list | tuple):
        return ["sequence", [_normalize_identity(item) for item in value]]
    raise TypeError(f"Unsupported prefix-cache identity value: {type(value).__name__}")


def hash_prefix_cache_value(value: object) -> bytes:
    """Deterministic, type-framed identity using vLLM's multimodal hasher."""

    encoded = json.dumps(_normalize_identity(value), ensure_ascii=True, separators=(",", ":"))
    return bytes.fromhex(MultiModalHasher.hash_kwargs("sha256", identity=encoded))


def get_cache_namespace(model_namespace: str, sampling: OmniDiffusionSamplingParams) -> str:
    """Isolate model semantics and the adapter actually activated by Worker.

    Native LoRA block keys use lora_name, but diffusion also supports a scale
    and identifies loaded adapters by lora_int_id. Keep that difference here.
    Seeds are NOT global identity inputs: models attach random state only to
    the multimodal features whose computation depends on it.
    """

    if not isinstance(model_namespace, str) or not model_namespace:
        raise ValueError("Model cache namespace must be non-empty")
    lora = sampling.lora_request
    adapter = None if lora is None else (lora.lora_int_id, float(sampling.lora_scale))
    return hash_prefix_cache_value(("diffusion-prefix-identity-v2", model_namespace, adapter)).hex()
