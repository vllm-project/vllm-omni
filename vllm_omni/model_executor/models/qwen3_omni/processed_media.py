# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Retain actual preprocessing identity in the existing processor cache.

Only fresh, CPU processor outputs are hashed. Cache hits (including sender-cache
hits that omit tensors) reuse the proof on their own resolved prompt updates.
There is no second cache, GPU transfer, or hashing of Thinker hidden states.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256

import numpy as np
import torch
from vllm.inputs import MultiModalHashes
from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange
from vllm.multimodal.processing.processor import MultiModalPromptUpdates, ResolvedPromptUpdate
from vllm.multimodal.utils import argsort_mm_positions

from vllm_omni.inputs.processed_media import ProcessedMediaIdentity, ProcessedMediaProvenance


def _cpu_value(value: object) -> object:
    """Unambiguous, order-independent description; never move a device tensor."""
    if isinstance(value, Enum):
        return ["enum", f"{type(value).__module__}.{type(value).__qualname__}", _cpu_value(value.value)]
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("media provenance requires CPU strided processor outputs")
        data = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        return ["torch", str(value.dtype), list(value.shape), sha256(memoryview(data)).hexdigest()]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("media provenance cannot hash object arrays")
        data = np.ascontiguousarray(value).reshape(-1).view(np.uint8)
        return ["numpy", value.dtype.str, list(value.shape), sha256(memoryview(data)).hexdigest()]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("media provenance requires string dictionary keys")
        return ["dict", [[key, _cpu_value(value[key])] for key in sorted(value)]]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_cpu_value(item) for item in value]]
    if value is None or type(value) in (str, int, float, bool):
        return [type(value).__name__, value]
    raise ValueError(f"unsupported processor value for media provenance: {type(value).__name__}")


def processor_value_digest(value: object) -> str:
    encoded = json.dumps(_cpu_value(value), ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _ProvenancePromptUpdate(ResolvedPromptUpdate):
    content_digest: str
    preprocessing_digest: str


def bind_processed_media_updates(
    updates: MultiModalPromptUpdates,
    kwargs: MultiModalKwargsItems,
    *,
    preprocessing_digest: str,
    unverified_modalities: set[str],
) -> MultiModalPromptUpdates:
    result: MultiModalPromptUpdates = {}
    for modality, items in updates.items():
        result[modality] = []
        for index, item_updates in enumerate(items):
            # Arbitrary precomputed embeddings are not trusted processor output.
            if modality in unverified_modalities:
                result[modality].append(item_updates)
                continue
            content_digest = processor_value_digest(kwargs[modality][index].get_data())
            result[modality].append(
                [
                    _ProvenancePromptUpdate(
                        modality=update.modality,
                        item_idx=update.item_idx,
                        mode=update.mode,
                        target=update.target,
                        content=update.content,
                        content_digest=content_digest,
                        preprocessing_digest=preprocessing_digest,
                    )
                    for update in item_updates
                ]
            )
    return result


def processed_media_provenance(
    prompt_ids: list[int],
    placeholders: Mapping[str, list[PlaceholderRange]],
    hashes: MultiModalHashes,
    updates: MultiModalPromptUpdates,
) -> ProcessedMediaProvenance | None:
    media = []
    routing_hashes = []
    for modality, index in argsort_mm_positions(placeholders):
        item_updates = updates[modality][index]
        proofs = [update for update in item_updates if isinstance(update, _ProvenancePromptUpdate)]
        if not proofs or len(proofs) != len(item_updates):
            return None
        proof = proofs[0]
        if any(
            (update.content_digest, update.preprocessing_digest) != (proof.content_digest, proof.preprocessing_digest)
            for update in proofs
        ):
            raise ValueError("conflicting provenance on a processed media item")
        position = placeholders[modality][index]
        media.append(
            ProcessedMediaIdentity(
                modality=modality,
                content_digest=proof.content_digest,
                preprocessing_digest=proof.preprocessing_digest,
                offset=position.offset,
                length=position.length,
                is_embed=None if position.is_embed is None else tuple(position.is_embed.tolist()),
            )
        )
        routing_hashes.append(hashes[modality][index])
    return ProcessedMediaProvenance(tuple(prompt_ids), tuple(media), tuple(routing_hashes))
