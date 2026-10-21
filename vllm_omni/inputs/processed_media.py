# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Processor-owned media identity, separate from caller-supplied extras."""

from dataclasses import dataclass

import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec


@dataclass(frozen=True)
class ProcessedMediaIdentity:
    modality: str
    content_digest: str
    preprocessing_digest: str
    offset: int
    length: int
    is_embed: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class ProcessedMediaProvenance:
    """Minted from processor results, not reconstructed from client JSON.

    Routing hashes bind the record to the renderer result; they are NOT content
    digests. They may contain arbitrary caller UUIDs. The content digests come
    from the actual processed CPU values retained with the processor cache item.
    Only ``media`` feeds conditioning identity; routing hashes must not do so.
    """

    prompt_token_ids: tuple[int, ...]
    media: tuple[ProcessedMediaIdentity, ...]
    routing_hashes: tuple[str, ...]

    def validate(self, prompt_token_ids: list[int] | None, features: list[MultiModalFeatureSpec] | None) -> None:
        if prompt_token_ids is None or tuple(prompt_token_ids) != self.prompt_token_ids:
            raise ValueError("processed-media provenance does not match the rendered prompt")
        features = features or []
        if len(features) != len(self.media) or len(features) != len(self.routing_hashes):
            raise ValueError("processed-media provenance does not cover the rendered features")
        for feature, item, routing_hash in zip(features, self.media, self.routing_hashes, strict=True):
            position = feature.mm_position
            if position.is_embed is not None and (
                position.is_embed.device.type != "cpu"
                or position.is_embed.ndim != 1
                or position.is_embed.dtype != torch.bool
            ):
                raise ValueError("processed-media provenance requires a CPU boolean position mask")
            mask = None if position.is_embed is None else tuple(position.is_embed.tolist())
            if (
                feature.modality != item.modality
                or feature.mm_hash != routing_hash
                or position.offset != item.offset
                or position.length != item.length
                or mask != item.is_embed
            ):
                raise ValueError("processed-media provenance does not match the rendered feature")
