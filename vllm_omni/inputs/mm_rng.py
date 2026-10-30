# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request RNG metadata, separate from cached multimodal processor data."""

import hashlib
from dataclasses import dataclass, replace

import msgspec
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem

MM_RNG_CONTEXT_KEY = "mm_rng_context"
HUNYUAN_VAE_RNG_POLICY = "hunyuan-ar-vae-v1"


@dataclass(frozen=True)
class MMRNGContext:
    policy: str
    seed: int
    retain_fields: tuple[str, ...]


def contextualize_mm_features(
    features: list[MultiModalFeatureSpec], context: MMRNGContext
) -> list[MultiModalFeatureSpec]:
    """Scope encoder identities to a seeded reference prefix, keeping processor keys."""
    prefix = b""
    result = []
    for feature in features:
        # Include the existing identifier, which already carries any encoder LoRA scope.
        prefix = hashlib.sha256(
            msgspec.msgpack.encode((context.policy, context.seed, prefix, feature.identifier))
        ).digest()
        result.append(replace(feature, identifier=f"{context.policy}:{prefix.hex()}"))
    return result


def retain_mm_rng_fields(
    features: list[MultiModalFeatureSpec],
    original_features: list[MultiModalFeatureSpec],
    context: MMRNGContext,
) -> list[MultiModalFeatureSpec]:
    """Restore compact RNG metadata after upstream strips prefix-covered payloads."""
    result = []
    for feature, original in zip(features, original_features, strict=True):
        if feature.identifier != original.identifier:
            raise ValueError("Multimodal RNG metadata does not match scheduled feature order")
        data, original_data = feature.data, original.data
        # SHM handles must remain intact for worker-side resolution and acknowledgement.
        if data is original_data or original_data is None or (data is not None and "address" in data):
            result.append(feature)
            continue
        retained = {key: original_data[key] for key in context.retain_fields if key in original_data}
        result.append(replace(feature, data=MultiModalKwargsItem({**(data or {}), **retained})))
    return result
