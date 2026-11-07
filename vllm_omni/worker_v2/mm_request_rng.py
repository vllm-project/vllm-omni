# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapt request RNG to MRv2's encoder inputs and request lifecycle."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

from vllm_omni.model_executor.mm_request_rng import (
    RandomDrawSpec,
    RequestRNGProvider,
    SupportsMMRequestRNG,
    tag_mm_rng_item,
)


@dataclass(frozen=True)
class _RequestPlan:
    seed: int | None
    references: tuple[tuple[str, str], ...]
    specs: dict[int, RandomDrawSpec]


class MMRequestRNGState:
    def __init__(self, model: SupportsMMRequestRNG) -> None:
        self.model = model
        self.provider = RequestRNGProvider(model.advance_mm_rng)
        self.plans: dict[str, _RequestPlan] = {}
        model.bind_mm_rng_provider(self.provider)

    def add_request(self, req_id: str, features: list[MultiModalFeatureSpec]) -> None:
        seed_field = self.model.mm_rng_seed_field
        seed = next(
            (int(f.data[seed_field].data.item()) for f in features if f.data is not None and seed_field in f.data),
            None,
        )
        # MRv2 releases consumed inputs, including keep_on_cpu fields. Retain
        # only the draw plan before that release, never the image tensors.
        plan = _RequestPlan(
            seed,
            tuple((f.modality, f.identifier) for f in features),
            self.model.get_mm_rng_specs(features, seed) if seed is not None else {},
        )
        previous = self.plans.get(req_id)
        # These are internal invariants after processor admission validation.
        if previous is not None and (previous.seed != seed or (seed is not None and previous != plan)):
            raise ValueError("Multimodal RNG plan changed within a request")
        self.plans[req_id] = plan

    @contextmanager
    def tagged_inputs(self, scheduled: dict[str, list[int]], cache: EncoderCache) -> Iterator[None]:
        originals: list[tuple[list[MultiModalFeatureSpec], int, MultiModalFeatureSpec]] = []
        try:
            for req_id, indices in scheduled.items():
                plan = self.plans[req_id]
                if plan.seed is None:
                    continue
                features = cache.mm_features[req_id]
                for index in indices:
                    feature = features[index]
                    if index not in plan.specs or feature.data is None or feature.identifier in cache.encoder_outputs:
                        continue
                    seed_field = self.model.mm_rng_seed_field
                    if seed_field not in feature.data or int(feature.data[seed_field].data.item()) != plan.seed:
                        raise ValueError("Multimodal RNG seed changed within a request")
                    handle = self.provider.handles.get(req_id)
                    if handle is None:
                        handle = self.provider.register(req_id, plan.seed, plan.specs)
                    originals.append((features, index, feature))
                    features[index] = replace(
                        feature, data=tag_mm_rng_item(feature.data, feature.modality, handle, index)
                    )
            yield
        finally:
            for features, index, feature in reversed(originals):
                features[index] = feature

    def discard(self, req_id: str) -> None:
        self.plans.pop(req_id, None)
        self.provider.discard(req_id)

    def clear(self) -> None:
        self.plans.clear()
        self.provider.clear()
