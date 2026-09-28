# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture first-code sampling while preserving each request's RNG stream."""

import torch
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.breeze_tts_2.depth_decoder import sample_logits
from vllm_omni.platforms import current_omni_platform


class BreezeFirstCodeSampler:
    def __init__(self) -> None:
        self._graphs: dict[tuple[torch.device, int, int, tuple[float, int, float]], _SamplerGraph] = {}

    def sample(
        self,
        logits: list[torch.Tensor],
        parameters: tuple[float, int, float],
        generators: list[torch.Generator],
    ) -> list[torch.Tensor]:
        if len(logits) != len(generators):
            raise ValueError("First-code sampling requires one generator per logit row")
        if not logits:
            return []
        first = logits[0]
        for row, generator in zip(logits, generators, strict=True):
            if (
                row.ndim != 2
                or row.shape[0] != 1
                or row.shape[1] == 0
                or row.shape != first.shape
                or row.dtype != torch.float32
                or row.device != first.device
                or generator.device.type != row.device.type
                or (generator.device.index is not None and generator.device.index != row.device.index)
            ):
                raise ValueError("First-code sampling requires aligned FP32 [1, vocab] rows and generators")
        capture = (
            parameters[0] > 0
            and first.is_cuda
            and current_platform.is_cuda()
            and hasattr(torch.cuda.CUDAGraph, "register_generator_state")
            and len({id(generator) for generator in generators}) == len(generators)
        )
        if capture:
            bucket = 1 << (len(logits) - 1).bit_length()
            key = (first.device, first.shape[1], bucket, parameters)
            entry = self._graphs.get(key)
            # Sampling parameters are static graph branches. Bound their cache
            # without evicting buffers retained by existing captured graphs.
            with torch.accelerator.device_index(first.device.index):
                if entry is None and len(self._graphs) < 32:
                    entry = _SamplerGraph(first, bucket, parameters)
                    self._graphs[key] = entry
                if entry is not None:
                    return entry.replay(logits, generators)
        # Shared generators must retain sequential draw ordering, including
        # when a matching graph is already cached for independent requests.
        return [sample_logits(row, *parameters, generator) for row, generator in zip(logits, generators, strict=True)]


class _SamplerGraph:
    def __init__(self, row: torch.Tensor, bucket: int, parameters: tuple[float, int, float]) -> None:
        self.logits = row.new_zeros((bucket, row.shape[1]))
        self.generators = [torch.Generator(device=row.device) for _ in range(bucket)]

        def sample_rows() -> torch.Tensor:
            # Keep the original one-row softmax, filtering, multinomial
            # validation and RNG draw geometry; batched reductions can change
            # seeded output trajectories even when probabilities are close.
            return torch.cat(
                [
                    sample_logits(self.logits[index : index + 1], *parameters, generator)
                    for index, generator in enumerate(self.generators)
                ]
            )

        for _ in range(3):
            sample_rows()
        current_omni_platform.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream(device=row.device)
        for generator in self.generators:
            self.graph.register_generator_state(generator)
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self.output = sample_rows()

    def replay(self, logits: list[torch.Tensor], generators: list[torch.Generator]) -> list[torch.Tensor]:
        live = len(logits)
        self.logits[:live].copy_(torch.cat(logits))
        # Captures retain the placeholder state objects. Update their values
        # rather than replacing those objects with request-owned RNG states.
        for placeholder, request in zip(self.generators[:live], generators, strict=True):
            placeholder.manual_seed(request.initial_seed())
            placeholder.set_offset(request.get_offset())
        self.graph.replay()
        for placeholder, request in zip(self.generators[:live], generators, strict=True):
            request.set_offset(placeholder.get_offset())
        # Padded rows have valid logits but no request state to advance. Clone
        # the live results so later replays cannot overwrite a returned token.
        return list(self.output[:live].clone().split(1))
