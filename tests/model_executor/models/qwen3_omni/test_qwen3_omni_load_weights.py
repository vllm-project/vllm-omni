# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Iterator

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _WeightLoader(nn.Module):
    def __init__(self, events: list[str]):
        super().__init__()
        self.events = events
        self.names: list[str] = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.events.append("load")
        self.names = [name for name, _ in weights]
        return set()


def test_load_weights_streams_the_active_stage():
    events: list[str] = []

    def weight_source() -> Iterator[tuple[str, torch.Tensor]]:
        for name in ("thinker.layer.weight", "talker.layer.weight"):
            events.append("yield")
            yield name, torch.empty(0)

    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "thinker"
    model.thinker = _WeightLoader(events)
    model.talker = None
    model.code2wav = None

    model.load_weights(weight_source())

    assert events[0] == "load"
    assert model.thinker.names == ["thinker.layer.weight"]
