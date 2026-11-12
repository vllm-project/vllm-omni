# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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


@pytest.mark.parametrize("decoder_first", [False, True])
def test_load_weights_preserves_talker_first_frame_decoder(monkeypatch, decoder_first):
    events: list[str] = []
    decoder_weights: list[tuple[str, torch.Tensor]] = []
    source_weight = torch.ones(1)
    names = ["thinker.layer.weight", "talker.layer.weight", "code2wav.layer.weight"]
    if decoder_first:
        names.reverse()

    def weight_source() -> Iterator[tuple[str, torch.Tensor]]:
        for name in names:
            events.append("yield")
            yield name, source_weight

    def load_decoder_weights(weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        decoder_weights.extend(weights)
        return {"layer.weight"}

    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "talker"
    model.thinker = None
    model.talker = _WeightLoader(events)
    model.code2wav = None
    model.first_frame_decoder = _WeightLoader(events)
    monkeypatch.setattr(model.first_frame_decoder, "load_weights", load_decoder_weights)
    monkeypatch.setattr(model, "_init_special_tokens_embeddings", lambda: set())

    loaded = model.load_weights(weight_source())

    assert events[0] == "load"
    assert events.count("yield") == len(names)
    assert model.talker.names == ["talker.layer.weight"]
    assert [name for name, _ in decoder_weights] == ["code2wav.layer.weight"]
    decoder_weight = decoder_weights[0][1]
    assert decoder_weight.device.type == "cpu"
    assert decoder_weight.data_ptr() != source_weight.data_ptr()
    torch.testing.assert_close(decoder_weight, source_weight)
    assert "first_frame_decoder.code2wav.layer.weight" in loaded
