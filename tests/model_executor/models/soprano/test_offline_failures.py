# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regressions for observed P11 frontend and decoder worker failures."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm import SamplingParams

from vllm_omni.model_executor.models.soprano.contracts import transform_prompt
from vllm_omni.model_executor.models.soprano.soprano_decoder import SopranoDecoderForGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_raw_numbers_are_normalized_by_the_production_frontend():
    pytest.importorskip("inflect")
    pytest.importorskip("unidecode")
    result = transform_prompt("I ordered 12 books and paid 45 dollars.", [SamplingParams()])
    assert result["prompt"] == "[STOP][TEXT]i ordered twelve books and paid forty-five dollars.[START]"


class SmallDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Module()
        self.head.out = nn.Linear(1, 1)

    def forward(self, hidden):
        return torch.ones(1, 2048 * (hidden.shape[-1] - 1))


def test_short_or_invalid_requests_return_errors_without_killing_decoder():
    model = object.__new__(SopranoDecoderForGeneration)
    nn.Module.__init__(model)
    model.decoder = SmallDecoder()
    payloads = [
        {"hidden_states": {"output": torch.empty(0, 512)}},
        {"hidden_states": {"output": torch.ones(1, 512)}},
        {"additional_information": {"soprano_error": "invalid trajectory"}},
        {"hidden_states": {"output": torch.ones(2, 512)}, "ids": {"output": [10, 11, 3]}},
    ]
    result = model(runtime_additional_information=payloads, request_ids=["zero", "one", "bad", "good"])
    mm = result.multimodal_outputs
    assert [a.numel() for a in mm["model_outputs"]] == [0, 0, 0, 2048]
    assert [bool(x) for x in mm["audio_synthesis_error"]] == [True, True, True, False]
    later = model(runtime_additional_information=[payloads[-1]], request_ids=["after-error"])
    assert later.multimodal_outputs["model_outputs"][0].numel() == 2048
    assert not bool(later.multimodal_outputs["audio_synthesis_error"][0])


def test_official_sentence_boundaries_are_preserved_without_truncation():
    pytest.importorskip("inflect")
    pytest.importorskip("unidecode")
    from vllm_omni.model_executor.models.soprano.text import prepare_prompts

    text = (
        "Every morning, the small bakery opens before sunrise. "
        "The baker checks the oven, prepares the dough, and places fresh bread near the window. "
        "By the time the first customers arrive, the room is warm and the shelves are full. "
        "Today, a child chooses a round loaf to share with her family."
    )
    prompts = prepare_prompts(text)
    assert len(prompts) == 4
    assert prompts[0] == "[STOP][TEXT]every morning, the small bakery opens before sunrise.[START]"
    assert prompts[-1].endswith("to share with her family.[START]")
    with pytest.raises(ValueError, match="synthesize|prepare_prompts"):
        transform_prompt(text, [SamplingParams()])


def test_frontend_failure_is_explicit_and_engine_can_be_reused():
    pytest.importorskip("inflect")
    pytest.importorskip("unidecode")
    from vllm_omni.model_executor.models.soprano.offline import SopranoSynthesisError, synthesize

    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": True, "truncation": False}
            return [3, 1, 20, 2]

    class Engine:
        calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            failed = self.calls == 1
            mm = {
                "audio": torch.empty(0) if failed else torch.ones(2048),
                "sr": 32000,
                "audio_synthesis_error": [torch.tensor([failed])],
                "audio_truncated": [torch.tensor([False])],
            }
            return [
                SimpleNamespace(
                    final_output_type="audio",
                    request_id=str(self.calls),
                    outputs=[SimpleNamespace(multimodal_output=mm)],
                )
            ]

    engine = Engine()
    with pytest.raises(SopranoSynthesisError):
        synthesize(engine, "Hello.", Tokenizer())
    result = synthesize(engine, "Good morning.", Tokenizer())
    assert engine.calls == 2
    assert result.audio.shape == (2048,)
    assert result.request_ids == ("2",)
    assert result.sample_rate == 32000
    assert not result.truncated


def test_overlong_sentence_is_rejected_before_any_request_is_admitted():
    pytest.importorskip("inflect")
    pytest.importorskip("unidecode")
    from vllm_omni.model_executor.models.soprano.offline import synthesize

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [10] * 257

    with pytest.raises(ValueError, match="no tokens were truncated"):
        synthesize(None, "Hello.", Tokenizer())


def test_unvalidated_precision_is_rejected_before_loading_weights(monkeypatch):
    from vllm_omni.model_executor.models.soprano.soprano_ar import SopranoForCausalLM

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(hidden_size=512, vocab_size=8192, eos_token_id=3),
            async_chunk=False,
            dtype=torch.bfloat16,
        )
    )
    with pytest.raises(ValueError, match="BF16 is not validated"):
        SopranoForCausalLM(vllm_config=config)
    config.model_config.dtype = torch.float32
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    with pytest.raises(ValueError, match="TRITON_F32_DEFAULT=ieee"):
        SopranoForCausalLM(vllm_config=config)
