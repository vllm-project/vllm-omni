# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Official single-request stream traces with controlled categorical draws.

The fixture records distributions from the unchanged official sampler and
tokens returned by its full generation loop. Scripted full-vocabulary logits
exercise sampling semantics, not model inference or runner integration.
"""

import json
from pathlib import Path

import pytest
import torch

from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioSpecialTokens
from vllm_omni.model_executor.models.kimi_audio.sampling import KimiAudioSamplingParams, sample_kimi_audio_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]
REFERENCE = json.loads((Path(__file__).parent / "fixtures/sampling_reference.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_official_sampling_distributions_and_stream_trace(case, monkeypatch):
    special = KimiAudioSpecialTokens.from_vocab(REFERENCE["special_tokens"])
    params = KimiAudioSamplingParams(**case["params"])
    expected = case["reference"]
    generator = torch.Generator().manual_seed(case["seed"])
    draw_index = 0

    def replay_draw(weights, num_samples, *, generator):
        nonlocal draw_index
        assert generator is request_generator and num_samples == 1
        # Check what reaches the draw, not just the winning token. Replaying
        # its selected column avoids depending on torch RNG version details.
        draw = expected["draws"][draw_index]
        draw_index += 1
        normalized = weights[0] / weights[0].sum()
        support = normalized.nonzero().flatten()
        assert support.tolist() == draw["positions"]
        torch.testing.assert_close(normalized[support], torch.tensor(draw["probabilities"]), rtol=1e-6, atol=1e-7)
        return torch.tensor([[draw["choice"]]], dtype=torch.long)

    request_generator = generator
    monkeypatch.setattr(torch, "multinomial", replay_draw)
    text_history, audio_history = [], []
    text_finished = False
    for index, expected_step in enumerate(expected["steps"]):
        logits = []
        for stream in ("text", "audio"):
            scores = torch.full((REFERENCE["vocab_size"],), -float("inf"))
            for token, score in case["logits"][index][stream]:
                scores[token] = score
            logits.append(scores)
        originals = [scores.clone() for scores in logits]
        histories_before = (list(text_history), list(audio_history))
        result = sample_kimi_audio_step(
            *logits,
            text_history=text_history,
            audio_history=audio_history,
            text_finished=text_finished,
            output_type=case["output_type"],
            special_tokens=special,
            audio_delay=REFERENCE["audio_delay"],
            params=params,
            generator=generator,
        )
        assert result.text_token == expected_step["text_token"]
        assert result.audio_token == expected_step["audio_token"]
        text_finished = text_finished or result.text_token == special.kimia_text_eos
        assert result.text_finished == text_finished
        # The official loop returned early only in terminal cases. Exhausting
        # its caller-supplied token budget must not manufacture a model EOS.
        assert result.finished == (index == len(expected["steps"]) - 1 and len(expected["steps"]) < len(case["logits"]))
        assert (text_history, audio_history) == histories_before
        for scores, original in zip(logits, originals):
            torch.testing.assert_close(scores, original)
        text_history.append(result.text_token)
        audio_history.append(result.audio_token)

    assert draw_index == len(expected["draws"])
    eos_index = (
        text_history.index(special.kimia_text_eos) if special.kimia_text_eos in text_history else len(text_history)
    )
    assert text_history[:eos_index] == expected["text"]
    audio_output = audio_history[REFERENCE["audio_delay"] :] if case["output_type"] == "both" else []
    assert audio_output == expected["audio"]
