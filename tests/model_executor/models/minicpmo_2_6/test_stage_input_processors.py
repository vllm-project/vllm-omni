# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiniCPM-o 2.6 stage 0 -> stage 1 bridge.

Covers ``vllm_omni.model_executor.stage_input_processors.minicpmo_2_6_omni.llm2tts``:

  - empty ``source_outputs`` raises
  - latent fallback to ``hidden_states`` when ``multimodal_output`` is empty
  - both inputs missing -> raises
  - additional_information payload carries the keys the talker expects
    (prompt_embeds, prompt_token_ids, llm_output_token_ids, llm_output_text)
  - dummy talker prompt ``[BOS, PAD, EOS] = [1, 0, 2]`` (single prefill step)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.minicpmo_2_6_omni import llm2tts

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HIDDEN_DIM = 4


def _make_thinker_output(
    *,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    text: str = "",
    request_id: str = "req-0",
    latent: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
):
    output = SimpleNamespace(
        multimodal_output={"latent": latent} if latent is not None else {},
        token_ids=output_token_ids,
        text=text,
    )
    if hidden_states is not None:
        output.hidden_states = hidden_states
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        outputs=[output],
    )


class TestInputValidation:
    def test_empty_source_outputs_raises(self) -> None:
        with pytest.raises(ValueError, match="source_outputs cannot be empty"):
            llm2tts([], prompt=None)

    def test_missing_latent_and_hidden_states_raises(self) -> None:
        bad = _make_thinker_output(prompt_token_ids=[10, 11], output_token_ids=[20])
        with pytest.raises(ValueError, match="No latent or hidden_states"):
            llm2tts([bad], prompt=None)


class TestBasicShape:
    def test_returns_one_entry_per_input(self) -> None:
        hidden = torch.zeros((3, _HIDDEN_DIM))
        out = llm2tts(
            [
                _make_thinker_output(prompt_token_ids=[10, 11], output_token_ids=[20], hidden_states=hidden),
                _make_thinker_output(
                    prompt_token_ids=[12],
                    output_token_ids=[21, 22],
                    hidden_states=hidden,
                    request_id="req-1",
                ),
            ],
            prompt=None,
        )
        assert len(out) == 2

    def test_talker_prompt_token_ids_dummy_bos_pad_eos(self) -> None:
        hidden = torch.zeros((2, _HIDDEN_DIM))
        out = llm2tts(
            [_make_thinker_output(prompt_token_ids=[10], output_token_ids=[20], hidden_states=hidden)],
            prompt=None,
        )
        assert out[0]["prompt_token_ids"] == [1, 0, 2]

    def test_additional_information_carries_thinker_outputs(self) -> None:
        prompt_ids = [10, 11, 12]
        out_ids = [20, 21]
        hidden = torch.randn(len(prompt_ids) + len(out_ids), _HIDDEN_DIM)

        result = llm2tts(
            [
                _make_thinker_output(
                    prompt_token_ids=prompt_ids,
                    output_token_ids=out_ids,
                    text="hello",
                    hidden_states=hidden,
                )
            ],
            prompt=None,
        )
        ai = result[0]["additional_information"]
        assert ai["prompt_token_ids"] == prompt_ids
        assert ai["llm_output_token_ids"] == out_ids
        assert ai["llm_output_text"] == ["hello"]
        assert ai["prompt_embeds"].dtype == torch.float32
        assert ai["prompt_embeds"].shape == (len(prompt_ids), _HIDDEN_DIM)
        assert torch.equal(ai["prompt_embeds"], hidden[: len(prompt_ids)].to(torch.float32))

    def test_latent_in_multimodal_output_takes_precedence(self) -> None:
        prompt_ids = [10, 11]
        out_ids = [20]
        latent = torch.ones((len(prompt_ids) + len(out_ids), _HIDDEN_DIM))
        hidden = torch.zeros_like(latent)

        result = llm2tts(
            [
                _make_thinker_output(
                    prompt_token_ids=prompt_ids,
                    output_token_ids=out_ids,
                    latent=latent,
                    hidden_states=hidden,
                )
            ],
            prompt=None,
        )
        ai = result[0]["additional_information"]
        assert torch.equal(ai["prompt_embeds"], latent[: len(prompt_ids)].to(torch.float32))


class TestPromptAndMultiModal:
    def test_prompt_can_be_single_dict_not_a_list(self) -> None:
        hidden = torch.zeros((2, _HIDDEN_DIM))
        llm2tts(
            [_make_thinker_output(prompt_token_ids=[10], output_token_ids=[20], hidden_states=hidden)],
            prompt={"multi_modal_data": {"audio": "ignored"}},
            requires_multimodal_data=False,
        )

    def test_multimodal_dropped_when_not_requested(self) -> None:
        hidden = torch.zeros((2, _HIDDEN_DIM))
        out = llm2tts(
            [_make_thinker_output(prompt_token_ids=[10], output_token_ids=[20], hidden_states=hidden)],
            prompt={"multi_modal_data": {"audio": "should-be-ignored"}},
            requires_multimodal_data=False,
        )
        assert out[0]["multi_modal_data"] is None

    def test_multimodal_forwarded_when_requested(self) -> None:
        hidden = torch.zeros((2, _HIDDEN_DIM))
        mm = {"audio": "forward-me"}
        out = llm2tts(
            [_make_thinker_output(prompt_token_ids=[10], output_token_ids=[20], hidden_states=hidden)],
            prompt={"multi_modal_data": mm},
            requires_multimodal_data=True,
        )
        assert out[0]["multi_modal_data"] == mm

    def test_streaming_context_is_accepted_and_ignored(self) -> None:
        hidden = torch.zeros((2, _HIDDEN_DIM))
        out = llm2tts(
            [_make_thinker_output(prompt_token_ids=[10], output_token_ids=[20], hidden_states=hidden)],
            prompt=None,
            streaming_context=object(),
        )
        assert len(out) == 1
