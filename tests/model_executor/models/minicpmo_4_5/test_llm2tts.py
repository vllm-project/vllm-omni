# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiniCPM-o 4.5 stage 0 -> stage 1 bridge.

Covers ``vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni.llm2tts``:

  - empty ``source_outputs`` raises
  - latent fallback to ``hidden_states`` when ``multimodal_output`` is empty
  - both inputs missing -> raises
  - additional_information payload carries the keys the talker expects
    (prompt_embeds, prompt_token_ids, llm_output_token_ids, llm_output_text)
  - dummy talker prompt ``[BOS, PAD, EOS] = [1, 0, 2]`` (single prefill step)
  - MiniCPM-o 4.5 TTS region detection on 151703 / 151704 tokens
  - MiniCPM-o 2.6 fallback detection on 151691 / 151692 when no 4.5 markers
  - No TTS markers present -> no ``tts_token_ids`` / ``tts_hidden_states`` keys
  - prompt arg is normalized to a list and ``multi_modal_data`` is gated by
    ``requires_multimodal_data``
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import llm2tts

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
    """Construct a minimal mock of a thinker engine output entry.

    The real ``llm2tts`` only reads a tight slice of fields:
      - top-level: ``request_id``, ``prompt_token_ids``, ``outputs[0]``
      - per-output: ``multimodal_output`` (dict), ``hidden_states`` (opt),
        ``text``, ``token_ids``
    """
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
        bad = _make_thinker_output(
            prompt_token_ids=[10, 11],
            output_token_ids=[20],
        )
        with pytest.raises(ValueError, match="No latent or hidden_states"):
            llm2tts([bad], prompt=None)


class TestBasicShape:
    def test_returns_one_entry_per_input(self) -> None:
        hidden = torch.zeros((3, _HIDDEN_DIM))
        out = llm2tts(
            [
                _make_thinker_output(prompt_token_ids=[10, 11], output_token_ids=[20], hidden_states=hidden),
                _make_thinker_output(
                    prompt_token_ids=[12], output_token_ids=[21, 22], hidden_states=hidden, request_id="req-1"
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
        # The talker AR framework needs *some* tokens to do a single prefill;
        # this is the agreed [BOS, PAD, EOS] minimal payload.
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
        # prompt_embeds = float32 view of the prompt-portion of hidden states.
        assert ai["prompt_embeds"].dtype == torch.float32
        assert ai["prompt_embeds"].shape == (len(prompt_ids), _HIDDEN_DIM)
        assert torch.equal(ai["prompt_embeds"], hidden[: len(prompt_ids)].to(torch.float32))

    def test_latent_in_multimodal_output_takes_precedence(self) -> None:
        # When both ``multimodal_output["latent"]`` and ``hidden_states`` are
        # present, the latent payload must win (this is the steady-state path
        # produced by the thinker stage).
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
        # latent (ones) won over hidden_states (zeros)
        assert torch.equal(ai["prompt_embeds"], latent[: len(prompt_ids)].to(torch.float32))


class TestTtsRegionDetection:
    """The bridge auto-detects which MiniCPM-o generation produced the output
    by sniffing for 4.5-specific BOS/EOS token ids (151703 / 151704); when
    those are absent it falls back to the 2.6 ids (151691 / 151692).

    This is a regression guard: a single off-by-one or wrong branch
    precedence here causes the talker to receive an empty / wrong-region
    slice and emit silent audio.
    """

    def _run(self, prompt_token_ids, output_token_ids):
        total = len(prompt_token_ids) + len(output_token_ids)
        hidden = torch.arange(total * _HIDDEN_DIM, dtype=torch.float32).reshape(total, _HIDDEN_DIM)
        result = llm2tts(
            [
                _make_thinker_output(
                    prompt_token_ids=prompt_token_ids,
                    output_token_ids=output_token_ids,
                    hidden_states=hidden,
                )
            ],
            prompt=None,
        )
        return result[0]["additional_information"], hidden

    def test_4_5_markers_detected(self) -> None:
        # prompt:        [10, 11]
        # output:        [151703, 30, 31, 151704, 40]
        # full sequence: [10, 11, 151703, 30, 31, 151704, 40]
        #                  0   1     2    3   4     5     6
        # 4.5 BOS at idx 2 -> slice starts at 3; EOS at idx 5 -> slice ends at 5.
        ai, hidden = self._run([10, 11], [151703, 30, 31, 151704, 40])
        assert "tts_token_ids" in ai and "tts_hidden_states" in ai
        assert ai["tts_token_ids"].tolist() == [30, 31]
        assert torch.equal(ai["tts_hidden_states"], hidden[3:5])

    def test_4_5_takes_precedence_when_both_markers_present(self) -> None:
        # If both 2.6 and 4.5 markers appear (shouldn't really happen on a
        # real checkpoint, but exercises the loop's break), the 4.5 markers
        # win because the source loop pins on the first 4.5 id seen.
        # sequence index: 0   1   2     3   4     5    6   7    8
        ai, hidden = self._run([10], [151691, 70, 151692, 151703, 30, 151704, 40])
        assert ai["tts_token_ids"].tolist() == [30]
        assert torch.equal(ai["tts_hidden_states"], hidden[5:6])

    def test_2_6_fallback(self) -> None:
        # No 4.5 markers anywhere -> use 2.6 BOS/EOS pair.
        # full sequence: [10, 151691, 30, 31, 151692, 40]
        #                  0     1    2   3     4     5
        # BOS at idx 1 -> slice starts at 2; EOS at idx 4 -> slice ends at 4.
        ai, hidden = self._run([10], [151691, 30, 31, 151692, 40])
        assert ai["tts_token_ids"].tolist() == [30, 31]
        assert torch.equal(ai["tts_hidden_states"], hidden[2:4])

    def test_bos_without_eos_runs_to_end(self) -> None:
        # When BOS is found but EOS is missing (typical for an in-flight or
        # truncated decode), the slice should extend to the end of the
        # hidden-state matrix instead of being dropped.
        # sequence: [10, 11, 151703, 30, 31]
        ai, hidden = self._run([10, 11], [151703, 30, 31])
        assert ai["tts_token_ids"].tolist() == [30, 31]
        assert torch.equal(ai["tts_hidden_states"], hidden[3:5])

    def test_no_tts_markers_omits_slice_keys(self) -> None:
        # If neither marker pair is present, the bridge should NOT populate
        # ``tts_token_ids`` / ``tts_hidden_states`` — the talker should fall
        # through to the dummy path.
        ai, _ = self._run([10, 11], [20, 21, 22])
        assert "tts_token_ids" not in ai
        assert "tts_hidden_states" not in ai


class TestPromptAndMultiModal:
    def test_prompt_can_be_single_dict_not_a_list(self) -> None:
        hidden = torch.zeros((2, _HIDDEN_DIM))
        # A single (non-list) prompt should be auto-wrapped without raising.
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
        # The framework passes ``streaming_context`` for streaming-capable
        # pipelines; MiniCPM-o 4.5 is turn-taking so the bridge must accept
        # but discard it without crashing.
        hidden = torch.zeros((2, _HIDDEN_DIM))
        out = llm2tts(
            [_make_thinker_output(prompt_token_ids=[10], output_token_ids=[20], hidden_states=hidden)],
            prompt=None,
            streaming_context=object(),
        )
        assert len(out) == 1
