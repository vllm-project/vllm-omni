# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiniCPM-o 4.5 stage 0 -> stage 1 bridge.

Covers ``vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni.llm2tts``:

  - empty ``source_outputs`` raises
  - latent fallback to ``hidden_states`` when ``multimodal_output`` is empty
  - both inputs missing -> raises
  - structured model_intermediate_buffer carries the thinker ids and text
  - scheduler prompt tokens follow the selected TTS region or output tokens
  - MiniCPM-o 4.5 TTS region detection on 151703 / 151704 tokens
  - No TTS markers present -> no ``tts_token_ids`` / ``tts_hidden_states`` keys
  - prompt arg is normalized to a list and ``multi_modal_data`` is gated by
    ``requires_multimodal_data``
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import llm2tts

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_HIDDEN_DIM = 4


def test_native_duplex_talker_emits_empty_text_metadata_to_clear_previous_segment() -> None:
    class _Talker:
        _ar_last_chunk_flags = [True]
        _ar_turn_end_flags = [True]

        def __call__(self, **kwargs):
            del kwargs
            return None, torch.zeros(0, dtype=torch.float32)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "tts"
    model.config = SimpleNamespace(hidden_size=_HIDDEN_DIM)
    model.talker = _Talker()

    output = model.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        runtime_additional_information=[
            {
                "native_duplex": True,
                "duplex": {"turn_id": 1, "epoch": 0},
                "meta": {"native_duplex_segment_text": ""},
            }
        ],
    )

    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs["meta.llm_output_text_utf8"].numel() == 0
    assert output.multimodal_outputs["meta.audio_text_total_chars"].tolist() == [0]


def test_tts_embed_input_ids_uses_active_talker_embedding(monkeypatch) -> None:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "tts"
    expected = torch.ones((2, _HIDDEN_DIM))

    monkeypatch.setattr(
        model,
        "get_input_embeddings",
        lambda input_ids: expected,
    )

    actual = model.embed_input_ids(torch.tensor([11, 12]))

    assert actual is expected


def _make_thinker_output(
    *,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    text: str = "",
    request_id: str = "req-0",
    latent: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    multimodal_output: dict[str, object] | None = None,
):
    """Construct a minimal mock of a thinker engine output entry.

    The real ``llm2tts`` only reads a tight slice of fields:
      - top-level: ``request_id``, ``prompt_token_ids``, ``outputs[0]``
      - per-output: ``multimodal_output`` (dict), ``hidden_states`` (opt),
        ``text``, ``token_ids``
    """
    mm_output = dict(multimodal_output or {})
    if latent is not None:
        mm_output["latent"] = latent
    output = SimpleNamespace(
        multimodal_output=mm_output,
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

    def test_talker_scheduler_prompt_uses_output_tokens(self) -> None:
        hidden = torch.zeros((2, _HIDDEN_DIM))
        out = llm2tts(
            [_make_thinker_output(prompt_token_ids=[10], output_token_ids=[20], hidden_states=hidden)],
            prompt=None,
        )
        assert out[0]["prompt_token_ids"] == [20]
        assert "stream_output" not in out[0]["model_intermediate_buffer"]

    def test_model_intermediate_buffer_carries_thinker_outputs(self) -> None:
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
        buffer = result[0]["model_intermediate_buffer"]
        assert buffer["ids"]["prompt"] == prompt_ids
        assert buffer["ids"]["output"] == out_ids
        assert buffer["llm_output_text"] == ["hello"]

    def test_latent_in_multimodal_output_takes_precedence(self) -> None:
        # When both ``multimodal_output["latent"]`` and ``hidden_states`` are
        # present, the latent payload must win (this is the steady-state path
        # produced by the thinker stage).
        prompt_ids = [10, 11]
        out_ids = [151703, 30, 151704]
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
        buffer = result[0]["model_intermediate_buffer"]
        # latent (ones) won over hidden_states (zeros)
        assert torch.equal(torch.tensor(buffer["hidden_states"]["tts"]), latent[3:4].to(torch.float32))


class TestTtsRegionDetection:
    """The bridge detects MiniCPM-o 4.5 BOS/EOS ids (151703 / 151704).

    This is a regression guard: a single off-by-one or wrong branch
    causes the talker to receive an empty / wrong-region slice and emit
    silent audio.
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
        return result[0]["model_intermediate_buffer"], hidden

    def test_4_5_markers_detected(self) -> None:
        # prompt:        [10, 11]
        # output:        [151703, 30, 31, 151704, 40]
        # full sequence: [10, 11, 151703, 30, 31, 151704, 40]
        #                  0   1     2    3   4     5     6
        # 4.5 BOS at idx 2 -> slice starts at 3; EOS at idx 5 -> slice ends at 5.
        buffer, hidden = self._run([10, 11], [151703, 30, 31, 151704, 40])
        assert buffer["ids"]["tts"] == [30, 31]
        assert torch.equal(torch.tensor(buffer["hidden_states"]["tts"]), hidden[3:5])

    def test_bos_without_eos_runs_to_end(self) -> None:
        # When BOS is found but EOS is missing (typical for an in-flight or
        # truncated decode), the slice should extend to the end of the
        # hidden-state matrix instead of being dropped.
        # sequence: [10, 11, 151703, 30, 31]
        buffer, hidden = self._run([10, 11], [151703, 30, 31])
        assert buffer["ids"]["tts"] == [30, 31]
        assert torch.equal(torch.tensor(buffer["hidden_states"]["tts"]), hidden[3:5])

    def test_no_tts_markers_omits_slice_keys(self) -> None:
        # If neither marker pair is present, the bridge should NOT populate
        # ``tts_token_ids`` / ``tts_hidden_states`` — the talker should fall
        # through to the dummy path.
        buffer, _ = self._run([10, 11], [20, 21, 22])
        assert "tts" not in buffer["ids"]
        assert "tts" not in buffer.get("hidden_states", {})


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

    def test_internal_streaming_context_is_accepted(self) -> None:
        hidden = torch.zeros((2, _HIDDEN_DIM))
        out = llm2tts(
            [_make_thinker_output(prompt_token_ids=[10], output_token_ids=[20], hidden_states=hidden)],
            prompt=None,
            _streaming_context=object(),
        )
        assert len(out) == 1
