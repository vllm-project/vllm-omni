# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-alignment tests for MiniCPM-o 4.5's native Talker."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
    _max_audio_tokens,
)
from vllm_omni.utils.mm_outputs import to_payload_element

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeNativeTalker(nn.Module):
    has_preprocess = True

    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs = None

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return torch.ones(2, 4)


def test_wrapper_always_delegates_talker_to_native_ar_path() -> None:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "tts"
    model.talker = _FakeNativeTalker()

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.arange(2),
        model_intermediate_buffer=[{"request_id": "req"}],
    )

    assert output.shape == (2, 4)
    assert model.talker.forward_kwargs["model_intermediate_buffer"][0]["request_id"] == "req"


def _make_talker() -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker._num_audio_tokens = 8
    talker._batch_stop_logits = None
    talker._request_generators = {}
    talker._request_audio_states = {}
    talker._deferred_cleanup_ids = set()
    return talker


def _routed(output, index: int):
    return to_payload_element(
        output.multimodal_outputs,
        index,
        index,
        index + 1,
        seq_len=2,
        scheduled_seq_len=2,
    )


@pytest.mark.parametrize(
    ("condition_tokens", "expected"),
    [(3, 64), (100, 1000), (1000, 2048)],
)
def test_audio_token_limit_scales_with_condition_length(
    condition_tokens: int,
    expected: int,
) -> None:
    assert _max_audio_tokens(condition_tokens) == expected


def test_talker_emits_request_aligned_codec_deltas_after_compaction(mocker) -> None:
    talker = _make_talker()
    seen: list[tuple[str, list[float], list[int]]] = []

    def sample(hidden, history, request_id, step):
        assert step == 0
        seen.append((request_id, hidden.reshape(-1).tolist(), history.tolist()))
        return torch.tensor(2 if request_id == "req-a" else 3)

    mocker.patch.object(talker, "_sample_audio_code", side_effect=sample)
    infos = [
        {"request_id": "req-a", "audio_codes": {"accumulated": torch.tensor([1])}},
        {"request_id": "req-b", "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)}},
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
    )

    assert seen == [
        ("req-a", [2.0, 0.0], [1]),
        ("req-b", [3.0, 0.0], []),
    ]
    assert infos[0]["audio_codes"]["accumulated"].tolist() == [1, 2]
    assert infos[1]["audio_codes"]["accumulated"].tolist() == [3]
    assert set(output.multimodal_outputs) == {"codes", "meta"}
    assert "model_outputs" not in output.multimodal_outputs
    assert "sr" not in output.multimodal_outputs
    assert _routed(output, 0)["codes"]["audio"].tolist() == [[2]]
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[3]]
    assert _routed(output, 0)["meta"]["finished"].item() is False
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [0, 0]


def test_incomplete_prefill_emits_no_code_and_does_not_advance_state(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    infos = [
        {
            "request_id": "req-prefill",
            "audio_state": {"step": 0},
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
        {
            "request_id": "req-decode",
            "audio_state": {"step": 4},
            "audio_codes": {"accumulated": torch.tensor([1])},
        },
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
        request_sample_eligible=[False, True],
    )

    sample.assert_called_once()
    assert sample.call_args.args[2] == "req-decode"
    assert infos[0]["audio_state"]["step"] == 0
    assert infos[0]["audio_codes"]["accumulated"].numel() == 0
    assert infos[1]["audio_state"]["step"] == 5
    assert _routed(output, 0)["codes"]["audio"].shape == (0, 1)
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[2]]


def test_eos_is_terminal_once_and_never_enters_codec_history(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(7))
    info = {
        "request_id": "req-stop",
        "audio_state": {"step": 3},
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    first = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    first_logits = talker.compute_logits(first.text_hidden_states)
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    sample.assert_called_once()
    assert info["audio_codes"]["accumulated"].tolist() == [4, 5]
    assert first.multimodal_outputs["codes"]["audio"][0].shape == (0, 1)
    assert first.multimodal_outputs["meta"]["finished"][0].item() is True
    assert second.multimodal_outputs["meta"]["finished"][0].item() is False
    assert first_logits.argmax(dim=-1).tolist() == [1]
    assert talker.compute_logits(second.text_hidden_states).argmax(dim=-1).tolist() == [1]


def test_max_token_terminal_includes_only_new_codec_delta(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(3))
    info = {
        "request_id": "req-limit",
        "audio_state": {"step": 1, "max_tokens": 2},
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    assert info["audio_codes"]["accumulated"].tolist() == [4, 5, 3]
    assert output.multimodal_outputs["codes"]["audio"][0].tolist() == [[3]]
    assert output.multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [1]


def test_request_local_state_survives_missing_runner_buffer_update(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(3))
    first_info = {
        "request_id": "req-local-state",
        "audio_state": {"step": 1, "max_tokens": 3},
        "audio_codes": {"accumulated": torch.tensor([4])},
    }

    talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[first_info],
        request_token_spans=[(0, 1)],
    )
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-local-state"}],
        request_token_spans=[(0, 1)],
    )

    assert second.multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker._request_audio_states["req-local-state"]["step"] == 3


def test_missing_conditioning_fails_clearly() -> None:
    talker = _make_talker()

    with pytest.raises(ValueError, match="tts_token_ids and tts_hidden_states"):
        talker.preprocess(
            torch.tensor([0]),
            None,
            _omni_is_prefill=True,
            request_id="req-invalid",
        )


def test_request_cleanup_evicts_ar_rng_and_decode_state() -> None:
    talker = _make_talker()
    talker._request_generators["req-done"] = torch.Generator()
    talker._request_audio_states["req-done"] = {"step": 1}

    talker.on_requests_finished(["req-done"])
    talker._flush_deferred_cleanup()

    assert "req-done" not in talker._request_generators
    assert "req-done" not in talker._request_audio_states
