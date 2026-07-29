# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongCat-Next audio-generation plumbing.

These cover the two places where a silent regression drops all generated
audio: make_omni_output must hand the runner an OmniOutput (a bare tensor
makes extract_multimodal_outputs yield ``{}``), and _advance_audio_gen must
reproduce the reference's delay/text_end gating.
"""

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import (
    AUDIOGEN_START_TOKEN_ID,
    AUDIOTEXT_PAD_TOKEN_ID,
    AUDIOTEXT_START_TOKEN_ID,
)
from vllm_omni.model_executor.models.longcat_next.modeling_longcat_next import (
    LongcatNextForCausalLM,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

HIDDEN = 4
NUM_LEVELS = 8


def _model(**attrs):
    """Stub carrying only what the methods under test touch."""
    base = {
        "_audio_gen": {},
        "_audio_debug": False,
        "_audio_delay_default": 0,
        "config": SimpleNamespace(hidden_size=HIDDEN),
    }
    base.update(attrs)
    return SimpleNamespace(**base)


def _frame(value: int) -> torch.Tensor:
    return torch.full((1, NUM_LEVELS), value, dtype=torch.long)


def _make_omni_output(model, hidden, buffer):
    return LongcatNextForCausalLM.make_omni_output(
        model, hidden, model_intermediate_buffer=buffer
    )


def _advance(model, req_id, last_token):
    return LongcatNextForCausalLM._advance_audio_gen(
        model, req_id, last_token, device=torch.device("cpu"), dtype=torch.float32
    )


# ---------------------------------------------------------------- #
# make_omni_output
# ---------------------------------------------------------------- #


def test_make_omni_output_returns_omni_output_with_codes():
    """The regression that silently dropped every frame: returning the bare
    hidden-states tensor makes the runner's extract_multimodal_outputs fall
    through to ``{}``, so codes never reach the request."""
    hidden = torch.zeros(1, HIDDEN)
    buffer = [{"codes": {"audio": _frame(7)}}]

    out = _make_omni_output(_model(), hidden, buffer)

    assert isinstance(out, OmniOutput)
    assert torch.equal(out.text_hidden_states, hidden)
    assert torch.equal(out.multimodal_outputs["codes"]["audio"], _frame(7))


def test_make_omni_output_consumes_frame_so_it_is_not_re_emitted():
    """The buffer entry persists across steps; leaving it in place would
    re-emit the same frame on every later step where talker_mtp did not run
    and duplicate audio."""
    model = _model()
    buffer = [{"codes": {"audio": _frame(7)}}]

    first = _make_omni_output(model, torch.zeros(1, HIDDEN), buffer)
    second = _make_omni_output(model, torch.zeros(1, HIDDEN), buffer)

    assert first.multimodal_outputs["codes"]["audio"].shape[0] == 1
    assert second.multimodal_outputs == {}


def test_make_omni_output_drops_discarded_rows():
    """talker_mtp marks discarded frames with an all -1 row to keep the
    returned tensor batch-aligned; those are not real codes."""
    buffer = [{"codes": {"audio": _frame(-1)}}]

    out = _make_omni_output(_model(), torch.zeros(1, HIDDEN), buffer)

    assert out.multimodal_outputs == {}


def test_make_omni_output_keeps_only_real_rows_in_mixed_batch():
    mixed = torch.cat([_frame(-1), _frame(5)], dim=0)
    buffer = [{"codes": {"audio": mixed}}]

    out = _make_omni_output(_model(), torch.zeros(2, HIDDEN), buffer)

    assert torch.equal(out.multimodal_outputs["codes"]["audio"], _frame(5))


def test_make_omni_output_empty_without_codes():
    out = _make_omni_output(_model(), torch.zeros(1, HIDDEN), [{}])

    assert isinstance(out, OmniOutput)
    assert out.multimodal_outputs == {}


def test_make_omni_output_stashes_hidden_state_for_next_step():
    """talker_mtp conditions on the previous step's last hidden state."""
    model = _model(_audio_gen={"r0": {"terminal": False}})
    hidden = torch.arange(2 * HIDDEN, dtype=torch.float32).reshape(2, HIDDEN)

    _make_omni_output(model, hidden, [{}])

    assert torch.equal(model._audio_gen["r0"]["last_hidden"], hidden[-1:])


# ---------------------------------------------------------------- #
# _advance_audio_gen
# ---------------------------------------------------------------- #


def test_audiogen_start_creates_state():
    model = _model()

    _advance(model, "r0", AUDIOGEN_START_TOKEN_ID)

    assert "r0" in model._audio_gen


def test_delay_zero_defers_first_real_frame_by_one_step():
    """With delay=0 the reference enables audio at the *end* of step 0, after
    that step's codes were discarded, so the first kept frame is step 1 and
    step 0 is the one carrying audiotext_start."""
    model = _model()
    _advance(model, "r0", AUDIOGEN_START_TOKEN_ID)
    state = model._audio_gen["r0"]

    assert state["ext_id"] == AUDIOTEXT_START_TOKEN_ID
    assert state["audio_start"] is False

    _advance(model, "r0", 123)

    assert state["ext_id"] == AUDIOTEXT_PAD_TOKEN_ID
    assert state["audio_start"] is True


def test_text_end_set_on_first_audiotext_pad():
    """The first AUDIOTEXT_PAD sampled as the visible token ends the spoken
    transcript (reference output_processor.py:233-237)."""
    model = _model()
    _advance(model, "r0", AUDIOGEN_START_TOKEN_ID)
    state = model._audio_gen["r0"]
    assert state["text_end"] is False

    _advance(model, "r0", AUDIOTEXT_PAD_TOKEN_ID)

    assert state["text_end"] is True


def test_advance_emits_mtp_inputs_so_runner_calls_talker():
    model = _model()

    update = _advance(model, "r0", AUDIOGEN_START_TOKEN_ID)

    assert "mtp_inputs" in update
    last_hidden, text_step = update["mtp_inputs"]
    assert last_hidden.shape == (1, HIDDEN)
    assert text_step.shape == (1, HIDDEN)


def test_terminal_request_stops_emitting_mtp_inputs():
    model = _model()
    _advance(model, "r0", AUDIOGEN_START_TOKEN_ID)
    model._audio_gen["r0"]["terminal"] = True

    update = _advance(model, "r0", 123)

    assert "mtp_inputs" not in update
