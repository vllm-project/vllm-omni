# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.easymagpie.test_decode_preparation import _model
from vllm_omni.model_executor.models.easymagpie.device_decode_state import DeviceDecodeState
from vllm_omni.model_executor.models.easymagpie.feedback_snapshot import is_owned_feedback
from vllm_omni.model_executor.models.easymagpie.runner import EasyMagpieGPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def setup(width=2):
    model = _model("cpu", width)
    model._batched_owned_feedback_enabled = True
    model._single_stage_audio = False
    model._out_codes = torch.arange(32 * 4).reshape(32, 4)
    model._device_decode_state = state = DeviceDecodeState(model, 3, 32, 32)
    if width:
        state.row_slots_host = {2: 0, 7: 1}
        state.row_slots[2], state.row_slots[7] = 0, 1
        state.ended[0], state.ended[1] = True, False
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = model
    runner.requests = {name: SimpleNamespace() for name in ("a", "b", "prefill")}
    runner.model_intermediate_buffer = {}
    hidden = torch.zeros(32, 3)
    return model, runner, hidden


def test_batch_snapshot_survives_delayed_postprocess_scratch_and_slot_reuse():
    model, runner, hidden = setup()
    expected_audio, expected_phonemes = model._out_codes.clone(), model._dec_phoneme_tokens.clone()
    batch = model.make_omni_output(hidden)
    audio = batch.multimodal_outputs["codes"]["audio"]
    model._out_codes.fill_(999)
    model._dec_phoneme_tokens.fill_(998)
    model._device_decode_state.ended.logical_not_()
    model._device_decode_state.row_slots_host.clear()
    for req_id, row in [("a", 2), ("b", 7), ("prefill", 11)]:
        update = model.postprocess(hidden[row : row + 1], batch.multimodal_outputs)
        assert all(is_owned_feedback(value, key) for key, value in update.items())
        runner._update_intermediate_buffer(req_id, update)
        stored = runner.model_intermediate_buffer[req_id]
        assert stored["last_audio_codes"].data_ptr() == audio[row : row + 1].data_ptr()
        torch.testing.assert_close(stored["last_audio_codes"], expected_audio[row : row + 1], rtol=0, atol=0)
        torch.testing.assert_close(stored["last_phoneme_token"], expected_phonemes[row : row + 1], rtol=0, atol=0)
        if row in (2, 7):
            assert stored["phoneme_ended"].item() is (row == 2)
        else:
            assert "phoneme_ended" not in stored
    del batch, audio
    gc.collect()
    torch.testing.assert_close(runner.model_intermediate_buffer["a"]["last_audio_codes"], expected_audio[2:3])
    # An unscheduled request can resume/reseed after its original DDS slot is reused.
    state = model._device_decode_state
    info = dict(runner.model_intermediate_buffer["a"], decode_offset=8, text_tokens=[1, 2, 3])
    model.prepare_decode_batch([(17, info)], torch.device("cpu"), request_keys=[("a", 4)])
    slot = state.row_slots_host[17]
    torch.testing.assert_close(state.audio[slot], expected_audio[2])
    torch.testing.assert_close(state.phoneme[slot], expected_phonemes[2])
    assert state.ended[slot].item() is True


def test_next_batch_cannot_overwrite_previous_owned_feedback():
    model, runner, hidden = setup()
    first = model.make_omni_output(hidden)
    first_update = model.postprocess(hidden[2:3], first.multimodal_outputs)
    runner._update_intermediate_buffer("a", first_update)
    expected = {key: value.clone() for key, value in runner.model_intermediate_buffer["a"].items()}
    model._out_codes.fill_(77)
    model._dec_phoneme_tokens.fill_(78)
    model._device_decode_state.ended.zero_()
    second = model.make_omni_output(hidden)
    second_update = model.postprocess(hidden[7:8], second.multimodal_outputs)
    runner._update_intermediate_buffer("b", second_update)
    for key, value in expected.items():
        torch.testing.assert_close(runner.model_intermediate_buffer["a"][key], value, rtol=0, atol=0)
    assert runner.model_intermediate_buffer["b"]["last_audio_codes"].eq(77).all()


def test_only_certified_correct_key_and_storage_bypass_clone():
    model, runner, hidden = setup()
    output = model.make_omni_output(hidden)
    certified = model.postprocess(hidden[2:3], output.multimodal_outputs)["last_audio_codes"]
    assert is_owned_feedback(certified, "last_audio_codes")
    assert not is_owned_feedback(certified, "last_phoneme_token")
    external = certified.clone()
    external._easymagpie_owned_feedback = certified._easymagpie_owned_feedback
    assert not is_owned_feedback(external, "last_audio_codes")
    for value in (external, torch.ones(1, 4, dtype=torch.long)):
        stored: dict = {}
        runner._store_value(stored, "last_audio_codes", value, {"last_audio_codes"})
        assert stored["last_audio_codes"].data_ptr() != value.data_ptr()
    # Ordinary model output without our private snapshot still takes old ownership path.
    ordinary_audio = model._out_codes.clone()
    update = model.postprocess(hidden[2:3], {"codes.audio": ordinary_audio})
    assert not any(is_owned_feedback(value, key) for key, value in update.items())


@pytest.mark.parametrize("single_stage", [False, True])
@pytest.mark.parametrize("phoneme_width", [0, 2])
def test_modes_and_snapshot_owner_has_no_reference_cycle(single_stage, phoneme_width):
    model, runner, hidden = setup(phoneme_width)
    model._single_stage_audio = single_stage
    output = model.make_omni_output(hidden)
    audio = output.multimodal_outputs["model_outputs"] if single_stage else output.multimodal_outputs["codes"]["audio"]
    snapshot = audio._easymagpie_feedback_snapshot
    audio_ref, snapshot_ref = weakref.ref(audio), weakref.ref(snapshot)
    update = model.postprocess(hidden[2:3], output.multimodal_outputs)
    assert is_owned_feedback(update["last_audio_codes"], "last_audio_codes")
    assert ("last_phoneme_token" in update) is bool(phoneme_width)
    del output, audio, snapshot, update
    assert audio_ref() is None and snapshot_ref() is None
