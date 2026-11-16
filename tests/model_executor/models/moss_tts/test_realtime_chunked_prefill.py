# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_talker_preserves_state_across_chunked_prefill() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSRealtimeTalkerForGeneration,
    )

    talker = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    talker.n_vq = 2
    talker.audio_pad_token = 1024
    talker.text_vocab_size = 16
    talker.text_pad_id = 2
    talker._batch_state = None
    talker._batch_state_spans = None

    embedded_audio: list[torch.Tensor | None] = []

    def build_input_embeds(input_ids, audio_codes):
        embedded_audio.append(audio_codes.clone() if audio_codes is not None else None)
        return torch.zeros((input_ids.shape[0], 1))

    generated: list[None] = []

    def generate_frame(*args, **kwargs):
        generated.append(None)
        return torch.ones((1, talker.n_vq), dtype=torch.long)

    talker._build_input_embeds = build_input_embeds
    talker.local_transformer = SimpleNamespace(generate_frame=generate_frame)
    talker.local_lm_heads = []

    ref_codes = torch.arange(10, dtype=torch.long).reshape(5, 2)
    info: dict[str, Any] = {
        "codes": {"ref": ref_codes},
        "ids": {"all": [7, 8]},
        "max_new_frames": [3],
    }
    chunks = [(0, 2, False), (2, 2, False), (4, 1, True)]

    state: dict[str, Any] | None = None
    for computed, length, eligible in chunks:
        _, _, update = talker.preprocess(
            torch.ones(length, dtype=torch.long),
            None,
            _omni_is_prefill=True,
            _omni_num_computed_tokens=computed,
            _omni_prompt_len=5,
            **info,
        )
        info.update(update)
        if state is None:
            state = info["audio_state"]
        else:
            assert info["audio_state"] is state

        output = talker.make_omni_output(
            torch.ones((length, 1)),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, length)],
            request_sample_eligible=[eligible],
        )
        if not eligible:
            assert output.multimodal_outputs == {}
        logits = talker.compute_logits(torch.ones((1, 1)))
        assert logits.argmax().item() == (7 if eligible else talker.text_pad_id)

    assert state is not None
    assert info["ref_offset"] == 5
    assert state["step"] == 1
    assert state["text_cursor"] == 1
    assert state["remaining_text"] == [7, 8]
    assert len(generated) == 1
    assert output.multimodal_outputs["codes"]["audio"][0].shape == (1, 2)
    for actual, expected in zip(embedded_audio, ref_codes.split((2, 2, 1)), strict=True):
        assert actual is not None
        torch.testing.assert_close(actual, expected)
