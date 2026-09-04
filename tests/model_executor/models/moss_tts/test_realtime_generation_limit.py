# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_talker_stops_at_max_new_frames() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSRealtimeTalkerForGeneration,
    )

    talker = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    talker.n_vq = 4
    talker.audio_pad_token = 1024
    talker.text_vocab_size = 8
    talker.text_pad_id = 2
    talker.audio_eos_id = 3
    talker._batch_state = None
    talker._build_input_embeds = lambda input_ids, audio_codes: torch.zeros((input_ids.shape[0], 1))

    generated: list[None] = []

    def generate_frame(*args, **kwargs):
        generated.append(None)
        return torch.ones((1, talker.n_vq), dtype=torch.long)

    talker.local_transformer = SimpleNamespace(generate_frame=generate_frame)
    talker.local_lm_heads = []

    _, _, info = talker.preprocess(
        torch.tensor([1, 2]),
        None,
        max_new_frames=[2],
    )
    assert info["audio_state"]["max_new_frames"] == 2

    for _ in range(3):
        output = talker.make_omni_output(
            torch.ones((1, 1)),
            runtime_additional_information=[info],
            request_token_spans=[(0, 1)],
        )

    assert len(generated) == 2
    assert info["audio_state"]["step"] == 2
    assert info["audio_state"]["is_stopping"] is True
    assert output.multimodal_outputs == {}
    assert talker.compute_logits(torch.ones((1, 1))).argmax().item() == talker.audio_eos_id
