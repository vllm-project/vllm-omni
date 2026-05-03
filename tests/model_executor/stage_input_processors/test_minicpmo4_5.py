# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.minicpmo4_5 import (
    TTS_BOS_ID,
    TTS_EOS_ID,
    talker2code2wav,
    thinker2talker,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_talker2code2wav_forwards_token_ids_and_canonical_ref_audio():
    output = SimpleNamespace(token_ids=[101, 102, 103])
    stage = SimpleNamespace(
        engine_outputs=[SimpleNamespace(outputs=[output], finished=True)],
    )
    prompt = [
        {
            "additional_information": {
                "ref_audio": [(torch.tensor([0.1, -0.2], dtype=torch.float32), torch.tensor(16000))]
            }
        }
    ]

    prompts = talker2code2wav(stage_list=[stage], engine_input_source=[0], prompt=prompt)

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [101, 102, 103]
    ref_audio = prompts[0]["additional_information"]["ref_audio"]
    assert ref_audio["sr"] == 16000
    assert ref_audio["wav"] == pytest.approx([0.1, -0.2], rel=1e-5, abs=1e-6)


def test_talker2code2wav_leaves_ref_audio_unset_when_absent():
    output = SimpleNamespace(token_ids=[201, 202])
    stage = SimpleNamespace(
        engine_outputs=[SimpleNamespace(outputs=[output], finished=True)],
    )
    prompt = [{"additional_information": {"text": ["hello"]}}]

    prompts = talker2code2wav(stage_list=[stage], engine_input_source=[0], prompt=prompt)

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [201, 202]
    assert prompts[0]["additional_information"] is None


def test_thinker2talker_allows_final_token_without_matching_latent():
    latent = torch.randn(5, 4096, dtype=torch.float32)
    output = SimpleNamespace(
        token_ids=[TTS_BOS_ID, 11, 12, TTS_EOS_ID],
        multimodal_output={"latent": latent},
    )
    stage = SimpleNamespace(
        engine_outputs=[SimpleNamespace(prompt_token_ids=[101, 102], outputs=[output])],
    )

    prompts = thinker2talker(stage_list=[stage], engine_input_source=[0], prompt=None)

    assert len(prompts) == 1
    info = prompts[0]["additional_information"]
    assert info["llm_tokens"].tolist() == [11, 12]
    assert tuple(info["tts_hidden_states"].shape) == (2, 4096)
