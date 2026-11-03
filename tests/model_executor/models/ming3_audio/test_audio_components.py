# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.common.ming.audio_lengths import (
    max_mel_frames_for_audio_tokens,
    ming3_audio_token_count,
)
from vllm_omni.model_executor.models.common.ming.audio_projector import (
    Ming3AudioProjector,
)
from vllm_omni.model_executor.models.common.ming.whisper_encoder import (
    WhisperAudioEncoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_AUDIO_ARGS = {
    "chunk_size": 3000,
    "kernels": [3, 3, 3],
    "strides": [2, 2, 2],
    "paddings": [1, 1, 1],
}


def _reference_token_count(mel_length: int) -> int:
    total = 0
    for offset in range(0, mel_length, 3000):
        length = min(3000, mel_length - offset)
        length = (length + 2 - 3) // 2 + 1
        for _ in range(3):
            length = (length + 2 - 3) // 2 + 1
        total += length
    return total


@pytest.mark.parametrize("mel_length", (1, 15, 16, 17, 2999, 3000, 3001, 6000, 6001))
def test_ming3_audio_lengths_match_release_chunk_boundaries(mel_length: int):
    assert ming3_audio_token_count(mel_length, **_AUDIO_ARGS) == _reference_token_count(mel_length)


@pytest.mark.parametrize("token_budget", (1, 187, 188, 189, 376, 512))
def test_inverse_audio_budget_is_exact(token_budget: int):
    frames = max_mel_frames_for_audio_tokens(token_budget, **_AUDIO_ARGS)
    assert ming3_audio_token_count(frames, **_AUDIO_ARGS) <= token_budget
    assert ming3_audio_token_count(frames + 1, **_AUDIO_ARGS) > token_budget


def test_projector_names_and_shapes_match_checkpoint():
    projector = Ming3AudioProjector(
        audio_dim=8,
        llm_dim=16,
        kernels=[3, 3, 3],
        strides=[2, 2, 2],
        paddings=[1, 1, 1],
    )
    state = projector.state_dict()

    assert "proj.0.weight" in state
    assert "proj.2.weight" in state
    assert "proj.4.weight" in state
    assert "proj.7.weight" in state
    assert projector(torch.zeros(2, 11, 8)).shape == (2, 2, 16)
    assert projector.output_length(11) == 2


def test_shared_whisper_encoder_supports_both_execution_paths():
    encoder = WhisperAudioEncoder(
        n_mels=4,
        n_ctx=16,
        n_state=8,
        n_head=2,
        n_layer=1,
        use_flash_attn=False,
        learned_positional_embedding=True,
    )
    padded, lengths = encoder.forward_padded(torch.zeros(2, 9, 4), [9, 5])
    packed = encoder(
        [torch.zeros(4, 9), torch.zeros(4, 5)],
        [9, 5],
    )

    assert padded.shape == (2, 5, 8)
    assert lengths == [5, 3]
    assert packed.shape == (8, 8)
