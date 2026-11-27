# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.model_executor.models.lfm3_omni.backbone import RequestState
from vllm_omni.model_executor.models.lfm3_omni.model import Lfm3OmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class FrameConfig:
    num_codebooks: int = 12
    maskgit_schedule: tuple[int, ...] = (4, 4, 4)


@pytest.mark.parametrize("penalty", [0.8, 1.0, 1.2])
def test_vectorized_greedy_preserves_reveal_order_and_repetition(penalty):
    generator = torch.Generator().manual_seed(17)
    base_logits = torch.randn(1, 12, 32, generator=generator)

    def depth(hidden, codes, masked):
        # Each reveal pass depends on the codes selected by earlier passes.
        return base_logits.roll(int(codes.sum()), dims=-1) + masked[..., None] * 0.25

    models, states = [], []
    for vectorized in (False, True):
        model = Lfm3OmniForConditionalGeneration.__new__(Lfm3OmniForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.config = FrameConfig()
        model._vectorized_greedy = vectorized
        model._depth_logits = depth
        models.append(model)
        states.append(
            RequestState(
                generator=torch.Generator().manual_seed(42),
                audio_temperature=0.0,
                audio_repetition_penalty=penalty,
                # Duplicated IDs must receive one penalty, not one per occurrence.
                audio_history=[torch.zeros(12, dtype=torch.long)] * 3,
            )
        )
    for _ in range(12):
        frames = [model._audio_frame(torch.zeros(8), state) for model, state in zip(models, states)]
        torch.testing.assert_close(*frames, rtol=0, atol=0)
        assert torch.equal(states[0].generator.get_state(), states[1].generator.get_state())
        for frame, state in zip(frames, states):
            state.audio_history.append(frame)


def test_finished_request_releases_state_even_when_last_batch_retains_it():
    model = Lfm3OmniForConditionalGeneration.__new__(Lfm3OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    state = RequestState(
        conv={0: torch.ones(1, 8, 6)},
        kv={2: (torch.ones(1, 1, 4, 8), torch.ones(1, 1, 4, 8))},
        next_audio_embed=torch.ones(8),
        audio_history=[torch.zeros(12, dtype=torch.long)],
    )
    other = RequestState(conv={0: torch.ones(1, 8, 6)})
    model._states = {"finished": state, "running": other}
    model._batch_states = [state, other]
    model.on_requests_finished(["finished", "finished"])
    assert set(model._states) == {"running"}
    assert other.conv
    assert not state.conv and not state.kv and not state.audio_history
    assert state.next_audio_embed is None
