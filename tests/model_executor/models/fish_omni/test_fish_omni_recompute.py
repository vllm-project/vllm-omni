# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for resuming a Drama 3 request after preemption.

vLLM recomputes a preempted request as one prefill of its prompt plus every
token it generated. The rebuilt rows must carry the codes of the frames that
were already produced, the one token that was sampled but never fed back gets
its frame now, and the tower history is kept rather than prefilled again.
"""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.fish_omni.codebook_tower import CodebookTower
from vllm_omni.model_executor.models.fish_omni.configuration_fish_omni import FishOmniAudioDecoderConfig
from vllm_omni.model_executor.models.fish_omni.fish_omni_model import FishOmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEMANTIC_BEGIN = 100
PROMPT_ROWS = 6
DIM = 32


def _model() -> FishOmniForConditionalGeneration:
    """A model instance with only the tower state the recompute path reads."""
    torch.manual_seed(0)
    config = FishOmniAudioDecoderConfig(
        dim=DIM,
        n_head=4,
        n_local_heads=2,
        head_dim=8,
        intermediate_size=48,
        num_codebooks=4,
        vocab_size=16,
        shared_enrichment_layers=2,
        rotary_percent=0.5,
    )
    model = FishOmniForConditionalGeneration.__new__(FishOmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.tower = CodebookTower(config, semantic_begin_id=SEMANTIC_BEGIN, cache_slots=4, max_len=64).eval()
    model._tower_graphs = None
    model._request_states = {}
    return model


def _prompt_frames() -> torch.Tensor:
    frames = torch.zeros((PROMPT_ROWS, 5), dtype=torch.long)
    frames[:, 0] = torch.arange(PROMPT_ROWS)  # text rows
    frames[2:4, 1:] = torch.randint(0, 16, (2, 4))  # a reference audio span
    frames[2:4, 0] = SEMANTIC_BEGIN + frames[2:4, 1]
    return frames


@torch.inference_mode()
def test_recompute_rebuilds_generated_frames_and_keeps_tower_history() -> None:
    model = _model()
    frames = _prompt_frames()
    hidden = torch.randn(PROMPT_ROWS + 3, DIM)
    state = model._prefill_tower(hidden[:PROMPT_ROWS], frames)
    generated = torch.tensor([SEMANTIC_BEGIN + 3, SEMANTIC_BEGIN + 7, SEMANTIC_BEGIN + 1])

    # Two decode steps feed the first two generated tokens back.
    first = model._tower_step([state], generated[0:1], hidden[PROMPT_ROWS - 1 : PROMPT_ROWS])[0]
    second = model._tower_step([state], generated[1:2], hidden[PROMPT_ROWS : PROMPT_ROWS + 1])[0]
    position_before = state.next_position

    # Preempted after sampling the third token: the recompute schedules prompt + 3 rows.
    info = {"drama3_tower": state, "hidden_states": {"last": hidden[PROMPT_ROWS + 1]}}
    rows = model._recompute_frames(generated, info, PROMPT_ROWS, PROMPT_ROWS + 3)

    assert rows.shape == (3, 5)
    assert torch.equal(rows[:, 0], generated)
    assert torch.equal(rows[0, 1:], first[0])
    assert torch.equal(rows[1, 1:], second[0])
    # The never-fed token got its frame from one more tower step.
    assert len(state.generated) == 3
    assert state.next_position == position_before + 1
    assert rows[2, 1] == generated[2] - SEMANTIC_BEGIN  # code 0 is the main token's offset


@torch.inference_mode()
def test_recompute_without_tower_history_fails_loudly() -> None:
    model = _model()
    frames = _prompt_frames()
    state = model._prefill_tower(torch.randn(PROMPT_ROWS, DIM), frames)
    model.tower.cache.release(state.slot, state)
    info = {"drama3_tower": state, "hidden_states": {"last": torch.randn(DIM)}}
    with pytest.raises(RuntimeError, match="preempted request"):
        model._recompute_frames(torch.tensor([SEMANTIC_BEGIN]), info, PROMPT_ROWS, PROMPT_ROWS + 1)


@torch.inference_mode()
def test_request_finished_before_first_decode_releases_its_slot() -> None:
    """Prefill, then finish with no decode step (max_tokens=1, immediate end of
    speech, or a cancel): the public hooks alone must free the slot and drop the
    queued prompt rows."""
    model = _model()
    frames = _prompt_frames()
    update = model.postprocess(torch.randn(PROMPT_ROWS, DIM), request_id="r", drama3_prompt={"frames": frames})
    state = update["drama3_tower"]
    assert model.tower.cache.owns(state.slot, state)
    assert state.pending is not None
    model.on_requests_finished({"r"})
    assert not model.tower.cache.owns(state.slot, state)
    assert state.pending is None
    assert "r" not in model._request_states


@torch.inference_mode()
def test_finished_request_after_decode_releases_its_slot() -> None:
    model = _model()
    hidden = torch.randn(PROMPT_ROWS + 1, DIM)
    update = model.postprocess(hidden[:PROMPT_ROWS], request_id="r", drama3_prompt={"frames": _prompt_frames()})
    state = update["drama3_tower"]
    model._tower_step([state], torch.tensor([SEMANTIC_BEGIN + 3]), hidden[PROMPT_ROWS - 1 : PROMPT_ROWS])
    model.on_requests_finished({"r"})
    assert not model.tower.cache.owns(state.slot, state)
