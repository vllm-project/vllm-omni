# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero's per-forward RoPE table and the two forms of its rotation."""

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero.causal_wan_model import (
    causal_rope_action_freqs,
    rope_apply,
    rope_params,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HEAD_DIM = 8
VIDEO_LEN = 12
NUM_ACTION = 3
NUM_STATE = 2
STEP = 1


def _tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    video = rope_params(VIDEO_LEN, HEAD_DIM).view(VIDEO_LEN, 1, -1)
    return video, rope_params(64, HEAD_DIM), rope_params(16, HEAD_DIM)


def _step_table(action_register_length: int | None) -> torch.Tensor:
    video, action, state = _tables()
    return causal_rope_action_freqs(video, action, state, action_register_length, NUM_ACTION, NUM_STATE, STEP)


def test_video_only_table_is_the_video_table() -> None:
    video, _, _ = _tables()
    assert torch.equal(_step_table(None), torch.view_as_real(video))


def test_action_table_appends_this_steps_rows() -> None:
    video, action, state = _tables()
    table = _step_table(NUM_ACTION + NUM_STATE)

    assert table.shape == (VIDEO_LEN + NUM_ACTION + NUM_STATE, 1, HEAD_DIM // 2, 2)
    assert torch.equal(table[:VIDEO_LEN], torch.view_as_real(video))
    step_rows = torch.cat(
        [
            action[STEP * NUM_ACTION : (STEP + 1) * NUM_ACTION],
            state[STEP * NUM_STATE : (STEP + 1) * NUM_STATE],
        ]
    )
    assert torch.equal(table[VIDEO_LEN:], torch.view_as_real(step_rows.view(-1, 1, HEAD_DIM // 2)))


def test_action_table_rejects_wrong_register_length() -> None:
    with pytest.raises(ValueError, match="action_register_length"):
        _step_table(NUM_ACTION + NUM_STATE + 1)


@pytest.mark.parametrize("action_register_length", [None, NUM_ACTION + NUM_STATE])
def test_eager_and_compiled_forms_agree_to_one_bfloat16_ulp(monkeypatch, action_register_length) -> None:
    table = _step_table(action_register_length)
    torch.manual_seed(0)
    x = torch.randn(2, table.shape[0], 3, HEAD_DIM, dtype=torch.bfloat16)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    eager = rope_apply(x, table).to(torch.bfloat16)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    compiled = rope_apply(x, table).to(torch.bfloat16)

    torch.testing.assert_close(compiled, eager, rtol=2**-7, atol=1e-5)
