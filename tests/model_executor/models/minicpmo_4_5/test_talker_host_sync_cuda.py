# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MiniCPM-o 4.5 Talker's per-step path must not block the host on the GPU."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.minicpmo_4_5.test_talker_host_sync import _make_talker

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)]


def test_talker_condition_upload_does_not_synchronize_cuda() -> None:
    talker = _make_talker("cuda")
    talker.emb_text = nn.Embedding(16, 4).cuda()
    talker.projector_semantic = nn.Linear(6, 4).cuda()
    talker._normalize = True
    talker._text_eos_id = 14
    talker._tts_bos_id = 15
    token_ids = torch.tensor([2, 3, 4])
    hidden_states = torch.randn(3, 6)
    expected = talker._build_condition_embeddings(token_ids.cuda(), hidden_states.cuda())
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.inference_mode():
            condition = talker._build_condition_embeddings(token_ids, hidden_states)
            boundary = talker._build_condition_embeddings(token_ids[:0], hidden_states[:0])
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.equal(condition, expected)
    assert boundary.shape == (2, 4)
