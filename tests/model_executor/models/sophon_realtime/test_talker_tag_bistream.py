# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tag talker adapter: bistream items -> embeddings, constrained sampling."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.sophon_realtime.core.bistream import BistreamSchedule, BistreamTokens
from vllm_omni.model_executor.models.sophon_realtime.talker_tag import SophonTagTalker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

T = BistreamTokens(speech_vocab=20)  # sos 20, eos 21, task 22, fill 23


def _talker() -> SophonTagTalker:
    t = object.__new__(SophonTagTalker)
    torch.nn.Module.__init__(t)
    t.model_stage = "cosyvoice3_talker"
    t.tokens = T
    t._streams = {}
    t._last_fed = {}
    t._batch_req_ids = []
    t._sampling_eps = 1e-5
    t.config = SimpleNamespace(llm={"sampling": {"win_size": 10, "tau_r": 0.1, "top_p": 0.8, "top_k": 25}})
    # stub embeddings: speech ids -> [id, 0], text ids -> [0, id]
    t._speech_embed = lambda ids: torch.tensor([[float(i), 0.0] for i in ids])
    t._text_embed = lambda ids: torch.tensor([[0.0, float(i)] for i in ids])
    return t


def _metadata(rows: int):
    return SimpleNamespace(temperature=torch.zeros(rows), top_p=None, top_k=None, generators={})


def _prefer(token: int) -> torch.Tensor:
    row = torch.full((T.speech_vocab + 200,), -10.0)
    row[token] = 10.0
    return row


def _sample(t: SophonTagTalker, rid: str, row: torch.Tensor) -> int:
    t.set_batch_req_ids([rid])
    return int(t.sample(row.unsqueeze(0).clone(), _metadata(1)).sampled_token_ids.reshape(-1)[0])


def _start(t: SophonTagTalker, rid: str, text, finished: bool) -> BistreamSchedule:
    st = BistreamSchedule(T)
    st.update_text(text, finished)
    t._streams[rid] = st
    t._last_fed[rid] = torch.tensor([[9.0, 9.0]])
    return st


def test_items_become_embeddings_and_repeat_reuses_the_last_input():
    t = _talker()
    st = _start(t, "r", [100, 101], True)
    st.out = [1] * 15
    assert _sample(t, "r", _prefer(2)) == T.fill
    fed = [t.preprocess(torch.tensor([0]), None, request_id="r")[1] for _ in range(4)]
    assert torch.equal(fed[0], torch.tensor([[9.0, 9.0]]))
    assert torch.equal(fed[1], torch.tensor([[0.0, 100.0]]))
    assert torch.equal(fed[3], torch.tensor([[float(T.task), 0.0]]))
    row = _prefer(T.fill)
    row[T.eos] = 5.0
    assert _sample(t, "r", row) == T.eos  # fill is masked in the final phase


def test_eos_is_masked_while_streaming():
    t = _talker()
    _start(t, "r", list(range(100, 110)), False)
    row = _prefer(T.eos)
    row[4] = 1.0
    assert _sample(t, "r", row) == 4


def test_intermediate_positions_are_discarded():
    t = _talker()
    st = _start(t, "r", list(range(100, 110)), False)
    st.out = [1] * 15
    assert _sample(t, "r", _prefer(3)) == T.fill
    t.preprocess(torch.tensor([0]), None, request_id="r")
    assert _sample(t, "r", _prefer(5)) == T.fill  # intermediate: forced fill, not recorded
    assert st.out[-1] == T.fill and len(st.out) == 16


def test_rows_without_a_stream_fall_back_to_the_stock_sampler(monkeypatch):
    t = _talker()
    called = []
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.cosyvoice3.cosyvoice3.CosyVoice3Model.sample",
        lambda self, logits, md, **kw: called.append(1),
    )
    t.set_batch_req_ids(["unknown"])
    t.sample(_prefer(1).unsqueeze(0), _metadata(1))
    assert called
