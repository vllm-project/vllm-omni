# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRV2 codes-delivery mechanics for the Higgs Audio v2 talker.

Drives the real split/postprocess pair with synthetic batches: cu_num_logits
is an inclusive prefix of length num_reqs + 1, and mixed prefill+decode
steps give some requests no logits rows. Wrong slicing shifts every
request's audio codes onto its neighbor — the batch>1 slot-shift failure.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _talker():
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
        HiggsAudioV2TalkerForConditionalGeneration as C,
    )

    t = C.__new__(C)
    torch.nn.Module.__init__(t)
    return t


def _batch(req_ids, cu):
    return SimpleNamespace(req_ids=list(req_ids), num_reqs=len(req_ids), cu_num_logits=torch.tensor(cu))


def test_decode_step_maps_each_request_to_its_own_row():
    t = _talker()
    # Three decode rows; row content is unique per request.
    codes = torch.tensor([[11, 11], [22, 22], [33, 33]], dtype=torch.long)
    pending = t._split_codes_by_request(codes, _batch(["a", "b", "c"], [0, 1, 2, 3]))
    assert set(pending) == {"a", "b", "c"}
    assert pending["a"][0, 0].item() == 11
    assert pending["b"][0, 0].item() == 22
    assert pending["c"][0, 0].item() == 33


def test_mixed_prefill_step_skips_request_without_logits_rows():
    t = _talker()
    # "a" is still prefilling (no logits row); "b" and "c" decode. Their rows
    # are 0 and 1 — a positional-cursor consumer would hand row 0 to "a".
    codes = torch.tensor([[22, 22], [33, 33]], dtype=torch.long)
    pending = t._split_codes_by_request(codes, _batch(["a", "b", "c"], [0, 0, 1, 2]))
    assert set(pending) == {"b", "c"}
    assert pending["b"][0, 0].item() == 22
    assert pending["c"][0, 0].item() == 33


def test_warmup_requests_are_excluded():
    t = _talker()
    codes = torch.tensor([[11, 11], [22, 22]], dtype=torch.long)
    pending = t._split_codes_by_request(codes, _batch(["_warmup_abc", "real"], [0, 1, 2]))
    assert set(pending) == {"real"}
    assert pending["real"][0, 0].item() == 22


def _audio_row(update):
    return update["codes"]["audio"][0].tolist()


def test_postprocess_serves_pending_map_and_falls_back_to_cursor():
    t = _talker()
    t._pending_codes_by_req = {
        "a": torch.tensor([[11, 11]], dtype=torch.long),
        "b": torch.tensor([[-1, -1]], dtype=torch.long),
    }
    # Hit: real codes for "a"; placeholder row for "b" emits nothing; "c" has no entry.
    assert _audio_row(t.postprocess(torch.zeros(1, 4), req_id="a")) == [11, 11]
    assert t.postprocess(torch.zeros(1, 4), req_id="b") == {}
    assert t.postprocess(torch.zeros(1, 4), req_id="c") == {}

    # V1 fallback: without the pending map, the positional cursor walks rows.
    t._pending_codes_by_req = None
    t._last_audio_codes = torch.tensor([[11, 11], [22, 22]], dtype=torch.long)
    t._postprocess_cursor = 0
    first = t.postprocess(torch.zeros(1, 4), req_id="a")
    second = t.postprocess(torch.zeros(1, 4), req_id="b")
    assert _audio_row(first) == [11, 11]
    assert _audio_row(second) == [22, 22]


def test_audio_state_survives_batch_row_shift():
    """req_id-keyed state must survive MRV2's batch compaction.

    Request "a" at batch row 0 accumulates audio state. Between steps,
    another request finishes and "a" shifts to row 1 (compaction). The
    batch-row-keyed view rebuilt by the hook must still resolve "a"'s state
    at its new row — a batch-row-keyed store would have lost it.
    """
    t = _talker()

    # Step N: "a" at batch row 0, "b" at row 1. State written by req_id.
    t._audio_state = {
        "a": {"num_delay": 3, "num_remaining_delays": None, "audio_out_ids": torch.tensor([[1, 2, 3]]).T},
        "b": {"num_delay": 1, "num_remaining_delays": None, "audio_out_ids": torch.tensor([[4, 5]]).T},
    }

    # Between steps, "b" finishes. Compaction: "a" moves to row 0 in a
    # single-request batch. The hook rebuilds the view.
    batch_after = SimpleNamespace(req_ids=["a"], num_reqs=1)
    t._rebuild_audio_state_view(batch_after)

    # embed_input_ids looks up by flat position 0 (= batch row 0).
    # The view must contain "a"'s state at row 0, not "b"'s stale entry.
    assert 0 in t._audio_state_view
    assert t._audio_state_view[0]["num_delay"] == 3
    assert t._audio_state_view[0]["audio_out_ids"].tolist() == [[1], [2], [3]]
    # "b" must not appear anywhere in the view.
    assert len(t._audio_state_view) == 1
