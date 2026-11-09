# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRV2 codes-delivery mechanics for the Higgs Audio v2 talker.

Drives the real split/postprocess pair with synthetic batches: cu_num_logits
is an inclusive prefix of length num_reqs + 1 and stays the authority for
row-to-request ownership. Wrong slicing shifts every request's audio codes
onto its neighbor. Also locks the req_id keying for bias state lookup and
delay-pattern attribution, the mixed-step audio-feedback owner walk, the V1
bias walks, and the finished-request state cleanup.
"""

from types import SimpleNamespace
from unittest import mock

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


def _batch(req_ids, cu, with_np=False):
    ns = SimpleNamespace(req_ids=list(req_ids), num_reqs=len(req_ids), cu_num_logits=torch.tensor(cu))
    if with_np:
        import numpy as np

        ns.cu_num_logits_np = np.asarray(cu, dtype=np.int32)
    return ns


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
    # Layout where prefilling "a" owns no row while "b" and "c" own rows 0
    # and 1: the cu prefix stays the authority, and a positional-cursor
    # consumer would hand row 0 to "a".
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
    another request finishes and "a" shifts to row 1. The batch-row-keyed
    view rebuilt by the hook must still resolve "a"'s state at its new row;
    a batch-row-keyed store would have lost it.
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


def test_view_tracks_current_forward_batch_after_compaction():
    """The per-forward rebuild must realign the view to the forward's batch.

    Step N samples with batch [a, b]; "a" finishes, so the next forward's
    batch is [b] with "b" moved to row 0. vLLM frees finished requests and
    re-sorts the batch between the sampling hook and the next forward, so a
    view left over from the sampling step maps row 0 to "a".
    HiggsAudioV2ModelState.run_preprocess rebuilds the view from the layout
    the forward consumes; without it, "b" embeds "a"'s last audio frame at
    its feedback position.
    """
    from vllm_omni.model_executor.models.higgs_audio_v2.model_state import (
        HiggsAudioV2ModelState,
    )
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    t = _talker()
    t.config = SimpleNamespace(audio_token_id=7, num_codebooks=8, codebook_size=16)
    t.audio_codebook_embeddings = torch.nn.Embedding(8 * 16, 4)
    frame_a = torch.full((8, 1), 1, dtype=torch.long)
    frame_b = torch.full((8, 1), 2, dtype=torch.long)
    t._audio_state = {"a": {"audio_out_ids": frame_a}, "b": {"audio_out_ids": frame_b}}
    # Worst case left over from the sampling step: the previous layout.
    t._rebuild_audio_state_view(SimpleNamespace(req_ids=["a", "b"], num_reqs=2))
    assert t._audio_state_view[0] is t._audio_state["a"]

    ms = object.__new__(HiggsAudioV2ModelState)
    ms.model = t
    with mock.patch.object(OmniModelState, "run_preprocess", return_value=None):
        ms.run_preprocess(SimpleNamespace(req_ids=["b"], num_reqs=1), {})

    assert t._audio_state_view == {0: t._audio_state["b"]}
    # This forward's layout: one request, one decode token.
    t._last_step_query_start_loc = torch.tensor([0, 1])
    out = t._maybe_apply_audio_feedback(torch.zeros(1, 4), torch.tensor([[7]]))
    assert torch.equal(out[0], t.embed_audio_codes(frame_b)[0])
    assert not torch.equal(out[0], t.embed_audio_codes(frame_a)[0])


def test_ras_resample_reads_req_id_keyed_state():
    """RAS re-sampling must resolve state through the req_id key.

    ``_audio_state`` is keyed by req_id under MRV2, and RAS is the
    anti-codebook-collapse guard V1 keeps. Here every codebook samples the
    value that filled its 7-frame history, so RAS must re-sample all of
    them from logits peaked at a different index.
    """
    t = _talker()
    audio_token_id = 3
    t.config = SimpleNamespace(
        audio_token_id=audio_token_id,
        audio_delay_token_id=4,
        num_codebooks=8,
        audio_stream_bos_id=30,
        audio_stream_eos_id=31,
    )
    # Every codebook's logits are one-hot at index 5, so the RAS re-sample
    # is deterministic.
    cb = torch.full((1, 8, 32), float("-inf"))
    cb[:, :, 5] = 10.0
    t.audio_codebook_logits = lambda hidden, mask: cb
    # Collapse: every codebook samples 0, matching the all-zero history.
    t._sample_audio_codes_upstream = lambda cb2d, temperature, top_k, top_p: torch.zeros(
        cb2d.shape[0], dtype=torch.long
    )
    t._audio_state = {
        "req-1": {
            # Past the leading delay phase: no BOS padding of the tail rows.
            "num_delay": 7,
            "num_remaining_delays": None,
            "audio_out_ids": torch.zeros(8, 7, dtype=torch.long),
        }
    }
    # Mark the request as ongoing: the MRV2 per-request reset treats a
    # req_id absent from _slot_output_len as new and would re-seed it.
    t._slot_output_len = {"req-1": 0}

    out = t._dispatch_audio_codes(
        SimpleNamespace(sampled_token_ids=torch.tensor([[audio_token_id]])),
        torch.randn(1, 4),
        None,
        row_req_ids=["req-1"],
    )
    assert out is not None
    assert t._last_audio_codes.tolist() == [[5] * 8]
    assert t._audio_state["req-1"]["audio_out_ids"][:, -1].tolist() == [5] * 8


def test_row_req_ids_maps_rows_through_cu_num_logits():
    t = _talker()
    # Every request owns exactly one row: identity mapping.
    assert t._row_req_ids(_batch(["a", "b", "c"], [0, 1, 2, 3])) == ["a", "b", "c"]
    # A prefilling request owning no row must not steal its row-neighbors'
    # identity: rows 0/1 belong to "b"/"c", so indexing req_ids by row
    # directly would attribute row 0 to "a".
    assert t._row_req_ids(_batch(["a", "b", "c"], [0, 0, 1, 2])) == ["b", "c"]
    # The numpy fast path and the tensor fallback agree.
    for cu in ([0, 1, 2, 3], [0, 0, 1, 2], [0, 2, 2, 5]):
        assert t._row_req_ids(_batch(["a", "b", "c"], cu, with_np=True)) == t._row_req_ids(_batch(["a", "b", "c"], cu))


_BIAS_CONFIG = SimpleNamespace(audio_bos_token_id=10, audio_token_id=11, audio_eos_token_id=128012)


def test_bias_forces_audio_eos_when_ramp_down_completed():
    """The MRV2 termination path must consume req_id-keyed ``should_terminate``.

    The state dict is keyed by req_id under MRV2, so an int-row lookup never
    finds the flag and the LM token is never forced to audio_eos.
    """
    t = _talker()
    t.config = _BIAS_CONFIG
    t._audio_state = {"req-7": {"audio_out_ids": torch.zeros(8, 5, dtype=torch.long), "should_terminate": True}}
    logits = torch.zeros(1, 128256)
    t._apply_audio_mode_bias(logits, None, row_req_ids=["req-7"], row_prev_tokens=[11])
    assert bool(torch.isfinite(logits[0, 128012]))
    assert int(torch.isfinite(logits[0]).sum()) == 1
    # The flag is consumed exactly once.
    assert not t._audio_state["req-7"]["should_terminate"]


def test_bias_keeps_forcing_audio_token_mid_span():
    t = _talker()
    t.config = _BIAS_CONFIG
    t._audio_state = {"req-7": {"audio_out_ids": torch.zeros(8, 5, dtype=torch.long)}}
    logits = torch.zeros(1, 128256)
    t._apply_audio_mode_bias(logits, None, row_req_ids=["req-7"], row_prev_tokens=[11])
    assert bool(torch.isfinite(logits[0, 11]))
    assert not bool(torch.isfinite(logits[0, 128012]))


def test_bias_resolves_rows_through_owning_request_in_mixed_step():
    """Mixed step where row i is not request i in the cu layout.

    Prev tokens arrive per row, and the decode row's state must be resolved
    through its owning request id; an int-row lookup into the req_id-keyed
    dict would never fire the ramp-down termination.
    """
    t = _talker()
    t.config = _BIAS_CONFIG
    # "a" is still prefilling; rows 0/1 belong to "b" (prefill-final) and
    # "c" (decoding, ramp-down just completed).
    t._audio_state = {"c": {"audio_out_ids": torch.zeros(8, 5, dtype=torch.long), "should_terminate": True}}
    logits = torch.zeros(2, 128256)
    logits[0, 5] = 3.0  # "b"'s prefill-final row must stay untouched
    t._apply_audio_mode_bias(logits, None, row_req_ids=["b", "c"], row_prev_tokens=[7, 11])
    assert bool(torch.isfinite(logits[0, 5]))
    assert bool(torch.isfinite(logits[1, 128012]))
    assert int(torch.isfinite(logits[1]).sum()) == 1


def test_dispatch_keys_state_by_row_owner_in_mixed_step():
    """Delay-pattern state must land under the row owner, not ``req_ids[row]``."""
    t = _talker()
    audio_token_id = 3
    t.config = SimpleNamespace(
        audio_token_id=audio_token_id,
        audio_delay_token_id=4,
        num_codebooks=8,
        audio_stream_bos_id=30,
        audio_stream_eos_id=31,
    )
    cb = torch.full((1, 8, 32), 0.0)
    cb[:, :, 5] = 10.0
    t.audio_codebook_logits = lambda hidden, mask: cb.expand(int(mask.sum()), -1, -1)
    t._sample_audio_codes_upstream = lambda cb2d, temperature, top_k, top_p: torch.full(
        (cb2d.shape[0],), 5, dtype=torch.long
    )
    t._audio_state = {}
    t._slot_output_len = {}
    batch = _batch(["a", "b", "c"], [0, 0, 1, 2])  # "a" prefilling, owns no row
    row_req_ids = t._row_req_ids(batch)
    assert row_req_ids == ["b", "c"]

    out = t._dispatch_audio_codes(
        SimpleNamespace(sampled_token_ids=torch.tensor([[audio_token_id], [audio_token_id]])),
        torch.randn(2, 4),
        None,
        row_req_ids=row_req_ids,
    )
    assert out is not None
    assert set(t._audio_state) == {"b", "c"}
    assert set(t._slot_output_len) == {"b", "c"}
    assert t._audio_state["b"]["audio_out_ids"].shape == (8, 1)
    assert t._audio_state["c"]["audio_out_ids"].shape == (8, 1)


def test_on_requests_finished_drops_only_finished_state():
    t = _talker()
    t._audio_state = {
        "a": {"num_delay": 3, "num_remaining_delays": None, "audio_out_ids": torch.zeros(8, 2, dtype=torch.long)},
        "b": {"num_delay": 0, "num_remaining_delays": None, "audio_out_ids": torch.zeros(8, 4, dtype=torch.long)},
    }
    t._slot_output_len = {"a": 5, "b": 6}
    # "a" was aborted after its last sampling step: its codes row waits in
    # the pending map for a postprocess walk that will never come.
    t._pending_codes_by_req = {
        "a": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
        "b": torch.tensor([[2, 2, 2, 2, 2, 2, 2, 2]], dtype=torch.long),
    }

    t.on_requests_finished({"a"})
    assert set(t._audio_state) == {"b"}
    assert set(t._slot_output_len) == {"b"}
    assert set(t._pending_codes_by_req) == {"b"}

    # Unknown ids are no-ops; a preempted (not finished) id keeps its state.
    t.on_requests_finished({"nope"})
    assert set(t._audio_state) == {"b"}
    assert set(t._pending_codes_by_req) == {"b"}
    assert t._audio_state["b"]["audio_out_ids"].shape == (8, 4)


def _feedback_talker():
    t = _talker()
    t.config = SimpleNamespace(audio_token_id=7, num_codebooks=8, codebook_size=16)
    t.audio_codebook_embeddings = torch.nn.Embedding(8 * 16, 4)
    return t


def test_audio_feedback_resolves_owner_row_in_mixed_step():
    """The qsl owner walk replaces the flat-position state lookup.

    Layout: "a" prefills 2 tokens with the audio placeholder at pos 1, while
    "b" and "c" decode. A positional lookup would cross-inject neighboring
    requests' frames here.
    """
    t = _feedback_talker()
    frame_b = torch.full((8, 1), 1, dtype=torch.long)
    frame_c = torch.full((8, 1), 2, dtype=torch.long)
    t._audio_state = {"b": {"audio_out_ids": frame_b}, "c": {"audio_out_ids": frame_c}}
    t._rebuild_audio_state_view(SimpleNamespace(req_ids=["a", "b", "c"], num_reqs=3))
    t._last_step_query_start_loc = torch.tensor([0, 2, 3, 4])
    hidden = torch.zeros(4, 4)
    # a: [text token, prompt placeholder]; b: decode token; c: decode token.
    input_ids = torch.tensor([5, 7, 7, 7])

    out = t._maybe_apply_audio_feedback(hidden, input_ids)
    assert torch.equal(out[0], hidden[0])
    assert torch.equal(out[1], hidden[1]), "a's prompt placeholder must stay untouched (a owns no audio state)"
    assert torch.equal(out[2], t.embed_audio_codes(frame_b)[0]), "b must get its own frame, not c's"
    assert torch.equal(out[3], t.embed_audio_codes(frame_c)[0]), "c must get its own frame"
    assert not torch.equal(out[2], out[3])


def test_audio_feedback_owner_walk_works_under_v1_row_keys():
    """The same walk must resolve through plain batch-row keys (V1 keying)."""
    t = _feedback_talker()
    frame_b = torch.full((8, 1), 1, dtype=torch.long)
    frame_c = torch.full((8, 1), 2, dtype=torch.long)
    # V1 keys the state dict by batch row directly; no view is published.
    t._audio_state = {0: {"audio_out_ids": frame_b}, 1: {"audio_out_ids": frame_c}}
    t._last_step_query_start_loc = torch.tensor([0, 2, 3])
    hidden = torch.zeros(3, 4)
    # b prefills 2 tokens (placeholder first), c decodes 1.
    input_ids = torch.tensor([7, 5, 7])

    out = t._maybe_apply_audio_feedback(hidden, input_ids)
    assert torch.equal(out[0], t.embed_audio_codes(frame_b)[0])
    assert torch.equal(out[1], hidden[1])
    assert torch.equal(out[2], t.embed_audio_codes(frame_c)[0]), "c's token at pos 2 must map to row 1"


def test_audio_feedback_pure_decode_still_substitutes():
    """pos == row in pure decode steps: the walk must not over-correct."""
    t = _feedback_talker()
    frames = [torch.full((8, 1), i + 1, dtype=torch.long) for i in range(2)]
    t._audio_state = {"b": {"audio_out_ids": frames[0]}, "c": {"audio_out_ids": frames[1]}}
    t._rebuild_audio_state_view(SimpleNamespace(req_ids=["b", "c"], num_reqs=2))
    t._last_step_query_start_loc = torch.tensor([0, 1, 2])

    out = t._maybe_apply_audio_feedback(torch.zeros(2, 4), torch.tensor([7, 7]))
    assert torch.equal(out[0], t.embed_audio_codes(frames[0])[0])
    assert torch.equal(out[1], t.embed_audio_codes(frames[1])[0])


def test_audio_feedback_fails_closed_without_layout():
    """No usable qsl stash: skip the substitution entirely, no exception.

    A positional fallback would reintroduce cross-request injection in mixed
    steps; skipping one feedback frame matches the pre-existing miss case.
    """
    t = _feedback_talker()
    t._audio_state = {"b": {"audio_out_ids": torch.full((8, 1), 1, dtype=torch.long)}}
    t._rebuild_audio_state_view(SimpleNamespace(req_ids=["b"], num_reqs=1))
    t._last_step_query_start_loc = None

    hidden = torch.zeros(1, 4)
    out = t._maybe_apply_audio_feedback(hidden, torch.tensor([7]))
    assert torch.equal(out, hidden)


def test_bias_v1_prev_token_walks_and_int_row_key():
    """The V1 path (row kwargs None) keeps its three prev-token sources
    and its int-row state key; the MRV2 kwargs never engage."""
    term = {"audio_out_ids": torch.zeros(8, 5, dtype=torch.long), "should_terminate": True}

    # 1) output_token_ids walk: request 0's last output token is audio_token.
    t = _talker()
    t.config = _BIAS_CONFIG
    t._audio_state = {0: dict(term)}
    logits = torch.zeros(1, 128256)
    t._apply_audio_mode_bias(logits, SimpleNamespace(prompt_token_ids=None, output_token_ids=[[11]]))
    assert int(torch.isfinite(logits[0]).sum()) == 1 and bool(torch.isfinite(logits[0, 128012]))
    assert not t._audio_state[0]["should_terminate"]

    # 2) prompt_token_ids walk: no outputs yet; the prompt ends on audio_bos.
    t2 = _talker()
    t2.config = _BIAS_CONFIG
    t2._audio_state = {0: {"audio_out_ids": torch.zeros(8, 1, dtype=torch.long)}}
    logits2 = torch.zeros(1, 128256)
    t2._apply_audio_mode_bias(logits2, SimpleNamespace(prompt_token_ids=[[5, 10]], output_token_ids=[[]]))
    assert int(torch.isfinite(logits2[0]).sum()) == 1 and bool(torch.isfinite(logits2[0, 11]))
    assert bool(t2._last_first_audio_after_bos[0]), "prev == audio_bos flagged"

    # 3) stash walk: metadata None + the forward stashes.
    t3 = _talker()
    t3.config = _BIAS_CONFIG
    t3._audio_state = {1: dict(term)}
    t3._last_step_input_ids = torch.tensor([5, 11])
    t3._last_step_query_start_loc = torch.tensor([0, 1, 2])
    logits3 = torch.zeros(2, 128256)
    t3._apply_audio_mode_bias(logits3, None)
    assert int(torch.isfinite(logits3[1]).sum()) == 1 and bool(torch.isfinite(logits3[1, 128012]))
    assert int(torch.isfinite(logits3[0]).sum()) == logits3.shape[1], "row 0's prev is text: untouched"
    assert not t3._audio_state[1]["should_terminate"]

    # A string "1" key must NOT be consumed by the V1 int-row lookup.
    t4 = _talker()
    t4.config = _BIAS_CONFIG
    t4._audio_state = {"1": dict(term)}
    t4._last_step_input_ids = torch.tensor([5, 11])
    t4._last_step_query_start_loc = torch.tensor([0, 1, 2])
    logits4 = torch.zeros(2, 128256)
    t4._apply_audio_mode_bias(logits4, None)
    assert int(torch.isfinite(logits4[1]).sum()) == 1 and bool(torch.isfinite(logits4[1, 11]))
    assert t4._audio_state["1"]["should_terminate"], "string key untouched by V1 lookup"
