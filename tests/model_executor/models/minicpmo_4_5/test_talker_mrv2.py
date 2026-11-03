# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model Runner V2 path of MiniCPM-o 4.5's Talker: device-side codec output,
EOS control and the 16-frame codec penalty must match the V1 host path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _OFFLINE_CODEC_MAX_NEW_TOKENS,
    MiniCPMO45OmniTTSForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_EOS = 6561


def _talker(max_reqs: int = 8, max_position_embeddings: int = 4096):
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    talker._codec_eos_id = _EOS
    talker._tts_config = SimpleNamespace(max_position_embeddings=max_position_embeddings)
    talker._mrv2_empty_speech = torch.zeros(max_reqs, dtype=torch.bool)
    talker._mrv2_code_runs = torch.zeros((max_reqs, 2), dtype=torch.long)
    talker._num_audio_tokens = _EOS + 1
    talker._codec_classes("cpu")
    talker._mrv2_forced_eos = None
    talker._mrv2_decode_rows_logged = False
    return talker


def _batch(rows: list[dict], *, pad_to: int | None = None):
    """rows: one dict per request with slot, prompt_len, span (token ids) and prefill flag."""
    ids = [token for row in rows for token in row["span"]]
    num_tokens = len(ids)
    padded = pad_to or num_tokens
    starts = np.cumsum([0] + [len(row["span"]) for row in rows])
    seq_lens = [row["computed"] + len(row["span"]) for row in rows]
    return SimpleNamespace(
        num_reqs=len(rows),
        has_prefill=any(row["prefill"] for row in rows),
        is_prefilling_np=np.array([row["prefill"] for row in rows]),
        idx_mapping_np=np.array([row["slot"] for row in rows]),
        idx_mapping=torch.tensor([row["slot"] for row in rows]),
        input_ids=torch.tensor(ids + [0] * (padded - num_tokens), dtype=torch.int32),
        logits_indices=torch.tensor(starts[1:] - 1),
        seq_lens=torch.tensor(seq_lens + [0] * 2, dtype=torch.int32),
    ), padded


def _req_states(prompt_lens: dict[int, int], max_reqs: int = 8, token_ids: dict[int, list[int]] | None = None):
    prompt_len = torch.zeros(max_reqs, dtype=torch.int32)
    for slot, value in prompt_lens.items():
        prompt_len[slot] = value
    all_token_ids = torch.zeros((max_reqs, 64), dtype=torch.int32)
    for slot, ids in (token_ids or {}).items():
        all_token_ids[slot, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(prompt_len=SimpleNamespace(gpu=prompt_len), all_token_ids=SimpleNamespace(gpu=all_token_ids))


def test_mrv2_output_emits_decode_input_codes_with_validity_and_forced_eos() -> None:
    talker = _talker()
    rows = [
        # Final prefill chunk: no code yet (V1 emits an empty delta).
        dict(slot=2, prompt_len=5, computed=0, span=[0, 0, 0, 0, 0], prefill=True),
        # Decode with a codec id: emitted this step.
        dict(slot=0, prompt_len=4, computed=6, span=[17], prefill=False),
        # Decode whose input is codec EOS: request ended, force EOS again.
        dict(slot=1, prompt_len=4, computed=8, span=[_EOS], prefill=False),
    ]
    batch, padded = _batch(rows, pad_to=8)
    buffers = [{"audio_state": {"finished": False}}, {}, {}]
    hidden = torch.zeros((padded, 4))
    out = talker.make_omni_output_mrv2(
        hidden,
        input_batch=batch,
        req_states=_req_states({cast(int, row["slot"]): cast(int, row["prompt_len"]) for row in rows}),
        model_intermediate_buffer=buffers,
    )
    assert isinstance(out, OmniOutput)
    codes = out.multimodal_outputs["codes"]["audio"]
    valid = out.multimodal_outputs["meta"]["codec_frame_valid"]
    assert codes.shape == (padded, 1) and valid.shape == (padded,)
    assert codes[:, 0].tolist() == batch.input_ids.tolist()
    assert valid.tolist() == [False] * 5 + [True, False] + [False]
    assert talker.take_mrv2_forced_eos(batch, None, 3).tolist() == [False, False, True]
    # Consumed once; a warmup sampler call sees nothing.
    assert talker.take_mrv2_forced_eos(batch, None, 3) is None


def test_mrv2_output_empty_condition_and_length_cap() -> None:
    talker = _talker()
    prompt_lens = {3: 6, 4: 4096 - 100}
    # Empty Thinker condition: preprocess marks the request finished at prefill.
    batch, _ = _batch([dict(slot=3, prompt_len=6, computed=0, span=[0] * 6, prefill=True)])
    talker.make_omni_output_mrv2(
        torch.zeros((6, 4)),
        input_batch=batch,
        req_states=_req_states(prompt_lens),
        model_intermediate_buffer=[{"audio_state": {"finished": True}}],
    )
    assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [True]
    # Its later decode rows stay forced and emit nothing, whatever the input id.
    batch, _ = _batch([dict(slot=3, prompt_len=6, computed=6, span=[42], prefill=False)])
    out = talker.make_omni_output_mrv2(
        torch.zeros((1, 4)), input_batch=batch, req_states=_req_states(prompt_lens), model_intermediate_buffer=[{}]
    )
    assert out.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == [False]
    assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [True]

    # Offline cap: min(2048, context - prompt) samples, the last one EOS.
    limit = min(_OFFLINE_CODEC_MAX_NEW_TOKENS, 100) - 1
    for step, forced in ((limit - 1, False), (limit, True)):
        batch, _ = _batch(
            [dict(slot=4, prompt_len=prompt_lens[4], computed=prompt_lens[4] + step - 1, span=[5], prefill=False)]
        )
        out = talker.make_omni_output_mrv2(
            torch.zeros((1, 4)), input_batch=batch, req_states=_req_states(prompt_lens), model_intermediate_buffer=[{}]
        )
        assert out.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == [True]
        assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [forced]


@pytest.mark.parametrize("forced", [None, [False, True]])
def test_sampler_adapter_keeps_upstream_counts_and_only_forces_codec_eos(mocker, forced):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import MiniCPMO45TalkerSampler

    output = SimpleNamespace(sampled_token_ids=torch.tensor([[2], [3]]), num_sampled=torch.tensor([1, 0]))
    base = mocker.Mock(return_value=output)
    base.req_states = object()
    mask = None if forced is None else torch.tensor(forced)
    talker = SimpleNamespace(_codec_eos_id=7, take_mrv2_forced_eos=mocker.Mock(return_value=mask))
    sampler = MiniCPMO45TalkerSampler(base, talker)
    logits, batch = torch.zeros(2, 8), object()
    assert sampler(logits, batch) is output
    assert output.sampled_token_ids.tolist() == [[2], [3 if forced is None else 7]]
    assert output.num_sampled.tolist() == [1, 0]
    base.assert_called_once_with(logits, batch)
    talker.take_mrv2_forced_eos.assert_called_once_with(batch, base.req_states, 2)


@pytest.mark.parametrize("condition_seq", [9, None])
def test_mrv2_talker_native_duplex_output(condition_seq) -> None:
    talker = _talker()
    rows = [
        dict(slot=0, prompt_len=4, computed=6, span=[17], prefill=False),
    ]
    batch, padded = _batch(rows, pad_to=1)
    buffers = [
        {
            "native_duplex": True,
            "duplex": {"epoch": 2, "turn_id": 5},
            "meta": {"native_duplex_segment_text": "hello", "streaming_condition_seq": condition_seq},
        }
    ]
    hidden = torch.zeros((1, 4))
    out = talker.make_omni_output_mrv2(
        hidden,
        input_batch=batch,
        req_states=_req_states({0: 4}),
        model_intermediate_buffer=buffers,
    )
    assert isinstance(out, OmniOutput)
    meta = out.multimodal_outputs["meta"]
    assert meta["native_duplex"][0].item() is True
    assert meta["duplex_epoch"][0].item() == 2
    assert meta["duplex_turn_id"][0].item() == 5
    assert meta["streaming_condition_seq"][0].item() == (9 if condition_seq is not None else -1)
    assert bytes(meta["llm_output_text_utf8"][0].tolist()).decode("utf-8") == "hello"
    assert bytes(meta["native_duplex_segment_text"][0].tolist()).decode("utf-8") == "hello"


@pytest.mark.parametrize("fence", [{"epoch": -1, "turn_id": 5}, {"epoch": True, "turn_id": 5}, {"turn_id": 5}])
def test_mrv2_talker_rejects_native_duplex_without_fence_identity(fence) -> None:
    """Same contract as V1 make_omni_output: condition fencing needs non-negative int identities."""
    batch, _ = _batch([dict(slot=0, prompt_len=4, computed=6, span=[17], prefill=False)], pad_to=1)
    with pytest.raises(RuntimeError, match="requires non-negative integer epoch and turn_id"):
        _talker().make_omni_output_mrv2(
            torch.zeros((1, 4)),
            input_batch=batch,
            req_states=_req_states({0: 4}),
            model_intermediate_buffer=[{"native_duplex": True, "duplex": fence}],
        )


def test_mrv2_context_with_only_eos_slot_forces_eos_at_prefill():
    talker = _talker(max_position_embeddings=5)
    batch, _ = _batch([dict(slot=2, prompt_len=4, computed=0, span=[0] * 4, prefill=True)])
    talker.make_omni_output_mrv2(
        torch.zeros((4, 4)),
        input_batch=batch,
        req_states=_req_states({2: 4}),
        model_intermediate_buffer=[{"audio_state": {"finished": False}}],
    )
    assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [True]


@pytest.mark.parametrize(
    "step,turn_start,turn_end,masked",
    [
        (step, False, True, masked)
        for step, masked in [(24, False), (25, True), (29, True), (30, False), (50, True), (55, False)]
    ]
    + [(0, True, False, False), (0, False, False, True), (24, False, False, True), (25, False, False, False)],
)
def test_mrv2_turn_end_drain_masks_cadence_eos(mocker, step, turn_start, turn_end, masked):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import MiniCPMO45TalkerSampler

    talker = _talker()
    batch, _ = _batch([dict(slot=2, prompt_len=4, computed=3 + step, span=[17], prefill=False)])
    states = _req_states({2: 4})
    talker.make_omni_output_mrv2(
        torch.zeros((1, 4)),
        input_batch=batch,
        req_states=states,
        model_intermediate_buffer=[
            {
                "native_duplex": True,
                "duplex": {"epoch": 3, "turn_id": 7},
                "meta": {"turn_start": turn_start, "turn_end": turn_end},
            }
        ],
    )
    base = mocker.Mock(side_effect=lambda logits, _: SimpleNamespace(sampled_token_ids=logits.argmax(-1)[:, None]))
    base.req_states = states
    sampler = MiniCPMO45TalkerSampler(base, talker)
    logits = torch.zeros(1, _EOS + 1)
    logits[0, _EOS] = 10
    output = sampler(logits, batch)
    assert (output.sampled_token_ids.item() != _EOS) is masked


def _mrv2_turn_end_samples_eos(mocker, talker, *, step: int, code: int, wants_eos: bool = True) -> bool:
    """One decode step of a turn-end condition whose turn ended a sentence; True if EOS is sampled."""
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import MiniCPMO45TalkerSampler

    batch, _ = _batch([dict(slot=2, prompt_len=4, computed=3 + step, span=[code], prefill=False)])
    states = _req_states({2: 4})
    talker.make_omni_output_mrv2(
        torch.zeros((1, 4)),
        input_batch=batch,
        req_states=states,
        model_intermediate_buffer=[
            {
                "native_duplex": True,
                "duplex": {"epoch": 3, "turn_id": 7},
                "meta": {"turn_end": True, "turn_text_ends_sentence": True},
            }
        ],
    )
    base = mocker.Mock(
        side_effect=lambda logits, _: SamplerOutput(
            sampled_token_ids=logits.argmax(-1)[:, None],
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.ones(1, dtype=torch.int32),
            num_rejected=torch.zeros(1, dtype=torch.int32),
        )
    )
    base.req_states = states
    logits = torch.zeros(1, _EOS + 1)
    logits[0, _EOS if wants_eos else 0] = 10
    return MiniCPMO45TalkerSampler(base, talker)(logits, batch).sampled_token_ids.item() == _EOS


def test_mrv2_turn_end_drain_waits_for_silence(mocker):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _SILENT_CODEC_IDS

    talker, silent = _talker(), _SILENT_CODEC_IDS[0]
    # While the Talker is speaking, its cadence EOS is masked at and between boundaries.
    assert not _mrv2_turn_end_samples_eos(mocker, talker, step=24, code=17)
    assert not _mrv2_turn_end_samples_eos(mocker, talker, step=25, code=17)
    # One silent code is not enough; after two in a row the model may stop.
    assert not _mrv2_turn_end_samples_eos(mocker, talker, step=26, code=silent)
    assert _mrv2_turn_end_samples_eos(mocker, talker, step=27, code=silent)
    # Speech resets the run.
    assert not _mrv2_turn_end_samples_eos(mocker, talker, step=28, code=17)


_TURN_END_BUFFER = {
    "native_duplex": True,
    "duplex": {"epoch": 3, "turn_id": 7},
    "meta": {"turn_end": True, "turn_text_ends_sentence": True},
}


def _turn_end_stop_rule(talker, *, slot: int, span: list[int], computed: int, prefill: bool, history=None):
    """One Talker step of a sentence-final turn-end condition (prompt length 4): its (forced, masked) EOS."""
    batch, _ = _batch([dict(slot=slot, prompt_len=4, computed=computed, span=span, prefill=prefill)])
    talker.make_omni_output_mrv2(
        torch.zeros((len(span), 4)),
        input_batch=batch,
        req_states=_req_states({slot: 4}, token_ids={slot: history} if history else None),
        model_intermediate_buffer=[_TURN_END_BUFFER],
    )
    forced, masked = talker._mrv2_forced_eos, talker._mrv2_masked_eos
    talker._mrv2_forced_eos = talker._mrv2_masked_eos = None
    return bool(forced[0]), bool(masked[0])


def test_mrv2_turn_end_drain_stops_the_same_way_after_preemption():
    """A request resumed after preemption re-prefills its codes; its EOS decisions match an uninterrupted run."""
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _QUIET_CODEC_IDS, _SILENT_CODEC_IDS

    silent = _SILENT_CODEC_IDS[0]
    quiet = next(code for code in _QUIET_CODEC_IDS if code not in _SILENT_CODEC_IDS)
    # Speech, two silent codes (EOS may be sampled at step 5), then quiet codes up to the forced stop at step 11.
    codes = [17, 17, 17, silent, silent] + [quiet] * 6
    prompt = [0] * 4

    def decode(talker, slot: int, step: int):
        return _turn_end_stop_rule(talker, slot=slot, span=[codes[step - 1]], computed=3 + step, prefill=False)

    uninterrupted_talker = _talker()
    uninterrupted = [decode(uninterrupted_talker, 2, step) for step in range(1, len(codes) + 1)]
    assert uninterrupted == [(False, True)] * 4 + [(False, False)] + [(False, True)] * 5 + [(True, False)]

    # Preempted after step 4, then resumed in another slot by re-prefilling the prompt and the codes so far.
    talker = _talker()
    resumed = [decode(talker, 2, step) for step in range(1, 5)]
    history = prompt + codes[:5]
    resumed.append(_turn_end_stop_rule(talker, slot=5, span=history, computed=0, prefill=True, history=history))
    resumed += [decode(talker, 5, step) for step in range(6, len(codes) + 1)]
    assert resumed == uninterrupted


def test_mrv2_turn_end_drain_new_condition_starts_from_zero():
    """A new condition's prefill has no codes yet, so runs left in its slot by an earlier condition do not count."""
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _SILENT_CODEC_IDS

    talker, silent = _talker(), _SILENT_CODEC_IDS[0]
    for step in (1, 2):
        _turn_end_stop_rule(talker, slot=2, span=[silent], computed=3 + step, prefill=False)
    assert talker._mrv2_code_runs[2].tolist() == [2, 2]
    _turn_end_stop_rule(talker, slot=2, span=[0] * 4, computed=0, prefill=True, history=[0] * 4)
    assert talker._mrv2_code_runs[2].tolist() == [0, 0]
    # One silent code of the new condition is not enough to stop.
    assert _turn_end_stop_rule(talker, slot=2, span=[silent], computed=4, prefill=False) == (False, True)


def test_mrv2_turn_end_drain_forces_eos_after_a_quiet_run(mocker):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _QUIET_CODEC_IDS,
        _SILENT_CODEC_IDS,
        _TURN_END_QUIET_STOP_FRAMES,
    )

    talker = _talker()
    quiet = next(code for code in _QUIET_CODEC_IDS if code not in _SILENT_CODEC_IDS)
    for step in range(10, 10 + _TURN_END_QUIET_STOP_FRAMES - 1):
        assert not _mrv2_turn_end_samples_eos(mocker, talker, step=step, code=quiet, wants_eos=False)
    assert _mrv2_turn_end_samples_eos(
        mocker, talker, step=10 + _TURN_END_QUIET_STOP_FRAMES - 1, code=quiet, wants_eos=False
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("output_count", [0, 3, 16, 20])
def test_codec_penalty_carries_previous_segment_through_reordered_slots(device, output_count):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _apply_batched_repetition_penalty,
        _apply_codec_window_penalty_gpu,
    )

    prompt_lens = torch.tensor([3, 0, 2], device=device)
    lengths = prompt_lens + output_count
    tokens = torch.zeros(3, 64, dtype=torch.long, device=device)
    prefixes = torch.full((3, 16), -1, dtype=torch.long, device=device)
    histories = []
    for slot, prefix in [(2, [5, 5, 5, 9]), (0, [7, 7])]:
        output = ([1, 7, 2] * 7)[:output_count]
        start = int(prompt_lens[slot])
        tokens[slot, start : start + output_count] = torch.tensor(output, device=device)
        prefixes[slot, -len(prefix) :] = torch.tensor(prefix, device=device)
        histories.append(torch.tensor(prefix + output))
    logits = torch.linspace(-3, 3, 32, device=device).repeat(2, 1)
    expected = _apply_batched_repetition_penalty(logits.cpu(), histories, penalty=1.05, window_size=16)
    _apply_codec_window_penalty_gpu(
        logits,
        torch.tensor([2, 0], device=device),
        tokens,
        lengths,
        prompt_lens,
        torch.full((3,), 1.05, device=device),
        window_size=16,
        history_prefix=prefixes,
    )
    torch.testing.assert_close(logits.cpu(), expected)


def test_mrv2_prefill_seeds_codec_history_for_the_actual_slot(mocker):
    talker = _talker()
    talker._mrv2_penalty_state = mocker.Mock()
    batch, _ = _batch([dict(slot=3, prompt_len=2, computed=0, span=[0, 0], prefill=True)])
    talker.make_omni_output_mrv2(
        torch.zeros(2, 4),
        input_batch=batch,
        req_states=_req_states({3: 2}),
        model_intermediate_buffer=[{"audio_state": {"recent_codes": [5, 7, 5]}}],
    )
    talker._mrv2_penalty_state.set_history_prefix.assert_called_once_with([3], [[5, 7, 5]])


def test_turn_decode_does_not_build_duplex_limit_tensors(monkeypatch):
    import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts as module

    def reject_copy(*args, **kwargs):
        raise AssertionError("ordinary turn decode must not upload duplex limits")

    talker = _talker()
    monkeypatch.setattr(module, "index_to_device", reject_copy)
    batch, _ = _batch([dict(slot=0, prompt_len=4, computed=5, span=[1], prefill=False)])
    talker.make_omni_output_mrv2(
        torch.zeros(1, 4),
        input_batch=batch,
        req_states=_req_states({0: 4}),
        model_intermediate_buffer=[{}],
    )
    assert talker._mrv2_masked_eos is None
