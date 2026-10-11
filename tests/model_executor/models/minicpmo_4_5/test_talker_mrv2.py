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


@pytest.mark.parametrize("padded", [False, True])
def test_native_partition_matches_generic_partition_and_owns_metadata(padded):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _DUPLEX_OUTPUT_META_KEYS,
        _mrv2_duplex_output_partition,
    )
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

    n = 2 + int(padded)
    meta = {key: [torch.tensor([1]), torch.tensor([2])] for key in _DUPLEX_OUTPUT_META_KEYS}
    outputs = {
        "codes": {"audio": torch.arange(n).view(-1, 1)},
        "meta": {"codec_frame_valid": torch.ones(n, dtype=torch.bool), "finished": torch.tensor([False, True])},
    }
    generic = {"codes": outputs["codes"], "meta": {**outputs["meta"], **meta}}
    generic["meta"]["finished"] = list(outputs["meta"]["finished"].unbind())
    expected = OmniARModelRunner._build_async_chunk_outputs_from_mm(generic, np.array([0, 1, 2]), np.ones(2), 2, 2, n)
    actual = _mrv2_duplex_output_partition(outputs, meta, [(0, 1), (1, 1)], {n})
    for got_rows, want_rows in zip((actual.inter_stage, actual.client), expected, strict=True):
        for got, want in zip(got_rows, want_rows, strict=True):
            assert list(got) == list(want)
            for key in got:
                torch.testing.assert_close(got[key], want[key], rtol=0, atol=0)
    for inter, client in zip(actual.inter_stage, actual.client, strict=True):
        assert all(client[key] is inter[key] for key in client)
    meta["llm_output_text_utf8"][0].zero_()
    assert actual.inter_stage[0]["meta.llm_output_text_utf8"].tolist() == [1]


def _talker(max_reqs: int = 8, max_position_embeddings: int = 4096):
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    talker._codec_eos_id = _EOS
    talker._tts_config = SimpleNamespace(max_position_embeddings=max_position_embeddings)
    talker._mrv2_empty_speech = torch.zeros(max_reqs, dtype=torch.bool)
    talker._mrv2_forced_eos = None
    talker._mrv2_decode_rows_logged = False
    talker._deferred_cleanup_ids = set()
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
        query_start_loc_np=starts,
        num_scheduled_tokens=np.diff(starts),
    ), padded


def _req_states(prompt_lens: dict[int, int], max_reqs: int = 8):
    prompt_len = torch.zeros(max_reqs, dtype=torch.int32)
    for slot, value in prompt_lens.items():
        prompt_len[slot] = value
    return SimpleNamespace(prompt_len=SimpleNamespace(gpu=prompt_len))


def test_mrv2_prefill_keeps_interleaved_session_conditions_and_codec_history_separate(mocker):
    talker = _talker()
    torch.nn.Module.__init__(talker)
    talker._request_condition_states = {}
    talker._request_audio_states = {}
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=torch.zeros(2, 4))
    # One session advances while another starts. The V2 runner supplies only
    # req_id; falling back to a shared default key makes seq 3 follow seq 0.
    for request_id, seq in [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("a", 3)]:
        _, _, updates = talker.preprocess(
            torch.zeros(2, dtype=torch.long),
            None,
            req_id=request_id,
            _omni_is_prefill=True,
            _omni_prompt_len=2,
            native_duplex=True,
            ids={"tts": torch.tensor([7])},
            hidden_states={"tts": torch.zeros(1, 4)},
            meta={"streaming_condition_seq": seq, "turn_start": seq == 0},
        )
        assert talker._request_audio_states[request_id] is updates["audio_state"]
        updates["audio_state"]["recent_codes"] = [11 if request_id == "a" else 22]
    assert talker._request_condition_states["a"]["condition_seq"] == 3
    assert talker._request_condition_states["b"]["condition_seq"] == 0
    assert talker._request_audio_states["a"]["recent_codes"] == [11]
    assert talker._request_audio_states["b"]["recent_codes"] == [22]


@pytest.mark.parametrize("fallback", ["expanded", "speculative", "ordinary", "unseeded"])
def test_seeded_codec_unsupported_batches_keep_original_sampler(monkeypatch, mocker, fallback):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex import mrv2

    monkeypatch.setattr(mrv2, "LegacySampler", mocker.Mock())
    original = mocker.Mock(spec=object.__new__(Sampler))
    original.num_speculative_tokens = 2 if fallback == "speculative" else 1
    original.req_states = mocker.Mock(spec=RequestState, index_to_req_id={2: "live"})
    model = mocker.Mock(
        spec=MiniCPMO45OmniTTSForConditionalGeneration,
        _mrv2_output_infos=[{"native_duplex": fallback != "ordinary"}],
    )
    core = mrv2.MiniCPMO45SeededCodecSampler(original, model)
    core.add_request(2, SamplingParams(seed=None if fallback == "unseeded" else 42))
    batch = mocker.Mock(spec=InputBatch, idx_mapping_np=np.array([2]))
    logits = torch.ones(2 if fallback == "expanded" else 1, 10)
    output = core(logits, batch)
    assert output is original.return_value
    original.assert_called_once_with(logits, batch)
    assert not core._generators


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


@pytest.mark.parametrize("speech_tokens", [None, 50])
@pytest.mark.parametrize("turn_end", [False, True])
@pytest.mark.parametrize("step", [24, 25, 29, 49, 50, 54])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_mrv2_native_unit_keeps_codec_budget_and_final_eos(mocker, speech_tokens, turn_end, step, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for the fused native codec output")
    talker = _talker()
    torch.nn.Module.__init__(talker)
    talker._request_condition_states = {}
    talker._request_audio_states = {}
    talker._mrv2_penalties = SimpleNamespace(prefix_history=torch.full((8, 16), -1, dtype=torch.long))
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=torch.zeros(2, 4))
    meta = {"turn_start": False, "turn_end": turn_end, "gander_context_version": 7}
    if speech_tokens is not None:
        meta["gander_speech_tokens"] = speech_tokens
    info = {"native_duplex": True, "duplex": {"epoch": 3, "turn_id": 5}, "meta": meta}
    _, _, updates = talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        req_id="r",
        _omni_is_prefill=True,
        _omni_prompt_len=2,
        ids={"tts": torch.tensor([7])},
        hidden_states={"tts": torch.zeros(1, 4)},
        **info,
    )
    info.update(updates)
    batch, _ = _batch([dict(slot=3, prompt_len=2, computed=0, span=[0, 0], prefill=True)])
    reqs = _req_states({3: 2})
    talker.make_omni_output_mrv2(
        torch.zeros(2, 4), input_batch=batch, req_states=reqs, model_intermediate_buffer=[info]
    )
    batch, _ = _batch([dict(slot=3, prompt_len=2, computed=2 + step - 1, span=[5], prefill=False)])
    for name in ("input_ids", "idx_mapping", "seq_lens", "logits_indices"):
        setattr(batch, name, getattr(batch, name).to(device))
    reqs.prompt_len.gpu = reqs.prompt_len.gpu.to(device)
    talker._mrv2_empty_speech = talker._mrv2_empty_speech.to(device)
    talker._mrv2_codec_controls = talker._mrv2_codec_controls.to(device)
    out = talker.make_omni_output_mrv2(
        torch.zeros(1, 4, device=device), input_batch=batch, req_states=reqs, model_intermediate_buffer=[info]
    )
    if turn_end:
        forced = False
        masked = speech_tokens is None and step in {25, 29, 50, 54}
    else:
        budget = 50 if speech_tokens == 50 else 25
        forced, masked = step >= budget, step < budget
    assert talker.take_mrv2_forced_eos(batch, reqs, 1).tolist() == [forced]
    assert talker._mrv2_mask_eos.tolist() == [masked]
    assert out.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == [True]
    assert talker._mrv2_output_meta[1]["gander_context_version"][0].item() == 7
    assert talker._mrv2_output_meta[1]["duplex_epoch"][0].item() == 3
    assert talker._mrv2_output_meta[1]["duplex_turn_id"][0].item() == 5


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


@pytest.mark.cuda
def test_mrv2_sampler_applies_codec_window_penalty_instead_of_stock_penalty():
    """The real MRv2 sampler pipeline scores the V1 16-frame codec penalty.

    The stock penalty would tax every code in the prompt (the scheduler's
    placeholder id 0) and in the whole output once; the Talker's penalty taxes
    ``penalty ** count`` over the last 16 sampled codes only.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _CODEC_PENALTY_WINDOW,
        _apply_batched_repetition_penalty,
        _install_mrv2_talker_sampler,
    )

    device, vocab, penalty = torch.device("cuda"), 32, 1.05
    reqs = RequestState(4, 64, 64, 0, vocab, device)
    sampler = Sampler(VllmConfig(), 4, vocab, device, reqs)
    talker = _talker(max_reqs=4)
    wrapped = _install_mrv2_talker_sampler(sampler, talker)

    prompt = [0, 0, 0]
    # Code 5 only before the window, code 7 three times and code 9 once inside it.
    output = [5, 5, 5, 5] + [7, 1, 2, 7, 3, 4, 7, 6, 8, 10, 11, 12, 13, 14, 15, 9]
    assert len(output) - 4 == _CODEC_PENALTY_WINDOW
    reqs.add_request("talker", len(prompt), prompt + output, len(prompt) + len(output), 64)
    slot = reqs.req_id_to_index["talker"]
    wrapped.add_request(slot, SamplingParams(repetition_penalty=penalty, temperature=1.0))
    reqs.apply_staged_writes()
    wrapped.apply_staged_writes()

    logits = torch.linspace(-3.0, 3.0, vocab, device=device).reshape(1, vocab)
    idx = torch.tensor([slot], dtype=torch.int32, device=device)
    processed = wrapped.apply_sampling_params(
        logits.clone(),
        idx,
        idx,
        np.array([slot]),
        torch.tensor([len(prompt) + len(output)], device=device),
        torch.tensor([output[-1]], dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        np.array([len(prompt) + len(output) + 1]),
        skip_top_k_top_p=True,
    )
    expected = _apply_batched_repetition_penalty(
        logits.cpu(), [torch.tensor(output)], penalty=penalty, window_size=_CODEC_PENALTY_WINDOW
    )
    torch.testing.assert_close(processed.cpu(), expected)
    # Neither the placeholder prompt id nor codes outside the window are taxed.
    assert processed[0, 0].item() == pytest.approx(logits[0, 0].item())
    assert processed[0, 5].item() == pytest.approx(logits[0, 5].item())
    assert processed[0, 7].item() == pytest.approx(logits[0, 7].item() * penalty**3)
    # The runner still finds the output bin counts on ``penalties_state``.
    assert sampler.penalties_state.output_bin_counts is not None


def test_history_survives_real_intermediate_buffer_merge_and_next_condition(mocker):
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    talker = _talker()
    torch.nn.Module.__init__(talker)
    talker._request_condition_states = {}
    talker._request_audio_states = {}
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=torch.zeros(2, 4))
    buffer = OmniIntermediateBuffer(1)
    buffer.add_request(
        0,
        SimpleNamespace(
            req_id="r",
            mm_features=[],
            model_intermediate_buffer={
                "native_duplex": True,
                "ids": {"tts": torch.tensor([7])},
                "hidden_states": {"tts": torch.zeros(1, 4)},
                "meta": {"streaming_condition_seq": 0, "turn_start": True},
            },
        ),
    )

    def prefill():
        _, _, updates = talker.preprocess(
            torch.zeros(2, dtype=torch.long),
            None,
            **buffer.buffers[0],
            _omni_is_prefill=True,
            _omni_prompt_len=2,
        )
        buffer.update(0, updates)
        return updates["audio_state"]

    previous = prefill()
    # The real buffer copies/merges dictionary fields; its audio_state is
    # not the model-owned object even immediately after preprocess.
    assert buffer.buffers[0]["audio_state"] is not previous
    batch, _ = _batch([dict(slot=0, prompt_len=2, computed=2, span=[8], prefill=False)])
    finalize = talker.mrv2_codec_history_finalizer(batch, buffer.gather(batch))
    for code in [8, 9, 8]:
        payload = {"codes.audio": torch.tensor([[code]]), "meta.codec_frame_valid": torch.tensor([True])}
        assert finalize(payload, [1]) is payload
    assert previous["recent_codes"] == [8, 9, 8]

    buffer.update(0, {"meta": {"streaming_condition_seq": 1, "turn_start": False}})
    current = prefill()
    assert current is not previous
    assert current["recent_codes"] == [8, 9, 8]
    # A delayed copy still belongs to the old condition despite both buffer
    # dictionaries being copies; it must not pollute the successor's prefix.
    finalize(payload, [1])
    assert previous["recent_codes"] == current["recent_codes"] == [8, 9, 8]
    talker._deferred_cleanup_ids = set()
    talker.on_requests_finished({"r"})
    talker._flush_deferred_cleanup()
    assert not talker._request_audio_states and not talker._request_condition_states


@pytest.mark.parametrize("valid_rows", [[True], [False], [True, False, True], [False, False]])
def test_history_finalizer_batched_read_matches_per_row_masks(valid_rows):
    talker = _talker()
    states = {"a": {"recent_codes": list(range(14))}, "b": {"recent_codes": [3]}}
    talker._request_audio_states = dict(states)
    n = len(valid_rows)
    rows = [dict(slot=0, prompt_len=4, computed=6, span=list(range(50, 50 + n)), prefill=False)]
    rows.append(dict(slot=1, prompt_len=4, computed=6, span=[90], prefill=False))
    batch, _ = _batch(rows)
    finalize = talker.mrv2_codec_history_finalizer(batch, [{"req_id": "a"}, {"req_id": "b"}])
    payload = {
        "codes": {"audio": torch.tensor([[50 + i] for i in range(n)] + [[90]])},
        "meta": {"codec_frame_valid": torch.tensor([*valid_rows, True])},
    }
    assert finalize(payload, [1, 1]) is payload
    kept = [50 + i for i, keep in enumerate(valid_rows) if keep]
    assert states["a"]["recent_codes"] == (list(range(14)) + kept)[-16:]
    assert states["b"]["recent_codes"] == [3, 90]
