"""No-device regression tests for the MiniCPM-o Talker K-step feedback chain.

Layers pinned down here:

- ``make_omni_output`` must record each frame's codec sample in the request
  state (``last_code``) while the multi-frame loop owns sampling: the rows
  vLLM schedules for the next step carry placeholder continue ids, so
  without the record every frame after the first embeds a placeholder
  instead of the real previous frame (silent audio corruption).
- The decode preprocess must prefer that state record over input_ids.
- The scheduler-side K-frame guard must drop continuation drafts whenever
  prefill work is pending, so no step ever mixes prefill rows with
  multi-row decode spans (the mixed batch that crashed a 64/4 run).
- The multi-frame gate matrix: non-uniform decode spans stay blocked at
  ``applies()``; the runner raise remains the assertion of last resort for
  a combination the guard makes unschedulable.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NUM_AUDIO_TOKENS = 6562
_EOS_ID = _NUM_AUDIO_TOKENS - 1


def _make_talker(*, k_step_frames: int, scripted_samples: list[int]):
    """Bare talker instance: no config, no weights, deterministic sampler.

    ``emb_code`` is crafted so the placeholder fallback (row 0) is all
    zeros while real codec rows are not -- a sharp contrast for asserting
    which id got embedded.
    """
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model._k_step_frames = k_step_frames
    model.supports_multi_frame_decode = k_step_frames > 0
    model._codec_temperature = 0.0
    model._codec_min_tokens = 0
    model._codec_max_tokens = 4032
    model._num_audio_tokens = _NUM_AUDIO_TOKENS
    model._codec_eos_id = _EOS_ID
    model._request_audio_states = {}
    emb = nn.Embedding(_NUM_AUDIO_TOKENS, 4)
    with torch.no_grad():
        emb.weight.zero_()
        emb.weight[42] = 1.0
        emb.weight[43] = 2.0
    model.emb_code = nn.ModuleList([emb])
    queue = iter(scripted_samples)

    def _greedy(_hidden, _codes, _request_id, _step, _min_tokens, _max_tokens):
        return torch.tensor(next(queue), dtype=torch.long)

    model._sample_audio_code_greedy = _greedy
    return model


def _frame_call(model, hidden):
    return model.make_omni_output(
        hidden,
        model_intermediate_buffer=[{"request_id": "r1"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
    )


def test_kstep_frame_sample_is_recorded_for_next_frame():
    model = _make_talker(k_step_frames=8, scripted_samples=[42, 43, _EOS_ID])
    state = {"step": 0, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state
    hidden = torch.randn(1, 8)

    out0 = _frame_call(model, hidden)
    assert state["last_code"] == 42
    assert state["step"] == 1
    assert out0.multimodal_outputs["codes"]["audio"][0].reshape(-1).tolist() == [42]

    out1 = _frame_call(model, hidden)
    assert state["last_code"] == 43
    assert out1.multimodal_outputs["codes"]["audio"][0].reshape(-1).tolist() == [43]

    # The EOS frame terminates the request: the record keeps the last real
    # sample (there is no next frame to feed), and the delta goes empty.
    out2 = _frame_call(model, hidden)
    assert state["last_code"] == 43
    assert state["finished"] is True
    assert out2.multimodal_outputs["codes"]["audio"][0].numel() == 0


def test_legacy_single_frame_path_does_not_touch_last_code():
    model = _make_talker(k_step_frames=0, scripted_samples=[42])
    state = {"step": 0, "codes": torch.tensor([10, 11])}
    model._request_audio_states["r1"] = state

    _frame_call(model, torch.randn(1, 8))
    assert "last_code" not in state


def test_decode_preprocess_embeds_state_last_code():
    model = _make_talker(k_step_frames=8, scripted_samples=[])
    state = {"step": 3, "codes": torch.tensor([42]), "last_code": 42}
    model._request_audio_states["r1"] = state
    # The scheduled row carries the placeholder continue id (0), not the
    # real previous frame's sample.
    input_ids = torch.zeros(1, dtype=torch.long)

    _, embeds, out = model.preprocess(
        input_ids,
        None,
        request_id="r1",
        audio_state=state,
        _omni_is_prefill=False,
    )
    assert torch.equal(embeds, model.emb_code[0](torch.tensor([42])))
    # Row 0 (the placeholder fallback) is all zeros in this fixture, so a
    # non-zero embed proves the state record won.
    assert torch.count_nonzero(embeds) == embeds.numel()
    assert out["codes"]["audio"].reshape(-1).tolist() == [42]


def _make_scheduler(*, num_spec: int, waiting=(), running=(), stage: str = "tts"):
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._omni_talker_kstep_cache = None
    # The armed check reads the stage and the n-gram fingerprint off the engine
    # config; the real Scheduler exposes neither as a plain attribute.
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage=stage),
        speculative_config=SimpleNamespace(method="ngram", num_speculative_tokens=num_spec),
    )
    # vLLM's Scheduler stores the count here (vllm_config.num_speculative_tokens);
    # the guard predicts row widths from it.
    sched.num_spec_tokens = num_spec
    sched.max_model_len = 40960
    sched.waiting = list(waiting)
    sched.running = list(running)
    return sched


def _req(*, computed: int, prompt: int, spec: list[int], total: int | None = None):
    return SimpleNamespace(
        num_computed_tokens=computed,
        prompt_token_ids=[0] * prompt,
        # num_tokens is prompt + generated; the guard reads the difference to
        # predict how many rows this request will schedule this step.
        num_tokens=prompt if total is None else total,
        spec_token_ids=list(spec),
    )


def test_guard_drops_drafts_when_waiting_request_pending():
    decode_req = _req(computed=100, prompt=100, spec=[0] * 7)
    sched = _make_scheduler(num_spec=7, waiting=[object()], running=[decode_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert decode_req.spec_token_ids == []


def test_guard_keeps_drafts_when_no_prefill_pending():
    decode_req = _req(computed=100, prompt=100, spec=[0] * 7, total=101)
    sched = _make_scheduler(num_spec=7, waiting=[], running=[decode_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert decode_req.spec_token_ids == [0] * 7


def test_guard_drops_drafts_when_chunked_prefill_in_flight():
    decoding_req = _req(computed=100, prompt=100, spec=[0] * 7, total=101)
    chunking_req = _req(computed=50, prompt=100, spec=[])
    sched = _make_scheduler(num_spec=7, waiting=[], running=[decoding_req, chunking_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert decoding_req.spec_token_ids == []


def test_guard_noop_for_text_stage_spec_config():
    # The text stage may carry its own n-gram config; only the Talker stage runs
    # the multi-frame loop, so the guard must stay a no-op there.
    text_req = _req(computed=100, prompt=100, spec=[0] * 15)
    sched = _make_scheduler(num_spec=15, stage="llm", waiting=[object()], running=[text_req])

    sched._drop_talker_drafts_if_prefill_pending()
    assert text_req.spec_token_ids == [0] * 15


def test_guard_arms_from_vllm_config_and_drops_drafts_for_a_running_chunk():
    # Real vLLM Scheduler instances expose no .speculative_config attribute
    # (and SchedulerConfig has no num_speculative_tokens), so the armed check
    # must read vllm_config.speculative_config -- otherwise the whole guard
    # silently stays off while the engine-side loop is armed.
    #
    # Steady-state decode sharing a step with a 7-token streaming chunk: both
    # spec_token_ids lists look innocent ([] and 7 placeholders), so the guard
    # must predict row widths from num_tokens - num_computed_tokens instead.
    from types import SimpleNamespace

    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._omni_talker_kstep_cache = None
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage="tts"),
        speculative_config=SimpleNamespace(method="ngram", num_speculative_tokens=7),
    )
    assert sched._talker_kstep_armed() is True
    sched.num_spec_tokens = 7
    sched.max_model_len = 40960

    steady = _req(computed=100, prompt=100, spec=[0] * 7, total=101)  # 1 token -> 8 rows
    chunk = _req(computed=107, prompt=107, spec=[], total=114)  # 7-token chunk -> 7 rows
    sched.waiting = []
    sched.running = [steady, chunk]

    sched._drop_talker_drafts_if_prefill_pending()
    assert steady.spec_token_ids == []
    assert chunk.spec_token_ids == []


def test_guard_keeps_drafts_when_all_decodes_share_the_padded_width():
    # A first-step decode (no placeholders yet) and a steady-state decode both
    # schedule 1 token this step, so vLLM gives both the same 1+num_spec row
    # width: spans stay uniform and nothing may drop.
    steady = _req(computed=100, prompt=100, spec=[0] * 7, total=101)
    first_step = _req(computed=100, prompt=100, spec=[], total=101)
    sched = _make_scheduler(num_spec=7, waiting=[], running=[steady, first_step])

    sched._drop_talker_drafts_if_prefill_pending()
    assert steady.spec_token_ids == [0] * 7
    assert first_step.spec_token_ids == []


def test_talker_stop_token_ids_match_the_multi_frame_head():
    """Stage 1's stop id must be one the head that actually runs can emit.

    The pipeline constraint used to pin ``stop_token_ids`` to the codec EOS
    (6561) unconditionally. With the multi-frame loop on, the vLLM-level head
    is the two-wide continue/stop row, so the only sampleable ids are 0/1,
    `check_stop` never matched, and every request ran to ``max_tokens`` (142s
    per request, model finished at frame ~116). The pipeline default therefore
    follows the multi-frame head; a deployment that turns the loop off
    overrides stage 1's stop token per stage.
    """
    from vllm_omni.model_executor.models.minicpmo_4_5 import pipeline as mcp_pipeline
    from vllm_omni.platforms.npu.worker import talker_multiframe

    assert mcp_pipeline._talker_stop_token_ids() == [talker_multiframe.STOP_TOKEN_ID]
    assert talker_multiframe.STOP_TOKEN_ID == 1


def test_multiframe_gate_matrix():
    from vllm_omni.platforms.npu.worker import talker_multiframe

    model = SimpleNamespace(supports_multi_frame_decode=True)
    state = {"step": 5}

    # Uniform K-row decode: the loop engages with K=8.
    uniform = {
        "request_token_spans": [(0, 8), (8, 16)],
        "model_intermediate_buffer": [
            {"audio_state": dict(state)},
            {"audio_state": dict(state)},
        ],
    }
    assert talker_multiframe.applies(model, uniform) == 8

    # Mixed decode spans (an 8-row drafted decode plus a 1-row decode):
    # blocked, and flagged as a multi-token decode step.
    mixed_decode = {
        "request_token_spans": [(0, 8), (8, 9)],
        "model_intermediate_buffer": [
            {"audio_state": dict(state)},
            {"audio_state": dict(state)},
        ],
    }
    assert talker_multiframe.applies(model, mixed_decode) == 0
    assert talker_multiframe.is_multi_token_decode(model, mixed_decode) is True

    # Prefill rows mixed with a drafted decode: blocked the same way.
    mixed_prefill = {
        "request_token_spans": [(0, 515), (515, 523)],
        "model_intermediate_buffer": [
            {"audio_state": dict(state), "_omni_is_prefill": True},
            {"audio_state": dict(state)},
        ],
    }
    assert talker_multiframe.applies(model, mixed_prefill) == 0
    assert talker_multiframe.is_multi_token_decode(model, mixed_prefill) is True


def _vocab_runner(*, supports_multi_frame: bool, vocab_size: int):
    return SimpleNamespace(
        model=SimpleNamespace(supports_multi_frame_decode=supports_multi_frame),
        input_batch=SimpleNamespace(vocab_size=vocab_size),
    )


def test_stop_vocab_gate_reports_the_two_wide_stop_row_not_the_hidden_width():
    """``input_batch.vocab_size`` must be the stop row's width (2), not hidden.

    ``InputBatch.add_request`` keeps ``top_k = vocab_size`` as its "no top-k"
    sentinel and ``RejectionSampler.parse_output`` filters accepted tokens
    against it; both only hold at 0 or 2. The gate used to be handed
    ``text_hidden_states``, so it wrote the hidden width (768): that passes the
    ``parse_output`` filter by accident while taking ``top_k`` out of its
    sentinel range, and the two-wide stop row then never reaches the request's
    token list -- every request runs to ``max_tokens`` instead of stopping on
    EOS (the 910C scene where stage 1 reported ``finished_reason=length`` for
    all 34 requests).
    """
    from vllm_omni.platforms.npu.worker import talker_multiframe

    assert talker_multiframe.STOP_ROW_WIDTH == 2

    runner = _vocab_runner(supports_multi_frame=True, vocab_size=0)
    # Hidden-width rows on purpose: the width of this tensor is not the answer.
    talker_multiframe.ensure_stop_token_vocab(runner, torch.randn(4, 768))
    assert runner.input_batch.vocab_size == 2

    # Idempotent: arming again must not move it.
    talker_multiframe.ensure_stop_token_vocab(runner, torch.randn(4, 768))
    assert runner.input_batch.vocab_size == 2


def test_stop_vocab_gate_stays_off_without_multi_frame_or_rows():
    from vllm_omni.platforms.npu.worker import talker_multiframe

    # Single-frame models (K=1, stage 0/2) never take the branch that reads
    # vocab_size; the gate must leave them exactly as they were.
    single = _vocab_runner(supports_multi_frame=False, vocab_size=0)
    talker_multiframe.ensure_stop_token_vocab(single, torch.randn(4, 768))
    assert single.input_batch.vocab_size == 0

    # No rows at hand: None must not be read as "width unknown, arm anyway".
    armed = _vocab_runner(supports_multi_frame=True, vocab_size=0)
    talker_multiframe.ensure_stop_token_vocab(armed, None)
    assert armed.input_batch.vocab_size == 0


class _MinTokensLogitsProcessor:
    """The name is the contract: the implementation matches on ``type(proc).__name__``."""

    def __init__(self, min_toks):
        self.min_toks = min_toks


def test_kstep_min_tokens_neutralization_clears_the_censor_list():
    """The vLLM-level ``min_tokens`` mask list must be cleared under multi-frame decode.

    It masks the request's only stop signal (id 1 of the binary stop row) and
    unmasks only once ``len(output_token_ids) >= min_tokens`` -- a counter owned
    by vLLM's spec bookkeeping, so a request whose counter does not advance can
    never stop (observed: stage 1 all ``length``, zero ``stop``). The model-side
    guard already holds the codec EOS back with ``state.step < min_tokens``, so
    this layer is redundant.
    """
    from vllm_omni.platforms.npu.worker import talker_multiframe

    censor = _MinTokensLogitsProcessor({0: (50, [0, 0, 0], {1})})
    untouched = _MinTokensLogitsProcessor({0: (50, [0], {1})})
    untouched.__class__ = type("SomeOtherProcessor", (object,), {})  # not the target processor

    talker_multiframe.neutralize_kstep_min_tokens(SimpleNamespace(non_argmax_invariant=[untouched, censor]))
    assert censor.min_toks == {}
    assert untouched.min_toks == {0: (50, [0], {1})}

    # An empty list and odd inputs must not raise either.
    talker_multiframe.neutralize_kstep_min_tokens(SimpleNamespace(non_argmax_invariant=[]))
    talker_multiframe.neutralize_kstep_min_tokens(None)
