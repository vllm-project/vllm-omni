# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA multi-frame Talker decode vs. the single-frame path it replaces.

A tiny fake Talker -- the real MiniCPM-o Talker class with toy codec
embedding/head tables and a causal recurrent "backbone" -- is decoded twice:
once a frame per step through the model's own preprocess / make_omni_output /
compute_logits / sample and a check_stop emulation, once K frames per step
through ``talker_kstep`` (frame 0 on the single-frame path, frames
1..K-1 replayed over a K-row buffer whose later rows are stale). The codec
streams, scheduler tokens and final Talker state must be identical.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.minicpmo_4_5 import talker_frame_plan
from vllm_omni.model_executor.models.minicpmo_4_5 import talker_kstep as mf
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

VOCAB, HIDDEN, EOS = 16, 8, 15
BIG = 10**6

# (vLLM max_tokens, vLLM min_tokens, Talker state max_tokens, state min_tokens)
REQUESTS = [
    (BIG, 6, None, None),  # EOS is favoured: stops on a natural EOS at output 7
    (BIG, 0, None, 4),  # the Talker's own min_tokens holds EOS back until step 4
    (BIG, 8, 4, 10),  # chunk limit forces EOS at step 3, past both min_tokens masks
    (6, 0, None, 100),  # vLLM max_tokens truncates mid-window, never an EOS
]


def _talker(seed: int) -> MiniCPMO45OmniTTSForConditionalGeneration:
    torch.manual_seed(seed)
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.emb_code = nn.ModuleList([nn.Embedding(VOCAB, HIDDEN)])
    head = nn.Linear(HIDDEN, VOCAB, bias=True)
    with torch.no_grad():
        head.bias.zero_()
        head.bias[EOS] = 4.0  # EOS wins whenever nothing masks it
    talker.head_code = nn.ModuleList([head])
    talker._codec_eos_id = EOS
    talker._num_audio_tokens = VOCAB
    talker._k_step_frames = 0
    talker._request_audio_states = {}
    talker._force_eos_rows = talker._mask_eos_rows = talker._pending_force_eos_rows = None
    talker._penalty_histories = None
    return talker.requires_grad_(False)


class _Backbone:
    """Causal toy LM: row state = 0.7 * previous row state + input embedding."""

    def __init__(self, seed: int):
        gen = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(HIDDEN, HIDDEN, generator=gen)

    def row(self, prev_state: torch.Tensor, embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = 0.7 * prev_state + embeds
        return state, torch.tanh(state @ self.proj) * 3.0


def _min_tokens_proc(outputs: list[list[int]]) -> MinTokensLogitsProcessor:
    proc = MinTokensLogitsProcessor(None, torch.device("cpu"), False)
    for index, (_, min_tokens, _, _) in enumerate(REQUESTS):
        if min_tokens:
            proc.min_toks[index] = (min_tokens, outputs[index], {EOS}, False)
    proc.min_toks[-1] = (1, [0], set(), False)  # already reached: forces the first rebuild
    proc.update_state(None)
    return proc


def _metadata(outputs, proc, greedy: bool, generators) -> SamplingMetadata:
    b = len(outputs)
    return SamplingMetadata(
        temperature=None if greedy else torch.full((b,), 0.8),
        all_greedy=greedy,
        all_random=not greedy,
        top_p=None if greedy else torch.full((b,), 0.9),
        top_k=None if greedy else torch.full((b,), 5, dtype=torch.int32),
        generators=generators,
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=torch.zeros((b, 2), dtype=torch.long),
        frequency_penalties=torch.zeros(b),
        presence_penalties=torch.zeros(b),
        repetition_penalties=torch.full((b,), 1.3),
        output_token_ids=[list(o) for o in outputs],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors([proc]),
    )


class _Run:
    """Request state shared by both drivers: a fresh talker, same weights."""

    def __init__(self, greedy: bool):
        self.talker = _talker(0)
        self.backbone = _Backbone(1)
        self.rids = [f"r{i}" for i in range(len(REQUESTS))]
        gen = torch.Generator().manual_seed(2)
        self.prefix = torch.randn(len(REQUESTS), HIDDEN, generator=gen)
        self.last = [3, 5, 7, 9]  # codec ids the prefill sampled
        self.outputs = [[t] for t in self.last]
        self.alive = [True] * len(REQUESTS)
        self.streams: list[list[int]] = [[] for _ in REQUESTS]
        self.flags = [False] * len(REQUESTS)
        self.final: list[tuple | None] = [None] * len(REQUESTS)
        for rid, (_, _, max_tokens, min_tokens) in zip(self.rids, REQUESTS):
            self.talker._request_audio_states[rid] = {
                "finished": False,
                "step": 0,
                "max_tokens": max_tokens,
                "min_tokens": min_tokens,
                "turn_end_drain": False,
            }
        self.proc = _min_tokens_proc(self.outputs)
        self.greedy = greedy
        self.generators = {} if greedy else {i: torch.Generator().manual_seed(100 + i) for i in range(len(REQUESTS))}

    def preprocess(self, span: int):
        embeds, infos = [], []
        for rid, last in zip(self.rids, self.last):
            state = self.talker._request_audio_states[rid]
            _, emb, upd = self.talker.preprocess(
                torch.tensor([last] * span), None, request_id=rid, audio_state=state, _omni_is_prefill=False
            )
            embeds.append(emb[0])
            infos.append({"request_id": rid, "audio_state": state, "codes": upd["codes"]})
        return torch.stack(embeds), infos

    def record(self, index: int, output, frame_flag: bool):
        audio = output.multimodal_outputs["codes"]["audio"][index]
        self.streams[index].extend(int(t) for t in audio.reshape(-1).tolist())
        self.flags[index] |= frame_flag

    def accept(self, index: int, tokens: list[int]) -> None:
        """check_stop, token by token; the loop must never need truncating."""
        max_tokens = REQUESTS[index][0]
        for position, token in enumerate(tokens):
            self.outputs[index].append(token)
            if token == EOS or len(self.outputs[index]) >= max_tokens:
                assert position == len(tokens) - 1, "the loop emitted a frame past check_stop"
                self.alive[index] = False
                self.final[index] = self.state()[index]
        self.last[index] = tokens[-1]

    def state(self):
        states = self.talker._request_audio_states
        return [
            (states[r]["step"], list(states[r].get("recent_codes", [])), bool(states[r]["finished"])) for r in self.rids
        ]


def _single_frame(greedy: bool) -> _Run:
    run = _Run(greedy)
    spans = [(i, i + 1) for i in range(len(REQUESTS))]
    for _ in range(64):
        if not any(run.alive):
            break
        run.proc.update_state(None)
        embeds, infos = run.preprocess(1)
        state, hidden = run.backbone.row(run.prefix, embeds)
        output = run.talker._make_omni_output_single_frame(hidden, infos, spans)
        logits = run.talker.compute_logits(hidden)
        md = _metadata(run.outputs, run.proc, greedy, run.generators)
        sampled = run.talker.sample(logits, md).sampled_token_ids[:, 0].tolist()
        for i in range(len(REQUESTS)):
            if run.alive[i]:
                run.record(i, output, bool(output.multimodal_outputs["meta"]["finished"][i]))
                run.prefix[i] = state[i]
                run.accept(i, [int(sampled[i])])
    assert not any(run.alive)
    return run


def _multi_frame(greedy: bool, frames: int) -> _Run:
    run = _Run(greedy)
    num = len(REQUESTS)
    spans = [(i * frames, (i + 1) * frames) for i in range(num)]
    for _ in range(64):
        if not any(run.alive):
            break
        run.proc.update_state(None)
        embeds0, infos = run.preprocess(frames)
        buf = torch.randn(frames, num, HIDDEN)  # stale rows past the current frame
        buf[0] = embeds0
        row_states: list[torch.Tensor] = []

        def forward(k: int, embeds: torch.Tensor, buf=buf, row_states=row_states) -> torch.Tensor:
            buf[k] = embeds
            state, hiddens = run.prefix, []
            row_states.clear()
            for j in range(frames):
                state, hidden = run.backbone.row(state, buf[j])
                row_states.append(state)
                hiddens.append(hidden)
            return hiddens[k]

        hidden0 = forward(0, embeds0)
        output = run.talker._make_omni_output_single_frame(hidden0, infos, spans)
        md = _metadata(run.outputs, run.proc, greedy, run.generators)
        first = run.talker.sample(run.talker.compute_logits(hidden0), md).sampled_token_ids[:, 0].long()
        plan = talker_frame_plan.plan_codec_frames(run.talker, infos, frames)
        controls, _ = mf.build_controls(
            plan,
            frames=frames,
            vocab_size=VOCAB,
            budgets=[REQUESTS[i][0] - len(run.outputs[i]) for i in range(num)],
            stop_ids=[{EOS}] * num,
            min_tokens_state=run.proc.min_toks,
            penalties=md.repetition_penalties,
            device=torch.device("cpu"),
        )
        frame_md = mf.frame_sampling_metadata(md)
        sampled, emitted = mf.decode_frames(
            first,
            frames,
            controls,
            embed=run.talker.emb_code[0],
            forward=forward,
            codec_logits=run.talker.compute_logits,
            sample=lambda logits: mf.sample_frame(logits, frame_md),
        )
        ids = torch.where(emitted, sampled, -1).tolist()
        tokens = [[t for t in row if t >= 0] for row in ids]
        assert all(row[: len(t)] == t for row, t in zip(ids, tokens)), "emitted frames must be a prefix"
        forwarded = [t[:-1] for t in tokens]
        flags = talker_frame_plan.commit_codec_frames(run.talker, plan, forwarded)
        for i in range(num):
            if run.alive[i]:
                run.record(i, output, bool(output.multimodal_outputs["meta"]["finished"][i]) or flags[i])
                run.streams[i].extend(forwarded[i])
                run.prefix[i] = row_states[len(tokens[i]) - 1][i]
                run.accept(i, tokens[i])
    assert not any(run.alive)
    return run


# ---------------------------------------------------------------------------
# End to end through the runner hook: maybe_run, vLLM's split of the sampled
# ids, the scheduler's accept/rollback and the next step's drafts.
# ---------------------------------------------------------------------------


class _Model:
    """What maybe_run / propose_drafts read off the stage-1 model wrapper."""

    supports_multi_frame_decode = True
    requires_request_sample_eligibility = True
    codec_eos_token_id = EOS
    codec_vocab_size = VOCAB

    def __init__(self, talker: MiniCPMO45OmniTTSForConditionalGeneration):
        self.talker = talker

    def plan_codec_frames(self, infos, frames):
        return talker_frame_plan.plan_codec_frames(self.talker, infos, frames)

    def commit_codec_frames(self, plan, forwarded):
        return talker_frame_plan.commit_codec_frames(self.talker, plan, forwarded)

    def compute_logits(self, hidden):
        return self.talker.compute_logits(hidden)

    def embed_input_ids(self, ids):
        return self.talker.emb_code[0](ids)


class _Runner:
    """The GPUARModelRunner surface the hook touches.

    ``batch_vocab`` is ``input_batch.vocab_size``. The real stage 1 reports 0:
    MiniCPMTTSConfig has no ``vocab_size``, so vLLM's ``get_vocab_size()``
    falls back to 0 for it.
    """

    def __init__(self, run: _Run, frames: int, batch_vocab: int):
        self.model = _Model(run.talker)
        self.num_spec_tokens = frames - 1
        self.use_async_scheduling = False
        self.max_model_len = BIG
        self.input_batch = SimpleNamespace(
            num_reqs=len(run.rids), req_ids=list(run.rids), vocab_size=batch_vocab, sampling_metadata=None
        )
        self.requests = {rid: SimpleNamespace(sampling_params=_params()) for rid in run.rids}
        arch = SimpleNamespace(vocab_size=batch_vocab)
        self.model_config = SimpleNamespace(model_arch_config=arch, get_vocab_size=lambda: arch.vocab_size)

    def _sample(self, logits, spec_decode_metadata):
        # GPUARModelRunner._sample without drafts: the model's own sampler.
        assert spec_decode_metadata is None
        return self.model.talker.sample(logits, self.input_batch.sampling_metadata)


def _bookkeep(ids: torch.Tensor, vocab_size: int) -> list[list[int]]:
    """GPUModelRunner._bookkeeping_sync's split of the step's sampled ids."""
    from vllm.v1.sample.rejection_sampler import RejectionSampler

    if ids.shape[-1] == 1:
        return ids.tolist()
    return RejectionSampler.parse_output(ids, vocab_size)[0]


def _engine(greedy: bool, frames: int, *, batch_vocab: int, force_drafts: bool = False, prepare=None):
    """Drive the real maybe_run/propose_drafts the way stage 1 does.

    Each step schedules ``[last] + drafts`` per request; the runner's
    preprocess embeds ``last`` at the span's first row (later rows are stale),
    make_omni_output does frame 0's bookkeeping and maybe_run the rest. The
    ids are split as vLLM splits them, and the scheduler emulation accepts what
    comes back: a request's KV is committed up to its last accepted id, and an
    empty row rolls the whole span back (OmniARScheduler's empty-row path), so
    the next step forwards the same codec id again.
    ``force_drafts`` schedules drafts every step, whatever propose_drafts said.
    Returns the run and the scheduled span width of every step.
    """
    run = _Run(greedy)
    runner = _Runner(run, frames, batch_vocab)
    if prepare is not None:
        prepare(runner)
    num = len(REQUESTS)
    drafts: list[list[int]] = [[] for _ in range(num)]
    stale = torch.Generator().manual_seed(3)
    widths: list[int] = []
    for _ in range(64):
        if not any(run.alive):
            break
        run.proc.update_state(None)
        width = 1 + len(drafts[0])
        widths.append(width)
        embeds0, infos = run.preprocess(width)
        starts = [i * width for i in range(num)]
        spans = [(s, s + width) for s in starts]
        buf = torch.randn(num * width, HIDDEN, generator=stale)
        buf[starts] = embeds0
        row_states: list[list[torch.Tensor]] = [[run.prefix[i]] * width for i in range(num)]

        def run_model(buf=buf, width=width, starts=starts, row_states=row_states) -> torch.Tensor:
            hidden = torch.empty(num * width, HIDDEN)
            for i, start in enumerate(starts):
                state = run.prefix[i]
                for j in range(width):
                    state, hidden[start + j] = run.backbone.row(state, buf[start + j])
                    row_states[i][j] = state
            return hidden

        hidden = run_model()
        output = run.talker._make_omni_output_single_frame(hidden, infos, spans)
        md = _metadata(run.outputs, run.proc, greedy, run.generators)
        runner.input_batch.sampling_metadata = md
        extra = {
            "request_token_spans": spans,
            "model_intermediate_buffer": infos,
            "request_max_tokens_remaining": [REQUESTS[i][0] - len(run.outputs[i]) for i in range(num)],
        }
        output = mf.maybe_run(runner, output, run_model=run_model, inputs_embeds=buf, model_kwargs_extra=extra)
        stash = mf.take_sampler_output(runner)
        if stash is None:
            assert width == 1, "a multi-row Talker step must come back through the stash"
            ids = run.talker.sample(run.talker.compute_logits(hidden), md).sampled_token_ids
        else:
            ids = stash.sampled_token_ids
        valid = _bookkeep(ids, runner.input_batch.vocab_size)
        for i in range(num):
            if not run.alive[i]:
                continue
            run.record(i, output, bool(output.multimodal_outputs["meta"]["finished"][i]))
            if valid[i]:
                run.prefix[i] = row_states[i][len(valid[i]) - 1]
                run.accept(i, valid[i])
        proposed = mf.propose_drafts(runner, valid)
        if force_drafts:
            proposed = [[run.last[i]] * (frames - 1) for i in range(num)]
        drafts = proposed or [[] for _ in range(num)]
    assert not any(run.alive)
    return run, widths


def _assert_same(got: _Run, ref: _Run) -> None:
    assert got.streams == ref.streams, "codec frames sent to Code2Wav differ"
    assert got.outputs == ref.outputs
    assert got.flags == ref.flags
    assert got.final == ref.final


@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "seeded"])
def test_engaged_through_maybe_run_matches_single_frame(greedy: bool):
    ref = _single_frame(greedy)
    got, widths = _engine(greedy, 4, batch_vocab=VOCAB)
    _assert_same(got, ref)
    assert widths[0] == 1 and set(widths[1:]) == {4}, widths


@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "seeded"])
def test_declined_multi_row_step_is_one_single_frame_step(greedy: bool):
    """The real stage 1 (input_batch.vocab_size 0) with drafts scheduled anyway.

    maybe_run must decline, and the step must be exactly one single-frame
    step: the same codes, one frame per request to Code2Wav, and the same
    Talker state. A -1-padded (B, K) output went through parse_output, which
    drops every id >= vocab_size; the scheduler rolled the whole span back and
    the next step re-sent the same codec id, so every frame reached Code2Wav
    twice.
    """
    ref = _single_frame(greedy)
    got, widths = _engine(greedy, 4, batch_vocab=0, force_drafts=True)
    _assert_same(got, ref)
    assert set(widths[1:]) == {4}, widths


def test_no_drafts_when_the_loop_would_decline():
    """With vocab_size 0 the next step is not drafted in the first place."""
    ref = _single_frame(greedy=True)
    got, widths = _engine(True, 4, batch_vocab=0)
    _assert_same(got, ref)
    assert set(widths) == {1}, widths


@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "seeded"])
def test_codec_vocab_lets_a_vocab0_stage_engage(greedy: bool):
    """ensure_codec_vocab on a stage whose config reports vocab_size 0."""
    ref = _single_frame(greedy)
    got, widths = _engine(greedy, 4, batch_vocab=0, prepare=mf.ensure_codec_vocab)
    _assert_same(got, ref)
    assert set(widths[1:]) == {4}, widths


@pytest.mark.parametrize("frames", [2, 3, 4, 8])
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "seeded"])
def test_multi_frame_matches_single_frame(greedy: bool, frames: int):
    ref = _single_frame(greedy)
    _assert_same(_multi_frame(greedy, frames), ref)


def _runner(req_params: list[SimpleNamespace], num_spec: int = 3, batch_vocab: int = VOCAB):
    md = SimpleNamespace(
        max_num_logprobs=None,
        logprob_token_ids=None,
        bad_words_token_ids={},
        allowed_token_ids_mask=None,
        thinking_budget_state_holder=None,
        logitsprocs=LogitsProcessors(),
    )
    req_ids = [f"r{i}" for i in range(len(req_params))]
    return SimpleNamespace(
        model=SimpleNamespace(
            supports_multi_frame_decode=True,
            requires_request_sample_eligibility=True,
            plan_codec_frames=lambda *a: None,
            codec_eos_token_id=EOS,
            codec_vocab_size=VOCAB,
        ),
        num_spec_tokens=num_spec,
        use_async_scheduling=False,
        input_batch=SimpleNamespace(
            num_reqs=len(req_ids), req_ids=req_ids, sampling_metadata=md, vocab_size=batch_vocab
        ),
        requests={r: SimpleNamespace(sampling_params=p) for r, p in zip(req_ids, req_params)},
    )


def _params(**overrides):
    base = dict(
        frequency_penalty=0.0,
        presence_penalty=0.0,
        structured_outputs=None,
        repetition_detection=None,
        stop_token_ids=[EOS],
        eos_token_id=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_drafts_repeat_last_codec_for_the_whole_batch_or_none():
    runner = _runner([_params(), _params()])
    assert mf.propose_drafts(runner, [[4], [7, 9]]) == [[4, 4, 4], [9, 9, 9]]
    # One request sampled nothing: nobody drafts, the next step is one frame.
    assert mf.propose_drafts(runner, [[4], []]) == [[], []]
    # A request whose stop set lacks the codec EOS cannot take the loop.
    assert mf.propose_drafts(_runner([_params(), _params(stop_token_ids=[1])]), [[4], [7]]) == [[], []]
    assert mf.propose_drafts(_runner([_params(frequency_penalty=0.5)]), [[4]]) == [[]]
    # Other stages (and K=1) fall through to vLLM's drafter.
    assert mf.propose_drafts(_runner([_params()], num_spec=0), [[4]]) is None
    other = _runner([_params()])
    other.model = SimpleNamespace(supports_multi_frame_decode=False)
    assert mf.propose_drafts(other, [[4]]) is None


def test_no_drafts_for_a_step_maybe_run_would_decline(monkeypatch):
    """Everything maybe_run declines on that is known at draft time stops the
    drafts, and each distinct reason is logged once, with its numbers."""
    log = SimpleNamespace(calls=[])
    monkeypatch.setattr(mf, "logger", SimpleNamespace(info=lambda *a: log.calls.append(a[0] % a[1:])))
    monkeypatch.setattr(mf, "_LOGGED", set())
    # The real stage 1 without ensure_codec_vocab: input_batch.vocab_size 0.
    narrow = _runner([_params(), _params()], batch_vocab=0)
    assert mf.propose_drafts(narrow, [[4], [7]]) == [[], []]
    assert mf.propose_drafts(narrow, [[5], [8]]) == [[], []]
    assert log.calls == [
        "[minicpmo] CUDA multi-frame Talker decode: no drafts: input batch vocab is narrower than the codec "
        f"head (input batch vocab 0, codec head {VOCAB})"
    ]
    asynchronous = _runner([_params()])
    asynchronous.use_async_scheduling = True
    assert mf.propose_drafts(asynchronous, [[4]]) == [[]]
    no_budget = _runner([_params()])
    no_budget.model.requires_request_sample_eligibility = False
    assert mf.propose_drafts(no_budget, [[4]]) == [[]]
    penalty = _runner([_params(), _params(presence_penalty=0.5)])
    assert mf.propose_drafts(penalty, [[4], [7]]) == [[], []]
    assert len(log.calls) == 4
    assert "(stage 1 needs async_scheduling: false)" in log.calls[1]
    assert "requires_request_sample_eligibility" in log.calls[2]
    assert "request r1: frequency 0.0, presence 0.5" in log.calls[3]


def test_ensure_codec_vocab_raises_only_a_narrow_talker_vocab():
    runner = _runner([_params()], batch_vocab=0)
    arch = SimpleNamespace(vocab_size=0)
    runner.model_config = SimpleNamespace(model_arch_config=arch, get_vocab_size=lambda: arch.vocab_size)
    mf.ensure_codec_vocab(runner)
    # model_config too: initialize_kv_cache rebuilds the InputBatch from it.
    assert (runner.input_batch.vocab_size, arch.vocab_size) == (VOCAB, VOCAB)
    wide = _runner([_params()], batch_vocab=32000)
    wide_arch = SimpleNamespace(vocab_size=32000)
    wide.model_config = SimpleNamespace(model_arch_config=wide_arch, get_vocab_size=lambda: wide_arch.vocab_size)
    mf.ensure_codec_vocab(wide)
    assert (wide.input_batch.vocab_size, wide_arch.vocab_size) == (32000, 32000)
    other = _runner([_params()], batch_vocab=0)
    other.model = SimpleNamespace(supports_multi_frame_decode=False, codec_vocab_size=VOCAB)
    mf.ensure_codec_vocab(other)
    assert other.input_batch.vocab_size == 0


def test_frame_budgets_bound_a_request_without_max_tokens_by_the_context():
    runner = _runner([_params(), _params()])
    runner.max_model_len = 100
    runner.requests["r1"].num_tokens = 97
    assert mf.frame_budgets(runner, {"request_max_tokens_remaining": [5, None]}, 2) == ([5, 3], None)
    budgets, decline = mf.frame_budgets(runner, {"request_max_tokens_remaining": [5]}, 2)
    assert budgets == [] and str(decline) == "no per-request token budget (1 budgets for 2 requests)"


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_kstep_overlay_arms_cuda_stage1_only(platform: str):
    """The opt-in overlay arms K=8 on CUDA stage 1 and keeps the codec EOS as
    the only stop id; the base config (and NPU through the overlay) is as before."""
    from pathlib import Path

    from tests.helpers.stage_config import get_deploy_config_path
    from vllm_omni.config.pipeline_registry import resolve_pipeline_config
    from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config, merge_pipeline_deploy

    def stage1(name: str):
        deploy = _apply_platform_overrides(load_deploy_config(Path(get_deploy_config_path(name))), platform=platform)
        return merge_pipeline_deploy(resolve_pipeline_config("minicpmo_4_5"), deploy)[1]

    overlay, base = stage1("minicpmo_4_5_kstep.yaml"), stage1("minicpmo_4_5.yaml")
    if platform == "cuda":
        assert "speculative_config" not in base.yaml_engine_args
        assert overlay.yaml_engine_args["speculative_config"]["num_speculative_tokens"] == 7
        assert overlay.yaml_engine_args["async_scheduling"] is False
        assert overlay.yaml_extras["default_sampling_params"]["stop_token_ids"] == [6561]
    else:
        assert overlay.yaml_engine_args == base.yaml_engine_args
        assert overlay.yaml_extras["default_sampling_params"] == base.yaml_extras["default_sampling_params"]
