# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 native duplex sampling: batched parameter reads and the vectorized repetition penalty."""

import copy
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

Model = MiniCPMO45OmniForConditionalGeneration


@pytest.mark.parametrize(
    "value",
    [torch.tensor([0.5, 0.9, 1.3]), torch.tensor([0.25]), torch.tensor(0.7), torch.tensor([]), 0.4, None],
)
def test_batched_rows_match_per_row_reads(value):
    md = SimpleNamespace(temperature=value)
    rows = Model._sampling_metadata_rows(md, "temperature", 4, 0.8)
    assert rows == [Model._sampling_metadata_value(md, "temperature", row, 0.8) for row in range(4)]


def _row_model(generated: list[int]) -> Model:
    model = Model.__new__(Model)
    model._minicpmo45_duplex_state_for_row = lambda row: SimpleNamespace(generated_tokens=list(generated))
    model._record_minicpmo45_duplex_generation_token = lambda row, token: None
    model._maybe_cut_minicpmo45_native_duplex_text_chunk = lambda sampled, recent, token_ids: sampled
    model._finalize_minicpmo45_native_duplex_sample = lambda row, sampled, token_ids: sampled
    model._minicpmo45_native_forbidden_token_ids = lambda token_ids: []
    return model


def _reference_sample(logits, generated, temperature, top_k, top_p, generator):
    """A straight-line stage-2 draw: penalty, temperature, candidates, multinomial.

    Uses the same candidate-space primitives the production paths use, so the
    seeded-equality tests below stay meaningful for the batching/vectorization
    they cover; the candidate distribution itself is pinned against the
    full-vocabulary filter by test_candidate_distribution_matches_the_filter.
    """
    logits = logits.clone()
    for token_id in set(generated):
        if 0 <= token_id < logits.shape[-1]:
            logits[0, token_id] /= 1.05
    probs, indices = Model._duplex_top_k_top_p_candidates(logits / temperature, top_k=top_k, top_p=top_p)
    draw = torch.multinomial(probs[0], 1, generator=generator)
    return int(indices[0].gather(0, draw).item())


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    ("top_k", "top_p"),
    [(20, 0.85), (0, 0.9), (10, 1.0), (0, 1.0), (5, 0.5), (1, 0.8)],
)
def test_candidate_distribution_matches_the_filter(seed, top_k, top_p):
    """Candidate-space sampling is the same categorical as filter + softmax over the vocabulary."""
    vocab = 4096
    logits = torch.randn(1, vocab, generator=torch.Generator().manual_seed(seed)) * 4
    filtered = torch.softmax(Model._top_k_top_p_filter(logits.clone(), top_k=top_k, top_p=top_p), dim=-1)
    probs, indices = Model._duplex_top_k_top_p_candidates(logits.clone(), top_k=top_k, top_p=top_p)
    dense = torch.zeros_like(filtered)
    dense[0].scatter_(0, indices[0], probs[0])
    support = filtered[0] > 0
    assert torch.equal(support, dense[0] > 0)
    assert torch.allclose(filtered[0][support], dense[0][support], atol=1e-5)
    assert abs(dense[0].sum().item() - 1.0) < 1e-5


@pytest.mark.parametrize("seed", range(4))
def test_boundary_chunk_eos_probability_matches_full_softmax(seed):
    """exp(l_eos - logsumexp(l)) is softmax(l)[chunk_eos], the boundary Bernoulli's p."""
    vocab = 2048
    logits = torch.randn(1, vocab, generator=torch.Generator().manual_seed(seed)) * 3
    logits[0, 7] += 4.0  # make chunk_eos competitive, not a denormal corner
    chunk_eos = 7
    full = torch.softmax(logits, dim=-1)[0, chunk_eos]
    fast = Model._duplex_boundary_chunk_eos_probs(logits, chunk_eos)[0]
    assert torch.allclose(full, fast, atol=1e-6)
    assert 0.0 < float(fast) < 1.0


@pytest.mark.parametrize("seed", range(6))
def test_vectorized_repetition_penalty_samples_like_the_per_token_loop(seed):
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy

    vocab = 64
    logits = torch.randn(1, vocab, generator=torch.Generator().manual_seed(seed)) * 3
    generated = [
        int(t) for t in torch.randint(0, vocab + 4, (40,), generator=torch.Generator().manual_seed(100 + seed))
    ]
    generated += [3, 3, 3]  # repeats are penalized once
    history = MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE
    md = SimpleNamespace(
        all_greedy=False,
        generators={0: torch.Generator().manual_seed(7)},
        output_token_ids=[[]],
    )
    sampled = _row_model(generated)._sample_minicpmo45_native_duplex_row(
        logits.clone(), md, row_idx=0, token_ids={}, params=(0.9, 20, 0.85)
    )
    expected = _reference_sample(logits, generated[-history:], 0.9, 20, 0.85, torch.Generator().manual_seed(7))
    assert sampled == expected


def test_row_without_params_still_reads_the_metadata():
    md = SimpleNamespace(
        all_greedy=False,
        generators={0: torch.Generator().manual_seed(3)},
        output_token_ids=[[]],
        temperature=torch.tensor([0.8]),
        top_k=torch.tensor([10]),
        top_p=torch.tensor([0.9]),
    )
    logits = torch.randn(1, 32, generator=torch.Generator().manual_seed(1))
    model = _row_model([])
    a = model._sample_minicpmo45_native_duplex_row(logits.clone(), md, row_idx=0, token_ids={})
    md.generators = {0: torch.Generator().manual_seed(3)}
    b = model._sample_minicpmo45_native_duplex_row(logits.clone(), md, row_idx=0, token_ids={}, params=(0.8, 10, 0.9))
    assert a == b


# --- _sample_minicpmo45_native_duplex_rows: the batched Stage-0 core -------
#
# The reference is `_sample_minicpmo45_native_duplex_row`, driven one row at
# a time like the pre-batching Stage-0 loop: the returned tokens and the host
# state it mutates must match.

TOKEN_IDS = {
    "unit_token_id": 1,
    "chunk_eos_token_id": 5,
    "listen_token_id": 6,
    "tts_bos_token_id": 7,
    "turn_eos_token_id": 8,
    "chunk_tts_eos_token_id": 9,
    "tts_pad_token_id": 10,
}
VOCAB = 40


def _rows_model(
    states: dict[int, SimpleNamespace],
    *,
    forbidden: list[int] | None = None,
    max_new_speak_tokens_per_chunk: int | None = None,
) -> Model:
    """A model with only the host-data lookups mocked; the samplers themselves are real."""
    model = Model.__new__(Model)
    model._minicpmo45_duplex_state_for_row = lambda row_idx: states.get(row_idx)
    model._minicpmo45_duplex_payload_for_row = lambda row_idx: None
    model._minicpmo45_duplex_row_request_max_tokens = lambda row_idx: None
    model._minicpmo45_native_forbidden_token_ids = lambda token_ids: list(forbidden or [])
    # Avoid a tokenizer dependency; the chunk-boundary/forced-chunk_eos paths
    # under test do not depend on chunk-cutting by decoded text length.
    model._maybe_cut_minicpmo45_native_duplex_text_chunk = lambda sampled, recent, token_ids: sampled
    if max_new_speak_tokens_per_chunk is not None:
        model.max_new_speak_tokens_per_chunk = max_new_speak_tokens_per_chunk
    return model


def _mixed_scenario(seed: int):
    """Six rows exercising every branch: plain sampling, temperature<=0,
    a forced chunk_eos (speak-length cap), a boundary-forced chunk_eos, and
    a repetition-penalized row -- built twice (reference vs. vectorized)
    with independent but identically-seeded state so the two runs are
    directly comparable."""
    torch.manual_seed(seed)
    logits = torch.randn(6, VOCAB) * 2
    # Row 3: make chunk_eos overwhelmingly likely so the boundary stage picks
    # it on its own, without ever reaching stage 2.
    logits[3, TOKEN_IDS["chunk_eos_token_id"]] += 50.0

    output_token_ids = [
        [1, 2, 3],
        [4, 4, 5, 6],
        [1, 1],  # row 2: length 2 trips the max_new_speak_tokens_per_chunk=3 cap
        [2, 3],
        [7, 8, 9],
        [11, 11, 11, 12, 13],
    ]
    row_params = [
        (0.8, 10, 0.9),  # row 0: plain sampled
        (1.1, 0, 1.0),  # row 1: sampled, top_k/top_p disabled
        (0.7, 100, 0.8),  # row 2: forced chunk_eos, params unused
        (0.7, 100, 0.8),  # row 3: boundary-forced chunk_eos, params unused
        (0.0, 100, 0.8),  # row 4: row-level greedy despite all_greedy=False
        (0.9, 20, 0.85),  # row 5: repetition-penalized sampled row
    ]
    seeds = [11, 22, 33, 44, 55, 66]

    def build():
        states = {
            row_idx: SimpleNamespace(generated_tokens=list(tokens) if row_idx == 5 else [])
            for row_idx, tokens in enumerate([[], [], [], [], [], [11, 11, 12]])
        }
        model = _rows_model(states, forbidden=[TOKEN_IDS["tts_pad_token_id"]], max_new_speak_tokens_per_chunk=3)
        md = SimpleNamespace(
            all_greedy=False,
            generators={row_idx: torch.Generator().manual_seed(s) for row_idx, s in enumerate(seeds)},
            output_token_ids=[list(tokens) for tokens in output_token_ids],
        )
        return model, states, md

    return logits.clone(), row_params, build


def _reference_rows(model, logits, md, row_idxs, row_params):
    """The pre-batching Stage-0 loop body: one `_sample_minicpmo45_native_duplex_row` call per row."""
    out = []
    for row_idx in row_idxs:
        row_logits = logits[row_idx : row_idx + 1].clone()
        sampled = model._sample_minicpmo45_native_duplex_row(
            row_logits,
            md,
            row_idx=row_idx,
            token_ids=TOKEN_IDS,
            params=row_params[row_idx],
        )
        out.append(sampled)
    return out


@pytest.mark.parametrize("seed", range(4))
def test_batched_rows_match_the_reference_per_row_loop(seed):
    logits, row_params, build = _mixed_scenario(seed)
    row_idxs = list(range(6))

    ref_model, ref_states, ref_md = build()
    expected = _reference_rows(ref_model, logits, ref_md, row_idxs, row_params)

    vec_model, vec_states, vec_md = build()
    actual = vec_model._sample_minicpmo45_native_duplex_rows(
        logits, vec_md, row_idxs=row_idxs, token_ids=TOKEN_IDS, row_params=row_params
    )

    assert actual == expected
    # Row 2 and row 3 resolve without ever reaching stage 2, so nothing is
    # recorded for them; every other row's generation history must match.
    for row_idx in row_idxs:
        assert vec_states[row_idx].generated_tokens == ref_states[row_idx].generated_tokens


def test_batched_rows_match_the_reference_when_all_greedy():
    logits, row_params, build = _mixed_scenario(seed=0)
    row_idxs = list(range(6))

    ref_model, ref_states, ref_md = build()
    ref_md.all_greedy = True
    expected = _reference_rows(ref_model, logits, ref_md, row_idxs, row_params)

    vec_model, vec_states, vec_md = build()
    vec_md.all_greedy = True
    actual = vec_model._sample_minicpmo45_native_duplex_rows(
        logits, vec_md, row_idxs=row_idxs, token_ids=TOKEN_IDS, row_params=row_params
    )

    assert actual == expected
    for row_idx in row_idxs:
        assert vec_states[row_idx].generated_tokens == ref_states[row_idx].generated_tokens


def test_batched_rows_match_the_reference_without_chunk_eos():
    """When the model has no chunk_eos token, every row skips the boundary
    stage entirely and goes straight to stage 2 (same as the reference)."""
    logits, row_params, build = _mixed_scenario(seed=1)
    row_idxs = list(range(6))
    token_ids_no_chunk_eos = {k: v for k, v in TOKEN_IDS.items() if k != "chunk_eos_token_id"}

    ref_model, ref_states, ref_md = build()
    expected = [
        ref_model._sample_minicpmo45_native_duplex_row(
            logits[row_idx : row_idx + 1].clone(),
            ref_md,
            row_idx=row_idx,
            token_ids=token_ids_no_chunk_eos,
            params=row_params[row_idx],
        )
        for row_idx in row_idxs
    ]

    vec_model, vec_states, vec_md = build()
    actual = vec_model._sample_minicpmo45_native_duplex_rows(
        logits, vec_md, row_idxs=row_idxs, token_ids=token_ids_no_chunk_eos, row_params=row_params
    )
    assert actual == expected


# --- The deferred (device-decided) Stage-0 path ------------------------------
#
# Reference: the synchronous path of `_sample_minicpmo45_native_duplex_stage0`
# (the deferred path forced off). Both run the same two consecutive steps; the
# tokens, the host state after the commit, and every generator's position must
# match, so a rewound extra draw can not change any later token.


class _CharTokenizer:
    """Token t decodes to (t % 3) + 1 copies of one CJK character; special ids decode to ""."""

    clean_up_tokenization_spaces = False
    bad_token_ids: list[int] = []

    def __init__(self, special: set[int], vocab: int):
        self.all_special_ids = sorted(special)
        self._special = special
        self._vocab = vocab

    def __len__(self):
        return self._vocab

    def _piece(self, token_id: int, skip_special_tokens: bool) -> str:
        if token_id in self._special:
            return "" if skip_special_tokens else f"<{token_id}>"
        return chr(0x4E00 + token_id) * (token_id % 3 + 1)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._piece(int(t), skip_special_tokens) for t in ids)

    def batch_decode(self, batch, skip_special_tokens=False):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]


def _stage0_model(states: dict[int, SimpleNamespace], *, max_chars: int) -> Model:
    model = Model.__new__(Model)
    model.model_stage = "llm"
    model._minicpmo45_native_duplex_token_ids_cache = dict(TOKEN_IDS)
    model._minicpmo45_tokenizer_cache = _CharTokenizer(set(TOKEN_IDS.values()), VOCAB)
    model._minicpmo45_duplex_state_for_row = lambda row_idx: states.get(row_idx)
    model._minicpmo45_duplex_payload_for_row = lambda row_idx: None
    model._minicpmo45_duplex_row_request_max_tokens = lambda row_idx: None
    model.max_new_speak_tokens_per_chunk = 5
    model.max_speak_chars_per_chunk = max_chars
    return model


def _stage0_scenario(seed: int, *, all_greedy: bool = False):
    torch.manual_seed(seed)
    steps = []
    for _ in range(2):
        logits = torch.randn(6, VOCAB) * 2
        logits[3, TOKEN_IDS["chunk_eos_token_id"]] += 50.0  # row 3: boundary picks chunk_eos
        logits[1, TOKEN_IDS["listen_token_id"]] += 40.0  # row 1: listen, rewritten to tts_bos
        steps.append(logits)
    histories = [[11, 12], [2], [1, 2, 3, 4], [2, 3], [7, 8, 9], [20, 21, 22]]
    temperature = torch.tensor([0.8, 1.1, 0.7, 0.7, 0.0, 0.9])

    def build():
        states = {
            row_idx: SimpleNamespace(
                generated_tokens=[11, 11, 12] if row_idx == 5 else [],
                current_turn_ended=row_idx != 1,
                pending_speech_context=False,
            )
            for row_idx in range(6)
        }
        md = SimpleNamespace(
            all_greedy=all_greedy,
            generators={row_idx: torch.Generator().manual_seed(100 + row_idx) for row_idx in range(6)},
            output_token_ids=[list(h) for h in histories],
            temperature=temperature.clone(),
            top_k=torch.tensor([10, 0, 100, 100, 100, 20]),
            top_p=torch.tensor([0.9, 1.0, 0.8, 0.8, 0.8, 0.85]),
        )
        return _stage0_model(states, max_chars=6), states, md

    return steps, build


def _run_two_steps(model, md, steps):
    tokens = []
    for logits in steps:
        out = model._sample_minicpmo45_native_duplex_stage0(logits.clone(), md, duplex_rows=list(range(6)))
        step_tokens = out.sampled_token_ids.squeeze(-1).tolist()
        tokens.append(step_tokens)
        model._commit_minicpmo45_duplex_pending_samples()
        for row_idx, token in enumerate(step_tokens):
            md.output_token_ids[row_idx].append(token)
    return tokens


def _snapshot(states, md):
    return (
        {row: copy.deepcopy(vars(state)) for row, state in states.items()},
        {row: gen.get_state().clone() for row, gen in md.generators.items()},
    )


@pytest.mark.parametrize("all_greedy", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_deferred_stage0_matches_the_synchronous_path(seed, all_greedy, monkeypatch):
    steps, build = _stage0_scenario(seed, all_greedy=all_greedy)

    sync_model, sync_states, sync_md = build()
    monkeypatch.setattr(sync_model, "_sample_minicpmo45_native_duplex_rows_deferred", lambda *a, **k: None)
    expected = _run_two_steps(sync_model, sync_md, steps)

    model, states, md = build()
    calls = {"deferred": 0}
    real = model._sample_minicpmo45_native_duplex_rows_deferred

    def counting(*args, **kwargs):
        out = real(*args, **kwargs)
        calls["deferred"] += out is not None
        return out

    monkeypatch.setattr(model, "_sample_minicpmo45_native_duplex_rows_deferred", counting)
    actual = _run_two_steps(model, md, steps)

    assert calls["deferred"] == 2
    assert actual == expected
    got_states, got_gens = _snapshot(states, md)
    want_states, want_gens = _snapshot(sync_states, sync_md)
    assert got_states == want_states
    for row in want_gens:
        assert torch.equal(got_gens[row], want_gens[row]), f"generator {row} advanced differently"


def test_deferred_stage0_cuts_and_rewrites_like_the_host():
    """Row 1's listen becomes tts_bos (its turn is open); a long chunk is cut to chunk_eos."""
    steps, build = _stage0_scenario(0, all_greedy=True)
    model, states, md = build()
    md.output_token_ids[0] = [14, 14]  # 3 + 3 chars already in the chunk: any text token reaches 6
    logits = steps[0].clone()
    logits[0, 20] += 40.0  # row 0 samples a text token
    out = model._sample_minicpmo45_native_duplex_stage0(logits, md, duplex_rows=list(range(6)))
    tokens = out.sampled_token_ids.squeeze(-1).tolist()
    assert tokens[1] == TOKEN_IDS["tts_bos_token_id"]
    assert tokens[0] == TOKEN_IDS["chunk_eos_token_id"]
    assert tokens[3] == TOKEN_IDS["chunk_eos_token_id"]


def test_deferred_stage0_defers_the_host_state_until_commit():
    steps, build = _stage0_scenario(1)
    model, states, md = build()
    before = _snapshot(states, md)[0]
    model._sample_minicpmo45_native_duplex_stage0(steps[0].clone(), md, duplex_rows=list(range(6)))
    assert _snapshot(states, md)[0] == before
    # An append of an unrelated session does not wait for the step.
    model._minicpmo45_duplex_row_sessions = {row: f"s{row}" for row in range(6)}
    model._minicpmo45_duplex_pending_samples.row_sessions = {row: f"s{row}" for row in range(6)}
    model._commit_minicpmo45_duplex_pending_samples(session_ids={"other"})
    assert model._minicpmo45_duplex_pending_samples is not None
    model._commit_minicpmo45_duplex_pending_samples(session_ids={"s2"})
    assert model._minicpmo45_duplex_pending_samples is None
    assert _snapshot(states, md)[0] != before


def test_deferred_stage0_falls_back_for_the_process_generator():
    steps, build = _stage0_scenario(2)
    model, states, md = build()
    del md.generators[0]  # row 0 samples from the process-wide generator
    assert (
        model._sample_minicpmo45_native_duplex_rows_deferred(
            steps[0].clone(),
            md,
            row_idxs=list(range(6)),
            token_ids=TOKEN_IDS,
            row_params=model._minicpmo45_duplex_row_params(md, 6),
        )
        is None
    )


def test_deferred_stage0_falls_back_inside_a_character():
    steps, build = _stage0_scenario(3)
    model, states, md = build()
    tokenizer = model._minicpmo45_tokenizer_cache
    original = tokenizer.decode
    tokenizer.decode = lambda ids, skip_special_tokens=False: original(ids, skip_special_tokens) + "�"
    assert (
        model._sample_minicpmo45_native_duplex_rows_deferred(
            steps[0].clone(),
            md,
            row_idxs=list(range(6)),
            token_ids=TOKEN_IDS,
            row_params=model._minicpmo45_duplex_row_params(md, 6),
        )
        is None
    )


def test_row_params_prefer_the_host_copies(monkeypatch):
    model = Model.__new__(Model)
    md = SimpleNamespace(temperature=torch.tensor([0.5, 0.9]), top_k=None, top_p=torch.tensor([0.7, 0.8]))
    model._minicpmo45_duplex_row_sampling_host = {0: (0.5, 3, 0.7), 1: (0.9, 4, 0.8)}
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self: pytest.fail("device read"))
    # top_k has no tensor: the default, as the device read would give.
    assert model._minicpmo45_duplex_row_params(md, 2) == [(0.5, 100, 0.7), (0.9, 100, 0.8)]
