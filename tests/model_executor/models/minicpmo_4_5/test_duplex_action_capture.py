# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Action capture from the MiniCPM-o 4.5 duplex stage-0 sampler.

The parity test recomputes every recorded log-probability independently in
float32 (raw-softmax Bernoulli for stage 1, forbidden-masked logits at the
row's temperature for stage 2) and must match within tolerance on all sampled
tokens; a second test pins that truncation and the repetition penalty steer
the draw but not the record. Taxonomy, seeded replay, cap-without-draw,
force-listen-without-draw, same-step publication, the generator device and
the per-unit capture seq are pinned beside it.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.model_executor.models.minicpmo_4_5.test_native_duplex_sampling import (
    CHUNK_EOS,
    LISTEN,
    TTS_PAD,
    _setup,
)
from vllm_omni.model_executor.duplex_sampling import DuplexSamplingRow
from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni
from vllm_omni.protocol.duplex.action_records import decode_unit_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

Model = minicpmo_4_5_omni.MiniCPMO45OmniForConditionalGeneration


def _capture_row(
    row: int, *, seq: int | None = None, epoch: int = 0, payload: dict[str, object] | None = None
) -> DuplexSamplingRow:
    """A capture-on row as ``DuplexSamplingHelper.rows`` builds it from the request envelope."""
    return DuplexSamplingRow(
        row_idx=row,
        request_id=f"req-{row}",
        session_id="sess-capture",
        seq=100 + row if seq is None else seq,
        payload=payload,
        max_tokens=None,
        epoch=epoch,
        action_capture={"enabled": True, "seed": 11, "response_index": 0},
    )


def _enable_capture(model, *, seed: int | None = 11, response_index: int = 0, rows: range = range(6)) -> None:
    """Publish the capture row context exactly as ``prepare_duplex_sampling`` would."""
    model._minicpmo45_duplex_row_capture = {
        row: {"enabled": True, "seed": seed, "response_index": response_index} for row in rows
    }
    model._minicpmo45_duplex_row_request_ids = {row: f"req-{row}" for row in rows}
    model._minicpmo45_duplex_row_identity = {row: ("sess-capture", 0, 100 + row) for row in rows}
    model._minicpmo45_duplex_capture_generators = model._minicpmo45_duplex_make_capture_generators(
        tuple(_capture_row(row) for row in rows), device=torch.device("cpu")
    )


def _take_payloads(model) -> dict[str, Any]:
    stash = getattr(model, "_minicpmo45_duplex_capture_step", None) or {}
    model._minicpmo45_duplex_capture_step = None
    return stash


def _published(model) -> dict[str, Any]:
    """The payloads queued for this step's publish (the MRv2 ``sample_step`` reads this stash).

    Capture rows force the synchronous path, so the payloads are complete the
    moment the sampler returns; ``MiniCPMO45DuplexSampler.sample_step`` puts
    them on the reserved key (pinned by the MRv2 test beside this file).
    """
    model._commit_minicpmo45_duplex_pending_samples()
    stash = getattr(model, "_minicpmo45_duplex_capture_step", None)
    assert isinstance(stash, dict) and stash, "no capture payloads published"
    model._minicpmo45_duplex_capture_step = None
    return dict(stash)


def _stage2_reference_logprob(logits_row: torch.Tensor, temperature: float, token: int) -> float:
    """Independent recompute in float32: ``log_softmax(forbidden-masked logits / T)[token]``."""
    masked = logits_row.clone()
    masked[TTS_PAD] = float("-inf")
    masked[CHUNK_EOS] = float("-inf")
    return float(torch.log_softmax(masked.float() / temperature, dim=-1)[token])


@pytest.mark.parametrize("seed", range(3))
def test_capture_logprobs_match_independent_recomputation(seed: int) -> None:
    model, states, md, steps = _setup(seed)
    _enable_capture(model)
    out = model._sample_minicpmo45_native_duplex_stage0(steps[0].clone(), md, duplex_rows=list(range(6)))
    tokens = out.sampled_token_ids.squeeze(-1).tolist()
    entries = _published(model)

    row_params = list(
        zip(
            md.temperature.tolist(),
            md.top_k.tolist(),
            md.top_p.tolist(),
            strict=True,
        )
    )
    for row in range(6):
        payload = decode_unit_payload(entries[f"req-{row}"])
        token = tokens[row]
        assert payload["token_ids"] == [token]
        assert payload["session_id"] == "sess-capture"
        assert payload["seq"] == 100 + row
        assert payload["effective_seed"] is not None
        kind, reason = payload["action_kinds"][0], payload["reason_codes"][0] or None
        logprob = payload["logprobs"][0]
        extension = payload["model"]["minicpmo_duplex"]
        temperature = float(row_params[row][0])
        p0 = float(torch.softmax(steps[0][row], dim=-1)[CHUNK_EOS])
        if reason in ("chunk_token_cap", "chunk_char_cap", "request_token_cap"):
            # A budget cap resolved before any draw: forced, no logprob.
            assert kind == "forced" and logprob is None
            continue
        if row == 1:
            # Sampled listen rewritten to tts_bos mid-turn: forced emission of
            # the policy's own draw, recorded as that draw.
            assert kind == "forced" and reason == "mid_turn_listen_rewrite"
            assert extension["sampled_token"] == LISTEN
            token = LISTEN
        else:
            assert kind == "sampled", (row, kind, reason)
        if token == CHUNK_EOS:
            # Boundary hit: the exact Bernoulli parameter on the raw softmax.
            assert math.isclose(logprob, math.log(p0), rel_tol=1e-4, abs_tol=1e-5)
            continue
        # Stage 2: the pre-truncation measure; a greedy row's argmax takes temperature 1.
        stage_temperature = 1.0 if temperature <= 0 else temperature
        joint = math.log(max(1.0 - p0, torch.finfo(torch.float32).tiny)) + _stage2_reference_logprob(
            steps[0][row], stage_temperature, token
        )
        assert math.isclose(logprob, joint, rel_tol=1e-3, abs_tol=1e-4), (row, logprob, joint)


def test_capture_logprobs_ignore_truncation_and_repetition_penalty() -> None:
    """Top-k/top-p and the repetition history steer the draw, not the recorded measure."""
    runs = []
    for top_k, top_p, history in (
        (None, None, {}),
        (3, 0.5, {}),
        (0, 1.0, {}),
        # The drawn tokens themselves in the penalty window.
        (None, None, {0: [23, 22, 21], 4: [22, 21, 20]}),
    ):
        model, states, md, steps = _setup(0)
        _enable_capture(model)
        if top_k is not None:
            md.top_k, md.top_p = torch.full((6,), top_k), torch.full((6,), top_p)
        for row, tokens in history.items():
            states[row].generated_tokens = list(tokens)
        out = model._sample_minicpmo45_native_duplex_stage0(steps[0].clone(), md, duplex_rows=list(range(6)))
        runs.append(
            (
                out.sampled_token_ids.squeeze(-1).tolist(),
                {k: decode_unit_payload(v) for k, v in _published(model).items()},
            )
        )
    tokens, baseline = runs[0]
    drawn = [row for row in range(6) if baseline[f"req-{row}"]["logprobs"][0] is not None]
    assert drawn, "no row recorded a draw"
    for varied_tokens, entries in runs[1:]:
        assert varied_tokens == tokens, "a varied input changed the draw; the record comparison needs the same token"
        for row in drawn:
            record, reference = entries[f"req-{row}"]["logprobs"][0], baseline[f"req-{row}"]["logprobs"][0]
            assert math.isclose(record, reference, abs_tol=1e-6), (row, record, reference)


def test_capture_records_the_effective_sampling_params() -> None:
    model, states, md, steps = _setup(0)
    _enable_capture(model)
    model._sample_minicpmo45_native_duplex_stage0(steps[0].clone(), md, duplex_rows=list(range(6)))
    entries = _published(model)
    payload = decode_unit_payload(entries["req-0"])
    assert set(payload["streams"][0]) >= {"name", "stage", "modality"}
    assert payload["sampling_params"] == pytest.approx({"temperature": 0.8, "top_k": 10, "top_p": 0.9})


def test_capture_off_publishes_nothing() -> None:
    model, states, md, steps = _setup(1)
    # Capture off: the publish stash stays empty; sample_step passes through.
    model._sample_minicpmo45_native_duplex_stage0(steps[0].clone(), md, duplex_rows=list(range(6)))
    assert getattr(model, "_minicpmo45_duplex_capture_step", None) is None


def test_seeded_capture_replays_identical_tokens_and_logprobs() -> None:
    runs = []
    for _ in range(2):
        model, states, md, steps = _setup(2)
        _enable_capture(model, seed=99, response_index=3)
        out = model._sample_minicpmo45_native_duplex_stage0(steps[0].clone(), md, duplex_rows=list(range(6)))
        entries = _published(model)
        runs.append(
            (out.sampled_token_ids.squeeze(-1).tolist(), {k: decode_unit_payload(v) for k, v in entries.items()})
        )
    assert runs[0][0] == runs[1][0]
    assert {k: p["logprobs"] for k, p in runs[0][1].items()} == {k: p["logprobs"] for k, p in runs[1][1].items()}
    # The derived effective seed is recorded and differs per response index.
    seeds = {p["effective_seed"] for p in runs[0][1].values()}
    assert len(seeds) == 1
    model2, _, md2, steps2 = _setup(2)
    _enable_capture(model2, seed=99, response_index=4)
    model2._sample_minicpmo45_native_duplex_stage0(steps2[0].clone(), md2, duplex_rows=list(range(6)))
    entries2 = {k: decode_unit_payload(v) for k, v in _published(model2).items()}
    assert next(iter(seeds)) != next(iter({p["effective_seed"] for p in entries2.values()}))


def test_terminator_reemit_records_no_draw() -> None:
    model, states, md, steps = _setup(3)
    _enable_capture(model)
    # Row 0's last accepted token is a chunk terminator: the lookahead
    # re-emits it without consuming RNG and the record says so.
    md.output_token_ids[0].append(CHUNK_EOS)
    out = model._sample_minicpmo45_native_duplex_stage0(steps[0].clone(), md, duplex_rows=list(range(6)))
    assert out.sampled_token_ids.squeeze(-1).tolist()[0] == CHUNK_EOS
    payload = decode_unit_payload(_published(model)["req-0"])
    assert payload["action_kinds"] == ["reemit"]
    assert payload["logprobs"] == [None]
    assert payload["terminators"] == [{"token_id": CHUNK_EOS, "kind": "chunk_eos"}]


def test_capture_rows_force_the_synchronous_path() -> None:
    """records arrive per unit at unit time.

    The deferred step commits host state one sampling step later, so a
    deferred capture payload would ride the next unit's delivery and the
    response-terminal unit's record would seal into the wrong scope (or
    surface as a missing payload). Capture rows therefore never defer, even
    with per-row generators installed.
    """
    model, states, md, steps = _setup(4)
    _enable_capture(model, seed=7, response_index=1)
    out = model._sample_minicpmo45_native_duplex_stage0(steps[0].clone(), md, duplex_rows=list(range(6)))
    assert getattr(model, "_minicpmo45_duplex_pending_samples", None) is None
    tokens = out.sampled_token_ids.squeeze(-1).tolist()
    entries = {k: decode_unit_payload(v) for k, v in _published(model).items()}
    # Same-step publication: every capture row's payload is already stashed.
    assert set(entries) == {f"req-{row}" for row in range(6)}
    for request_id, payload in entries.items():
        if payload["action_kinds"][0] == "sampled":
            assert payload["logprobs"][0] is not None and math.isfinite(payload["logprobs"][0])
        else:
            assert payload["reason_codes"][0]
    assert tokens


_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
]


@pytest.mark.parametrize("device", _DEVICES)
def test_capture_generators_live_on_the_logits_device(device: str) -> None:
    """The boundary and stage-2 draws run on the logits device with these generators.

    A CPU generator (the former default: no device was ever sourced) makes the
    first ``torch.rand(..., device=logits.device)`` raise on an accelerator."""
    model, _, _, _ = _setup(0)
    model._minicpmo45_duplex_row_capture = {row: {"enabled": True, "seed": 11, "response_index": 0} for row in range(6)}
    logits = torch.empty(0, device=device)
    generators = model._minicpmo45_duplex_make_capture_generators(
        tuple(_capture_row(row) for row in range(6)), device=logits.device
    )
    assert set(generators) == set(range(6))
    assert all(generator.device == logits.device for _, generator in generators.values())
    # Reusing the derived generator of a (session, epoch, response) keeps its device.
    again = model._minicpmo45_duplex_make_capture_generators(
        tuple(_capture_row(row) for row in range(6)), device=logits.device
    )
    assert all(again[row][1] is generators[row][1] for row in range(6))


@pytest.mark.parametrize("device", _DEVICES)
def test_capture_generators_fall_back_to_the_module_parameter_device(device: str) -> None:
    """Without a tensor at hand, the module's parameter device is the draw device."""
    model, _, _, _ = _setup(0)
    torch.nn.Module.__init__(model)
    model.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1, device=device)))
    model.register_buffer("spare", torch.zeros(1, dtype=torch.bool, device=device))
    model._minicpmo45_duplex_row_capture = {0: {"enabled": True, "seed": 11, "response_index": 0}}
    generators = model._minicpmo45_duplex_make_capture_generators((_capture_row(0),))
    assert generators[0][1].device == model._parameters["anchor"].device


def test_capture_seq_advances_per_unit_within_a_session_epoch() -> None:
    """The envelope seq is frozen at request creation; the capture seq is not.

    Every unit of one (session, epoch) repeats the frozen envelope seq, and the
    ledger rejects the second (``seq N is not after N`` per scope). The capture
    seq seeds at the envelope seq (the ledger joins a unit to its input
    annotation by it), advances per stashed unit, lifts to a later request's
    envelope seq when that outran the counter, and restarts with a new epoch:
    ledger scopes never span epochs.
    """
    model, _, md, steps = _setup(0)
    rows = (_capture_row(0, seq=7), *(replace(_capture_row(row), action_capture=None) for row in range(1, 6)))
    seqs = []
    for logits in steps:
        model.prepare_duplex_sampling(logits, md, rows)
        model._sample_minicpmo45_native_duplex_stage0(logits.clone(), md, duplex_rows=list(range(6)))
        seqs.append(decode_unit_payload(_published(model)["req-0"])["seq"])
    assert model._minicpmo45_duplex_row_identity[0] == ("sess-capture", 0, 7)
    assert seqs[0] == 7
    assert seqs[1] == seqs[0] + 1

    def sample_req0(epoch: int, seq: int | None) -> int:
        step = tuple(replace(rows[row], epoch=epoch, seq=seq) if row == 0 else rows[row] for row in range(6))
        model.prepare_duplex_sampling(steps[0], md, step)
        model._sample_minicpmo45_native_duplex_stage0(steps[0].clone(), md, duplex_rows=list(range(6)))
        return decode_unit_payload(_published(model)["req-0"])["seq"]

    assert sample_req0(epoch=0, seq=7) == seqs[1] + 1  # same request: the counter resumes
    assert sample_req0(epoch=0, seq=20) == 20  # a later request whose envelope seq outran the counter
    assert sample_req0(epoch=0, seq=7) == 21  # ... and the counter stays above it
    assert sample_req0(epoch=1, seq=3) == 3  # a new epoch restarts at its own envelope seq


def test_finished_session_frees_its_capture_seq_counter() -> None:
    model, _, _, _ = _setup(0)
    model.model = SimpleNamespace()
    model._minicpmo45_duplex_capture_seq = {"sid-a": {0: 4}, "sid-b": {0: 9}}
    model._minicpmo45_duplex_request_sessions = {"req-a": "sid-a"}
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={"sid-a": object(), "sid-b": object()})
    model.on_requests_finished({"req-a"})
    assert model._minicpmo45_duplex_capture_seq == {"sid-b": {0: 9}}


@pytest.mark.parametrize("source", ["payload", "turn_ended"])
def test_capture_records_force_listen_as_forced_without_logprob(source: str) -> None:
    """A row the force-listen policy masked to p(listen)=1 is an override, not a decision.

    Both forcing classes pin the same way: the payload's ``force_listen`` flag
    (VAD speech) and the turn-ended latch on a silent unit. Sampled as-is, the
    pinned distribution yields ``sampled`` with logprob exactly 0 (log 1):
    indistinguishable from the policy choosing to listen. The record is
    ``forced``/``force_listen`` with the protocol's no-draw logprob slot. Only
    the record changes: the draws still run, so the generator advances as
    before (the trainer's replay mirrors that sequence) and the other rows'
    records are those of the same step without the mask.
    """
    forced = {0, 4}  # a sampled row and a greedy one: the mask, not the argmax, decides
    records: dict[bool, tuple[list[int], dict[str, dict[str, Any]]]] = {}
    pinned_rows: dict[bool, set[int]] = {}
    for pinned in (False, True):
        model, _, md, steps = _setup(0)
        if source == "turn_ended":
            # The latch forces a silent unit to listen; pending speech releases it.
            model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(
                sessions={"sess-capture": SimpleNamespace(current_turn_ended=True, pending_speech_context=not pinned)}
            )
            payload: dict[str, object] | None = {"is_speech": False}
        else:
            payload = {"force_listen": pinned}
        rows = tuple(_capture_row(row, payload=payload if row in forced else None) for row in range(6))
        model._minicpmo45_duplex_payload_for_row = lambda row, rows=rows: rows[row].payload
        logits = steps[0].clone()
        model.prepare_duplex_sampling(logits, md, rows)
        pinned_rows[pinned] = getattr(model, "_minicpmo45_force_listen_pinned_rows", set())
        before = {row: gen.get_state().clone() for row, (_, gen) in model._minicpmo45_duplex_capture_generators.items()}
        out = model._sample_minicpmo45_native_duplex_stage0(logits, md, duplex_rows=list(range(6)))
        records[pinned] = (
            out.sampled_token_ids.squeeze(-1).tolist(),
            {k: decode_unit_payload(v) for k, v in _published(model).items()},
        )
        if pinned:
            assert all(
                not torch.equal(model._minicpmo45_duplex_capture_generators[row][1].get_state(), before[row])
                for row in forced
            )

    tokens, entries = records[True]
    baseline_tokens, baseline = records[False]
    for row in forced:
        record = entries[f"req-{row}"]
        assert tokens[row] == LISTEN
        assert record["token_ids"] == [LISTEN]
        assert record["action_kinds"] == ["forced"]
        assert record["reason_codes"] == ["force_listen"]
        assert record["logprobs"] == [None]
        assert "sampled_token" not in record["model"]["minicpmo_duplex"]
        assert record["terminators"] == [{"token_id": LISTEN, "kind": "listen"}]
    for row in set(range(6)) - forced:
        assert tokens[row] == baseline_tokens[row]
        assert entries[f"req-{row}"] == baseline[f"req-{row}"]
    # The mask is per step: a step without it pins nothing.
    assert pinned_rows == {False: set(), True: forced}
