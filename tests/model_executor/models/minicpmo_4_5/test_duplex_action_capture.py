# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Action capture from the MiniCPM-o 4.5 duplex stage-0 sampler.

The parity test recomputes every recorded log-probability independently from
the recorded distribution inputs (raw-softmax Bernoulli, forbidden mask,
repetition penalty, top-k/top-p renormalized stage 2) and must match within
tolerance on all sampled tokens. Taxonomy, seeded replay, cap-without-draw,
and same-step publication are pinned beside it.
"""

from __future__ import annotations

import math
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
from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
from vllm_omni.protocol.duplex.action_records import decode_unit_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

Model = minicpmo_4_5_omni.MiniCPMO45OmniForConditionalGeneration


def _enable_capture(model, *, seed: int | None = 11, response_index: int = 0, rows: range = range(6)) -> None:
    """Publish the capture row context exactly as ``prepare_duplex_sampling`` would."""
    model._minicpmo45_duplex_row_capture = {
        row: {"enabled": True, "seed": seed, "response_index": response_index} for row in rows
    }
    model._minicpmo45_duplex_row_request_ids = {row: f"req-{row}" for row in rows}
    model._minicpmo45_duplex_row_identity = {row: ("sess-capture", 0, 100 + row) for row in rows}
    model._minicpmo45_duplex_capture_generators = model._minicpmo45_duplex_make_capture_generators(
        tuple(
            SimpleNamespace(
                row_idx=row,
                request_id=f"req-{row}",
                session_id="sess-capture",
                seq=100 + row,
                payload=None,
                max_tokens=None,
                epoch=0,
                action_capture={"enabled": True, "seed": seed, "response_index": response_index},
            )
            for row in rows
        )
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


def _stage2_reference_logprob(
    logits_row: torch.Tensor,
    recent: list[int],
    generated: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    token: int,
) -> float:
    """Independent recompute: masked, penalized, truncated, renormalized logprob of ``token``."""
    row_logits = logits_row.clone()
    row_logits[TTS_PAD] = float("-inf")
    row_logits[CHUNK_EOS] = float("-inf")
    history = list(dict.fromkeys((generated or recent)[-MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE :]))
    for token_id in history:
        row_logits[token_id] /= 1.05
    scaled = row_logits if temperature <= 0 else row_logits / temperature
    dense = torch.softmax(Model._top_k_top_p_filter(scaled.view(1, -1), top_k=top_k, top_p=top_p), dim=-1)[0]
    return float(torch.log(dense[token].clamp_min(torch.finfo(torch.float32).tiny)))


@pytest.mark.parametrize("seed", range(3))
def test_capture_logprobs_match_independent_recomputation(seed: int) -> None:
    model, states, md, steps = _setup(seed)
    _enable_capture(model)
    # The parity reference must see the pre-commit repetition history.
    pre_recent = {row: list(md.output_token_ids[row]) for row in range(6)}
    pre_generated = {row: list(states[row].generated_tokens) for row in range(6)}
    logits = steps[0].clone()
    out = model._sample_minicpmo45_native_duplex_stage0(logits.clone(), md, duplex_rows=list(range(6)))
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
        temperature, top_k, top_p = row_params[row]
        p0 = float(torch.softmax(steps[0][row], dim=-1)[CHUNK_EOS])
        if reason in ("chunk_token_cap", "chunk_char_cap", "request_token_cap"):
            # A budget cap resolved before any draw: forced, no logprob.
            assert kind == "forced" and logprob is None
            continue
        if row == 1:
            # Sampled listen rewritten to tts_bos mid-turn: forced emission of
            # the policy's own draw.
            assert kind == "forced" and reason == "mid_turn_listen_rewrite"
            assert extension["sampled_token"] == LISTEN
            token = LISTEN
        else:
            assert kind == "sampled", (row, kind, reason)
        if token == CHUNK_EOS:
            # Boundary hit: the exact Bernoulli parameter on the raw softmax.
            assert math.isclose(logprob, math.log(p0), rel_tol=1e-4, abs_tol=1e-5)
            assert math.isclose(extension["stage0_prob_chunk_eos"], p0, rel_tol=1e-4)
            continue
        recent = pre_recent[row]
        generated = pre_generated[row]
        reference = _stage2_reference_logprob(
            steps[0][row], recent, generated, float(temperature), int(top_k), float(top_p), token
        )
        joint = math.log(max(1.0 - p0, torch.finfo(torch.float32).tiny)) + reference
        assert math.isclose(logprob, joint, rel_tol=1e-3, abs_tol=1e-4), (row, logprob, joint)
        assert math.isclose(extension["stage2_logprob"], reference, rel_tol=1e-3, abs_tol=1e-4)
        # Every non-SamplingParams distribution input rides the extension.
        assert extension["repetition_penalty"] == 1.05
        assert extension["repetition_history_tokens"] == 512
        assert extension["forbidden_tokens"]


def test_capture_records_the_non_sampling_params_distribution_inputs() -> None:
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
