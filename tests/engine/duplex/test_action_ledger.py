# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the duplex action-capture ledger and wire payload."""

from __future__ import annotations

import pytest

from vllm_omni.engine.duplex.session.action_ledger import (
    RETAINED_SEALED_SETS,
    ActionLedgerError,
    SessionActionLedger,
)
from vllm_omni.protocol.duplex.action_records import (
    ACTION_CAPTURE_WIRE_VERSION,
    ActionPayloadError,
    DuplexActionRecordSet,
    build_unit_payload,
    decode_unit_payload,
    encode_unit_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _payload(
    *,
    seq: int,
    unit_index: int | None = None,
    session_id: str = "s1",
    stage_id: int = 0,
    epoch: int = 0,
    token_ids: tuple[int, ...] = (11,),
    logprobs: tuple[float | None, ...] = (-0.25,),
    kinds: tuple[str, ...] = ("sampled",),
    reasons: tuple[str | None, ...] | None = (None,),
    streams: tuple[dict, ...] | None = ({"name": "content", "stage": 0, "modality": "audio"},),
    terminators: tuple[dict, ...] | None = None,
    seed: int | None = 42,
    response_index: int | None = None,
    model: dict | None = None,
) -> bytes:
    return encode_unit_payload(
        build_unit_payload(
            session_id=session_id,
            stage_id=stage_id,
            unit_index=seq if unit_index is None else unit_index,
            seq=seq,
            epoch=epoch,
            token_ids=list(token_ids),
            logprobs=list(logprobs),
            action_kinds=list(kinds),
            reason_codes=list(reasons) if reasons is not None else None,
            streams=list(streams) if streams is not None else None,
            terminators=list(terminators) if terminators is not None else None,
            sampling_params={"temperature": 0.7, "top_p": 0.8, "top_k": 20},
            effective_seed=seed,
            response_index=response_index,
            model=model if model is not None else {"rep_penalty": 1.05},
        )
    )


def _ingest_r1(ledger: SessionActionLedger, payload: bytes) -> int:
    """Ingest into the ``r1`` scope of epoch 0 / turn 1, as the model channel does."""
    return ledger.ingest(payload, epoch=0, turn_id=1, response_id="r1", stage_id=0, input_modality="")


# ---- wire payload ----


def test_payload_roundtrip_and_strict_top_level_keys() -> None:
    data = _payload(seq=1)
    decoded = decode_unit_payload(data)
    assert decoded["schema_version"] == ACTION_CAPTURE_WIRE_VERSION
    assert decoded["token_ids"] == [11]
    assert decoded["unit_index"] == 1  # one model unit per payload unless a test stamps another

    bad = dict(decode_unit_payload(data))
    bad["mystery_field"] = 1
    with pytest.raises(ActionPayloadError, match="unknown action capture fields"):
        decode_unit_payload(bad)


def test_payload_rejects_unknown_schema_version_and_bad_columns() -> None:
    bad = dict(decode_unit_payload(_payload(seq=1)))
    bad["schema_version"] = 99
    with pytest.raises(ActionPayloadError, match="schema_version"):
        decode_unit_payload(bad)

    bad = dict(decode_unit_payload(_payload(seq=1)))
    bad["action_kinds"] = ["sampled", "sampled"]
    with pytest.raises(ActionPayloadError, match="parallel"):
        decode_unit_payload(bad)

    bad = dict(decode_unit_payload(_payload(seq=1)))
    bad["logprobs"] = [float("nan")]
    with pytest.raises(ActionPayloadError, match="finite"):
        decode_unit_payload(bad)


def test_payload_rejects_invalid_action_kind_and_namespaced_pass_through() -> None:
    bad = dict(decode_unit_payload(_payload(seq=1)))
    bad["action_kinds"] = ["telepathic"]
    with pytest.raises(ActionPayloadError, match="action kind"):
        decode_unit_payload(bad)

    good = dict(decode_unit_payload(_payload(seq=1)))
    good["model"] = {"minicpmo": {"stage1_prob": 0.2, "unknown_future_key": [1, 2]}}
    decoded = decode_unit_payload(good)
    assert decoded["model"]["minicpmo"]["unknown_future_key"] == [1, 2]


# ---- ledger semantics ----


def test_ingest_orders_units_and_derives_records() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_listening(epoch=0, turn_id=1)
    ledger.note_input_unit(0, 1, modality="audio", origin="client", commit_ns=101)
    ledger.note_input_unit(0, 2, modality="silence", origin="engine")
    ledger.ingest(_payload(seq=1), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")
    ledger.ingest(_payload(seq=2), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")
    # A new response beginning seals the listening period as completed.
    ledger.bind_response("r1", epoch=0, turn_id=2)

    [set_] = ledger.take_terminal_sets()
    assert set_.status == "completed"
    assert set_.num_units() == 2
    assert set_.input_origins == ("client", "engine")
    assert set_.input_modalities[1] == "silence"
    records = list(set_.records())
    assert [record.unit_index for record in records] == [0, 1]
    assert records[0].input_chunk.origin == "client"
    assert records[0].actions[0].token_id == 11
    assert records[0].effective_seed == 42
    assert records[0].model == {"rep_penalty": 1.05}


def test_reordering_and_stale_seq_are_session_fatal() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_listening(epoch=0, turn_id=1)
    ledger.ingest(_payload(seq=5), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")
    with pytest.raises(ActionLedgerError, match="not after"):
        ledger.ingest(_payload(seq=5), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")
    with pytest.raises(ActionLedgerError, match="not after"):
        ledger.ingest(_payload(seq=4), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")


def test_decode_steps_of_one_unit_fold_into_a_single_record() -> None:
    """A model unit arrives as one payload per decode step, all stamped with the unit's counter."""
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_response("r1", epoch=0, turn_id=1)
    ledger.note_input_unit(0, 1, modality="audio", origin="engine", commit_ns=101)
    boundary = {"token_id": 5, "kind": "chunk_eos"}
    restated = {"token_id": 5, "kind": "chunk_eos"}
    steps = (
        _payload(seq=1, unit_index=4),
        _payload(seq=2, unit_index=4, token_ids=(12,), logprobs=(-0.5,)),
        _payload(seq=3, unit_index=4, token_ids=(5,), logprobs=(-0.75,), terminators=(boundary,)),
        # The next unit's open restates the terminator under the closing unit's counter.
        _payload(
            seq=4,
            unit_index=4,
            token_ids=(5,),
            logprobs=(None,),
            kinds=("reemit",),
            reasons=("terminator_reemit",),
            terminators=(restated,),
        ),
    )
    # Each step joins unit 0 of the scope; the unit's row stays its first step's.
    assert [_ingest_r1(ledger, step) for step in steps] == [0, 0, 0, 0]
    ledger.seal_active_response("r1")
    [set_] = ledger.take_terminal_sets()
    assert set_.num_units() == 1
    assert set_.token_ids == (11, 12, 5, 5)
    assert set_.unit_offsets == (0, 4)
    assert set_.logprobs == (-0.25, -0.5, -0.75, None)
    assert set_.action_kinds == ("sampled", "sampled", "sampled", "reemit")
    assert set_.terminators == ((boundary, restated),)
    # The unannotated later steps leave the unit its first step's input annotation.
    assert (set_.input_origins, set_.input_commit_ns) == (("engine",), (101,))
    [record] = set_.records()
    assert [action.action_kind for action in record.actions] == ["sampled"] * 3 + ["reemit"]
    assert [action.token_id for action in record.actions] == [11, 12, 5, 5]


def test_a_new_unit_counter_opens_a_record_and_a_behind_one_is_fatal() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_response("r1", epoch=0, turn_id=1)
    assert _ingest_r1(ledger, _payload(seq=1, unit_index=7)) == 0
    assert _ingest_r1(ledger, _payload(seq=2, unit_index=7)) == 0
    assert _ingest_r1(ledger, _payload(seq=3, unit_index=8, token_ids=(13,), logprobs=(-0.5,))) == 1
    with pytest.raises(ActionLedgerError, match="unit_index 7 is behind the scope's unit 8") as exc_info:
        _ingest_r1(ledger, _payload(seq=4, unit_index=7))
    assert exc_info.value.code == "action_capture_invalid"
    ledger.seal_active_response("r1")
    [set_] = ledger.take_terminal_sets()
    assert set_.num_units() == 2
    assert set_.unit_offsets == (0, 2, 3)
    assert set_.token_ids == (11, 11, 13)


def test_epoch_mismatch_and_stage_mismatch_are_invalid() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_listening(epoch=0, turn_id=1)
    with pytest.raises(ActionLedgerError, match="epoch"):
        ledger.ingest(_payload(seq=1, epoch=3), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")
    with pytest.raises(ActionLedgerError, match="stage"):
        ledger.ingest(_payload(seq=1, stage_id=2), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")


def test_response_binding_promotes_listening_scope() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_listening(epoch=0, turn_id=1)
    ledger.ingest(_payload(seq=1), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")
    # The response binding trails the speaking unit; the listening scope is
    # promoted in place so ordering holds.
    ledger.ingest(_payload(seq=2), epoch=0, turn_id=1, response_id="r1", stage_id=0, input_modality="")
    ledger.bind_response("r1", epoch=0, turn_id=1)

    # Promoted in place: the scope keeps its units and now answers to r1.
    assert ledger.open_scope_response_id == "r1"
    ledger.seal_active_response("r1")
    [spoken] = ledger.take_terminal_sets()
    assert spoken.response_id == "r1"
    assert spoken.num_units() == 2


def test_bind_response_seals_a_pure_listening_scope() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_listening(epoch=0, turn_id=1)
    ledger.ingest(_payload(seq=1), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")
    # No promotion ever happened (the response's first payload has not
    # arrived); the engine opening the response seals the listening period.
    ledger.bind_response("r1", epoch=0, turn_id=2)
    [listening] = ledger.take_terminal_sets()
    assert listening.response_id is None
    assert listening.status == "completed"
    assert listening.num_units() == 1


def test_retention_window_is_bounded_and_overflow_is_loud() -> None:
    assert RETAINED_SEALED_SETS == 1
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_response("r1", epoch=0, turn_id=1)
    ledger.ingest(_payload(seq=1), epoch=0, turn_id=1, response_id="r1", stage_id=0, input_modality="")
    ledger.seal_active_response("r1")  # sealed slot now holds r1
    ledger.bind_response("r2", epoch=0, turn_id=2)
    ledger.ingest(_payload(seq=2), epoch=0, turn_id=2, response_id="r2", stage_id=0, input_modality="")
    with pytest.raises(ActionLedgerError, match="not spilled") as exc_info:
        ledger.seal_active_response("r2")  # overflow: r1 uncollected
    assert exc_info.value.code == "action_capture_retention_overflow"
    # Spilling the oldest sealed set clears the window; the next seal succeeds.
    assert ledger.pop_oldest_sealed() is not None
    ledger.seal_active_response("r2")


def test_seal_statuses_for_close_interruption_and_cancel() -> None:
    for seal, expected in (
        (lambda ledger: ledger.seal_for_close(), "truncated_close"),
        (lambda ledger: ledger.seal_for_interruption(), "truncated_interruption"),
        (lambda ledger: ledger.seal_for_cancel(), "truncated_cancel"),
    ):
        ledger = SessionActionLedger(session_id="s1")
        ledger.bind_response("r1", epoch=0, turn_id=1)
        ledger.ingest(_payload(seq=1), epoch=0, turn_id=1, response_id="r1", stage_id=0, input_modality="")
        seal(ledger)
        set_ = ledger.take_terminal_sets()[0]
        assert set_.status == expected
        assert set_.num_units() == 1  # partial set kept, never dropped


def test_seed_contradiction_within_scope_is_invalid() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_listening(epoch=0, turn_id=1)
    ledger.ingest(_payload(seq=1, seed=7), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")
    with pytest.raises(ActionLedgerError, match="seed"):
        ledger.ingest(_payload(seq=2, seed=8), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")


def test_record_set_wire_roundtrip() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_response("r1", epoch=0, turn_id=1)
    ledger.ingest(
        _payload(
            seq=1,
            token_ids=(11, 12),
            logprobs=(-0.25, None),
            kinds=("sampled", "reemit"),
            reasons=(None, ""),
            terminators=({"token_id": 12, "kind": "chunk_eos"},),
        ),
        epoch=0,
        turn_id=1,
        response_id="r1",
        stage_id=0,
        input_modality="",
    )
    ledger.seal_active_response("r1")
    set_ = ledger.take_terminal_sets()[0]
    wire = set_.to_wire()
    restored = DuplexActionRecordSet.from_wire(wire)
    assert restored.token_ids == set_.token_ids
    assert restored.logprobs == (-0.25, None)
    assert restored.action_kinds == ("sampled", "reemit")
    assert restored.terminators == (({"token_id": 12, "kind": "chunk_eos"},),)
    assert [record.actions[0].token_id for record in restored.records()] == [11]
    assert restored.unit_offsets == set_.unit_offsets


def test_pop_oldest_sealed_spills_in_seal_order_until_the_window_has_room() -> None:
    ledger = SessionActionLedger(session_id="s1")
    assert ledger.pop_oldest_sealed() is None
    ledger.bind_listening(epoch=0, turn_id=1)
    ledger.ingest(_payload(seq=1), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")
    ledger.bind_response("r1", epoch=0, turn_id=2)  # seals the listening period: window full
    spilled = ledger.pop_oldest_sealed()
    assert (spilled.response_id, spilled.num_units()) == (None, 1)
    ledger.ingest(_payload(seq=2), epoch=0, turn_id=2, response_id="r1", stage_id=0, input_modality="")
    ledger.seal_active_response("r1")
    next_ = ledger.pop_oldest_sealed()
    assert (next_.response_id, next_.num_units()) == ("r1", 1)
    # The window is empty again; there is nothing more to spill.
    assert ledger.pop_oldest_sealed() is None


def test_in_flight_listening_unit_joins_the_sealed_scope() -> None:
    """A unit sampled under the listening seed lands there after the response bind."""
    listen_seed, response_seed = 11, 22
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_listening(epoch=0, turn_id=1)
    ledger.note_input_unit(0, 1, modality="audio", origin="client", commit_ns=5)
    ledger.ingest(
        _payload(seq=1, unit_index=3, seed=listen_seed, response_index=0),
        epoch=0,
        turn_id=1,
        response_id=None,
        stage_id=0,
        input_modality="",
    )
    ledger.bind_response("r1", epoch=0, turn_id=1)
    # The step still in flight shares the listening unit's counter, then one more unit of that draw.
    ledger.ingest(
        _payload(seq=2, unit_index=3, seed=listen_seed, response_index=0, token_ids=(12,), logprobs=(-0.5,)),
        epoch=0,
        turn_id=1,
        response_id="r1",
        stage_id=0,
        input_modality="",
    )
    ledger.ingest(
        _payload(seq=3, unit_index=4, seed=listen_seed, response_index=0, token_ids=(13,), logprobs=(-0.4,)),
        epoch=0,
        turn_id=1,
        response_id="r1",
        stage_id=0,
        input_modality="",
    )
    ledger.ingest(
        _payload(seq=4, unit_index=5, seed=response_seed, response_index=1, token_ids=(14,), logprobs=(-0.1,)),
        epoch=0,
        turn_id=1,
        response_id="r1",
        stage_id=0,
        input_modality="",
    )
    ledger.seal_for_close()
    sets = ledger.take_terminal_sets()
    listening = next(item for item in sets if item.response_id is None)
    response = next(item for item in sets if item.response_id == "r1")
    assert listening.num_units() == 2
    assert listening.effective_seed == listen_seed
    assert listening.token_ids == (11, 12, 13)
    assert listening.unit_offsets == (0, 2, 3)
    assert response.num_units() == 1
    assert response.effective_seed == response_seed
    assert response.token_ids == (14,)


def test_non_monotone_listening_straggler_does_not_enter_the_response_scope() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_listening(epoch=0, turn_id=1)
    ledger.ingest(
        _payload(seq=2, unit_index=3, seed=11, response_index=0),
        epoch=0,
        turn_id=1,
        response_id=None,
        stage_id=0,
        input_modality="",
    )
    ledger.bind_response("r1", epoch=0, turn_id=1)
    with pytest.raises(ActionLedgerError, match="not a monotone continuation"):
        ledger.ingest(
            _payload(seq=1, unit_index=3, seed=11, response_index=0, token_ids=(12,), logprobs=(-0.5,)),
            epoch=0,
            turn_id=1,
            response_id="r1",
            stage_id=0,
            input_modality="",
        )
    assert ledger.open_scope_response_id == "r1"


def test_response_seeded_payload_does_not_join_the_listening_scope() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_listening(epoch=0, turn_id=1)
    ledger.ingest(
        _payload(seq=1, unit_index=1, seed=11),
        epoch=0,
        turn_id=1,
        response_id=None,
        stage_id=0,
        input_modality="",
    )
    ledger.bind_response("r1", epoch=0, turn_id=1)
    ledger.ingest(
        _payload(seq=2, unit_index=2, seed=22, token_ids=(9,), logprobs=(-0.2,)),
        epoch=0,
        turn_id=1,
        response_id="r1",
        stage_id=0,
        input_modality="",
    )
    ledger.seal_for_close()
    sets = ledger.take_terminal_sets()
    listening = next(item for item in sets if item.response_id is None)
    response = next(item for item in sets if item.response_id == "r1")
    assert listening.num_units() == 1
    assert listening.token_ids == (11,)
    assert response.token_ids == (9,)
    assert response.effective_seed == 22


def test_unreserved_capture_seq_is_engine_origin() -> None:
    """A later unit of one append consumes no new client chunk."""
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_response("r1", epoch=0, turn_id=1)
    ledger.note_input_unit(0, 1, modality="audio", origin="client", commit_ns=9)
    _ingest_r1(ledger, _payload(seq=1, unit_index=1))
    _ingest_r1(ledger, _payload(seq=2, unit_index=2, token_ids=(12,), logprobs=(-0.5,)))
    ledger.seal_active_response("r1")
    [set_] = ledger.take_terminal_sets()
    assert set_.input_origins == ("client", "engine")
    assert set_.input_commit_ns == (9, None)


def test_take_terminal_sets_drains_everything() -> None:
    ledger = SessionActionLedger(session_id="s1")
    ledger.bind_listening(epoch=0, turn_id=1)
    ledger.ingest(_payload(seq=1), epoch=0, turn_id=1, response_id=None, stage_id=0, input_modality="")
    ledger.bind_response("r1", epoch=0, turn_id=2)
    ledger.ingest(_payload(seq=2), epoch=0, turn_id=2, response_id="r1", stage_id=0, input_modality="")
    ledger.seal_for_close()
    sets = ledger.take_terminal_sets()
    assert [set_.response_id for set_ in sets] == [None, "r1"]
    assert ledger.take_terminal_sets() == []
