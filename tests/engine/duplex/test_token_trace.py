# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-unit token trace (``extra_body.trace_tokens``), with fake stage outputs and tokenizer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.contracts import DuplexFence, DuplexOutputContext, DuplexRequestIdentity
from vllm_omni.engine.duplex.plugin import (
    DuplexModelPlugin,
    DuplexRuntimeConfigError,
    DuplexUnitDecision,
    default_unit_token_trace,
    token_id_list,
)
from vllm_omni.engine.duplex.session import token_trace as token_trace_module
from vllm_omni.engine.duplex.session.input_clock import InputClock, StageProgress
from vllm_omni.engine.duplex.session.token_trace import (
    TokenTracer,
    check_token_trace_allowed,
    check_token_trace_unchanged,
    load_trace_tokenizer,
    token_trace_enabled,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
from vllm_omni.protocol.duplex.events import DuplexEvent, InputProcessed, UnitTokens

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION = "sess-trace"


class _Tokenizer:
    all_special_ids = [100]

    @staticmethod
    def get_added_vocab() -> dict[str, int]:
        return {"<unit>": 101}

    @staticmethod
    def convert_ids_to_tokens(ids: list[int]) -> list[str]:
        names = {100: "<|listen|>", 101: "<unit>"}
        return [names.get(i, f"tok{i}") for i in ids]


class _DefaultHooks(MiniCPMO45DuplexPlugin):
    """A real plugin with the contract's default input-clock and token-trace hooks."""

    unit_decision = DuplexModelPlugin.unit_decision
    unit_output_complete = DuplexModelPlugin.unit_output_complete
    unit_token_trace = DuplexModelPlugin.unit_token_trace

    def __init__(self) -> None:
        super().__init__(lambda *args: None)


class _PromptHooks(_DefaultHooks):
    """Splits the first unit's input into the session context and the unit, and labels codec ids."""

    def unit_token_trace(self, *, stage_id, output, context, cursor, runtime_config):
        trace = default_unit_token_trace(stage_id=stage_id, output=output, cursor=cursor)
        if stage_id == 0 and not cursor.get("seen"):
            cursor["seen"] = True
            trace["prompt_input"], trace["input"] = trace["input"][:2], trace["input"][2:]
        if stage_id == 1:
            trace["output"] = [[i, f"code:{i}"] for i in trace["output"]]
        return trace


def _stage_output(prompt: list[int], generated: list[int], request_id: str = "req-0") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id, prompt_token_ids=prompt, outputs=[SimpleNamespace(token_ids=generated)]
    )


def _session(plugin: DuplexModelPlugin, tokenizer: object = _Tokenizer) -> tuple[InputClock, list[DuplexEvent]]:
    sent: list[DuplexEvent] = []
    clock = InputClock(emit=sent.extend, plugin=plugin, runtime_config=dict)
    clock.tracer = TokenTracer(plugin=plugin, tokenizer_loader=lambda: tokenizer, emit=sent.extend)
    return clock, sent


def _progress(
    clock: InputClock,
    stage_id: int,
    output: object,
    *,
    decision: str | None = None,
    final: int = 1,
    segment_finished: bool = True,
) -> None:
    """A segment end of a traced session: the runner hands every stage's output along."""
    context = DuplexOutputContext(
        identity=DuplexRequestIdentity(session_id=SESSION, fence=DuplexFence(SESSION, epoch=0)),
        final_stage_id=final,
        segment_finished=segment_finished,
    )
    clock.on_stage_progress(
        StageProgress(
            stage_id=stage_id,
            final_stage_id=final,
            epoch=0,
            segment_finished=segment_finished,
            decision=DuplexUnitDecision(label=decision, ends_unit=True) if decision is not None else None,
            output=output,
            context=context,
        )
    )


def _input(clock: InputClock) -> None:
    ack = clock.admit_input("input_audio_buffer.append")
    clock.begin_input(ack)
    clock.note_input_audio(ack, samples=16000, sample_rate_hz=16000)
    clock.unit_submitted(clock.new_unit(input_us=1_000_000, epoch=0))
    clock.end_input(ack)


def test_opt_in_and_id_helpers() -> None:
    assert token_trace_enabled({"trace_tokens": True})
    assert not token_trace_enabled({"trace_tokens": "yes"}) and not token_trace_enabled(None)
    clocked = {"clock": "input", "trace_tokens": True}
    check_token_trace_allowed(clocked, debug_events_enabled=True)
    check_token_trace_allowed({}, debug_events_enabled=False)  # not asked for: nothing to check
    for extra_body, debug_events, code in [
        (clocked, False, "token_trace_disabled"),
        ({"trace_tokens": True}, True, "token_trace_requires_input_clock"),
    ]:
        with pytest.raises(DuplexRuntimeConfigError) as excinfo:
            check_token_trace_allowed(extra_body, debug_events_enabled=debug_events)
        assert excinfo.value.code == code
    check_token_trace_unchanged(clocked, dict(clocked))
    with pytest.raises(DuplexRuntimeConfigError) as excinfo:
        check_token_trace_unchanged({"clock": "input"}, clocked)
    assert excinfo.value.code == "input_clock_update_unsupported"
    assert token_id_list((1, "2", None)) == [1, 2]
    assert token_id_list(SimpleNamespace(tolist=lambda: [3])) == [3]
    assert token_id_list("12") == []


def test_the_default_trace_reports_deltas_since_the_previous_segment() -> None:
    cursor: dict[str, object] = {}
    first = default_unit_token_trace(stage_id=0, output=_stage_output([1, 2, 3], [100]), cursor=cursor)
    second = default_unit_token_trace(stage_id=0, output=_stage_output([1, 2, 3, 4, 5], [100, 7]), cursor=cursor)
    assert (first["input"], first["output"]) == ([1, 2, 3], [100])
    assert (second["input"], second["output"]) == ([4, 5], [7])

    # A prompt that shrank within the same request (e.g. a rebuilt context) is reported whole.
    shrunk = default_unit_token_trace(stage_id=0, output=_stage_output([9, 9], [100, 7, 8]), cursor=cursor)
    assert (shrunk["input"], shrunk["output"]) == ([9, 9], [8])


def test_the_default_trace_starts_over_for_a_new_stage_request() -> None:
    """A model that opens one request per turn: the next turn's longer prompt is not sliced by the old offset."""
    cursor: dict[str, object] = {}
    default_unit_token_trace(stage_id=0, output=_stage_output([1, 2, 3], [100], request_id="turn-0"), cursor=cursor)
    turn = default_unit_token_trace(
        stage_id=0, output=_stage_output([5, 6, 7, 8, 9], [41, 42], request_id="turn-1"), cursor=cursor
    )
    assert (turn["input"], turn["output"]) == ([5, 6, 7, 8, 9], [41, 42])


def test_each_unit_gets_one_event_before_its_acknowledgement() -> None:
    clock, sent = _session(_DefaultHooks())
    _input(clock)
    _progress(clock, 0, _stage_output([101, 5, 5], [100]), decision="listen")

    assert [type(e) for e in sent] == [UnitTokens, InputProcessed]
    wire = sent[0].to_realtime()
    assert wire["type"] == "debug.unit_tokens"
    assert (wire["unit_index"], wire["end_ms"], wire["decision"]) == (0, 1000, "listen")
    assert wire["stages"] == [
        {"stage": "stage0", "input": [[101, "<unit>"], [5, "tok5"], [5, "tok5"]], "output": [[100, "<|listen|>"]]}
    ]
    assert wire["special_ids"] == [100, 101]


def test_a_speaking_unit_carries_every_stage_and_the_prompt_is_sent_once() -> None:
    clock, sent = _session(_PromptHooks())
    _input(clock)
    _progress(clock, 0, _stage_output([1, 2, 101, 5], [8]), final=2)  # speaking: forwarded
    _progress(clock, 1, _stage_output([0, 0], [41, 42]), final=2)
    assert sent == [], "the final stage has not ended the unit yet"
    _progress(clock, 2, None, final=2)

    prompt, unit = (e.to_realtime() for e in sent[:2])
    assert (prompt["unit_index"], prompt["decision"], prompt["end_ms"]) == (-1, "prompt", 0)
    assert prompt["stages"] == [{"stage": "stage0", "input": [[1, "tok1"], [2, "tok2"]]}]
    assert unit["decision"] == "speak"
    assert [s["stage"] for s in unit["stages"]] == ["stage0", "stage1"]
    assert unit["stages"][1]["output"] == [[41, "code:41"], [42, "code:42"]]
    assert isinstance(sent[2], InputProcessed)

    _input(clock)
    _progress(clock, 0, _stage_output([1, 2, 101, 5, 101, 6], [8, 100]), decision="listen", final=2)
    assert [e.to_realtime()["unit_index"] for e in sent if isinstance(e, UnitTokens)] == [-1, 0, 1]


def test_settled_units_are_traced_with_their_decision_and_nothing_follows_the_close() -> None:
    clock, sent = _session(_DefaultHooks())
    _input(clock)
    clock.cancel_open("client_cancelled", before_epoch=1)
    assert [type(e) for e in sent] == [UnitTokens, InputProcessed]
    assert sent[0].to_realtime()["decision"] == "cancelled"

    clock, sent = _session(_DefaultHooks())
    _input(clock)
    clock.close("client_close")
    assert [type(e) for e in sent] == [UnitTokens, InputProcessed]
    assert sent[0].to_realtime()["decision"] == "aborted"
    clock.new_unit(input_us=1_000_000, epoch=0)
    _progress(clock, 0, _stage_output([1], [100]), decision="listen")
    assert len(sent) == 2, "nothing is traced after the teardown"


def test_a_failing_plugin_hook_does_not_break_the_session() -> None:
    class _Broken(_DefaultHooks):
        def unit_token_trace(self, *, stage_id, output, context, cursor, runtime_config):
            raise RuntimeError("boom")

    clock, sent = _session(_Broken())
    _input(clock)
    _progress(clock, 0, _stage_output([1], [100]), decision="listen")
    assert [type(e) for e in sent] == [UnitTokens, InputProcessed]
    assert sent[0].to_realtime()["stages"] == []


def test_without_a_tokenizer_ids_are_sent_with_their_number_as_text() -> None:
    clock, sent = _session(_DefaultHooks(), tokenizer=None)
    _input(clock)
    _progress(clock, 0, _stage_output([7], [9]), decision="listen")
    assert sent[0].to_realtime()["stages"][0]["input"] == [[7, "7"]]


class _UnnamingTokenizer(_Tokenizer):
    @staticmethod
    def convert_ids_to_tokens(ids: list[int]) -> list[str]:
        raise IndexError("piece table out of range")


class _BrokenVocabTokenizer(_Tokenizer):
    @staticmethod
    def get_added_vocab() -> dict[str, int]:
        raise RuntimeError("no added vocabulary")


@pytest.mark.parametrize("tokenizer", [_UnnamingTokenizer, _BrokenVocabTokenizer])
def test_a_tokenizer_failure_neither_withholds_the_acknowledgement_nor_breaks_the_teardown(tokenizer) -> None:
    clock, sent = _session(_DefaultHooks(), tokenizer=tokenizer)
    _input(clock)
    _progress(clock, 0, _stage_output([7], [9]), decision="listen")
    assert [type(e) for e in sent] == [UnitTokens, InputProcessed], "rendered without pieces, then acknowledged"
    assert sent[0].to_realtime()["stages"][0]["input"] == [[7, "7"]]

    _input(clock)
    clock.close("client_close")
    assert [type(e) for e in sent[2:]] == [UnitTokens, InputProcessed], "the teardown acknowledgement still goes out"


def test_a_failing_tracer_never_reaches_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    clock, sent = _session(_DefaultHooks())
    monkeypatch.setattr(clock.tracer, "emit_unit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    _input(clock)
    _progress(clock, 0, _stage_output([7], [9]), decision="listen")
    assert [type(e) for e in sent] == [InputProcessed]


def test_a_segment_end_that_overtakes_its_unit_is_traced_with_that_unit() -> None:
    """A Talker segment end held until its unit speaks is traced then, so later units' ids stay their own."""
    clock, sent = _session(_DefaultHooks())
    _input(clock)
    _progress(clock, 1, _stage_output([5], [6]), final=2)  # the unit's Talker segment end, ahead of Stage 0
    _progress(clock, 0, _stage_output([1], [2]), final=2)  # its Stage-0 segment end: it speaks, the held end replays
    _progress(clock, 2, _stage_output([], [9]), final=2)
    talker = [stage for stage in sent[0].to_realtime()["stages"] if stage["stage"] == "stage1"]
    assert talker == [{"stage": "stage1", "input": [[5, "tok5"]], "output": [[6, "tok6"]]}]


def test_the_tokenizer_is_loaded_once_off_the_loop_and_a_failure_only_drops_the_pieces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(model_config):
        raise OSError("no tokenizer files")

    monkeypatch.setattr(token_trace_module, "cached_tokenizer_from_config", _fail)
    assert load_trace_tokenizer(object()) is None
