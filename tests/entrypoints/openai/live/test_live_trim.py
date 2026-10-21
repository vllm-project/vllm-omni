# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Context trimming of Live sessions on ``context_length_exceeded``."""

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.live.audio import LiveAudioFormat
from vllm_omni.entrypoints.openai.live.processor import LiveSessionProcessor, ProcessorContext
from vllm_omni.entrypoints.openai.live.session import LiveSessionState, NativeUnitItem, SystemItem, UserTextItem
from vllm_omni.model_executor.models.minicpmo_4_5.live import processor as minicpmo_processor
from vllm_omni.model_executor.models.minicpmo_4_5.live.processor import MiniCPMO45LiveSessionProcessor
from vllm_omni.model_executor.models.personaplex.live.processor import PersonaPlexLiveSessionProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CharTokenizer:
    """One token per character, spaces included, as in byte-level BPE."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(i) for i in token_ids)


class _CostProcessor(LiveSessionProcessor):
    """Units cost ``unit.data["cost"]`` tokens; the prefix costs ``prefix``."""

    def __init__(self, max_model_len: int, prefix: int) -> None:
        super().__init__(ProcessorContext(tokenizer=None, max_model_len=max_model_len, model_name="m", model_path="m"))
        self.prefix = prefix
        self.dropped: list[list[NativeUnitItem]] = []

    def render_unit(self, state, unit, *, first, last_sampled_token):
        return {}

    def render_resume(self, state):
        return {}

    def prefix_tokens(self, state):
        return self.prefix

    def unit_tokens(self, unit):
        return unit.data["cost"]

    def on_units_dropped(self, state, units):
        self.dropped.append(units)


def _state(history) -> LiveSessionState:
    return LiveSessionState(model="m", audio_format=LiveAudioFormat("audio/pcm", 24_000), expires_at=0, history=history)


def _unit(index: int, cost: int = 0) -> NativeUnitItem:
    return NativeUnitItem(
        audio=np.zeros(0, dtype=np.float32), start_ms=index * 100, end_ms=(index + 1) * 100, data={"cost": cost}
    )


@pytest.mark.parametrize(
    ("costs", "kept"),
    [
        # budget = (1000 - 200) * 0.5 = 400
        pytest.param([100] * 8, 4, id="fills-budget"),
        pytest.param([100, 200, 150, 150], 2, id="next-would-exceed"),
        pytest.param([50, 900], 1, id="newest-kept-over-budget"),
        pytest.param([100, 100], 2, id="nothing-to-drop"),
    ],
)
def test_trim_keeps_newest_units_within_half_the_context(costs, kept):
    units = [_unit(i, cost) for i, cost in enumerate(costs)]
    state = _state(list(units))
    processor = _CostProcessor(max_model_len=1000, prefix=200)

    dropped = processor.trim_context(state)

    assert dropped == len(costs) - kept
    assert state.units == units[len(costs) - kept :]
    assert processor.dropped == ([units[: len(costs) - kept]] if dropped else [])


def test_trim_keeps_non_unit_history_in_order():
    system, user = SystemItem("be brief"), UserTextItem("hi")
    units = [_unit(i, 300) for i in range(3)]
    state = _state([system, units[0], user, units[1], units[2]])

    assert _CostProcessor(max_model_len=1000, prefix=200).trim_context(state) == 2
    assert state.history == [system, user, units[2]]


def test_trim_without_units_drops_nothing():
    state = _state([SystemItem("be brief")])
    processor = _CostProcessor(max_model_len=1000, prefix=200)

    assert processor.trim_context(state) == 0
    assert processor.dropped == []


def _personaplex() -> PersonaPlexLiveSessionProcessor:
    processor = object.__new__(PersonaPlexLiveSessionProcessor)
    processor.context = ProcessorContext(tokenizer=None, max_model_len=600, model_name="m", model_path="m")
    processor.prefill_slots = 100
    return processor


def test_personaplex_trim_restarts_from_the_last_frame():
    system = SystemItem("persona")
    units = [_unit(i) for i in range(5)]
    state = _state([system, *units])

    assert _personaplex().trim_context(state) == 4
    assert state.history == [system, units[-1]]


def test_personaplex_trim_single_frame_drops_nothing():
    unit = _unit(0)
    state = _state([unit])

    assert _personaplex().trim_context(state) == 0
    assert state.history == [unit]


def _minicpmo(tokenizer: _CharTokenizer) -> MiniCPMO45LiveSessionProcessor:
    processor = object.__new__(MiniCPMO45LiveSessionProcessor)
    processor.context = ProcessorContext(tokenizer=tokenizer, max_model_len=1024, model_name="m", model_path="m")
    processor.previous_text = ""
    return processor


def test_minicpmo_dropped_units_carry_forward_as_previous_text():
    tokenizer = _CharTokenizer()
    processor = _minicpmo(tokenizer)
    first, second, silent = _unit(0), _unit(1), _unit(2)
    first.token_ids = tokenizer.encode("hello there")
    second.token_ids = tokenizer.encode(" general kenobi")

    processor.on_units_dropped(_state([]), [first, second])
    assert processor.previous_text == "hello there general kenobi"

    processor.on_units_dropped(_state([]), [silent])
    assert processor.previous_text == "hello there general kenobi"


def test_minicpmo_previous_text_keeps_the_newest_tokens(monkeypatch):
    monkeypatch.setattr(minicpmo_processor, "_PREVIOUS_MAX_TOKENS", 5)
    tokenizer = _CharTokenizer()
    processor = _minicpmo(tokenizer)
    processor.previous_text = "one two"
    unit = _unit(0)
    unit.token_ids = tokenizer.encode("three")

    processor.on_units_dropped(_state([]), [unit])

    assert processor.previous_text == "three"
