# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.realtime_venus.delegation import DelegateParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("split", range(len("Before <delegate>12 + 30</delegate> after") + 1))
def test_call_is_recognized_at_every_transport_split(split):
    text = "Before <delegate>12 + 30</delegate> after"
    parser = DelegateParser()
    queries = [parser.feed(text[:split]), parser.feed(text[split:], end_of_turn=True)]
    assert [query for query in queries if query is not None] == ["12 + 30"]
    assert len(parser.pending) < len(parser.OPEN)


@pytest.mark.parametrize(
    "text, message",
    [
        ("<delegate>  </delegate>", "empty"),
        ("<delegate>unclosed", "before </delegate>"),
        ("<delegate>one<delegate>two</delegate>", "nested"),
        ("<delegate>" + "x" * 4001 + "</delegate>", "4000"),
        ("<delegate>one</delegate><delegate>two</delegate>", "one delegate"),
    ],
)
def test_malformed_call_is_not_executed(text, message):
    parser = DelegateParser()
    with pytest.raises(ValueError, match=message):
        parser.feed(text, end_of_turn=True)
    assert parser.feed("</delegate>") is None


def test_normal_speech_does_not_accumulate_in_parser():
    parser = DelegateParser()
    for _ in range(100):
        assert parser.feed("ordinary speech " * 100) is None
        assert parser.pending == ""


def test_limit_allows_split_closing_tag():
    parser = DelegateParser()
    assert parser.feed("<delegate>" + "x" * 4000 + "</delega") is None
    assert parser.feed("te>") == "x" * 4000


def test_repeated_span_while_waiting_for_backend_is_one_call():
    parser = DelegateParser()
    assert parser.feed("<delegate>lookup inventory</delegate>") == "lookup inventory"
    assert parser.feed("<delegate>lookup ") is None
    assert parser.feed("inventory</delegate>", end_of_turn=True) is None
    assert not parser.failed
