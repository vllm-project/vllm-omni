# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The turn question (framework-independent core): the exact wording the model was trained with, and reading
the answer from the option-letter logprobs."""

import ast
import hashlib
import math
import pathlib

import pytest

from vllm_omni.model_executor.models.sophon_realtime.core import turn_question as tq
from vllm_omni.model_executor.models.sophon_realtime.core.turn_question import (
    BACKCHANNEL,
    FINISHED,
    answer_probabilities,
    turn_question,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_module_imports_no_framework():
    tree = ast.parse(pathlib.Path(tq.__file__).read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not names & {"vllm", "vllm_omni", "torch"}


@pytest.mark.parametrize(
    ("lang", "order", "digest"), [("zh", [1, 3, 2, 0], "3d268a3d22b3ef7c"), ("en", [2, 1, 0, 3], "783e841f3714dfd5")]
)
def test_wording_and_option_order_match_training(lang, order, digest):
    # The model reads this text verbatim; any change in wording, layout or option order changes its answers.
    text, got = turn_question(lang, "")
    assert got == order
    assert hashlib.sha256(text.encode()).hexdigest()[:16] == digest
    assert text.count("<|audio|>") == 1


def test_previous_reply_is_shown_as_context_and_clipped():
    text, _ = turn_question("en", "  first   words  " + "x" * 400 + " last")
    assert text.startswith('The assistant had just said: "')
    context = text.split('"')[1]
    assert len(context) == 300 and context.endswith(" last")


def test_unknown_language_falls_back_to_english():
    assert turn_question("fr", "") == turn_question("en", "")


def test_probabilities_follow_the_shuffled_letters():
    _, order = turn_question("en", "")  # C = finished, A = back-channel
    probs = answer_probabilities({"C": math.log(0.6), " A": math.log(0.3), "B": math.log(0.1)}, order)
    assert probs[FINISHED] == pytest.approx(0.6)
    assert probs[BACKCHANNEL] == pytest.approx(0.3)
    assert sum(probs) == pytest.approx(1.0)


def test_duplicate_tokens_keep_the_best_and_non_letters_are_ignored():
    _, order = turn_question("en", "")
    probs = answer_probabilities({"C": -5.0, " C": -0.1, "The": -0.01, "D": -0.1}, order)
    assert probs[FINISHED] == pytest.approx(0.5)


def test_no_option_letter_is_an_error():
    _, order = turn_question("zh", "")
    with pytest.raises(ValueError):
        answer_probabilities({"Yes": -0.1, "No": -2.0}, order)
