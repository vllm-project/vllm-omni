# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The turn question: how Sophon decides whether the user has finished.

At a pause the thinker hears the user's current, uncommitted utterance and answers a four-option question with one
letter: finished / not finished / back-channel / hesitating. The options are shuffled with a fixed seed (the order
the model was trained with), and the answer is read from the logprobs of the option letters.

Framework-independent (no vllm / vllm_omni / torch imports).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping

LETTERS = "ABCDEFGHIJ"
FINISHED, NOT_FINISHED, BACKCHANNEL, HESITATING = range(4)

_QUESTION = {
    "zh": (
        "用户现在处于什么话轮状态？",
        ["话说完了，可以接话", "句子没说完，还在继续", "简短附和，不是要接话", "还在组织语言，或要求先等一下"],
    ),
    "en": (
        "Has the user finished speaking?",
        [
            "finished, the assistant can reply",
            "not finished, still talking",
            "just a backchannel, not taking the turn",
            "hesitating or asking to wait",
        ],
    ),
}
_TEMPLATE = {
    "en": (
        "Question: {q}",
        "Options:",
        "Answer with only the letter of the correct option.",
        "The user said: <|audio|>",
        'The assistant had just said: "{c}"',
    ),
    "zh": ("问题：{q}", "选项：", "请只回答正确选项的字母。", "用户说：<|audio|>", "助手刚才说的是：“{c}”"),
}
_SAID_CHARS = 300


def option_order(qid: str, options: list[str], seed: int = 0) -> list[int]:
    """Deterministic shuffle of the options: position i shows option order[i]."""
    key = "\x1f".join([qid, "\x1e".join(options), str(seed)]).encode()
    rng = random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))
    order = list(range(len(options)))
    rng.shuffle(order)
    return order


def turn_question(lang: str, said: str) -> tuple[str, list[int]]:
    """The question text (with one ``<|audio|>`` placeholder for the user's utterance) and the option order.

    ``said`` is what the assistant said last (its tail is shown as context).
    """
    lang = lang if lang in _QUESTION else "en"
    q, options = _QUESTION[lang]
    t_question, t_options, t_instruction, t_head, t_said = _TEMPLATE[lang]
    order = option_order("turn", options)
    context = " ".join((said or "").split())[-_SAID_CHARS:]
    lines = [t_question.format(q=q), "", t_options]
    lines += [f"{LETTERS[i]}. {options[j]}" for i, j in enumerate(order)]
    lines += ["", t_instruction]
    text = (t_said.format(c=context) + "\n\n" if context else "") + t_head + "\n\n" + "\n".join(lines)
    return text, order


def answer_probabilities(top_logprobs: Mapping[str, float], order: list[int]) -> list[float]:
    """P(option) for each option (in option order, not display order), from the logprobs of the first answer token.

    Only the option letters count; tokens are compared after stripping whitespace, and a missing letter has
    probability 0. Raises ``ValueError`` when no option letter is among the logprobs.
    """
    best: dict[str, float] = {}
    for token, logprob in top_logprobs.items():
        k = str(token).strip()
        if k and math.isfinite(logprob) and (k not in best or logprob > best[k]):
            best[k] = logprob
    letters = [LETTERS[i] for i in range(len(order))]
    present = [best[x] for x in letters if x in best]
    if not present:
        raise ValueError("no option letter among the top logprobs")
    m = max(present)
    weights = [math.exp(best[x] - m) if x in best else 0.0 for x in letters]
    total = sum(weights)
    by_position = [w / total for w in weights]
    probs = [0.0] * len(order)
    for position, option in enumerate(order):
        probs[option] = by_position[position]
    return probs
