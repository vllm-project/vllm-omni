# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bistream schedule of the Tag talker (framework-independent).

The Tag talker (a CosyVoice3 LM) reads the reply text while the thinker is still
writing it, on CosyVoice3's ``inference_bistream`` schedule:

* the model emits ``fill`` -> the next ``mix_text`` text tokens are fed ->
  ``mix_speech`` speech tokens are decoded -> ``fill`` is forced -> ...;
* once the thinker has finished, the input that produced the last ``fill`` is
  fed again, then the remaining text and ``task``, and the model decodes until
  ``eos`` (final phase);
* while text is streaming every special token except ``fill`` is masked; in the
  final phase only ``eos`` is allowed.

A runtime decodes one position per step, so a multi-token input is fed one item
per step; the samples of the intermediate positions are discarded (reported as
``fill``, which the vocoder ignores).

This module only decides *what* to feed and which token is forced or allowed.
Runtimes map the input items to embeddings and do the sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# One input position: a text token, a speech/special token, or the previous
# input again.
InputItem = tuple[Literal["text"], int] | tuple[Literal["speech"], int] | tuple[Literal["repeat"]]
REPEAT: InputItem = ("repeat",)


@dataclass(frozen=True)
class BistreamTokens:
    speech_vocab: int  # number of speech tokens; the special tokens follow
    mix_text: int = 5
    mix_speech: int = 15

    @property
    def sos(self) -> int:
        return self.speech_vocab

    @property
    def eos(self) -> int:
        return self.speech_vocab + 1

    @property
    def task(self) -> int:
        return self.speech_vocab + 2

    @property
    def fill(self) -> int:
        return self.speech_vocab + 3


@dataclass
class SamplingPlan:
    """What the sampler must do for one stream at this step."""

    forced: int | None = None  # emit this token (no sampling)
    final: bool = False  # final phase: only eos among the special tokens
    discard: bool = False  # intermediate position: the sample is not kept


@dataclass
class BistreamSchedule:
    tokens: BistreamTokens
    text: list[int] = field(default_factory=list)  # text ids received so far
    pos: int = 0  # text ids fed
    finished: bool = False  # the thinker has finished
    phase: Literal["bistream", "final"] = "bistream"
    out: list[int] = field(default_factory=list)  # sampled tokens of real positions
    queue: list[InputItem] = field(default_factory=list)
    intermediate: bool = False
    starved: int = 0  # times the talker ran out of text while the thinker was writing
    last_token: int | None = None
    next_fill: int = -1

    def __post_init__(self) -> None:
        if self.next_fill < 0:
            self.next_fill = self.tokens.mix_speech  # no prompt speech tokens

    # ------------------------------------------------------------------ text
    def update_text(self, ids: list[int] | None, finished: bool) -> None:
        """Cumulative text from the thinker (never shrinks)."""
        if ids is not None and len(ids) >= len(self.text):
            self.text = list(ids)
        self.finished = self.finished or finished

    @property
    def done(self) -> bool:
        return self.phase == "final" and self.last_token == self.tokens.eos

    # ------------------------------------------------------------------ inputs
    def next_input(self) -> InputItem | None:
        """The input of this decode step; ``None`` means "the last sampled
        speech token", as in plain decoding."""
        if self.queue:
            item = self.queue.pop(0)
            self.intermediate = bool(self.queue)
            return item
        self.intermediate = False
        return None

    def _feed_text(self) -> None:
        t = self.tokens
        avail = len(self.text) - self.pos
        if avail >= t.mix_text:
            self.queue = [("text", i) for i in self.text[self.pos : self.pos + t.mix_text]]
            self.pos += t.mix_text
            return
        if self.finished:
            self.phase = "final"
            rest: list[InputItem] = [("text", i) for i in self.text[self.pos :]]
            self.pos = len(self.text)
            self.queue = [REPEAT, *rest, ("speech", t.task)]
            return
        self.starved += 1
        if avail > 0:
            self.queue = [("text", i) for i in self.text[self.pos :]]
            self.pos = len(self.text)
        else:
            self.queue = [REPEAT]

    # ------------------------------------------------------------------ sampling
    def plan(self) -> SamplingPlan:
        if self.intermediate:
            return SamplingPlan(forced=self.tokens.fill, discard=True)
        if self.phase == "bistream" and len(self.out) == self.next_fill:
            return SamplingPlan(forced=self.tokens.fill)
        return SamplingPlan(final=self.phase == "final")

    def accept(self, token: int) -> None:
        """Record the token emitted at a real (non-intermediate) position."""
        self.out.append(token)
        if self.phase == "bistream" and token == self.tokens.fill:
            self.next_fill = len(self.out) + self.tokens.mix_speech
            self._feed_text()
        else:
            self.last_token = token
