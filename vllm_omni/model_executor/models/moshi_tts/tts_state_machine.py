"""TTS state machine (Delayed Streams Modeling).

Ported from moshi/moshi/moshi/models/tts.py (TokenIds, Entry, State, StateMachine,
script_to_entries). Pure-Python; no torch-engine deps.

Drives word-by-word text feeding into the LM: every step, the model predicts
either `pad` or `new_word`. The state machine watches those predictions, forces
padding when a word is still being fed, forces `new_word` when max padding is
reached, and converts each decision into the token the LM should see at the
next step.

MVP scope: batch_size=1 per TTSState. For multi-request batching, the caller
holds one state per sequence.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class TokenIds:
    """Special token ids for the DSM protocol.

    Attributes:
        card: text cardinality = tokenizer vocab size + 1.
        new_word: signals a word boundary.
        pad: padding (no text this step).
        main: marks the main speaker turn.
        other: marks a secondary speaker turn.
        zero: value embedded to exactly 0 (used for prefix masking).
        ungenerated: placeholder for not-yet-generated positions.
    """

    card: int
    new_word: int = 0
    pad: int = 3
    main: int = 1
    other: int = 2
    zero: int = -1
    ungenerated: int = -2


@dataclass
class Entry:
    """One word (or break) to synthesize.

    Attributes:
        tokens: tokenizer ids for this word (may include a leading speaker tag).
        text: the word as a string (for debug/transcript).
        padding: extra pad steps to force after this word is fully fed.
    """

    tokens: list[int]
    text: str
    padding: int = 0


@dataclass
class State:
    """Mutable runtime state for one sequence.

    Attributes:
        entries: queue of remaining words.
        remaining_padding: how many more pad tokens the model MAY sample before
            being forced to ask for a new word.
        forced_padding: how many more pad tokens the model MUST sample before a
            new word can start.
        queued: tokenizer ids waiting to be fed to the LM (current word).
        lookahead_queued: tokenizer ids waiting on the second text stream
            (only used if second_stream_ahead > 0).
        end_step: step at which generation finished (None while running).
        consumption_times: step index when each Entry was consumed.
        transcript: (word, step) pairs for each word actually spoken.
    """

    entries: deque
    remaining_padding: int
    forced_padding: int
    queued: deque = field(default_factory=deque)
    lookahead_queued: deque = field(default_factory=deque)
    end_step: int | None = None
    consumption_times: list[int] = field(default_factory=list)
    transcript: list[tuple[str, int]] = field(default_factory=list)

    def get_tokens_ahead(self, lookahead: int) -> list[int]:
        """Return the tokens of the Nth upcoming content word (1-indexed)."""
        assert lookahead > 0
        for entry in self.entries:
            if entry.tokens:
                lookahead -= 1
                if lookahead == 0:
                    return entry.tokens
        return []


@dataclass
class StateMachine:
    """Drives word-by-word text feeding for DSM TTS.

    Attributes:
        token_ids: special token ids (see TokenIds).
        second_stream_ahead: if > 0, a second (lookahead) text stream is used.
            The returned input tokens are multiplexed as
            ``(second + 1) * card + main`` and the LM's embedding demuxes them.
        max_padding: max pad tokens the model may sample in a row before being
            forced to start the next word.
        initial_padding: pad tokens forced at the very start (keeps the first
            word from being cut off).
    """

    token_ids: TokenIds
    second_stream_ahead: int = 0
    max_padding: int = 8
    initial_padding: int = 2

    def new_state(self, entries: Sequence[Entry]) -> State:
        return State(
            entries=deque(entries),
            lookahead_queued=deque(),
            remaining_padding=self.initial_padding,
            forced_padding=self.initial_padding,
        )

    def process(self, step: int, state: State, token: int) -> int:
        """Advance the state one step given the model's predicted text token.

        Args:
            step: current step index (starting from 0).
            state: state to mutate.
            token: text token sampled by the LM this step.

        Returns:
            next_input_token: what the LM should see on the text channel
            at the next step. When ``second_stream_ahead > 0`` the two streams
            are multiplexed together; the LM's embedding is expected to demux.
        """
        if token not in (self.token_ids.new_word, self.token_ids.pad):
            token = self.token_ids.pad

        if state.queued:
            token = self.token_ids.pad
        elif state.forced_padding > 0:
            token = self.token_ids.pad
        elif state.remaining_padding <= 0:
            token = self.token_ids.new_word

        if token == self.token_ids.new_word:
            if state.entries:
                entry = state.entries.popleft()
                state.consumption_times.append(step)
                if entry.tokens:
                    state.transcript.append((entry.text, step))
                    state.queued.extend(entry.tokens)
                    if self.second_stream_ahead:
                        state.lookahead_queued.extend(state.get_tokens_ahead(self.second_stream_ahead))
                    state.remaining_padding = self.max_padding
                else:
                    token = self.token_ids.pad
                state.forced_padding = entry.padding
            else:
                token = self.token_ids.pad
                if self.second_stream_ahead and state.end_step is None:
                    token = self.token_ids.new_word
                if state.end_step is None:
                    state.end_step = step

        output: int | None = None
        if token == self.token_ids.pad:
            if state.remaining_padding > 0:
                state.remaining_padding -= 1
            if state.forced_padding > 0:
                state.forced_padding -= 1
            if state.queued:
                output = state.queued.popleft()
            else:
                output = self.token_ids.pad
        elif token == self.token_ids.new_word:
            output = self.token_ids.new_word
        else:
            raise RuntimeError(f"Invalid token {token}")

        if self.second_stream_ahead:
            second = -1
            if output == self.token_ids.new_word:
                second = self.token_ids.new_word
                if state.queued:
                    output = state.queued.popleft()
                else:
                    output = self.token_ids.pad
            elif state.lookahead_queued:
                second = state.lookahead_queued.popleft()
            output = (second + 1) * self.token_ids.card + output

        assert output is not None
        return output


_EVENT_RE = re.compile(r"(?:<break\s+time=\"([0-9]+(?:.[0-9]*)?)s\"\s*/?>)|(?:\s+)")


def script_to_entries(
    tokenize: callable,
    token_ids: TokenIds,
    frame_rate: float,
    script: Sequence[str],
    multi_speaker: bool = True,
    padding_between: int = 0,
) -> list[Entry]:
    """Tokenize a multi-turn script into Entry objects.

    Args:
        tokenize: callable ``str -> list[int]`` (e.g. ``sp.encode``).
        token_ids: special token ids; ``main`` / ``other`` are inserted at
            speaker turns when ``multi_speaker=True``.
        frame_rate: codec frame rate (Hz), used to convert ``<break time="Xs"/>``
            into a pad-step count.
        script: list of utterances; index % 2 selects speaker when multi-speaker.
            Pass an empty first string to start with the "other" speaker.
        multi_speaker: insert speaker tags at turn boundaries.
        padding_between: extra pad steps to force between words (1-2 slows
            speech slightly, can improve articulation).

    Returns:
        list[Entry] ready for ``StateMachine.new_state``.
    """
    speaker_tokens = [token_ids.main, token_ids.other]
    last_speaker: int | None = None
    entries: list[Entry] = []

    def _add_entry(idx: int, word: str, first_content_ref: list[bool]) -> None:
        nonlocal last_speaker
        assert " " not in word and word
        tokens = list(tokenize(word))
        if first_content_ref[0]:
            speaker = idx % len(speaker_tokens)
            if multi_speaker and last_speaker != speaker:
                last_speaker = speaker
                tokens.insert(0, speaker_tokens[speaker])
            first_content_ref[0] = False
        padding = 0
        if padding_between > 0:
            padding = max(0, padding_between + len(tokens) - 1)
        entries.append(Entry(tokens=tokens, text=word, padding=padding))

    for idx, line in enumerate(script):
        first_content_ref = [True]
        line = line.replace("’", "'")
        line = line.replace(":", " ")
        line = line.replace("(", "")
        line = line.replace(")", "")
        while line:
            match = _EVENT_RE.search(line)
            if match is None:
                break
            word = line[: match.start()]
            line = line[match.end() :]
            if word:
                _add_entry(idx, word, first_content_ref)
            if match.group(1):
                break_duration = float(match.group(1))
                padding = int(round(break_duration * frame_rate))
                entries.append(Entry(tokens=[], text="", padding=padding))
        if line:
            _add_entry(idx, line, first_content_ref)
    return entries


__all__ = [
    "TokenIds",
    "Entry",
    "State",
    "StateMachine",
    "script_to_entries",
]
