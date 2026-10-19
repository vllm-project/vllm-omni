# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 duplex decoding for OpenAI Live sessions.

Reproduces MiniCPM-o-Demo's greedy ``DuplexCapability.streaming_generate``
loop (``StreamDecoder.decode(mode="greedy")``) on a vLLM resumable request,
where each streaming update is one ``<unit>`` and each segment is the model's
output for it. Per unit, token ``j``:

* the first ``force_listen_count`` units of a session are ``<|listen|>``;
* ``j == max_unit_tokens - 1`` is ``<|chunk_eos|>`` without decoding;
* otherwise: if the raw argmax is ``<|chunk_eos|>``, take it. Else mask the
  forbidden tokens (``<|tts_pad|>``, the tokenizer's bad tokens and
  ``<|chunk_eos|>``), divide the logits of the distinct tokens among the last
  512 decoded tokens by 1.05, apply the length penalty to ``<|turn_eos|>``
  and take the argmax (it joins the decoded history);
* a decoded ``<|listen|>`` before the turn has ended becomes ``<|tts_bos|>``;
* a non-terminator re-opens the turn, and ``<|turn_eos|>`` ends it;
* a token after the first whose addition would bring the unit's text to 28
  characters is replaced by ``<|chunk_eos|>``.

Every decision is applied as a one-hot row, so vLLM's greedy sampler takes
exactly the decided token. The decoded history and the turn state span the
whole session, across units and resubmissions: vLLM re-adds the request to
the batch on every streaming update, so they are kept per session id
(``extra_args["minicpmo_live"]["session"]``), not per batch row. Requests
without ``extra_args["minicpmo_live"]`` are untouched.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

REPETITION_PENALTY = 1.05
REPETITION_WINDOW = 512
LENGTH_PENALTY = 1.1
FORCE_LISTEN_COUNT = 3
UNIT_TEXT_LIMIT = 28
# Sessions whose decode state is kept; the oldest is evicted beyond this.
MAX_SESSIONS = 1024


@dataclass
class _SessionState:
    # A fresh official session starts with the turn ended.
    turn_ended: bool = True
    history: list[int] = field(default_factory=list)


@dataclass
class _RowState:
    session: _SessionState
    prefix: bool
    unit_index: int
    max_unit_tokens: int
    step: int = 0
    done: bool = False
    # Fed tokens of the unit after its first (the unit text for the limit).
    unit_ids: list[int] = field(default_factory=list)


class MiniCPMOLiveLogitsProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        spec = (sampling_params.extra_args or {}).get("minicpmo_live")
        if spec is None:
            return
        if not isinstance(spec, dict) or not spec.get("session"):
            raise ValueError("extra_args.minicpmo_live needs a session id")
        if not spec.get("prefix") and int(spec.get("max_unit_tokens", 0)) < 1:
            raise ValueError("extra_args.minicpmo_live.max_unit_tokens must be >= 1")

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        self.device = device
        self._model_config = vllm_config.model_config
        self._rows: dict[int, _RowState] = {}
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()
        self._tokenizer = None

    def is_argmax_invariant(self) -> bool:
        return False

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        config = self._model_config
        tok = AutoTokenizer.from_pretrained(
            config.tokenizer,
            trust_remote_code=bool(config.trust_remote_code),
            revision=getattr(config, "tokenizer_revision", None),
        )

        def token_id(token: str) -> int:
            value = tok.convert_tokens_to_ids(token)
            if value is None or value == getattr(tok, "unk_token_id", None):
                raise ValueError(f"MiniCPM-o tokenizer has no {token}")
            return int(value)

        self._tokenizer = tok
        self.listen = token_id("<|listen|>")
        self.tts_bos = token_id("<|tts_bos|>")
        self.chunk_eos = token_id("<|chunk_eos|>")
        self.chunk_tts_eos = token_id("<|chunk_tts_eos|>")
        self.turn_eos = token_id("<|turn_eos|>")
        self.terminators = {self.listen, self.chunk_eos, self.chunk_tts_eos}
        forbidden = [
            token_id("<|tts_pad|>"),
            *(int(t) for t in getattr(tok, "bad_token_ids", []) or []),
            self.chunk_eos,
        ]
        self._forbidden = torch.tensor(sorted(set(forbidden)), dtype=torch.long, device=self.device)

    def _session(self, key: str) -> _SessionState:
        state = self._sessions.get(key)
        if state is None:
            state = self._sessions[key] = _SessionState()
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(key)
        return state

    def _new_row(self, params: SamplingParams, _prompt: list[int] | None, _output: list[int]) -> _RowState | None:
        spec = (params.extra_args or {}).get("minicpmo_live") if params is not None else None
        if not spec:
            return None
        if self._tokenizer is None:
            self._load_tokenizer()
        return _RowState(
            session=self._session(str(spec["session"])),
            prefix=bool(spec.get("prefix")),
            unit_index=int(spec.get("unit_index", 0)),
            max_unit_tokens=int(spec.get("max_unit_tokens", 1)),
        )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self._rows, batch_update, self._new_row)

    def _decode(self, row: _RowState, logits: torch.Tensor) -> int:
        """One official decision; updates the session's history and turn."""
        session = row.session
        if row.prefix or (row.step == 0 and row.unit_index < FORCE_LISTEN_COUNT):
            return self.listen
        if row.step >= row.max_unit_tokens - 1:
            return self.chunk_eos
        if int(torch.argmax(logits)) == self.chunk_eos:
            return self.chunk_eos
        scores = logits.clone()
        scores[self._forbidden] = float("-inf")
        recent = session.history[-REPETITION_WINDOW:]
        if recent:
            index = torch.tensor(sorted(set(recent)), dtype=torch.long, device=scores.device)
            scores[index] = scores[index] / REPETITION_PENALTY
        turn_eos = scores[self.turn_eos]
        scores[self.turn_eos] = turn_eos / LENGTH_PENALTY if turn_eos > 0 else turn_eos * LENGTH_PENALTY
        decoded = int(torch.argmax(scores))
        session.history.append(decoded)
        if len(session.history) > REPETITION_WINDOW:
            del session.history[:-REPETITION_WINDOW]
        token = decoded
        if token == self.listen and not session.turn_ended:
            token = self.tts_bos
        if token in self.terminators:
            return token
        session.turn_ended = False
        if row.step != 0:
            text = self._tokenizer.decode([*row.unit_ids, token], skip_special_tokens=True)
            if len(text) >= UNIT_TEXT_LIMIT:
                return self.chunk_eos
            row.unit_ids.append(token)
        if token == self.turn_eos:
            session.turn_ended = True
        return token

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, row in self._rows.items():
            if index >= logits.shape[0] or row.done:
                continue
            token = self._decode(row, logits[index])
            row.step += 1
            row.done = token in self.terminators
            keep = logits[index, token].clone()
            logits[index] = float("-inf")
            logits[index, token] = keep if torch.isfinite(keep) else 0.0
        return logits
