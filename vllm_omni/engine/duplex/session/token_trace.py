# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in per-unit token trace (``session.extra_body.trace_tokens == true``).

For every model unit the session emits one ``debug.unit_tokens`` event with
the token ids each stage consumed and produced for that unit, plus (once) the
initial prompt as unit ``-1``: what the model actually saw, special tokens
included, which is what one needs to check a duplex model's chunk layout
(unit delimiters, audio placeholders, listen/speak control tokens, TTS
boundaries).

Units are the input clock's (one Stage-0 submission each, see
``input_clock``), so the trace is available on input-clocked sessions only:
the clock's preconditions (one Stage-0 segment end per accepted submission,
a new epoch for every cancel that calls off appends, no silence
continuation, the plugin's unit boundaries) are what make the mapping of ids
to units exact. An event goes out when its unit completes, before the
``input_audio_buffer.processed`` that covers the unit. Like the clock, the
option is fixed when the session is created.

The engine owns the unit mapping and the rendering (token pieces from the
model's tokenizer, special-id detection). Which ids a stage consumed and
produced for one segment is asked of the plugin
(``DuplexModelPlugin.unit_token_trace``).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config

from vllm_omni.engine.duplex.config import input_clocked
from vllm_omni.engine.duplex.contracts import DuplexOutputContext
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError, default_unit_token_trace
from vllm_omni.protocol.duplex.events import DuplexEvent, UnitTokens

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session.input_clock import InputClockUnit

logger = init_logger(__name__)

TRACE_KEY = "trace_tokens"


def token_trace_enabled(extra_body: object) -> bool:
    return isinstance(extra_body, Mapping) and extra_body.get(TRACE_KEY) is True


def check_token_trace_allowed(extra_body: object, *, debug_events_enabled: bool) -> None:
    """Refuse ``trace_tokens`` unless the deployment enabled debug events and the session is input-clocked."""
    if not token_trace_enabled(extra_body):
        return
    if not debug_events_enabled:
        raise DuplexRuntimeConfigError(
            "extra_body.trace_tokens needs a server with duplex_session.enable_debug_events: true",
            code="token_trace_disabled",
        )
    if not input_clocked(extra_body):
        raise DuplexRuntimeConfigError(
            'extra_body.trace_tokens needs an input-clocked session (extra_body.clock: "input")',
            code="token_trace_requires_input_clock",
        )


def check_token_trace_unchanged(current: Mapping[str, object], candidate: Mapping[str, object]) -> None:
    """Refuse a ``session.update`` that turns the trace on or off: like the clock, it is fixed at creation."""
    if token_trace_enabled(current) != token_trace_enabled(candidate):
        raise DuplexRuntimeConfigError(
            f"session.update cannot change extra_body.{TRACE_KEY}: it is fixed at session creation",
            code="input_clock_update_unsupported",
        )


def load_trace_tokenizer(model_config: object) -> TokenizerLike | None:
    """The model's tokenizer for naming traced ids (blocking: run it in an executor); ``None`` if it cannot load."""
    try:
        return cached_tokenizer_from_config(model_config)
    except Exception:
        logger.warning("token trace: no tokenizer; ids are sent without their pieces", exc_info=True)
        return None


class TokenTracer:
    """Collects per-stage token ids onto units and emits ``debug.unit_tokens``."""

    def __init__(
        self,
        *,
        plugin: object,
        tokenizer_loader: Callable[[], TokenizerLike | None],
        emit: Callable[[list[DuplexEvent]], None],
        runtime_config: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        self._plugin = plugin
        self._tokenizer_loader = tokenizer_loader
        self._tokenizer: TokenizerLike | None = None
        self._tokenizer_loaded = False
        self._special_ids: set[int] = set()
        self._emit = emit
        self._runtime_config = runtime_config
        self._cursors: dict[int, dict[str, object]] = {}
        self._prompt_emitted = False

    @staticmethod
    def on_arrival(output: object) -> object:
        """The output as the trace reads it later, in mailbox order: its token id lists copied now.

        A stage output's ``prompt_token_ids`` and its completions' ``token_ids``
        / ``cumulative_token_ids`` can alias the stage's live lists, which may
        hold the next unit's ids by the time the clock takes the output.
        """
        try:
            view = copy.copy(output)
            prompt = getattr(output, "prompt_token_ids", None)
            if isinstance(prompt, list):
                setattr(view, "prompt_token_ids", list(prompt))
            completions = getattr(output, "outputs", None)
            if isinstance(completions, list):
                copies = [copy.copy(completion) for completion in completions]
                for completion in copies:
                    for name in ("token_ids", "cumulative_token_ids"):
                        if isinstance(getattr(completion, name, None), list):
                            setattr(completion, name, list(getattr(completion, name)))
                setattr(view, "outputs", copies)
            return view
        except Exception:
            logger.debug("token trace: stage output %s not copied", type(output).__name__, exc_info=True)
            return output

    # ------------------------------------------------------------------ #
    # Collection                                                         #
    # ------------------------------------------------------------------ #

    def on_stage_segment(
        self, unit: InputClockUnit | None, *, stage_id: int, output: object, context: DuplexOutputContext | None
    ) -> None:
        """A stage ended its segment for ``unit``: collect the ids it consumed and produced.

        With ``unit`` None (a segment no unit is credited with) the hook still
        runs, so its per-stage cursor advances, and the ids are dropped.
        """
        if output is None:
            return
        cursor = self._cursors.setdefault(stage_id, {})
        runtime_config = self._runtime_config() if self._runtime_config is not None else {}
        hook = getattr(self._plugin, "unit_token_trace", None)
        try:
            trace = (
                hook(stage_id=stage_id, output=output, context=context, cursor=cursor, runtime_config=runtime_config)
                if callable(hook)
                else default_unit_token_trace(stage_id=stage_id, output=output, cursor=cursor)
            )
        except Exception:
            # A debug aid must never take the session down with it.
            logger.exception("token trace: unit_token_trace failed for stage %s", stage_id)
            return
        if not trace or unit is None:
            return
        trace = dict(trace)
        prompt_input = trace.pop("prompt_input", None)
        if isinstance(prompt_input, list) and prompt_input:
            unit.prompt_trace = {"stage": trace.get("stage", f"stage{stage_id}"), "input": prompt_input}
        if trace.get("input") or trace.get("output"):
            unit.traces.append(trace)

    # ------------------------------------------------------------------ #
    # Rendering                                                          #
    # ------------------------------------------------------------------ #

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer_loaded:
            return
        self._tokenizer_loaded = True
        try:
            tokenizer = self._tokenizer_loader()
            special: set[int] = set()
            if tokenizer is not None:
                special = {int(i) for i in getattr(tokenizer, "all_special_ids", None) or ()}
                get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
                if callable(get_added_vocab):
                    special.update(int(i) for i in get_added_vocab().values())
        except Exception:
            logger.warning("token trace: no tokenizer; ids are sent without their pieces", exc_info=True)
            tokenizer, special = None, set()
        self._tokenizer = tokenizer
        self._special_ids = special

    def _render(self, items: list[object], special: set[int]) -> list[list[object]]:
        """``[[id, text], ...]``: plugin-labelled pairs pass through, bare ids get the tokenizer piece."""
        bare = sorted({item for item in items if isinstance(item, int)})
        pieces: dict[int, str] = {}
        if bare and self._tokenizer is not None:
            try:
                names = self._tokenizer.convert_ids_to_tokens(bare)
                pieces = {token_id: str(name) for token_id, name in zip(bare, names, strict=False)}
            except Exception:
                # Ids the tokenizer cannot name (e.g. outside its vocabulary): sent with their number.
                logger.warning("token trace: the tokenizer could not name ids %s", bare[:8], exc_info=True)
        rendered: list[list[object]] = []
        for item in items:
            if isinstance(item, int):
                rendered.append([item, pieces.get(item, str(item))])
                if item in self._special_ids:
                    special.add(item)
            elif isinstance(item, list | tuple) and len(item) == 2:
                try:
                    rendered.append([int(item[0]), str(item[1])])
                except (TypeError, ValueError):
                    continue
        return rendered

    def _stage_payload(self, trace: Mapping[str, object], special: set[int]) -> dict[str, object]:
        payload: dict[str, object] = {"stage": str(trace.get("stage"))}
        for key in ("input", "output"):
            items = trace.get(key)
            if isinstance(items, list) and items:
                payload[key] = self._render(items, special)
        return payload

    def emit_unit(self, unit: InputClockUnit, *, end_ms: int) -> None:
        self._ensure_tokenizer()
        events: list[DuplexEvent] = []
        if not self._prompt_emitted and unit.prompt_trace is not None:
            self._prompt_emitted = True
            prompt_special: set[int] = set()
            events.append(
                UnitTokens(
                    unit_index=-1,
                    end_ms=0,
                    decision="prompt",
                    stages=(self._stage_payload(unit.prompt_trace, prompt_special),),
                    special_ids=tuple(sorted(prompt_special)),
                )
            )
        special: set[int] = set()
        stages = tuple(self._stage_payload(trace, special) for trace in unit.traces)
        events.append(
            UnitTokens(
                unit_index=unit.index,
                end_ms=end_ms,
                decision=unit.decision or "unknown",
                stages=stages,
                special_ids=tuple(sorted(special)),
            )
        )
        self._emit(events)


__all__ = [
    "TRACE_KEY",
    "TokenTracer",
    "check_token_trace_allowed",
    "check_token_trace_unchanged",
    "load_trace_tokenizer",
    "token_trace_enabled",
]
