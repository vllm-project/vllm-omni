# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``LiveSessionProcessor``: model-specific side of a native full-duplex session.

A processor cuts the session's input audio into the model's units, renders
each unit as a streaming update of the session's resumable request, renders a
full resubmission after the history was trimmed, and reads the model's output
back into the history.
"""

from __future__ import annotations

import copy
import importlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError
from vllm_omni.entrypoints.openai.live.session import LiveSessionState, NativeUnitItem

logger = init_logger(__name__)


@dataclass
class ProcessorContext:
    """What a processor may use from the serving process."""

    # HF tokenizer, or None for stages that run with skip_tokenizer_init.
    tokenizer: Any
    max_model_len: int
    model_name: str
    model_path: str

    @property
    def raw_tokenizer(self) -> Any:
        return getattr(self.tokenizer, "tokenizer", self.tokenizer)


def load_processor_class(path: str) -> type[LiveSessionProcessor]:
    module_name, _, attr = path.partition(":") if ":" in path else path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), attr)
    if not (isinstance(cls, type) and issubclass(cls, LiveSessionProcessor)):
        raise TypeError(f"{path} is not a LiveSessionProcessor")
    return cls


class LiveSessionProcessor(ABC):
    input_sample_rate: ClassVar[int] = 16_000
    output_sample_rate: ClassVar[int] = 24_000
    # Fixed unit length for the default ``push_audio`` / ``next_unit``.
    unit_ms: ClassVar[int | None] = None
    # On context_length_exceeded the units kept fill this share of the context
    # left after the session prefix.
    trim_fraction: ClassVar[float] = 0.5

    def __init__(self, context: ProcessorContext) -> None:
        self.context = context
        self._pending: list[np.ndarray] = []
        self._pending_samples = 0
        self._pending_start_ms = 0.0

    @classmethod
    def preload(cls, model_path: str) -> None:
        """Load per-model assets once at server startup (runs in a thread)."""

    # ---- session configuration -------------------------------------------------

    def resolve_voice(self, voice: Any) -> str | None:
        """Validate ``audio.output.voice`` and return the model voice id.

        OpenAI built-in names are rejected by the handler before this runs.
        """
        if voice is None:
            return None
        if isinstance(voice, dict):
            voice = voice.get("id")
        if not isinstance(voice, str) or not voice:
            raise LiveProtocolError("invalid_value", "Invalid voice.", param="audio.output.voice")
        return voice

    def configure(self, state: LiveSessionState) -> None:
        """Called once after ``session.start`` validation, before any unit."""

    def count_tokens(self, text: str) -> int:
        return len(self.context.raw_tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return self.context.raw_tokenizer.decode(token_ids, skip_special_tokens=True)

    def sampling_params_list(self, base: Sequence[Any]) -> list[Any]:
        """The session's per-stage sampling params (stage 0 is then adjusted
        per append by ``stage0_sampling_params``)."""
        return list(base)

    def stage0_sampling_params(self, state: LiveSessionState, base: Any) -> Any:
        """Per-append stage-0 sampling params (a clone of ``base``)."""
        return base.clone() if hasattr(base, "clone") else copy.deepcopy(base)

    def prefix_sampling_params(self, state: LiveSessionState, base: Any) -> Any:
        """Stage-0 sampling params of the prefix segment (``render_prefix``)."""
        return self.stage0_sampling_params(state, base)

    # ---- input units -----------------------------------------------------------

    def push_audio(self, samples: np.ndarray, start_ms: float) -> None:
        """Input audio at ``input_sample_rate``, contiguous with the last push."""
        if self._pending_samples == 0:
            self._pending_start_ms = start_ms
        if samples.size:
            self._pending.append(np.asarray(samples, dtype=np.float32))
            self._pending_samples += samples.size

    def push_silence(self, samples: int, start_ms: float) -> None:
        """Muted input: ``samples`` samples of silence at ``input_sample_rate``."""
        self.push_audio(np.zeros(samples, dtype=np.float32), start_ms)

    def next_unit(self) -> NativeUnitItem | None:
        """The next unit ready for the model, or None if more audio is needed."""
        if self.unit_ms is None:
            raise NotImplementedError(f"{type(self).__name__} must set unit_ms or override next_unit")
        window = self.input_sample_rate * self.unit_ms // 1000
        if self._pending_samples < window:
            return None
        joined = np.concatenate(self._pending)
        chunk, rest = joined[:window], joined[window:]
        start = self._pending_start_ms
        self._pending = [rest] if rest.size else []
        self._pending_samples = rest.size
        self._pending_start_ms = start + self.unit_ms
        return NativeUnitItem(audio=chunk, start_ms=int(start), end_ms=int(start + self.unit_ms))

    # ---- rendering -------------------------------------------------------------

    def render_prefix(self, state: LiveSessionState) -> dict[str, Any] | None:
        """First prompt of a request that opens with the session prefix alone.

        The orchestrator copies a request's first prompt to the downstream
        stages on every streaming update, so a processor whose units are heavy
        keeps them out of it: the prefix segment's sampled token is discarded
        and the units follow as streaming updates. None: the first unit's
        prompt carries the prefix itself.
        """
        return None

    @abstractmethod
    def render_unit(
        self,
        state: LiveSessionState,
        unit: NativeUnitItem,
        *,
        first: bool,
        last_sampled_token: int | None,
    ) -> dict[str, Any]:
        """Engine prompt for one unit. ``first``: the request's first unit,
        which also carries the session prefix unless ``render_prefix`` opened
        the request."""

    @abstractmethod
    def render_resume(self, state: LiveSessionState) -> dict[str, Any]:
        """Every unit in the history, the last one open for generation, for a
        new request: its first prompt (with the session prefix), or the update
        after ``render_prefix`` when the processor has one."""

    # ---- context ---------------------------------------------------------------

    @abstractmethod
    def prefix_tokens(self, state: LiveSessionState) -> int:
        """Engine tokens of the session prefix (system prompt, reference voice)."""

    @abstractmethod
    def unit_tokens(self, unit: NativeUnitItem) -> int:
        """Engine tokens one unit occupies, its output included."""

    def trim_context(self, state: LiveSessionState) -> int:
        """Drop the oldest units so the rest fit ``trim_fraction`` of the
        context after the prefix; the newest unit is always kept. Returns the
        number of units dropped."""
        prefix = self.prefix_tokens(state)
        budget = (self.context.max_model_len - prefix) * self.trim_fraction
        units = state.units
        kept = 0
        used = 0
        for unit in reversed(units):
            cost = self.unit_tokens(unit)
            if kept and used + cost > budget:
                break
            used += cost
            kept += 1
        drop = {id(unit) for unit in units[: len(units) - kept]}
        logger.debug(
            "Live session %s: context trim: max_model_len=%d prefix=%d budget=%d; "
            "keeping %d units (%d tokens, %d-%d ms), dropping %d units (%d tokens)",
            state.id,
            self.context.max_model_len,
            prefix,
            int(budget),
            kept,
            used,
            units[-kept].start_ms if kept else 0,
            units[-1].end_ms if units else 0,
            len(drop),
            sum(self.unit_tokens(unit) for unit in units[: len(units) - kept]),
        )
        if drop:
            self.on_units_dropped(state, [unit for unit in units if id(unit) in drop])
            state.history = [item for item in state.history if id(item) not in drop]
        return len(drop)

    def on_units_dropped(self, state: LiveSessionState, units: list[NativeUnitItem]) -> None:
        """Hook for processors that carry dropped units forward (as text)."""

    # ---- output ------------------------------------------------------------------

    def observe_stage0_output(self, state: LiveSessionState, token_ids: list[int], output: Any) -> None:
        """Record stage-0 output of the current unit into history."""
        units = state.units
        if units:
            units[-1].token_ids.extend(token_ids)

    def spoken_text(self, token_ids: list[int], text: str) -> str:
        """The part of a stage-0 delta that is the model's speech transcript."""
        return text

    def observe_multimodal_output(self, state: LiveSessionState, output: Any) -> str:
        """Record per-stream model outputs; returns transcript text, if any."""
        return ""
