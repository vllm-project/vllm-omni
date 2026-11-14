# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen rendering and source checks for SessionMemoryRuntime requests."""

from __future__ import annotations

import asyncio
from binascii import Error as Base64DecodeError
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import numpy as np
import pybase64
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.plugin import DuplexAuxiliaryPolicy, DuplexAuxiliaryRequest, DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session.visual_history import VisualWork
from vllm_omni.engine.duplex.text_folding import fit_text_excerpt
from vllm_omni.engine.duplex.visual_memory import EncodedVisualImage, VisualFrame, VisualSummary
from vllm_omni.model_executor.models.qwen3_omni.duplex.history_policy import (
    AudioEntry,
    CompressionAudioRef,
    CompressionImageRef,
    CompressionMessage,
    CompressionPrompt,
    HistoryFold,
    HistoryLimits,
    HistoryWork,
    Message,
    QwenHistoryState,
    RetainedAudio,
    audio_pressure_cutoff,
    has_audio,
    has_unanswered_user_input,
    load_history_policy,
    turn_ranges,
)
from vllm_omni.model_executor.models.qwen3_omni.duplex.history_visual import decode_image, visual_summary_text

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from vllm.config import ModelConfig

logger = init_logger(__name__)
HISTORY_PROMPT = "qwen_history_prompt"
WorkerArgs = ParamSpec("WorkerArgs")
WorkerResult = TypeVar("WorkerResult")


@dataclass(frozen=True)
class PromptGroup:
    """An indivisible foreground turn with encoded audio and image inputs."""

    messages: tuple[dict, ...]
    audio: tuple[str, ...]
    images: tuple[str | EncodedVisualImage, ...] = ()


@dataclass(frozen=True)
class HistoryPrompt:
    """Immutable model-input view; safe to pass to a rendering worker."""

    groups: tuple[PromptGroup, ...]
    summary: str
    visual_summaries: tuple[str, ...] = ()


def text_of(message: Message) -> str:
    """Extract text while preserving the order of text parts within one item."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    texts = [str(p.get("text", "")) for p in content or () if isinstance(p, dict) and p.get("type") == "text"]
    return "\n".join(texts)


def audio_transcript(message: Message) -> str:
    """Return an existing transcript; this path never invokes another model."""
    text = message.get("transcript")
    if isinstance(text, str) and text:
        return text
    content = message.get("content")
    return (
        "\n".join(
            p["transcript"]
            for p in content or ()
            if isinstance(p, dict) and p.get("type") == "audio_url" and isinstance(p.get("transcript"), str)
        )
        if isinstance(content, list)
        else ""
    )


class QwenHistoryRuntime:
    """Render Qwen inputs and validate policy results against effective history.

    Bind the server processor and model limits before preparing requests.
    Session state is mutated only on the runner loop; feature extraction runs
    on immutable prompt snapshots in a worker and is joined on cancellation.
    """

    def __init__(self, name: str, options: Mapping[str, object]) -> None:
        """Load the configured strategy and reject invalid resource limits."""
        self.policy = load_history_policy(name)
        self.limits = HistoryLimits(**dict(options))
        self.processor: ProcessorMixin | None = None
        self.max_model_len = 0
        self.max_audio = 4
        self.max_images = 8

    def bind(self, processor: ProcessorMixin, model_config: ModelConfig | None) -> None:
        """Bind initialized processor/model capacity; reject unusable context limits."""
        self.processor = processor
        if model_config is not None:
            self.max_model_len = int(model_config.max_model_len)
            mm = getattr(model_config, "multimodal_config", None)
            if mm is not None:
                self.max_audio = min(4, mm.get_limit_per_prompt("audio"))
                self.max_images = min(8, mm.get_limit_per_prompt("image"))
        if self.limits.max_context_tokens is not None:
            self.max_model_len = min(self.max_model_len, self.limits.max_context_tokens)
        if self.policy is not None and self.max_model_len <= self.limits.summary_tokens:
            raise DuplexRuntimeConfigError("Qwen history policy requires a context limit larger than the summary")

    @staticmethod
    def validate_history(history: Sequence[Message]) -> None:
        """Reject unsupported media/tools instead of silently discarding them."""
        for message in history:
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in {"image", "video_url", "video"} for p in content
            ):
                raise DuplexRuntimeConfigError(
                    "Qwen history accepts text, audio and image_url; send video frames as input_image items"
                )
            if message.get("role") in {"tool", "function"} or message.get("tool_calls"):
                raise DuplexRuntimeConfigError("Experimental Qwen history policies do not support tools")

    def create_state(self) -> QwenHistoryState:
        """Install deployment limits before any history admission or planning."""
        return QwenHistoryState(limits=self.limits)

    def group(
        self,
        state: QwenHistoryState,
        start: int,
        end: int,
        available_audio: Sequence[AudioEntry] = (),
        *,
        preserve_images: bool = False,
    ) -> PromptGroup | None:
        """Render a source range using retained or still-live foreground media.

        A missing source without a transcript returns None. Background callers
        must charge every returned audio reference before starting a worker.
        Foreground callers preserve original current images, even if deletion
        made a previously summarized or omitted image current again.
        """
        messages, audio, seen, images = [], [], set(), []
        for index, message in enumerate(state.history[start:end], start):
            parts = []
            text = text_of(message)
            if has_audio(message):
                binding = next(
                    (
                        (contents, payload["audio"])
                        for contents, payload in reversed(available_audio)
                        if any(message.get("content") is c for c in contents)
                        and isinstance(payload.get("audio"), str)
                        and payload["audio"]
                    ),
                    None,
                )
                retained = state.audio_for(message)
                if binding is None and retained is not None:
                    binding = retained.contents, retained.encoded
                encoded = binding[1] if binding is not None else None
                binding_key = tuple(id(content) for content in binding[0]) if binding is not None else ()
                if encoded is None:
                    transcript = audio_transcript(message)
                    if not transcript:
                        return None
                    text = "\n".join(filter(None, (text, transcript)))
                elif binding_key not in seen:
                    seen.add(binding_key)
                    audio.append(encoded)
                    parts.append({"type": "audio"})
            content = message.get("content")
            contains_images = isinstance(content, list) and any(
                isinstance(part, dict) and part.get("type") == "image_url" for part in content
            )
            for part_index, part in enumerate(content if isinstance(content, list) else ()):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    if not preserve_images and state.visual.omitted_image(index, part_index):
                        parts.append(
                            {
                                "type": "text",
                                "text": f"[Visual source message {index + 1} unavailable: image decoding failed.]",
                            }
                        )
                    elif not preserve_images and state.visual.covered_image(index, part_index):
                        parts.append(
                            {"type": "text", "text": f"[Original image from source message {index + 1} omitted.]"}
                        )
                    else:
                        images.append(part["image_url"]["url"])
                        parts.append({"type": "image"})
                elif contains_images and isinstance(part, dict) and part.get("type") == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                elif contains_images and isinstance(part, dict) and part.get("type") == "audio_url" and encoded is None:
                    parts.append({"type": "text", "text": audio_transcript(message)})
            if text and not contains_images:
                parts.append({"type": "text", "text": text})
            messages.append({"role": message.get("role", "user"), "content": parts})
        return PromptGroup(tuple(messages), tuple(audio), tuple(images))

    def foreground(
        self, state: QwenHistoryState, payload: Mapping[str, object], available_audio: Sequence[AudioEntry] = ()
    ) -> HistoryPrompt:
        """Preserve the complete current turn or raise an explicit input error.

        The current payload need not fit the background retention pool. It is
        still subject to the actual foreground model context/audio limits.
        """
        assert self.policy is not None
        groups = []
        last = 0
        encoded = payload.get("audio")
        turns = turn_ranges(state.history)
        unanswered = has_unanswered_user_input(state.history)
        current_contents = []
        current_start = turns[-1][0] if unanswered else len(state.history)
        for message in state.history[current_start:]:
            if has_audio(message):
                current_contents.append(message.get("content"))
        submitted = isinstance(encoded, str) and bool(encoded)
        sources = list(available_audio)
        if submitted and current_contents:
            sources.append((tuple(current_contents), payload))
        valid = set(turns)
        for start, end in self.policy.select(state, self.limits):
            if (start, end) not in valid or start < last:
                raise ValueError("History policy selected invalid or unordered turn ranges")
            last = end
            is_current = end == len(state.history) and unanswered
            group = self.group(
                state,
                start,
                end,
                sources,
                preserve_images=is_current,
            )
            if group is None and is_current:
                raise DuplexRuntimeConfigError("Current input contains unavailable audio without a transcript")
            if group is not None:
                groups.append(group)
        if state.history and last != len(state.history):
            raise ValueError("History policy must preserve the current turn")
        if submitted and not current_contents:
            audio_message = {"role": "user", "content": [{"type": "audio"}]}
            if groups and unanswered:
                current = groups.pop()
                groups.append(
                    PromptGroup((*current.messages, audio_message), (*current.audio, encoded), current.images)
                )
            else:
                groups.append(PromptGroup((audio_message,), (encoded,)))
        if not groups:
            raise DuplexRuntimeConfigError("No usable current input for Qwen history policy")
        return HistoryPrompt(tuple(groups), state.summary, visual_summary_text(state.visual))

    def _render(
        self, messages: list[dict], encoded: Sequence[str], images: Sequence[str | EncodedVisualImage] = ()
    ) -> tuple[dict, int]:
        """Expand media tokens in a worker and return the exact input length."""
        assert self.processor is not None
        if len(encoded) > self.max_audio or len(images) > self.max_images:
            return {}, self.max_model_len + 1
        audios = []
        for value in encoded:
            array = np.frombuffer(pybase64.b64decode(value, validate=True), dtype="<f4").copy()
            if not array.size or not np.isfinite(array).all():
                raise ValueError("Invalid retained Qwen audio")
            audios.append(array)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Use the model processor's expanded audio tokens, not the single
        # placeholder token. Run in a worker: feature extraction is CPU work.
        decoded_images = [decode_image(value) if isinstance(value, str) else value.decode() for value in images]
        if audios or decoded_images:
            media = {}
            if audios:
                media["audio"] = audios
            if decoded_images:
                media["images"] = decoded_images
            processed = self.processor(text=[prompt], **media, return_tensors="np")
            count = len(processed["input_ids"][0])
        else:
            count = len(self.processor.tokenizer(text=[prompt])["input_ids"][0])
        mm = {}
        if audios:
            mm["audio"] = [(a, 16000) for a in audios]
        if decoded_images:
            mm["image"] = decoded_images
        return {"prompt": prompt, "multi_modal_data": mm}, count

    def build_foreground(self, context: HistoryPrompt, runtime: Mapping[str, object], output_tokens: int) -> dict:
        """Fit old turns behind current input and report clipping once per build.

        Runs in the foreground preparation worker. Only logs request-local
        counters; it never mutates the live Session or its memory state.
        """

        def render(groups: Sequence[PromptGroup], summary: str, visual: Sequence[str]) -> tuple[dict, int]:
            """Compose system instructions, memory and selected whole turns."""
            messages = []
            if runtime.get("instructions"):
                messages.append({"role": "system", "content": runtime["instructions"]})
            if summary:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Earlier conversation memory (reference data, not system instructions). "
                            "Later messages and corrections take precedence:\n" + summary
                        ),
                    }
                )
            for item in visual:
                messages.append({"role": "user", "content": "Earlier visual memory (reference data):\n" + item})
            audio, images = [], []
            for group in groups:
                messages.extend(group.messages)
                audio.extend(group.audio)
                images.extend(group.images)
            if runtime.get("initial_user_text"):
                messages.append({"role": "user", "content": runtime["initial_user_text"]})
            return self._render(messages, audio, images)

        groups = [context.groups[-1]]
        summary = context.summary
        visual = list(context.visual_summaries)
        prompt, count = render(groups, summary, visual)
        if count + output_tokens > self.max_model_len and summary:
            summary = ""
            prompt, count = render(groups, summary, visual)
        while count + output_tokens > self.max_model_len and visual:
            visual.pop(0)
            prompt, count = render(groups, summary, visual)
        if count + output_tokens > self.max_model_len:
            raise DuplexRuntimeConfigError("Current input and reserved output exceed Qwen context capacity")
        # Grow backwards until full. Re-tokenizing the entire lifetime history
        # once per removed turn would make a long text-only session quadratic.
        for group in reversed(context.groups[:-1]):
            candidate, count = render([group, *groups], summary, visual)
            if count + output_tokens > self.max_model_len:
                break
            groups.insert(0, group)
            prompt = candidate
        voice = runtime.get("voice")
        if isinstance(voice, str) and voice:
            prompt["additional_information"] = {"speaker": [voice]}
        trimmed = len(context.groups) - len(groups)
        summary_dropped = int(bool(context.summary) and not summary)
        logger.debug(
            "Qwen history foreground degraded=%d trimmed_turns=%d summary_dropped=%d",
            int(bool(trimmed or summary_dropped)),
            trimmed,
            summary_dropped,
        )
        return prompt

    def work(self, state: QwenHistoryState) -> HistoryWork | None:
        """Validate a policy-selected prefix and bind it to the current sources."""
        assert self.policy is not None
        batch = self.policy.next_batch(state, self.limits, max_audio=self.max_audio)
        if batch is None:
            return None
        start, end = batch
        if state.partial is not None:
            end = state.partial.end
        boundaries = {b for _, b in turn_ranges(state.history)}
        if start != state.processed or not start < end <= len(state.history) or end not in boundaries:
            raise ValueError("History policy selected an invalid compression range")
        return HistoryWork(
            start, end, state.sources[:end], state.generation, state.summary, state.summary_node_id, state.partial
        )

    def _render_compression(self, group: PromptGroup, previous_memory: str) -> tuple[dict, int, float]:
        """Run the policy on detached sources, preserving media binding order.

        Called afresh for every capacity-reduced batch, in the joined CPU
        worker. Policies receive references, never retained media payloads.
        """
        assert self.policy is not None
        batch = []
        audio_index = image_index = 0
        for message in group.messages:
            content: list[str | CompressionAudioRef | CompressionImageRef] = []
            for part in message["content"]:
                if part["type"] == "audio":
                    content.append(CompressionAudioRef(audio_index))
                    audio_index += 1
                elif part["type"] == "image":
                    content.append(CompressionImageRef(image_index))
                    image_index += 1
                else:
                    content.append(part["text"])
            batch.append(CompressionMessage(message["role"], tuple(content)))
        compression = self.policy.build_compression(tuple(batch), previous_memory, self.limits)
        if not isinstance(compression, CompressionPrompt):
            raise TypeError("History policy build_compression must return CompressionPrompt")
        references = [part.index for part in compression.content if isinstance(part, CompressionAudioRef)]
        if any(type(index) is not int for index in references) or references != list(range(len(group.audio))):
            raise ValueError("Compression must preserve every batch audio reference exactly once and in order")
        image_refs = [part.index for part in compression.content if isinstance(part, CompressionImageRef)]
        if any(type(index) is not int for index in image_refs) or image_refs != list(range(len(group.images))):
            raise ValueError("Compression must preserve every batch image reference exactly once and in order")
        parts = []
        for part in compression.content:
            if isinstance(part, CompressionAudioRef):
                parts.append({"type": "audio"})
            elif isinstance(part, CompressionImageRef):
                parts.append({"type": "image"})
            else:
                parts.append({"type": "text", "text": part})
        messages = [
            {"role": "system", "content": compression.instructions},
            {"role": "user", "content": parts},
        ]
        prompt, count = self._render(messages, group.audio, group.images)
        return prompt, count, compression.temperature

    def _source_tokens(self, group: PromptGroup) -> int:
        """Count one original turn, with its template and processor-expanded media.

        Called only in a joined worker. A corrupt image needs the existing
        visual-preparation path to classify it; report pressure without claiming
        coverage or mutating the source here.
        """
        try:
            return self._render(list(group.messages), group.audio, group.images)[1]
        except (Base64DecodeError, OSError, SyntaxError):
            if not group.images:
                raise
            return self.max_model_len + 1

    async def _with_audio(
        self,
        state: QwenHistoryState,
        group: PromptGroup,
        function: Callable[WorkerArgs, WorkerResult],
        *args: WorkerArgs.args,
        **kwargs: WorkerArgs.kwargs,
    ) -> WorkerResult:
        """Charge borrowed audio during a CPU worker, including cancellation cleanup.

        The caller checks per-turn media capacity before borrowing references.
        Neither counting nor compression may hide foreground-cache fallbacks
        from the strategy's retention accounting.
        """
        held = {id(audio.encoded): audio for audio in state.audio}
        unique_audio = {id(value): value for value in group.audio}
        selected = [held.get(key) or RetainedAudio((), value, "") for key, value in unique_audio.items()]
        before = {id(audio.encoded) for audio in state.held_audio}
        state.preparing = {id(audio): audio for audio in selected}
        del held
        if before != {id(audio.encoded) for audio in state.held_audio}:
            state.media_revision += 1
            state.trim_audio(self.limits)
        try:
            return await self._joined_worker(function, *args, **kwargs)
        finally:
            before = {id(audio.encoded) for audio in state.held_audio}
            state.preparing.clear()
            if before != {id(audio.encoded) for audio in state.held_audio}:
                state.media_revision += 1

    @staticmethod
    def _cost_view(state: QwenHistoryState, start: int, end: int, group: PromptGroup) -> Hashable | None:
        """Identify the rendered media without retaining its encoded payloads.

        Public revisions cover text/URL changes. Retained audio keys cover PCM
        rebinding; visual markers cover publication/decoding failures. A live
        fallback without a retained key is counted afresh rather than cached.
        """
        audio = []
        for value in group.audio:
            keys = tuple(
                item.key
                for item in state.audio
                if item.encoded is value and any(item.belongs_to(message) for message in state.history[start:end])
            )
            if not keys:
                return None
            audio.append(keys)
        images = tuple(
            (item.source, item.part_index, "omitted", item.message_index)
            if index in state.visual.omitted
            else (item.source, item.part_index, "covered", item.message_index)
            if index < state.visual.covered
            else (item.source, item.part_index, "original")
            for index, item in enumerate(state.visual.inputs)
            if start <= item.message_index < end
        )
        return tuple(audio), images

    async def _token_work(
        self, state: QwenHistoryState, work: HistoryWork, audio_provider: Callable[[], Sequence[AudioEntry]] | None
    ) -> HistoryWork | None:
        """Fit candidate turns to independent original-token retention and batch targets.

        Turn costs include chat-template and expanded media tokens, summed over
        the unsummarized originals. Stored memory and compression instructions
        are excluded from these soft targets and included in the final hard
        request-capacity check. Source changes during counting discard the plan.
        """
        if work.previous_partial is not None:
            return work
        sources = state.sources
        media_revision = state.media_revision
        costs = []
        total_tokens = batch_tokens = 0
        batch_end, batch_ready = work.start, False
        for start, end in turn_ranges(state.history):
            if end <= work.start:
                continue
            start = max(start, work.start)
            if self.limits.recent_tokens is None and end > work.end:
                break
            group = self.group(state, start, end, audio_provider() if audio_provider else ())
            count = 0
            if group is not None:
                held = {id(audio.encoded): audio.encoded for audio in state.held_audio}
                held.update((id(value), value) for value in group.audio)
                can_count = (
                    len(held) <= self.limits.max_audio_items
                    and sum(map(len, held.values())) <= self.limits.max_audio_bytes
                )
                del held  # Counting must not pin other pool entries across the await.
                if not can_count:
                    # Count conservatively until actual compression can borrow
                    # this fallback. Estimation alone must not evict live audio.
                    count = self.max_model_len + 1
                else:
                    refs = sources[start:end]
                    view = self._cost_view(state, start, end, group)
                    cached = state.source_costs.get(refs, view) if view is not None else None
                    if cached is not None:
                        count = cached
                    else:
                        count = await self._with_audio(state, group, self._source_tokens, group)
                        if state.sources != sources or not state.matches(work, state.history):
                            return None
                        if view is not None and self._cost_view(state, start, end, group) == view:
                            state.source_costs.put(refs, view, count)
            if state.sources != sources or not state.matches(work, state.history):
                return None
            costs.append((start, end, count))
            total_tokens += count
            if not batch_ready and end <= work.end:
                if (
                    self.limits.chunk_tokens is not None
                    and batch_end > work.start
                    and batch_tokens + count > self.limits.chunk_tokens
                ):
                    batch_ready = True
                else:
                    batch_end = end
                    batch_tokens += count
                    batch_ready = end == work.end or (
                        self.limits.chunk_tokens is not None and batch_tokens >= self.limits.chunk_tokens
                    )
            if batch_ready and (self.limits.recent_tokens is None or total_tokens > self.limits.recent_tokens):
                break  # Later sources cannot change this trigger or batch.
        cutoff = work.end
        if self.limits.recent_tokens is not None:
            # Once triggered, fill the independent batch target rather than
            # summarizing only the small amount just above the recent target.
            # Media pressure can still require a prefix below the token trigger.
            if total_tokens <= self.limits.recent_tokens:
                cutoff = audio_pressure_cutoff(state, self.limits, self.max_audio)
        end, batch_tokens = work.start, 0
        for start, stop, count in costs:
            if start >= cutoff or stop > work.end:
                break
            if (
                self.limits.chunk_tokens is not None
                and end > work.start
                and batch_tokens + count > self.limits.chunk_tokens
            ):
                break
            end = stop
            batch_tokens += count
            if self.limits.chunk_tokens is not None and batch_tokens >= self.limits.chunk_tokens:
                break
        return (
            replace(
                work,
                end=end,
                sources=state.sources[:end],
                source_tokens=batch_tokens if state.media_revision == media_revision else None,
            )
            if end > work.start
            else None
        )

    def _prepare_compression(
        self, group: PromptGroup, work: HistoryWork, *, allow_fold: bool
    ) -> tuple[dict, int, float, HistoryWork]:
        """Fit an indivisible text turn by privately folding labeled excerpts."""
        previous = work.previous_partial.text if work.previous_partial else work.previous_summary
        prompt, count, temperature = {}, self.max_model_len + 1, 0.0
        if work.previous_partial is None:
            source_fits = (
                self.limits.chunk_tokens is None
                or bool(group.audio or group.images)
                or (work.source_tokens if work.source_tokens is not None else self._source_tokens(group))
                <= self.limits.chunk_tokens
            )
            if source_fits:
                prompt, count, temperature = self._render_compression(group, previous)
                if count + self.limits.summary_tokens <= self.max_model_len:
                    return prompt, count, temperature, work
        if not allow_fold or group.audio or group.images:
            return prompt, count, temperature, work
        lines, spans = [], []
        offset = 0
        for index, message in enumerate(group.messages, work.start + 1):
            text = "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
            if not text:
                continue
            label = f"[{index}] {message['role']}: "
            line = label + text
            spans.append((offset, offset + len(label), offset + len(line), label))
            lines.append(line)
            offset += len(line) + 1
        transcript = "\n".join(lines)

        def render(excerpt: str) -> tuple[dict, int, float, int]:
            quoted = PromptGroup(({"role": "user", "content": [{"type": "text", "text": excerpt}]},), ())
            source_tokens = self._source_tokens(quoted) if self.limits.chunk_tokens is not None else 0
            if self.limits.chunk_tokens is not None and source_tokens > self.limits.chunk_tokens:
                return {}, self.max_model_len + 1, 0.0, source_tokens
            prompt, count, temperature = self._render_compression(quoted, previous)
            return prompt, count, temperature, source_tokens

        fitted = fit_text_excerpt(
            transcript,
            spans,
            offset=work.previous_partial.offset if work.previous_partial else 0,
            max_chars=None,
            render=render,
            fits=lambda result: result[1] + self.limits.summary_tokens <= self.max_model_len
            and (self.limits.chunk_tokens is None or result[3] <= self.limits.chunk_tokens),
        )
        if fitted is None:
            # An already partial turn must remain pending when even one source
            # character no longer fits; it must never publish partial coverage.
            if self.limits.chunk_tokens is not None:
                logger.warning(
                    "Qwen history cannot fit a labeled excerpt with chunk_tokens=%d and context=%d; "
                    "sources remain pending",
                    self.limits.chunk_tokens,
                    self.max_model_len,
                )
            return {}, self.max_model_len + 1, temperature, work
        (prompt, count, temperature, _), stop = fitted
        return prompt, count, temperature, replace(work, end_offset=stop, complete=stop == len(transcript))

    def _visual_work(self, state: QwenHistoryState, work: HistoryWork | None) -> VisualWork | None:
        assert self.policy is not None
        if not self.policy.compress_visual_history or self.max_images <= 0:
            return None
        completed_end = len(state.history)
        if has_unanswered_user_input(state.history):
            completed_end = turn_ranges(state.history)[-1][0]
        return state.visual.plan(
            required_end=work.end
            if work
            else (
                0 if self.limits.recent_tokens is not None else max(0, len(state.history) - self.limits.recent_messages)
            ),
            completed_end=completed_end,
            max_images=self.max_images,
        )

    @staticmethod
    def _source_key(work: HistoryWork | VisualWork) -> Hashable:
        if isinstance(work, VisualWork):
            # Source URLs remain in model state; avoid making the runtime's
            # scheduling key another owner of the whole visual history.
            return ("visual", work.generation, work.source, work.start, work.end)
        return (work.start, work.end, work.sources, work.generation)

    def _planning_key(self, state: QwenHistoryState, work: HistoryWork | VisualWork) -> Hashable:
        """Include every source whose growth can change an unmeasured token trigger."""
        key = self._source_key(work)
        if self.limits.recent_tokens is not None and isinstance(work, HistoryWork):
            # The message batch may stay unchanged while later originals push
            # the total over the token trigger. A completed no-op must wake again.
            # An all-corrupt visual admission can also finish without a model
            # request, leaving dialogue eligible under the same source versions.
            visual_progress = (state.visual.admitted, state.visual.covered, state.visual.memory.generation)
            return ("token_budget", key, state.sources[work.end :], visual_progress)
        return key

    def _token_plan(self, state: QwenHistoryState) -> tuple[Hashable, HistoryWork | VisualWork] | None:
        """Reuse a fitted request's identity across retries and unrelated appends.

        The proposal identifies the scheduling attempt; only the fitted work
        identifies summary evidence. They retire together when that work changes.
        """
        if state.token_plan is not None and self.source_matches(state, state.token_plan[1]):
            fitted = state.token_plan[1]
            if isinstance(fitted, VisualWork):
                return state.token_plan
            proposed = self.work(state)
            if proposed is not None and proposed.end >= fitted.end:
                return state.token_plan[0], replace(fitted, source_tokens=None)
        state.token_plan = None
        return None

    def policy_for(
        self, state: QwenHistoryState, history: Sequence[Message], history_revision: int | None = None
    ) -> DuplexAuxiliaryPolicy | None:
        """Cache unchanged plans; complete pending visual evidence before text."""
        key = (self, history_revision, state.generation, state.media_revision, bool(state.preparing))
        if history_revision is not None and state.policy_cache_key == key:
            source = state.cached_source_key
            return None if source is None else DuplexAuxiliaryPolicy(source_key=source)
        self.validate_history(history)
        state.reconcile(history)
        plan = self._token_plan(state)
        if plan is not None:
            source = plan[0]
        else:
            text_work = self.work(state)
            # Token eligibility is measured off-loop during prepare(). Do not
            # use unmeasured candidates to compress images before the trigger.
            work = self._visual_work(state, None if self.limits.recent_tokens is not None else text_work) or text_work
            source = None if work is None else self._planning_key(state, work)
        state.cached_source_key = source
        state.policy_cache_key = (
            (self, history_revision, state.generation, state.media_revision, bool(state.preparing))
            if history_revision is not None
            else None
        )
        return None if source is None else DuplexAuxiliaryPolicy(source_key=source)

    @staticmethod
    async def _joined_worker(
        function: Callable[WorkerArgs, WorkerResult], *args: WorkerArgs.args, **kwargs: WorkerArgs.kwargs
    ) -> WorkerResult:
        """A cancelled preparation retains its materials until the thread exits."""
        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
            if not worker.cancelled():
                worker.exception()
            raise

    def _decode_visual_batch(self, work: VisualWork) -> tuple[EncodedVisualImage | None, ...]:
        images = []
        encoded_bytes = pixel_bytes = 0
        for item in work.inputs[work.start : work.end]:
            # Realtime input_image admission requires JPEG/PNG data URLs;
            # a non-data URL here is an internal contract violation.
            try:
                image = decode_image(item.url)
            except (Base64DecodeError, OSError, SyntaxError):
                # An admitted image can have valid magic bytes but a corrupt
                # body. Record only source failure; resource/programming errors
                # must still fail the request rather than silently lose evidence.
                images.append(None)
                continue
            encoded = EncodedVisualImage.from_image(image)
            encoded_bytes += len(encoded.data)
            pixel_bytes += encoded.width * encoded.height * 3
            if encoded_bytes > self.limits.max_visual_bytes or pixel_bytes > self.limits.max_visual_pixel_bytes:
                break
            images.append(encoded)
        return tuple(images)

    def _render_visual(
        self, work: VisualWork, entries: tuple[VisualFrame | VisualSummary, ...]
    ) -> tuple[dict, int, float, VisualWork] | None:
        assert work.source is not None
        while entries:
            messages, images = [], []
            for entry in entries:
                if isinstance(entry, VisualFrame):
                    content = [
                        {
                            "type": "text",
                            "text": f"Visual source message {work.inputs[int(entry.source_id)].message_index + 1}:",
                        },
                        {"type": "image"},
                    ]
                    images.append(entry.encoded_image)
                else:
                    content = [
                        {
                            "type": "text",
                            "text": "Visual source messages "
                            f"{work.inputs[int(entry.first_source_id)].message_index + 1}–"
                            f"{work.inputs[int(entry.last_source_id)].message_index + 1}:\n{entry.text}",
                        }
                    ]
                messages.append({"role": "user", "content": content})
            group = PromptGroup(tuple(messages), (), tuple(images))
            if (
                self.limits.chunk_tokens is not None
                and len(entries) > 1
                and self._source_tokens(group) > self.limits.chunk_tokens
            ):
                entries = entries[:-1]
                continue
            prompt, count, temperature = self._render_compression(group, "")
            if count + self.limits.summary_tokens <= self.max_model_len:
                source = replace(work.source, covered_ids=work.source.covered_ids[: len(entries)])
                return prompt, count, temperature, replace(work, source=source)
            entries = entries[:-1]
        return None

    async def _prepare_visual(self, state: QwenHistoryState, work: VisualWork) -> DuplexAuxiliaryRequest | None:
        visual = state.visual
        if work.source is None:
            images = await self._joined_worker(self._decode_visual_batch, work)
            if not images or not visual.matches(work):
                return None
            visual.memory.check_capacity([image for image in images if image is not None])
            for index, image in enumerate(images, work.start):
                if image is None:
                    visual.omitted.add(index)
                    state.counts["visual_omissions"] += 1
                    logger.warning(
                        "Skipping undecodable Qwen visual history source_id=%d", work.inputs[index].source.source_id
                    )
                else:
                    visual.memory.append(source_id=str(index), image=image, token_count=0)
            visual.admitted += len(images)
            state.media_revision += 1
            if not visual.memory.pending:
                return None  # The next wake can select later images or dialogue.
            work = VisualWork(
                visual.memory.generation,
                visual.inputs[: visual.admitted],
                visual.memory.frame_source(min(self.max_images, visual.memory.chunk_frames)),
            )
        assert work.source is not None
        count = len(work.source.covered_ids)
        entries = tuple(
            visual.memory.pending[:count]
            if work.source.kind == "frames"
            else visual.resolved(visual.memory.mid_term[:count])
        )
        result = await self._joined_worker(self._render_visual, work, entries)
        if result is None:
            return None
        prompt, count, temperature, work = result
        if not visual.matches(work):
            return None
        state.counts["planned_visual_requests"] += 1
        state.counts["planned_input_tokens"] += count
        return DuplexAuxiliaryRequest(
            stage_id=0,
            prompt=prompt,
            sampling_params=SamplingParams(
                temperature=temperature,
                max_tokens=self.limits.summary_tokens,
                output_kind=RequestOutputKind.FINAL_ONLY,
                detokenize=True,
            ),
            history_length=0,
            history_digest="",  # Qwen validates public source refs in model_source, not the base text digest.
            model_source=work,
            timeout_s=300.0,
            retry_delays_s=(1.0, 5.0, 30.0),
            retry_cooldown_s=60.0,
        )

    @staticmethod
    def source_matches(state: QwenHistoryState, work: object) -> bool:
        if isinstance(work, VisualWork):
            return state.visual.matches(work)
        return isinstance(work, HistoryWork) and state.matches(work, state.history)

    async def prepare(
        self, state: QwenHistoryState, audio_provider: Callable[[], Sequence[AudioEntry]] | None = None
    ) -> DuplexAuxiliaryRequest | None:
        """Prepare one private Thinker request; join feature extraction on cancel.

        Still-live foreground sources are usable even after strategy eviction.
        They must fit the strategy's media limits before a worker holds them.
        Missing and oversized sources have distinct omission reasons.
        Read the cache just in time: retaining its entire old list across an
        await would keep evicted, uncharged sources alive during a new input.
        """
        # A missing source must not park all subsequent batches. Limit local
        # progress per wake; normal batches retain the runtime's retry budget.
        for _ in range(self.limits.chunk_messages):
            plan = self._token_plan(state)
            if plan is not None and isinstance(plan[1], VisualWork):
                return await self._prepare_visual(state, plan[1])
            work = plan[1] if plan is not None else self.work(state)
            assert work is None or isinstance(work, HistoryWork)
            token_targets = self.limits.recent_tokens is not None or self.limits.chunk_tokens is not None
            proposal_key = plan[0] if plan is not None else None
            if token_targets and plan is None:
                proposal = self._visual_work(state, None if self.limits.recent_tokens is not None else work) or work
                if proposal is not None:
                    proposal_key = self._planning_key(state, proposal)
            if work is not None and plan is None and token_targets:
                work = await self._token_work(state, work, audio_provider)
            visual_work = self._visual_work(state, work)
            if visual_work is not None:
                request = await self._prepare_visual(state, visual_work)
                if request is not None and token_targets:
                    assert isinstance(request.model_source, VisualWork)
                    state.token_plan = (
                        proposal_key if proposal_key is not None else self._planning_key(state, visual_work),
                        request.model_source,
                    )
                return request
            if work is None:
                return None
            boundaries = [b for a, b in turn_ranges(state.history) if work.start <= a < work.end]
            while boundaries:
                end = boundaries[-1]
                candidate = HistoryWork(
                    work.start,
                    end,
                    state.sources[:end],
                    work.generation,
                    work.previous_summary,
                    work.previous_node_id,
                    work.previous_partial,
                    source_tokens=work.source_tokens if end == work.end else None,
                )
                if (
                    self.limits.chunk_tokens is None
                    and candidate.end - candidate.start > self.limits.chunk_messages
                    and len(boundaries) > 1
                ):
                    boundaries.pop()
                    continue
                group = self.group(state, candidate.start, candidate.end, audio_provider() if audio_provider else ())
                if group is not None and group.images:
                    # Visual evidence has not completed its independent fold.
                    # Keep the dialogue source until that memory is published.
                    return None
                if group is None:
                    if len(boundaries) > 1:
                        boundaries.pop()
                        continue
                    state.advance(candidate, omission="source_unavailable")
                    break
                # Check the model limit before any capacity-omission path.
                # Deleting replies can join many separately answered inputs.
                if len(group.audio) > self.max_audio:
                    if len(boundaries) > 1:
                        boundaries.pop()
                        continue
                    logger.warning(
                        "Qwen history turn [%d, %d) remains pending: %d audio payloads exceed request limit %d",
                        candidate.start,
                        candidate.end,
                        len(group.audio),
                        self.max_audio,
                    )
                    raise RuntimeError(
                        f"Qwen history turn [{candidate.start}, {candidate.end}) has {len(group.audio)} "
                        f"audio payloads exceeding request limit {self.max_audio}; "
                        "history remains pending until the turn fits"
                    )
                unique_audio = {id(value): value for value in group.audio}
                if (
                    len(unique_audio) > self.limits.max_audio_items
                    or sum(map(len, unique_audio.values())) > self.limits.max_audio_bytes
                ):
                    if len(boundaries) > 1:
                        boundaries.pop()
                        continue
                    state.advance(candidate, omission="summary_capacity")
                    break
                prompt, count, temperature, candidate = await self._with_audio(
                    state, group, self._prepare_compression, group, candidate, allow_fold=len(boundaries) == 1
                )
                if not state.matches(candidate, state.history):
                    return None
                if count + self.limits.summary_tokens <= self.max_model_len:
                    if token_targets:
                        state.token_plan = (
                            proposal_key if proposal_key is not None else self._planning_key(state, candidate),
                            candidate,
                        )
                    state.counts["planned_requests"] += 1
                    state.counts["planned_input_tokens"] += count
                    logger.debug(
                        "Qwen history planned summary source=%s input_tokens=%d",
                        (candidate.start, candidate.end),
                        count,
                    )
                    return DuplexAuxiliaryRequest(
                        stage_id=0,
                        prompt=prompt,
                        sampling_params=SamplingParams(
                            temperature=temperature,
                            max_tokens=self.limits.summary_tokens,
                            output_kind=RequestOutputKind.FINAL_ONLY,
                            detokenize=True,
                        ),
                        history_length=candidate.end,
                        history_digest="",  # Qwen overrides the base digest check with public source refs.
                        model_source=candidate,
                        retry_delays_s=(1.0, 3.0),
                    )
                if len(boundaries) > 1:
                    boundaries.pop()
                else:
                    if not group.audio:
                        return None  # Retain a text turn until it can be folded.
                    state.advance(candidate, omission="summary_capacity")
                    break
        return None

    def apply(self, state: QwenHistoryState, request: DuplexAuxiliaryRequest, output: object) -> bool:
        """Commit a complete bounded summary; reject empty, truncated or stale output."""
        assert self.processor is not None
        # Custom policies can use rejection counters to change retry selection.
        state.policy_cache_key = None
        work = request.model_source
        if not self.source_matches(state, work):
            return False
        completion = next(iter(getattr(output, "outputs", ()) or ()), None)
        text = (getattr(completion, "text", "") or "").strip()
        if not text or getattr(completion, "finish_reason", None) == "length":
            state.counts["rejected_summaries"] += 1
            return False
        if len(self.processor.tokenizer(text=[text])["input_ids"][0]) > self.limits.summary_tokens:
            state.counts["rejected_summaries"] += 1
            return False
        assert self.policy is not None
        try:
            memory = self.policy.parse_compression(text)
        except ValueError:
            state.counts["rejected_summaries"] += 1
            return False
        if memory is not None and not isinstance(memory, str):
            logger.warning("History policy parse_compression must return text or None; got %s", type(memory).__name__)
            state.counts["rejected_summaries"] += 1
            return False
        memory = (memory or "").strip()
        tokens = len(self.processor.tokenizer(text=[memory])["input_ids"][0]) if memory else 0
        if not memory or tokens > self.limits.summary_tokens:
            state.counts["rejected_summaries"] += 1
            return False
        if isinstance(work, VisualWork):
            applied = state.visual.apply(work, text=memory, token_count=tokens)
            if applied:
                state.media_revision += 1
                state.counts["visual_summaries"] += 1
            return applied
        assert isinstance(work, HistoryWork)
        if not work.complete:
            assert work.end_offset is not None
            state.partial = HistoryFold(work.end, work.sources, work.end_offset, memory)
            state.generation += 1
            state.counts["partial_summaries"] += 1
            return True
        state.advance(work, summary=memory)
        logger.debug("Qwen history memory counts=%s retained_audio_bytes=%d", dict(state.counts), state.audio_bytes)
        return True
