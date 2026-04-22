from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniCPMO4_5EnqueuedCondition:
    index: int
    tensor: torch.Tensor
    text_finished: bool


@dataclass
class MiniCPMO4_5ReadyAudioChunk:
    token_ids: list[int]
    condition_index: int | None
    condition_shape: list[int] | None
    text_finished: bool
    is_last_audio_chunk: bool


@dataclass
class MiniCPMO4_5PoppedToken:
    token_id: int
    condition_index: int | None
    condition_shape: list[int] | None
    text_finished: bool
    is_eos: bool = False


@dataclass
class _MiniCPMO4_5GeneratorState:
    pending_conditions: list[MiniCPMO4_5EnqueuedCondition] = field(default_factory=list)
    ready_audio_chunks: list[MiniCPMO4_5ReadyAudioChunk] = field(default_factory=list)
    current_chunk: MiniCPMO4_5ReadyAudioChunk | None = None
    current_chunk_offset: int = 0
    token_buffer: list[torch.Tensor] = field(default_factory=list)
    all_generated_tokens: list[torch.Tensor] = field(default_factory=list)
    past_key_values: Any = None
    text_start_pos: int = 0
    stream_finished: bool = False
    eos_pending: bool = False
    eos_emitted: bool = False
    current_condition_index: int | None = None
    current_condition_shape: list[int] | None = None
    current_condition_text_finished: bool = False
    last_generated_token: torch.Tensor | None = None
    step_index: int = 0
    sampling_generator: torch.Generator | None = None
    sampling_generator_device: str | None = None


def _shape_list(tensor: Any) -> list[int] | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    return [int(dim) for dim in tensor.shape]


def _cache_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    return int(past_key_values[0][0].shape[2])


def _topk_logits_summary(logits: torch.Tensor, probs: torch.Tensor, k: int) -> list[dict[str, Any]]:
    if logits.ndim != 2 or probs.ndim != 2 or logits.shape != probs.shape:
        return []
    vocab_size = int(logits.shape[-1])
    if vocab_size <= 0:
        return []
    k = max(1, min(int(k), vocab_size))
    top_vals, top_ids = torch.topk(logits, k=k, dim=-1)
    top_probs = probs.gather(-1, top_ids)
    rows: list[dict[str, Any]] = []
    for token_id, logit_val, prob_val in zip(top_ids[0], top_vals[0], top_probs[0], strict=False):
        rows.append(
            {
                "token_id": int(token_id.item()),
                "logit": float(logit_val.item()),
                "prob": float(prob_val.item()),
            }
        )
    return rows


class MiniCPMO4_5TTSStreamingGenerator:
    def __init__(
        self,
        *,
        tts_model: nn.Module,
        emb_text: nn.Embedding,
        emb_code: nn.Embedding,
        head_code: nn.Module,
        text_eos_token_id: int,
        audio_bos_token_id: int,
        eos_token_id: int,
        num_audio_tokens: int,
        chunk_size: int,
        do_sample: bool,
        temperature: float,
        sampling_seed: int | None,
        logits_processors: list[Any],
        logits_warpers: list[Any],
        debug_top_logprobs: int = 8,
        max_new_token: int = 500,
        condition_event_logger: Callable[[dict[str, Any]], None] | None = None,
        decode_step_logger: Callable[[dict[str, Any]], None] | None = None,
        decode_tensor_dumper: Callable[..., str | None] | None = None,
    ) -> None:
        self.tts_model = tts_model
        self.emb_text = emb_text
        self.emb_code = emb_code
        self.head_code = head_code
        self.text_eos_token_id = int(text_eos_token_id)
        self.audio_bos_token_id = int(audio_bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.num_audio_tokens = int(num_audio_tokens)
        self.chunk_size = int(chunk_size)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.sampling_seed = sampling_seed
        self.logits_processors = list(logits_processors)
        self.logits_warpers = list(logits_warpers)
        self.debug_top_logprobs = int(debug_top_logprobs)
        self.max_new_token = int(max_new_token)
        self.condition_event_logger = condition_event_logger
        self.decode_step_logger = decode_step_logger
        self.decode_tensor_dumper = decode_tensor_dumper
        self.state = _MiniCPMO4_5GeneratorState()

    def reset(self) -> None:
        self.state = _MiniCPMO4_5GeneratorState()

    @property
    def stream_finished(self) -> bool:
        return bool(self.state.stream_finished)

    def snapshot(self) -> dict[str, Any]:
        state = self.state
        pending_audio_token_buffer_size = int(len(state.token_buffer))
        if state.current_chunk is not None:
            pending_audio_token_buffer_size += max(
                0,
                int(len(state.current_chunk.token_ids) - state.current_chunk_offset),
            )
        pending_audio_token_buffer_size += sum(int(len(chunk.token_ids)) for chunk in state.ready_audio_chunks)
        return {
            "pending_condition_queue_size": int(len(state.pending_conditions)),
            "pending_audio_token_buffer_size": int(pending_audio_token_buffer_size),
            "current_condition_index": state.current_condition_index,
            "current_condition_shape": (
                None if state.current_condition_shape is None else list(state.current_condition_shape)
            ),
            "current_condition_text_finished": bool(state.current_condition_text_finished),
            "last_generated_token_id": (
                None if state.last_generated_token is None else int(state.last_generated_token.reshape(-1)[0].item())
            ),
            "all_generated_token_count": int(sum(int(tok.shape[1]) for tok in state.all_generated_tokens)),
            "stream_finished": bool(state.stream_finished),
            "text_finished": bool(state.current_condition_text_finished),
            "text_start_pos": int(state.text_start_pos),
            "cache_length": int(_cache_length(state.past_key_values)),
        }

    def enqueue_condition(
        self,
        condition: torch.Tensor,
        *,
        condition_index: int,
        text_finished: bool,
    ) -> None:
        state = self.state
        state.pending_conditions.append(
            MiniCPMO4_5EnqueuedCondition(
                index=int(condition_index),
                tensor=condition.detach().to("cpu").contiguous(),
                text_finished=bool(text_finished),
            )
        )
        self._drain_pending_conditions()

    def pop_token(self) -> MiniCPMO4_5PoppedToken:
        state = self.state
        self._promote_current_chunk()
        if state.current_chunk is None:
            self._drain_pending_conditions()
            self._promote_current_chunk()

        if state.current_chunk is not None:
            token_id = int(state.current_chunk.token_ids[state.current_chunk_offset])
            condition_index = state.current_chunk.condition_index
            condition_shape = (
                None if state.current_chunk.condition_shape is None else list(state.current_chunk.condition_shape)
            )
            text_finished = bool(state.current_chunk.text_finished)
            state.current_chunk_offset += 1
            if state.current_chunk_offset >= len(state.current_chunk.token_ids):
                state.current_chunk = None
                state.current_chunk_offset = 0
            return MiniCPMO4_5PoppedToken(
                token_id=token_id,
                condition_index=condition_index,
                condition_shape=condition_shape,
                text_finished=text_finished,
                is_eos=False,
            )

        if state.eos_pending and not state.eos_emitted:
            state.eos_emitted = True
            state.stream_finished = True
            return MiniCPMO4_5PoppedToken(
                token_id=self.eos_token_id,
                condition_index=state.current_condition_index,
                condition_shape=(
                    None if state.current_condition_shape is None else list(state.current_condition_shape)
                ),
                text_finished=True,
                is_eos=True,
            )

        raise RuntimeError("MiniCPM async TTS generator has no token ready to emit.")

    def generate_with_buffer(
        self,
        condition: torch.Tensor,
        text_finished: bool = False,
        max_new_token: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, bool]]:
        state = self.state
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype

        max_new_token = int(self.max_new_token if max_new_token is None else max_new_token)
        condition = condition.to(device=device, dtype=dtype, non_blocking=True)
        if condition.ndim == 3:
            if int(condition.shape[0]) != 1:
                raise ValueError(f"Expected batch size 1 condition, got shape {tuple(condition.shape)}")
            condition = condition[0]

        if bool(text_finished):
            condition = torch.cat([condition, self._get_text_eos_embed(device=device, dtype=dtype)], dim=0)

        condition = torch.cat([condition, self._get_audio_bos_embed(device=device, dtype=dtype)], dim=0).unsqueeze(0)

        condition_length = int(condition.shape[1])
        prefill_len = condition_length
        chunk_generated_tokens: list[torch.Tensor] = []
        saw_audio_eos = False

        for t in range(max_new_token):
            if t == 0:
                inputs_embeds = condition
                pos_ids = torch.arange(
                    int(state.text_start_pos),
                    int(state.text_start_pos) + condition_length,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                input_kind = "condition"
                input_audio_token_id = None
            else:
                if not state.all_generated_tokens:
                    break
                last = state.all_generated_tokens[-1]
                inputs_embeds = self.emb_code(last).reshape(1, 1, -1)
                pos_ids = torch.tensor(
                    [int(state.text_start_pos) + prefill_len + t - 1],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                input_kind = "audio_feedback"
                input_audio_token_id = int(last.reshape(-1)[0].item())

            step_index = int(state.step_index)
            state.step_index += 1
            cache_len_before = _cache_length(state.past_key_values)
            text_start_pos_before = int(state.text_start_pos)

            outputs = self.tts_model(
                position_ids=pos_ids,
                past_key_values=state.past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            state.past_key_values = outputs.past_key_values

            raw_logits = self.head_code(hidden_states[:, -1:, :]).reshape(1, -1).to(dtype=torch.float32)
            sampling_logits = raw_logits.clone()
            if self.do_sample:
                sampling_logits = sampling_logits / max(float(self.temperature), 1e-5)

            generated = None
            if state.all_generated_tokens:
                generated = torch.cat(state.all_generated_tokens, dim=1).to(
                    device=sampling_logits.device,
                    dtype=torch.long,
                )
                for processor in self.logits_processors:
                    sampling_logits = processor(generated, sampling_logits)
                for warper in self.logits_warpers:
                    sampling_logits = warper(generated, sampling_logits)

            probs = F.softmax(sampling_logits, dim=-1)
            greedy_token = torch.argmax(sampling_logits, dim=-1, keepdim=True)
            if self.do_sample:
                generator = self._get_sampling_generator(device=probs.device)
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            else:
                next_token = greedy_token

            next_id = int(next_token.reshape(-1)[0].item())
            if next_id == self.eos_token_id:
                saw_audio_eos = True
            else:
                next_tok = next_token.reshape(1, 1).to(dtype=torch.long, device=device)
                state.last_generated_token = next_tok.detach().clone()
                state.all_generated_tokens.append(next_tok.clone())
                chunk_generated_tokens.append(next_tok.clone())
                state.token_buffer.append(next_tok.clone())

            cache_len_after = _cache_length(state.past_key_values)
            decode_tensor_dump_dir: str | None = None
            if self.decode_tensor_dumper is not None:
                decode_tensor_dump_dir = self.decode_tensor_dumper(
                    step_index=step_index,
                    input_kind=input_kind,
                    inputs_embeds=inputs_embeds,
                    position_ids=pos_ids,
                    hidden_states=hidden_states,
                    raw_logits=raw_logits.detach(),
                    sampling_logits=sampling_logits.detach(),
                    probs=probs.detach(),
                    condition_index=state.current_condition_index,
                    condition_shape=state.current_condition_shape,
                    condition_text_finished=bool(state.current_condition_text_finished),
                )

            if self.decode_step_logger is not None:
                self.decode_step_logger(
                    {
                        "step_index": int(step_index),
                        "input_kind": input_kind,
                        "input_audio_token_id": input_audio_token_id,
                        "input_embeds_shape": _shape_list(inputs_embeds),
                        "condition_index": state.current_condition_index,
                        "condition_shape": state.current_condition_shape,
                        "condition_text_finished": bool(state.current_condition_text_finished),
                        "pending_condition_queue_size_after_pop": int(len(state.pending_conditions)),
                        "pending_audio_token_buffer_size_after_append": int(
                            self.snapshot()["pending_audio_token_buffer_size"]
                        ),
                        "cache_len_before": int(cache_len_before),
                        "cache_len_after": int(cache_len_after),
                        "text_start_pos_before": int(text_start_pos_before),
                        "text_start_pos_after": int(state.text_start_pos),
                        "hidden_state_shape": _shape_list(hidden_states),
                        "decode_tensor_dump_dir": decode_tensor_dump_dir,
                        "sampled_token_id": int(next_id),
                        "sampling_do_sample": bool(self.do_sample),
                        "temperature": float(self.temperature),
                        "generated_token_count_before": int(0 if generated is None else generated.shape[1]),
                        "generated_token_tail": (
                            []
                            if generated is None or generated.numel() == 0
                            else [int(tok) for tok in generated[0, -16:].tolist()]
                        ),
                        "raw_top_tokens": _topk_logits_summary(
                            raw_logits,
                            torch.softmax(raw_logits, dim=-1),
                            self.debug_top_logprobs,
                        ),
                        "sample_top_tokens": _topk_logits_summary(
                            sampling_logits,
                            probs,
                            self.debug_top_logprobs,
                        ),
                        "greedy_token_id": int(greedy_token.reshape(-1)[0].item()),
                        "sample_matches_greedy": bool(int(greedy_token.reshape(-1)[0].item()) == int(next_id)),
                    }
                )

            if len(state.token_buffer) == 0:
                if bool(text_finished) and saw_audio_eos:
                    yield torch.empty((1, 0), dtype=torch.long, device=device), True
                    break
                if saw_audio_eos:
                    break
                continue

            if len(state.token_buffer) >= self.chunk_size:
                batch = torch.cat(state.token_buffer[: self.chunk_size], dim=1)
                yield batch, False
                state.token_buffer = state.token_buffer[self.chunk_size :]
                if saw_audio_eos:
                    break
            else:
                if saw_audio_eos:
                    if bool(text_finished):
                        batch = torch.cat(state.token_buffer, dim=1)
                        yield batch, True
                        state.token_buffer = []
                    break
                continue

        state.text_start_pos += prefill_len + len(chunk_generated_tokens)
        if bool(text_finished) and saw_audio_eos:
            state.eos_pending = True

    def _drain_pending_conditions(self) -> None:
        state = self.state
        while state.pending_conditions and not state.stream_finished:
            condition_chunk = state.pending_conditions.pop(0)
            condition_shape = (
                [1, int(condition_chunk.tensor.shape[0]), int(condition_chunk.tensor.shape[1])]
                if condition_chunk.tensor.ndim == 2
                else [int(dim) for dim in condition_chunk.tensor.shape]
            )
            state.current_condition_index = int(condition_chunk.index)
            state.current_condition_shape = list(condition_shape)
            state.current_condition_text_finished = bool(condition_chunk.text_finished)
            if self.condition_event_logger is not None:
                self.condition_event_logger(
                    {
                        "event": "consume",
                        "condition_index": int(condition_chunk.index),
                        "condition_shape": list(condition_shape),
                        "pending_condition_queue_size": int(len(state.pending_conditions)),
                        "pending_audio_token_buffer_size": int(self.snapshot()["pending_audio_token_buffer_size"]),
                        "text_finished": bool(condition_chunk.text_finished),
                    }
                )
            for audio_token_chunk, is_last_audio_chunk in self.generate_with_buffer(
                condition_chunk.tensor,
                text_finished=condition_chunk.text_finished,
            ):
                token_ids = self._coerce_token_chunk(audio_token_chunk)
                if token_ids:
                    state.ready_audio_chunks.append(
                        MiniCPMO4_5ReadyAudioChunk(
                            token_ids=token_ids,
                            condition_index=int(condition_chunk.index),
                            condition_shape=list(condition_shape),
                            text_finished=bool(condition_chunk.text_finished),
                            is_last_audio_chunk=bool(is_last_audio_chunk),
                        )
                    )
                if self.condition_event_logger is not None:
                    self.condition_event_logger(
                        {
                            "event": "yield_chunk",
                            "condition_index": int(condition_chunk.index),
                            "condition_shape": list(condition_shape),
                            "audio_token_count": int(len(token_ids)),
                            "is_last_audio_chunk": bool(is_last_audio_chunk),
                            "pending_condition_queue_size": int(len(state.pending_conditions)),
                            "pending_audio_token_buffer_size": int(self.snapshot()["pending_audio_token_buffer_size"]),
                            "text_finished": bool(condition_chunk.text_finished),
                        }
                    )

    def _promote_current_chunk(self) -> None:
        state = self.state
        if state.current_chunk is not None and state.current_chunk_offset < len(state.current_chunk.token_ids):
            return
        state.current_chunk = None
        state.current_chunk_offset = 0
        if state.ready_audio_chunks:
            state.current_chunk = state.ready_audio_chunks.pop(0)

    def _get_sampling_generator(self, *, device: torch.device) -> torch.Generator | None:
        state = self.state
        if self.sampling_seed is None:
            return None
        device_str = str(device)
        if state.sampling_generator is None or state.sampling_generator_device != device_str:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.sampling_seed))
            state.sampling_generator = generator
            state.sampling_generator_device = device_str
        return state.sampling_generator

    def _get_text_eos_embed(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.emb_text(torch.tensor([self.text_eos_token_id], device=device, dtype=torch.long)).to(dtype=dtype)

    def _get_audio_bos_embed(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.emb_text(torch.tensor([self.audio_bos_token_id], device=device, dtype=torch.long)).to(dtype=dtype)

    @staticmethod
    def _coerce_token_chunk(chunk: Any) -> list[int]:
        if chunk is None:
            return []
        if isinstance(chunk, torch.Tensor):
            token_tensor = chunk.detach().cpu().to(dtype=torch.long)
        else:
            token_tensor = torch.as_tensor(chunk, dtype=torch.long)
        return [int(token) for token in token_tensor.reshape(-1).tolist()]
