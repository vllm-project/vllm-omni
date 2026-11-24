# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru_vllm/haru_vllm/runtime.py @ d3612b0.
"""Per-conversation state and the per-step audio work of the engine worker.

Each conversation is one resumable engine request. Its first append schedules
the whole prompt (system prompt + voice reference); every later append
schedules one position, the frame ``text[t] + model_audio[t] + user_audio[t]``.
After a position is sampled, the MeanFlow head and the codec produce the audio
of the next frame, which is both sent to the client and fed back as input.

One engine step is: ``begin_step`` -> ``add_row`` per scheduled request ->
forward -> ``adjust_logits`` -> sampler -> ``record_sampled`` -> ``finish_step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pybase64 as base64
import torch

from vllm_omni.model_executor.models.haru.loaders import load_tokenizer
from vllm_omni.model_executor.models.haru.prompt import build_prompt, silence_latent
from vllm_omni.model_executor.models.haru.sampling import DryConfig, DryState
from vllm_omni.model_executor.models.haru.vibevoice_codec import open_stream

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.haru.haru import HaruForConditionalGeneration


@dataclass
class SessionOptions:
    """What a client may set for one conversation (validated by the duplex plugin)."""

    system_prompt: str = ""
    reference_pcm16: str = ""  # base64 PCM16 mono at the model sample rate
    guidance_scale: float = 1.5
    eop_bias: float = 0.0
    speak_first: bool = False
    dry: bool = True
    seed: int | None = None

    @classmethod
    def from_runtime_config(cls, runtime_config: dict[str, Any]) -> SessionOptions:
        options = runtime_config.get("haru")
        return cls(**options) if isinstance(options, dict) else cls()

    def reference(self) -> torch.Tensor | None:
        if not self.reference_pcm16:
            return None
        pcm = np.frombuffer(base64.b64decode(self.reference_pcm16), dtype="<i2")
        return torch.from_numpy(pcm.astype(np.float32) / 32768.0)


@dataclass
class _Conversation:
    options: SessionOptions
    codec: Any  # frame-by-frame codec stream
    generator: torch.Generator
    prompt_embeds: torch.Tensor | None  # dropped once the prompt has run
    prompt_length: int
    progress: int = 0
    token: torch.Tensor | None = None  # last sampled text token, [1]
    latent: torch.Tensor | None = None  # last generated raw latent, [1, D]
    force_text: int = 0
    dry: DryState | None = None


@dataclass
class _Row:
    request_id: str
    conversation: _Conversation | None
    samples: bool  # this step reaches the end of the request's prompt


class HaruRuntime:
    def __init__(self, model: HaruForConditionalGeneration, model_path: str):
        self._model = model
        self._model_path = model_path
        self._tokenizer = None
        self._conversations: dict[str, _Conversation] = {}
        config = model.haru_config
        self._frame_samples = config.frame_samples
        self._stream_ids = [config.pad_token_id, config.eop_token_id]
        self._banned = sorted(set(self._stream_ids) | ({config.eos_token_id} - {None}))
        self.begin_step()

    @torch.no_grad()
    def warm_up(self) -> None:
        """Run every audio code path once, so the first conversation starts without a delay."""
        model = self._model
        self._tokenizer = load_tokenizer(self._model_path)
        stream = open_stream(model.codec, backend="auto")
        frame = torch.zeros(1, self._frame_samples)
        for _ in range(2):
            stream.decode(model.codec.sample(stream.encode(frame)).float())
        silence_latent(model.codec)
        model.codec.encode(torch.zeros(30 * model.haru_config.sample_rate))

    # --------------------------------------------------------------- step

    @property
    def has_rows(self) -> bool:
        return bool(self._rows)

    def begin_step(self) -> None:
        self._rows: list[_Row] = []
        self._hidden: torch.Tensor | None = None
        self._sampled: list[list[int]] | None = None

    @torch.no_grad()
    def add_row(self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, info: dict[str, Any]) -> torch.Tensor:
        """Register one scheduled request; return the embeddings of its new positions."""
        request_id = str(info["request_id"])
        span = int(input_ids.shape[0])
        computed = int(info["_omni_num_computed_tokens"])
        samples = computed + span >= int(info["_omni_prompt_len"])
        duplex = info.get("duplex")
        if not isinstance(duplex, dict):
            # Not a conversation (engine warm-up): plain token embeddings.
            self._rows.append(_Row(request_id, None, samples))
            return input_embeds if input_embeds is not None else self._model.model.embed_input_ids(input_ids)

        conversation = self._conversations.get(request_id)
        if conversation is None:
            if computed != 0:
                raise RuntimeError(f"conversation {request_id!r} lost its state; it cannot be recomputed")
            conversation = self._open(duplex)
            self._conversations[request_id] = conversation
        if computed != conversation.progress:
            raise RuntimeError(f"conversation {request_id!r} was rescheduled out of order")
        conversation.progress = computed + span
        self._rows.append(_Row(request_id, conversation, samples))

        if computed < conversation.prompt_length:
            embeds = conversation.prompt_embeds[computed : computed + span]
            if embeds.shape[0] != span:
                raise RuntimeError("the scheduled prompt span does not match the prepared prompt")
            if conversation.progress >= conversation.prompt_length:
                conversation.prompt_embeds = None
            return embeds
        if span != 1:
            raise RuntimeError("a live frame is exactly one position")
        return self._frame_embeds(conversation, duplex)

    def _open(self, duplex: dict[str, Any]) -> _Conversation:
        model = self._model
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self._model_path)
        options = SessionOptions.from_runtime_config(duplex.get("runtime_config") or {})
        generator = torch.Generator(device=model.codec.device)
        if options.seed is None:
            generator.seed()
        else:
            generator.manual_seed(options.seed)
        prompt = build_prompt(
            model.haru_config,
            model.codec,
            self._tokenizer,
            options.system_prompt,
            options.reference(),
            options.speak_first,
            generator,
        )
        embeds = model.embed_streams(
            prompt.tokens, prompt.model_latents, prompt.user_latents, prompt.model_mask, prompt.user_mask
        )
        return _Conversation(
            options=options,
            codec=open_stream(model.codec, backend="auto"),
            generator=generator,
            prompt_embeds=embeds,
            prompt_length=prompt.length,
            force_text=1 if options.speak_first else 0,
            dry=DryState(DryConfig(), set(self._banned)) if options.dry else None,
        )

    def _frame_embeds(self, conversation: _Conversation, duplex: dict[str, Any]) -> torch.Tensor:
        model = self._model
        raw = base64.b64decode(duplex["payload"]["audio"])
        audio = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        if audio.shape[0] != self._frame_samples:
            raise ValueError(f"a frame is {self._frame_samples} samples, got {audio.shape[0]}")
        mean = conversation.codec.encode(audio[None])
        user_latent = model.codec.sample(mean, conversation.generator)[0].float()
        return model.embed_streams(conversation.token, conversation.latent, user_latent)

    @torch.no_grad()
    def adjust_logits(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Keep the step's hidden states and apply each conversation's text-stream rules."""
        if hidden.shape[0] != len(self._rows):
            raise RuntimeError("logit rows do not match the scheduled requests")
        self._hidden = hidden
        # float32 from here on: bfloat16 cannot hold the recomputed PAD / EOP logits.
        logits = logits.float()
        weight = self._model.lm_head.weight
        stream = torch.tensor(self._stream_ids, device=logits.device)
        for index, row in enumerate(self._rows):
            conversation = row.conversation
            if conversation is None or not row.samples:
                continue
            if conversation.dry is not None:
                conversation.dry.apply(logits[index])
            # The PAD / EOP decision is often closer than bfloat16 resolves.
            logits[index, stream] = hidden[index].float() @ weight[stream].float().T
            if conversation.options.eop_bias:
                logits[index, self._stream_ids[1]] += conversation.options.eop_bias
            if conversation.force_text > 0:
                logits[index, self._banned] = float("-inf")
                conversation.force_text -= 1
        return logits

    def record_sampled(self, sampled: list[list[int]]) -> None:
        if len(sampled) != len(self._rows):
            raise RuntimeError("sampled rows do not match the scheduled requests")
        self._sampled = [list(tokens) for tokens in sampled]

    @torch.no_grad()
    def finish_step(self, request_ids: list[str], invalid_req_indices: list[int]) -> dict[str, Any]:
        """Generate and decode the audio of every frame sampled in this step."""
        rows, sampled, hidden = self._rows, self._sampled, self._hidden
        self.begin_step()
        if request_ids != [row.request_id for row in rows] or sampled is None or hidden is None:
            raise RuntimeError("post-sample rows do not match the scheduled requests")
        # A conversation advances its stateful codec below, and its state cannot be recomputed, so a row the
        # engine discards must not have sampled. (Under synchronous scheduling vLLM leaves this list empty and
        # drops a partial-prefill token itself; such a row has ``samples`` False and never reaches the codec.)
        invalid = set(invalid_req_indices)
        for index, row in enumerate(rows):
            if row.conversation is not None and row.samples and index in invalid:
                raise RuntimeError(f"the engine discarded the sampled frame of conversation {row.request_id!r}")
        active = [
            index for index, row in enumerate(rows) if row.conversation is not None and row.samples and sampled[index]
        ]
        audio_ids: list[str] = []
        chunks: list[torch.Tensor] = []
        if active:
            model = self._model
            conversations = [rows[index].conversation for index in active]
            dim = model.haru_config.latent_dim
            dtype = model.head.cond_proj.weight.dtype
            noise = torch.cat(
                [torch.randn(1, dim, device=hidden.device, dtype=dtype, generator=c.generator) for c in conversations]
            )
            guidance = torch.tensor([c.options.guidance_scale for c in conversations], device=hidden.device)
            latents = model.unscale_latents(model.head.sample(hidden[active], noise, guidance))
            for position, (index, conversation) in enumerate(zip(active, conversations, strict=True)):
                token = sampled[index][0]
                latent = latents[position : position + 1]
                audio = conversation.codec.decode(latent[None])[0]
                conversation.token = torch.tensor([token], device=hidden.device)
                conversation.latent = latent
                if conversation.dry is not None:
                    conversation.dry.update(token)
                audio_ids.append(rows[index].request_id)
                chunks.append(audio.float().clamp(-1.0, 1.0).cpu())
        sample_rate = self._model.haru_config.sample_rate
        return {
            "model_outputs": chunks,
            "sr": [torch.tensor(sample_rate, dtype=torch.int32) for _ in chunks],
            "meta": {"req_id": audio_ids, "sparse_audio": ["1"]},
        }

    def release(self, request_ids: set[str] | list[str]) -> None:
        for request_id in request_ids:
            self._conversations.pop(request_id, None)
