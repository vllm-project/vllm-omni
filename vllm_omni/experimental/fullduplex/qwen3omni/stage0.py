# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy

logger = init_logger(__name__)

_SAMPLE_RATE = Qwen3OmniDuplexPolicy.SAMPLE_RATE_HZ
_SAMPLES_PER_AUDIO_TOKEN = Qwen3OmniDuplexPolicy.SAMPLES_PER_AUDIO_TOKEN


@dataclass
class _Qwen3OmniStage0SessionState:
    session_id: str
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_chunk_idx: int = 0
    context_embeds: list[torch.Tensor] = field(default_factory=list)
    context_token_ids: list[int] = field(default_factory=list)
    prepared_append_identity: tuple[int | None, int] | None = None
    prepared_inputs_embeds: torch.Tensor | None = None
    prepared_input_token_ids: list[int] = field(default_factory=list)
    prepared_result: dict[str, Any] = field(default_factory=dict)
    last_epoch: int = -1

    def reset_for_new_epoch(self, epoch: int) -> None:
        """Clear transient audio/cache state when epoch advances."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_chunk_idx = 0
        self.context_embeds = []
        self.context_token_ids = []
        self.prepared_append_identity = None
        self.prepared_inputs_embeds = None
        self.prepared_input_token_ids = []
        self.prepared_result = {}
        self.last_epoch = epoch


class Qwen3OmniStage0DuplexRuntime:
    """Build scheduler-owned Qwen3-Omni Stage0 duplex inputs.

    Unlike MiniCPM-o's streaming audio encoder, Qwen3-Omni uses a batch-only
    Whisper encoder without KV cache. Each audio segment is processed
    independently through the full feature extractor + audio tower pipeline.
    """

    def __init__(
        self,
        stage_model: Any,
        *,
        model_path: str | None = None,
        device: str = "cuda",
    ) -> None:
        self.stage_model = stage_model
        self.model_path = model_path
        self.device = device
        self.sessions: dict[tuple[str, int], _Qwen3OmniStage0SessionState] = {}

        self.thinker = getattr(stage_model, "thinker", None) or getattr(stage_model, "model", None) or stage_model

        self.processor = self._load_processor(model_path, stage_model)
        self.feature_extractor = (
            getattr(self.processor, "feature_extractor", None) if self.processor is not None else None
        )
        self.tokenizer = (
            getattr(self.processor, "tokenizer", None)
            if self.processor is not None
            else getattr(stage_model, "tokenizer", None)
        )

        self.im_end_token_id = Qwen3OmniDuplexPolicy.IM_END_TOKEN_ID
        self.audio_start_token_id = Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID
        self.audio_end_token_id = Qwen3OmniDuplexPolicy.AUDIO_END_TOKEN_ID
        self.audio_pad_token_id = Qwen3OmniDuplexPolicy.AUDIO_PAD_TOKEN_ID

    def _load_processor(self, model_path: str | None, stage_model: Any) -> Any:
        for target in (stage_model, getattr(stage_model, "thinker", None)):
            proc = getattr(target, "processor", None)
            if proc is not None:
                return proc
        if not model_path:
            return None
        try:
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            logger.warning(
                "Failed to load processor from %s for Qwen3-Omni duplex",
                model_path,
            )
            return None

    def _build_context_embeds(
        self,
        state: _Qwen3OmniStage0SessionState,
        session_config: dict[str, Any],
    ) -> None:
        if self.tokenizer is None:
            return
        state.context_embeds = []
        state.context_token_ids = []
        instructions = session_config.get("instructions")
        prefix, suffix = Qwen3OmniDuplexPolicy.session_context_texts(instructions)
        # prefix ends with "<|im_start|>user\n"
        for token_id in self._encode_text(prefix):
            state.context_embeds.append(self._embed_token(token_id))
            state.context_token_ids.append(token_id)
        history = session_config.get("conversation_history")
        last_role = "user"  # prefix ends with <|im_start|>user\n
        if isinstance(history, list):
            for msg in history:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "")
                if not role or not content:
                    continue
                if role == "user":
                    turn_text = f"{content}<|im_end|>\n"
                else:
                    turn_text = f"<|im_start|>assistant\n{content}<|im_end|>\n<|im_start|>user\n"
                last_role = role
                for token_id in self._encode_text(turn_text):
                    state.context_embeds.append(self._embed_token(token_id))
                    state.context_token_ids.append(token_id)
        if last_role == "user" and isinstance(history, list) and history:
            # user msg closed the user block; reopen for current turn's audio
            for token_id in self._encode_text("<|im_start|>user\n"):
                state.context_embeds.append(self._embed_token(token_id))
                state.context_token_ids.append(token_id)
        # current turn's audio follows
        audio_start_embed = self._embed_token(self.audio_start_token_id)
        state.context_embeds.append(audio_start_embed)
        state.context_token_ids.append(self.audio_start_token_id)
        self._suffix_text = suffix

    def _prepare_session_context(
        self,
        state: _Qwen3OmniStage0SessionState,
        session_config: dict[str, Any],
    ) -> None:
        self._build_context_embeds(state, session_config)

    def _append_suffix(
        self,
        embed_parts: list[torch.Tensor],
        token_ids: list[int],
    ) -> None:
        embed_parts.append(self._embed_token(self.audio_end_token_id))
        token_ids.append(self.audio_end_token_id)
        suffix = getattr(self, "_suffix_text", "<|im_end|>\n<|im_start|>assistant\n")
        for token_id in self._encode_text(suffix):
            embed_parts.append(self._embed_token(token_id))
            token_ids.append(token_id)

    def stage_prefill_embeddings(
        self,
        state: _Qwen3OmniStage0SessionState,
        audio_waveform: np.ndarray,
        *,
        epoch: int | None = None,
        seq: int | None = None,
        final: bool = False,
        session_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start_time = time.time()

        if epoch is not None and epoch != state.last_epoch:
            state.reset_for_new_epoch(epoch)
            if session_config is not None:
                self._build_context_embeds(state, session_config)

        append_identity = (epoch, seq) if seq is not None else None
        if (
            append_identity is not None
            and state.prepared_append_identity == append_identity
            and state.prepared_inputs_embeds is not None
        ):
            result = dict(state.prepared_result)
            result["inputs_embeds"] = state.prepared_inputs_embeds
            result["input_token_ids"] = list(state.prepared_input_token_ids)
            return result

        if audio_waveform is None or len(audio_waveform) == 0:
            return self._prefill_result(False, start_time, "empty audio")

        state.audio_buffer = np.concatenate([state.audio_buffer, np.asarray(audio_waveform, dtype=np.float32)])

        if len(state.audio_buffer) < _SAMPLE_RATE // 10 and not final:
            return self._prefill_result(
                False,
                start_time,
                f"buffering: {len(state.audio_buffer)} samples",
            )

        embed_parts: list[torch.Tensor] = []
        token_ids: list[int] = []

        if state.audio_chunk_idx == 0 and state.context_embeds:
            embed_parts.extend(state.context_embeds)
            token_ids.extend(state.context_token_ids)

        audio_to_encode = state.audio_buffer
        if len(audio_to_encode) > 0:
            audio_embeds = self._encode_audio_segment(audio_to_encode)
            if audio_embeds is not None:
                audio_embeds_2d = self._as_2d(audio_embeds)
                embed_parts.append(audio_embeds_2d)
                num_audio_tokens = int(audio_embeds_2d.shape[0])
                token_ids.extend([self.audio_pad_token_id] * num_audio_tokens)
                state.audio_buffer = np.array([], dtype=np.float32)
                state.audio_chunk_idx += 1
            else:
                return self._prefill_result(False, start_time, "audio encoding failed")

        if final:
            self._append_suffix(embed_parts, token_ids)

        if not embed_parts:
            return self._prefill_result(False, start_time, "no embeddings built")

        inputs_embeds = torch.cat([self._as_2d(e) for e in embed_parts], dim=0)

        result = self._prefill_result(True, start_time)
        result.update(
            {
                "inputs_embeds": inputs_embeds,
                "input_token_ids": token_ids,
                "special_token_ids": self._special_token_ids(),
                "num_input_tokens": int(inputs_embeds.shape[0]),
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "runtime_impl": "scheduler_data_plane",
            }
        )

        if append_identity is not None:
            state.prepared_append_identity = append_identity
            state.prepared_inputs_embeds = inputs_embeds
            state.prepared_input_token_ids = list(token_ids)
            state.prepared_result = {k: v for k, v in result.items() if k not in {"inputs_embeds", "input_token_ids"}}

        return result

    def _encode_audio_segment(self, audio: np.ndarray) -> torch.Tensor | None:
        if self.feature_extractor is None:
            logger.error("No feature extractor available for Qwen3-Omni duplex")
            return None

        try:
            hop_length = getattr(self.feature_extractor, "hop_length", 160)
            audio_length = len(audio)
            if audio_length % hop_length != 0:
                pad_length = hop_length - (audio_length % hop_length)
                audio = np.pad(audio, (0, pad_length), mode="constant", constant_values=0)
                audio_length = len(audio)
            actual_mel_frames = audio_length // hop_length

            features = self.feature_extractor(
                audio,
                sampling_rate=_SAMPLE_RATE,
                return_tensors="pt",
            )
            input_features = features.input_features.to(
                device=self._model_device(), dtype=self.thinker.audio_tower.dtype
            )
            if input_features.ndim == 3:
                input_features = input_features.squeeze(0)
            input_features = input_features[:, :actual_mel_frames]
            feature_lengths = torch.tensor([actual_mel_frames], dtype=torch.long, device=input_features.device)
            audio_output_lengths = self._get_output_lengths(feature_lengths)
            audio_features = self.thinker.audio_tower(
                input_features,
                feature_lens=feature_lengths,
                aftercnn_lens=audio_output_lengths,
            )
            total_tokens = int(audio_output_lengths.sum().item())
            return audio_features[:total_tokens]
        except Exception:
            logger.exception("Qwen3-Omni duplex audio encoding failed")
            return None

    @staticmethod
    def _get_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def _embed_token(self, token_id: int) -> torch.Tensor:
        token = torch.tensor([int(token_id)], dtype=torch.long, device=self._model_device())
        embedder = self._token_embedder()
        return self._as_2d(embedder(token))

    def _token_embedder(self) -> Any:
        for target in (self.thinker,):
            embed_fn = getattr(target, "get_input_embeddings", None)
            if callable(embed_fn):
                try:
                    embeddings = embed_fn()
                    if callable(embeddings):
                        return embeddings
                except TypeError:
                    return embed_fn
            embed_tokens = getattr(
                getattr(getattr(target, "language_model", None), "model", None),
                "embed_tokens",
                None,
            )
            if callable(embed_tokens):
                return embed_tokens
            embed_tokens = getattr(getattr(target, "model", None), "embed_tokens", None)
            if callable(embed_tokens):
                return embed_tokens
        raise AttributeError("Qwen3-Omni thinker does not expose token embeddings")

    def _model_device(self) -> torch.device:
        for target in (self.thinker, self.stage_model):
            try:
                return next(target.parameters()).device
            except (StopIteration, AttributeError):
                continue
        return torch.device(self.device)

    @staticmethod
    def _as_2d(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 1:
            return value.unsqueeze(0)
        if value.ndim == 3 and value.shape[0] == 1:
            return value.squeeze(0)
        return value

    def _encode_text(self, text: str) -> list[int]:
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            return list(encode(text, add_special_tokens=False))
        return []

    def _special_token_ids(self) -> dict[str, int]:
        return {
            "im_end_token_id": self.im_end_token_id,
            "audio_start_token_id": self.audio_start_token_id,
            "audio_end_token_id": self.audio_end_token_id,
            "audio_pad_token_id": self.audio_pad_token_id,
        }

    def stage_padding_token_id(self) -> int:
        return self.audio_pad_token_id

    def cleanup_session(self, session_key: tuple[str, int]) -> None:
        self.sessions.pop(session_key, None)

    @staticmethod
    def _prefill_result(success: bool, start_time: float, reason: str = "") -> dict[str, Any]:
        return {
            "success": success,
            "prefill_success": success,
            "is_buffering": not success,
            "reason": reason,
            "cost_all": time.time() - start_time,
            "stage_runtime_ready": True,
        }

    @staticmethod
    def _decode_audio_payload(payload: dict[str, Any]) -> np.ndarray:
        audio = payload.get("audio") or payload.get("data")
        if not isinstance(audio, str):
            raise ValueError("audio append payload requires base64 audio")
        fmt = payload.get("format") or "pcm_f32le"
        if fmt != "pcm_f32le":
            raise ValueError(f"Qwen3-Omni stage0 expects pcm_f32le audio, got {fmt!r}")
        return np.frombuffer(base64.b64decode(audio), dtype=np.float32)
