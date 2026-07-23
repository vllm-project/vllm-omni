from __future__ import annotations

import base64
import copy
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vllm_omni.experimental.fullduplex.minicpmo45.policy import MiniCPMO45DuplexPolicy

_MINICPMO45_SPECIAL_TOKEN_FIELDS = MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS
_MINICPMO45_OPTIONAL_TOKEN_FIELDS = MiniCPMO45DuplexPolicy.OPTIONAL_TOKEN_FIELDS


@dataclass
class _MiniCPMO45Stage0SessionState:
    session_id: str
    streaming_processor: Any | None = None
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_chunk_idx: int = 0
    context_embeds: list[Any] = field(default_factory=list)
    context_token_ids: list[int] = field(default_factory=list)
    current_turn_ended: bool = True
    prepared_append_identity: tuple[int | None, int] | None = None
    prepared_inputs_embeds: Any | None = None
    prepared_input_token_ids: list[int] = field(default_factory=list)
    prepared_result: dict[str, Any] = field(default_factory=dict)
    audio_past_key_values: Any | None = None
    pending_terminator_token: int | None = None
    last_terminator_token: int | None = None
    pending_speech_context: bool = False
    pending_speech_append_identity: tuple[int | None, int] | None = None
    pending_speech_response_open: bool = False
    generated_text_tokens: list[int] = field(default_factory=list)


class MiniCPMO45Stage0DuplexRuntime:
    """Build scheduler-owned MiniCPM-o 4.5 Stage0 duplex inputs."""

    def __init__(self, stage_model: Any, *, model_path: str | None = None, device: str = "cuda") -> None:
        self.stage_model = stage_model
        self.model_path = model_path
        self.device = device
        self.sessions: dict[tuple[str, int], _MiniCPMO45Stage0SessionState] = {}
        self.thinker = getattr(stage_model, "thinker", None) or getattr(stage_model, "model", None) or stage_model
        self.processor = (
            getattr(stage_model, "processor", None)
            or getattr(self.thinker, "processor", None)
            or self._load_processor_from_path(model_path)
        )
        self.tokenizer = (
            getattr(self.processor, "tokenizer", None)
            if self.processor is not None
            else getattr(stage_model, "tokenizer", None)
        )
        self._init_token_ids()

    def _stage_runtime_ready(self) -> bool:
        return self.processor is not None and self.tokenizer is not None and self.thinker is not None

    def _configure_streaming_processor(
        self,
        state: _MiniCPMO45Stage0SessionState | None = None,
    ) -> Any | None:
        processor = self.processor
        if processor is None:
            return None
        if state is not None:
            if state.streaming_processor is not None:
                return state.streaming_processor
            processor = copy.copy(processor)
            shared_mel = getattr(self.processor, "_streaming_mel_processor", None)
            if shared_mel is not None:
                processor._streaming_mel_processor = copy.deepcopy(shared_mel)
            state.streaming_processor = processor
        if processor is None:
            return
        set_streaming_mode = getattr(processor, "set_streaming_mode", None)
        if callable(set_streaming_mode):
            set_streaming_mode(
                mode="exact",
                chunk_ms=int(self._stage_param("chunk_ms", 1000)),
                first_chunk_ms=int(self._stage_param("first_chunk_ms", 1035)),
                cnn_redundancy_ms=int(self._stage_param("cnn_redundancy_ms", 20)),
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )
            # Match official init_streaming_processor: reset the streaming mel-processor
            # buffers at session init (modeling_minicpmo_unified.py:207).
            reset_streaming = getattr(processor, "reset_streaming", None)
            if callable(reset_streaming):
                reset_streaming()
            return processor
        configure_streaming = getattr(processor, "configure_streaming", None)
        if callable(configure_streaming):
            configure_streaming(
                chunk_ms=int(self._stage_param("chunk_ms", 1000)),
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )
        return processor

    def _prepare_session_context(
        self,
        state: _MiniCPMO45Stage0SessionState,
        session_config: dict[str, Any],
        *,
        runtime_config: dict[str, Any] | None = None,
    ) -> None:
        if not self._stage_runtime_ready():
            return
        self._require_special_token_ids()
        ref_audio = self._decode_ref_audio_from_session_config(runtime_config or {})
        # Matches MiniCPMODuplex.prepare() in the released checkpoint's
        # modeling_minicpmo.py: the <|audio_start|>/<|audio_end|> markers are
        # only present when reference audio is embedded between them. The
        # template is shared with the serving adapter so the first-append
        # scheduler reserve can count these tokens exactly.
        prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
            session_config.get("instructions"),
            ref_audio is not None,
        )
        for token_id in self._encode_text(prefix):
            state.context_embeds.append(self._embed_token(token_id))
            state.context_token_ids.append(token_id)
        if ref_audio is not None:
            ref_audio_embeds = self._stage_ref_audio_embeddings(ref_audio, state=state)
            if ref_audio_embeds is not None:
                ref_audio_embeds = self._as_2d_tensor(ref_audio_embeds)
                state.context_embeds.append(ref_audio_embeds)
                state.context_token_ids.extend([self.unit_token_id] * int(ref_audio_embeds.shape[0]))
        for token_id in self._encode_text(suffix):
            state.context_embeds.append(self._embed_token(token_id))
            state.context_token_ids.append(token_id)

    def _stage_prefill_embeddings_only(
        self,
        state: _MiniCPMO45Stage0SessionState,
        audio_waveform: Any,
        *,
        video_frames: list[Any] | None = None,
        epoch: int | None = None,
        seq: int | None = None,
        is_speech: bool = False,
        final: bool = False,
    ) -> dict[str, Any]:
        """Build scheduler-owned Stage0 input embeddings for one audio append.

        Unlike the legacy worker-control path, this method never calls an eager
        model forward. The normal vLLM runner consumes the returned embeddings
        and owns attention metadata, block tables, KV cache, and sampling.
        """
        start_time = time.time()
        processor = self._configure_streaming_processor(state)
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
        self._require_special_token_ids()
        if audio_waveform is None or len(audio_waveform) == 0:
            return self._stage_prefill_result(False, start_time, "empty audio")
        # Omni duplex: encode this append's camera frames up front so the unit
        # loop can interleave one <image> block per unit, mirroring official
        # streaming_prefill (feed <unit>, then image embeds, then audio).
        frame_blocks: list[Any] = []
        if video_frames:
            self._require_vision_token_ids()
            frame_blocks = self._stage_vision_embeddings(video_frames)
            if frame_blocks is None or len(frame_blocks) != len(video_frames):
                return self._stage_prefill_result(False, start_time, "streaming vision embedding failed")
        state.audio_buffer = np.concatenate([state.audio_buffer, np.asarray(audio_waveform, dtype=np.float32)])
        chunk_size = self._streaming_chunk_size(processor)
        self._pad_first_audio_chunk_if_needed(state, processor)
        if len(state.audio_buffer) < chunk_size:
            return self._stage_prefill_result(
                False,
                start_time,
                f"audio not enough: need {chunk_size} samples, only {len(state.audio_buffer)}",
            )

        embed_parts: list[Any] = []
        token_ids: list[int] = []
        if state.audio_chunk_idx == 0 and state.context_embeds:
            embed_parts.extend(state.context_embeds)
            token_ids.extend(state.context_token_ids)

        # Consume every complete chunk in the buffer so the appended span and
        # the scheduler's slot reservation for this append agree exactly. A
        # final append may zero-pad a real residual chunk, but it must not add
        # a whole silence unit after all input was already consumed: official
        # duplex generation runs once per microphone unit and client commit is
        # not an additional model decision.
        units_built = 0
        while True:
            if len(state.audio_buffer) < chunk_size:
                if not final or len(state.audio_buffer) == 0:
                    break
                pad = np.zeros(chunk_size - len(state.audio_buffer), dtype=np.float32)
                state.audio_buffer = np.concatenate([state.audio_buffer, pad])
            audio_chunk = state.audio_buffer[:chunk_size]
            batch_feature = self._process_streaming_audio(audio_chunk, state.audio_chunk_idx, processor=processor)
            for name, value in (
                ("chunk_idx", state.audio_chunk_idx),
                ("use_extra_context", True),
                ("prefix_extra_frames", 0 if state.audio_chunk_idx == 0 else 2),
                ("suffix_extra_frames", 2),
            ):
                with suppress(Exception):
                    setattr(batch_feature, name, value)
            audio_embeds = self._stage_audio_embeddings(batch_feature, state=state)
            if audio_embeds is None:
                if units_built == 0:
                    return self._stage_prefill_result(False, start_time, "streaming audio embedding returned empty")
                break
            if state.audio_chunk_idx > 0:
                # Official duplex closes every unit (finalize_unit feeds the
                # sampled terminator + </unit>) before the next <unit> opens.
                # The scheduler session update discards the previous segment's
                # sampled terminator token, so it is re-injected here ahead of
                # the closure; the model's listen/speak policy depends on
                # seeing its own past decisions in context.
                pending_terminator = state.pending_terminator_token
                if pending_terminator is not None and units_built == 0:
                    state.pending_terminator_token = None
                    embed_parts.append(self._embed_token(pending_terminator))
                    token_ids.append(int(pending_terminator))
                embed_parts.append(self._embed_token(self.unit_end_token_id))
                token_ids.append(self.unit_end_token_id)
            embed_parts.append(self._embed_token(self.unit_token_id))
            token_ids.append(self.unit_token_id)
            if frame_blocks:
                # Official order inside a unit: <image> + 64 resampler
                # embeddings + </image> ahead of the unit's audio embeddings
                # (max_slice_nums=1 in streaming, one frame per unit).
                vision_block = self._as_2d_tensor(frame_blocks.pop(0))
                embed_parts.append(self._embed_token(self.image_start_token_id))
                token_ids.append(int(self.image_start_token_id))
                embed_parts.append(vision_block)
                token_ids.extend([self._vision_embedding_placeholder_token_id()] * int(vision_block.shape[0]))
                embed_parts.append(self._embed_token(self.image_end_token_id))
                token_ids.append(int(self.image_end_token_id))
            embed_parts.append(audio_embeds)
            token_ids.extend(
                [self._audio_embedding_placeholder_token_id()] * int(self._as_2d_tensor(audio_embeds).shape[0])
            )
            state.audio_buffer = state.audio_buffer[
                self._consumed_audio_samples(state.audio_chunk_idx, chunk_size, processor=processor) :
            ]
            state.audio_chunk_idx += 1
            units_built += 1
            chunk_size = self._streaming_chunk_size(processor)
        if frame_blocks:
            # Each frame reserves one image block in the append plan. A frame
            # without a matching audio unit would desynchronize that plan.
            return self._stage_prefill_result(
                False,
                start_time,
                f"{len(frame_blocks)} video frame(s) left without a matching audio unit",
            )
        # Match official streaming_prefill: per chunk feed ONLY <unit>+audio. The assistant
        # turn is opened once at session init; re-emitting the turn-open prefix per chunk
        # re-opened the turn each chunk -> degenerate repetition. tts_bos/listen/turn_eos are
        # model-generated and tracked via current_turn_ended (mirrors streaming_generate).
        prompt_suffix_len = 0

        import torch

        inputs_embeds = torch.cat([self._as_2d_tensor(embed) for embed in embed_parts], dim=0)
        result = self._stage_prefill_result(True, start_time)
        result.update(
            {
                "inputs_embeds": inputs_embeds,
                "input_token_ids": token_ids,
                "special_token_ids": self._special_token_ids(),
                "num_input_tokens": int(inputs_embeds.shape[0]),
                "prompt_suffix_len": prompt_suffix_len,
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
                "runtime_impl": "scheduler_data_plane",
            }
        )
        if is_speech and (append_identity is None or state.pending_speech_append_identity != append_identity):
            state.pending_speech_context = True
            state.pending_speech_append_identity = append_identity
        if append_identity is not None:
            state.prepared_append_identity = append_identity
            state.prepared_inputs_embeds = inputs_embeds
            state.prepared_input_token_ids = list(token_ids)
            state.prepared_result = {k: v for k, v in result.items() if k not in {"inputs_embeds", "input_token_ids"}}
        return result

    @staticmethod
    def _stage_prefill_result(success: bool, start_time: float, reason: str = "") -> dict[str, Any]:
        return {
            "success": success,
            "prefill_success": success,
            "is_buffering": not success,
            "reason": reason,
            "cost_all": time.time() - start_time,
            "stage_runtime_ready": True,
        }

    @staticmethod
    def _as_2d_tensor(value: Any) -> Any:
        if value.ndim == 1:
            return value.unsqueeze(0)
        if value.ndim == 3 and value.shape[0] == 1:
            return value.squeeze(0)
        return value

    def _embed_token(self, token_id: int) -> Any:
        import torch

        token = torch.tensor([int(token_id)], dtype=torch.long, device=self._model_device())
        embedder = self._token_embedder()
        embeds = embedder(token)
        return self._as_2d_tensor(embeds)

    def _token_embedder(self) -> Any:
        nested_embed = getattr(getattr(getattr(self.thinker, "llm", None), "model", None), "embed_tokens", None)
        if callable(nested_embed):
            return nested_embed
        for target in (self.thinker, self.stage_model):
            embedder = getattr(target, "get_input_embeddings", None)
            if callable(embedder):
                try:
                    embeddings = embedder()
                    if callable(embeddings):
                        return embeddings
                except TypeError:
                    return embedder
        raise AttributeError("MiniCPM-o stage0 model does not expose token embeddings")

    def _model_device(self) -> Any:
        try:
            return next(self.thinker.parameters()).device
        except Exception:
            pass
        try:
            return next(self.stage_model.parameters()).device
        except Exception:
            pass
        return self.device

    def _streaming_chunk_size(self, processor: Any | None = None) -> int:
        processor = processor or self.processor
        get_chunk = getattr(processor, "get_streaming_chunk_size", None)
        if callable(get_chunk):
            return int(get_chunk())
        return 16000

    def _sample_rate(self, processor: Any | None = None) -> int:
        processor = processor or self.processor
        return int(
            self._stage_param(
                "sample_rate",
                getattr(getattr(processor, "_streaming_mel_processor", None), "sample_rate", 16000),
            )
        )

    def _first_chunk_samples(self, default_chunk_size: int, processor: Any | None = None) -> int:
        processor = processor or self.processor
        if getattr(processor, "_streaming_mel_processor", None) is None:
            return default_chunk_size
        return int(self._stage_param("first_chunk_ms", 1035) * self._sample_rate(processor) / 1000)

    def _pad_first_audio_chunk_if_needed(
        self,
        state: _MiniCPMO45Stage0SessionState,
        processor: Any | None = None,
    ) -> None:
        if state.audio_chunk_idx != 0 or len(state.audio_buffer) == 0:
            return
        first_chunk_samples = self._first_chunk_samples(
            self._streaming_chunk_size(processor),
            processor,
        )
        if len(state.audio_buffer) >= first_chunk_samples:
            return
        padding = np.zeros(first_chunk_samples - len(state.audio_buffer), dtype=np.float32)
        state.audio_buffer = np.concatenate([padding, state.audio_buffer])

    def _stage_param(self, name: str, default: Any) -> Any:
        for target in (self.stage_model, self.thinker, getattr(self.thinker, "llm", None)):
            value = getattr(target, name, None)
            if value is not None:
                return value
            value = getattr(target, name.upper(), None)
            if value is not None:
                return value
        return default

    def _consumed_audio_samples(
        self,
        chunk_idx: int,
        default_chunk_size: int,
        *,
        processor: Any | None = None,
    ) -> int:
        processor = processor or self.processor
        if chunk_idx != 0:
            chunk_ms = int(self._stage_param("chunk_ms", 1000))
            return int(chunk_ms * self._sample_rate(processor) / 1000)
        mel_processor = getattr(processor, "_streaming_mel_processor", None)
        get_config = getattr(mel_processor, "get_config", None)
        if callable(get_config):
            cfg = get_config()
            if isinstance(cfg, dict):
                consumed_ms = int(cfg.get("effective_first_chunk_ms", self._stage_param("first_chunk_ms", 1035)))
                return int(consumed_ms * self._sample_rate(processor) / 1000)
        return default_chunk_size

    def _process_streaming_audio(
        self,
        audio_chunk: Any,
        chunk_idx: int,
        *,
        processor: Any | None = None,
    ) -> Any:
        processor = processor or self.processor
        process = getattr(processor, "process_audio_streaming", None)
        if callable(process):
            try:
                return process(audio_chunk, reset=False, return_batch_feature=True)
            except TypeError:
                return process(audio_chunk, chunk_idx=chunk_idx)
        return {"audio_features": audio_chunk, "audio_feature_lens": [[len(audio_chunk)]]}

    def _stage_audio_embeddings(
        self,
        batch_feature: Any,
        *,
        state: _MiniCPMO45Stage0SessionState | None = None,
    ) -> Any | None:
        if hasattr(batch_feature, "to"):
            batch_feature = batch_feature.to(self.device)
        self._ensure_dynamic_cache_compat()
        has_audio_cache = state is not None and hasattr(self.thinker, "audio_past_key_values")
        previous_audio_past_key_values = (
            getattr(self.thinker, "audio_past_key_values", None) if has_audio_cache else None
        )
        if has_audio_cache:
            self.thinker.audio_past_key_values = state.audio_past_key_values
        try:
            for target in (self.stage_model, self.thinker):
                get_streaming = getattr(target, "get_audio_embedding_streaming", None)
                if callable(get_streaming):
                    try:
                        result = self._cat_nested_tensors(
                            get_streaming(
                                batch_feature,
                                use_extra_context=True,
                                prefix_extra_frames=0 if int(getattr(batch_feature, "chunk_idx", 0)) == 0 else 2,
                                suffix_extra_frames=2,
                            )
                        )
                        if has_audio_cache:
                            state.audio_past_key_values = getattr(self.thinker, "audio_past_key_values", None)
                        return result
                    except TypeError:
                        result = self._cat_nested_tensors(get_streaming(batch_feature))
                        if has_audio_cache:
                            state.audio_past_key_values = getattr(self.thinker, "audio_past_key_values", None)
                        return result
                get_hidden = getattr(target, "get_audio_hidden_states", None)
                if callable(get_hidden):
                    result = self._cat_nested_tensors(get_hidden(batch_feature))
                    if has_audio_cache:
                        state.audio_past_key_values = getattr(self.thinker, "audio_past_key_values", None)
                    return result
            return None
        finally:
            if has_audio_cache:
                self.thinker.audio_past_key_values = previous_audio_past_key_values

    @staticmethod
    def _decode_ref_audio_from_session_config(session_config: dict[str, Any]) -> Any | None:
        from vllm_omni.experimental.fullduplex.minicpmo45.input import decode_native_ref_audio_from_config

        return decode_native_ref_audio_from_config({"extra_body": session_config})

    def _stage_ref_audio_embeddings(
        self,
        ref_audio: Any,
        *,
        state: _MiniCPMO45Stage0SessionState | None = None,
    ) -> Any | None:
        process_audio = getattr(self.processor, "process_audio", None)
        if callable(process_audio):
            batch_feature = process_audio([ref_audio])
            if hasattr(batch_feature, "to"):
                batch_feature = batch_feature.to(self.device)
            self._ensure_dynamic_cache_compat()
            for target in (self.stage_model, self.thinker):
                get_audio_embedding = getattr(target, "get_audio_embedding", None)
                if callable(get_audio_embedding):
                    try:
                        chunk_length = getattr(getattr(target, "config", None), "audio_chunk_length", None)
                        if chunk_length is not None:
                            return self._cat_nested_tensors(
                                get_audio_embedding(batch_feature, chunk_length=chunk_length)
                            )
                    except TypeError:
                        pass
                    return self._cat_nested_tensors(get_audio_embedding(batch_feature))
                # The split vLLM stage0 wrapper ports official
                # get_audio_embedding(chunk_length=...) as
                # get_audio_hidden_states (chunk_length comes from config).
                get_hidden = getattr(target, "get_audio_hidden_states", None)
                if callable(get_hidden):
                    return self._cat_nested_tensors(get_hidden(batch_feature))
        # The split vLLM stage0 wrapper may only expose the streaming encoder
        # path.  Use it as a fallback so the server-resolved reference audio is
        # still represented in the same system context location as official
        # MiniCPM-o prepare().
        processor = state.streaming_processor if state is not None else self.processor
        batch_feature = self._process_streaming_audio(ref_audio, 0, processor=processor)
        for name, value in (
            ("chunk_idx", 0),
            ("use_extra_context", True),
            ("prefix_extra_frames", 0),
            ("suffix_extra_frames", 2),
        ):
            with suppress(Exception):
                setattr(batch_feature, name, value)
        return self._stage_audio_embeddings(batch_feature, state=state)

    @staticmethod
    def _ensure_dynamic_cache_compat() -> None:
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            return
        if hasattr(DynamicCache, "get_usable_length"):
            return

        def get_usable_length(self, new_seq_length: int | None = None, layer_idx: int = 0) -> int:
            get_seq_length = getattr(self, "get_seq_length", None)
            if not callable(get_seq_length):
                return 0
            try:
                return int(get_seq_length(layer_idx))
            except TypeError:
                return int(get_seq_length())

        DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]

    @staticmethod
    def _cat_nested_tensors(value: Any) -> Any | None:
        import torch

        tensors = []

        def collect(item: Any) -> None:
            if item is None:
                return
            if hasattr(item, "detach"):
                tensors.append(item)
                return
            if isinstance(item, dict):
                for child in item.values():
                    collect(child)
                return
            if isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)

        collect(value)
        if not tensors:
            return None
        return torch.cat([tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors], dim=0)

    def _encode_text(self, text: str) -> list[int]:
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            return list(encode(text, add_special_tokens=False))
        return []

    def _special_token_ids(self) -> dict[str, int]:
        return {
            name: value
            for name, value in {
                "unit_token_id": self.unit_token_id,
                "unit_end_token_id": self.unit_end_token_id,
                "listen_token_id": self.listen_token_id,
                "speak_token_id": self.speak_token_id,
                "tts_bos_token_id": self.tts_bos_token_id,
                "tts_eos_token_id": self.tts_eos_token_id,
                "tts_pad_token_id": self.tts_pad_token_id,
                "chunk_eos_token_id": self.chunk_eos_token_id,
                "chunk_tts_eos_token_id": self.chunk_tts_eos_token_id,
                "turn_eos_token_id": self.turn_eos_token_id,
            }.items()
            if isinstance(value, int) and value >= 0
        }

    @staticmethod
    def _load_processor_from_path(model_path: str | None) -> Any | None:
        if not model_path:
            return None
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
        try:
            from processing_minicpmo import MiniCPMOProcessor

            return MiniCPMOProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            return None

    def _init_token_ids(self) -> None:
        if self.tokenizer is None:
            for field_name in _MINICPMO45_SPECIAL_TOKEN_FIELDS:
                setattr(self, field_name, -1)
            for field_name in _MINICPMO45_OPTIONAL_TOKEN_FIELDS:
                setattr(self, field_name, -1)
        else:
            for field_name, token in _MINICPMO45_SPECIAL_TOKEN_FIELDS.items():
                setattr(self, field_name, self._resolve_special_token_id(token))
            for field_name, token in _MINICPMO45_OPTIONAL_TOKEN_FIELDS.items():
                setattr(self, field_name, self._resolve_special_token_id(token))

    def _resolve_special_token_id(self, token: str) -> int:
        if self.tokenizer is None:
            return -1

        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        candidate = None
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            value = convert(token)
            if isinstance(value, list):
                value = value[0] if len(value) == 1 else None
            with suppress(TypeError, ValueError):
                candidate = int(value)
        if candidate is not None and candidate >= 0 and candidate != unk_token_id:
            return candidate

        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            ids = list(encode(token, add_special_tokens=False))
            if len(ids) == 1:
                value = int(ids[0])
                if value >= 0 and value != unk_token_id:
                    return value
        return -1

    def _require_special_token_ids(self) -> None:
        missing = [
            token
            for field_name, token in _MINICPMO45_SPECIAL_TOKEN_FIELDS.items()
            if not isinstance(getattr(self, field_name, None), int) or getattr(self, field_name) < 0
        ]
        if missing:
            raise ValueError(
                "MiniCPM-o 4.5 native duplex requires tokenizer-defined special "
                f"tokens, missing or unknown: {', '.join(missing)}"
            )

    def _required_token_id(self, field_name: str) -> int:
        token_id = getattr(self, field_name, None)
        if not isinstance(token_id, int) or token_id < 0:
            token = _MINICPMO45_SPECIAL_TOKEN_FIELDS.get(field_name, field_name)
            raise ValueError(f"MiniCPM-o 4.5 missing required special token id for {token}")
        return token_id

    def stage_padding_token_id(self) -> int:
        return self._required_token_id("unit_end_token_id")

    def _audio_embedding_placeholder_token_id(self) -> int:
        token_id = getattr(self, "audio_placeholder_token_id", -1)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
        return self.stage_padding_token_id()

    @staticmethod
    def _decode_audio_payload(payload: dict[str, Any]) -> Any:
        audio = payload.get("audio") or payload.get("data")
        if not isinstance(audio, str):
            raise ValueError("audio append payload requires base64 audio")
        fmt = payload.get("format") or "pcm_f32le"
        if fmt != "pcm_f32le":
            raise ValueError(f"MiniCPM-o stage0 expects pcm_f32le audio, got {fmt!r}")
        return np.frombuffer(base64.b64decode(audio), dtype=np.float32)

    @staticmethod
    def _decode_video_frames_payload(payload: dict[str, Any]) -> list[Any]:
        """Decode omni-duplex camera frames (base64 JPEG/PNG) to PIL images."""
        frames = payload.get("video_frames")
        if not isinstance(frames, list) or not frames:
            return []
        from io import BytesIO

        from PIL import Image

        decoded: list[Any] = []
        for frame_b64 in frames:
            if not isinstance(frame_b64, str) or not frame_b64:
                continue
            try:
                raw = base64.b64decode(frame_b64, validate=True)
                image = Image.open(BytesIO(raw))
                image.load()
            except Exception as exc:  # noqa: BLE001 - normalized below
                raise ValueError("invalid omni duplex video frame payload") from exc
            decoded.append(image.convert("RGB"))
        return decoded

    def _vision_embedding_placeholder_token_id(self) -> int:
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        if isinstance(unk_token_id, int) and unk_token_id >= 0:
            return unk_token_id
        return self._required_optional_token_id("image_start_token_id")

    def _required_optional_token_id(self, field_name: str) -> int:
        token_id = getattr(self, field_name, None)
        if not isinstance(token_id, int) or token_id < 0:
            token = _MINICPMO45_OPTIONAL_TOKEN_FIELDS.get(field_name, field_name)
            raise ValueError(f"MiniCPM-o 4.5 missing required special token id for {token}")
        return token_id

    def _require_vision_token_ids(self) -> None:
        missing = [
            _MINICPMO45_OPTIONAL_TOKEN_FIELDS[field_name]
            for field_name in ("image_start_token_id", "image_end_token_id")
            if not isinstance(getattr(self, field_name, None), int) or getattr(self, field_name) < 0
        ]
        if missing:
            raise ValueError(
                "MiniCPM-o 4.5 omni duplex requires tokenizer-defined image tokens, "
                f"missing or unknown: {', '.join(missing)}"
            )

    def _stage_vision_embeddings(self, frames: list[Any]) -> list[Any] | None:
        """Encode camera frames for omni duplex via the loaded vision tower.

        Semantics mirror ``MiniCPMODuplex.streaming_prefill`` (one 64-embedding
        block per frame at ``max_slice_nums=1``), executed through the vLLM
        wrapper's ``get_vision_hidden_states`` (vpm + resampler) since the
        stage model does not expose the remote-code ``get_vision_embedding``.
        """
        process_image = getattr(self.processor, "process_image", None)
        if not callable(process_image):
            return None
        try:
            processed = process_image(frames, max_slice_nums=1)
        except Exception:  # noqa: BLE001 - prefill fails with a reason
            return None
        targets = (self.stage_model, self.thinker, getattr(self.stage_model, "model", None))
        for target in targets:
            if target is None:
                continue
            get_hidden = getattr(target, "get_vision_hidden_states", None)
            vpm = getattr(target, "vpm", None)
            if not callable(get_hidden) or vpm is None:
                continue
            try:
                import torch

                vpm_param = next(vpm.parameters())
                device, dtype = vpm_param.device, vpm_param.dtype
                pixel_nested = processed["pixel_values"]
                tgt_nested = processed["tgt_sizes"]
                flat_pixels: list[Any] = []
                flat_tgt: list[Any] = []
                for image_slices, image_tgt in zip(pixel_nested, tgt_nested):
                    for slice_pixels in image_slices:
                        flat_pixels.append(slice_pixels.to(device=device, dtype=dtype))
                    tgt_tensor = image_tgt if hasattr(image_tgt, "reshape") else torch.tensor(image_tgt)
                    flat_tgt.append(tgt_tensor.reshape(-1, 2))
                tgt_sizes = torch.cat(flat_tgt, dim=0).to(device=device, dtype=torch.int64)
                with torch.no_grad():
                    hidden = get_hidden({"pixel_values": flat_pixels, "tgt_sizes": tgt_sizes})
            except Exception:  # noqa: BLE001 - prefill fails with a reason
                return None
            expected = MiniCPMO45DuplexPolicy.VISION_EMBEDS_PER_FRAME
            out: list[Any] = []
            for block in hidden:
                block_2d = self._as_2d_tensor(block)
                if int(block_2d.shape[0]) != expected:
                    return None
                out.append(block_2d)
            return out
        return None
