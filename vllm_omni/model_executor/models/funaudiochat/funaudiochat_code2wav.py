# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FunAudioChat stage-1 bridge from codec-token chunks to waveform deltas."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig

from vllm_omni.data_entry_keys import to_struct
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config
from vllm_omni.transformers_utils.repo_utils import hf_api

_DEFAULT_SPEAKER = "中文女"
_SPEAKER_INFO_FILENAME = "new_spk2info.pt"
_SPEAKER_INFO_ENV = "FUN_AUDIO_CHAT_SPK2INFO_PATH"
_EMPTY_AUDIO = torch.empty(0, dtype=torch.float32)


def _scalar_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item()) if value.numel() else default
    if isinstance(value, (list, tuple)):
        return _scalar_bool(value[0], default) if value else default
    return bool(value) if value is not None else default


def _scalar_int(value: Any, default: int = 0) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item()) if value.numel() else default
    if isinstance(value, (list, tuple)):
        return _scalar_int(value[0], default) if value else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scalar_str(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        return _scalar_str(value[0]) if value else None
    return value if isinstance(value, str) and value else None


class FunAudioChatCosyVoice3Code2Wav(nn.Module):
    """Decode FunAudioChat codec streams with CosyVoice3's causal flow/vocoder.

    Streaming inputs contain a cumulative codec prefix plus ``left_context_size``.
    CosyVoice3 reuses that prefix as decoder context and returns only newly
    generated waveform samples. The HiFT cache is isolated by request ID and
    released on the terminal chunk (and on request cancellation).
    """

    input_modalities = "audio"
    have_multimodal_outputs = True
    requires_raw_input_tokens = True
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        self.model_dir = str(model_config.model)
        if not os.path.isdir(self.model_dir):
            self.model_dir = hf_api().snapshot_download(self.model_dir)

        self.config = model_config.hf_config
        if not isinstance(self.config, CosyVoice3Config) and not all(
            hasattr(self.config, key) for key in ("flow", "hift", "sample_rate")
        ):
            raise TypeError(
                "FunAudioChat's stage-1 model must use the official Fun-CosyVoice3 "
                "checkpoint/configuration (CosyVoice3Config)."
            )
        self.code2wav = CosyVoice3Code2Wav(self.config).eval()
        self.register_buffer("_default_speaker_embedding", torch.empty((1, 0)), persistent=False)
        self._stream_cache_by_req: dict[str, dict[str, torch.Tensor] | None] = {}
        self._stream_cache_lock = Lock()

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Provide the runner's placeholder embeddings; decoding uses codec IDs."""
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        del hidden_states, sampling_metadata
        return None

    def _speaker_info_path(self) -> Path:
        configured_path = getattr(self.config, "funaudiochat_speaker_info_path", None)
        candidates = [
            Path(os.environ[_SPEAKER_INFO_ENV]) if os.environ.get(_SPEAKER_INFO_ENV) else None,
            Path(configured_path) if configured_path else None,
            Path(self.model_dir) / _SPEAKER_INFO_FILENAME,
            Path(self.model_dir) / "utils" / _SPEAKER_INFO_FILENAME,
        ]
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "FunAudioChat's official CosyVoice3 default speaker embedding is missing. "
            f"Install upstream `{_SPEAKER_INFO_FILENAME}` (the `'{_DEFAULT_SPEAKER}'` "
            "entry from FunAudioChat's `utils/new_spk2info.pt`) beside the stage-1 "
            f"checkpoint or set {_SPEAKER_INFO_ENV} to its path. The FunAudioChat "
            "and Fun-CosyVoice3 Hugging Face repositories do not bundle this file; "
            "the official asset is at "
            "https://github.com/FunAudioLLM/Fun-Audio-Chat/blob/main/utils/new_spk2info.pt."
        )

    def _load_default_speaker_embedding(self) -> torch.Tensor:
        path = self._speaker_info_path()
        try:
            speaker_info = torch.load(path, map_location="cpu", weights_only=True)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Loading FunAudioChat's official speaker conditioning requires the "
                "PyTorch checkpoint deserializer and its serialized value types."
            ) from exc
        if not isinstance(speaker_info, dict):
            raise TypeError(f"Expected a speaker-info mapping in {path}, got {type(speaker_info).__name__}")
        speaker = speaker_info.get(_DEFAULT_SPEAKER)
        if not isinstance(speaker, dict) or "embedding" not in speaker:
            raise KeyError(f"{path} has no `{_DEFAULT_SPEAKER}.embedding` entry required by official FunAudioChat S2S")

        embedding = torch.as_tensor(speaker["embedding"], dtype=torch.float32).reshape(1, -1)
        expected_dim = int(self.config.flow["spk_embed_dim"])
        if embedding.shape[-1] != expected_dim:
            raise ValueError(
                f"Official FunAudioChat speaker embedding in {path} has dimension "
                f"{embedding.shape[-1]}, expected {expected_dim}."
            )
        if not torch.isfinite(embedding).all():
            raise ValueError(f"Official FunAudioChat speaker embedding in {path} contains non-finite values.")
        return embedding

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load CosyVoice3 decoder weights and required official speaker conditioning."""
        del weights
        embedding = self._load_default_speaker_embedding()
        device = torch.device(self.vllm_config.device_config.device)
        self.code2wav.load_weights(self.model_dir, device)
        self._default_speaker_embedding = embedding.to(device=device)
        return set()

    @staticmethod
    def _request_segments(input_ids: torch.Tensor, seq_token_counts: Any, num_requests: int) -> list[torch.Tensor]:
        if input_ids.ndim > 1 and seq_token_counts is None:
            rows = [row.reshape(-1) for row in input_ids]
            if len(rows) == num_requests:
                return rows
            raise ValueError(f"FunAudioChat received {len(rows)} codec-token rows for {num_requests} requests")
        flat_ids = input_ids.reshape(-1)
        if seq_token_counts is None:
            if num_requests == 1:
                return [flat_ids]
            raise ValueError("seq_token_counts is required to split FunAudioChat codec IDs across requests")
        try:
            counts = [int(count) for count in seq_token_counts]
        except (TypeError, ValueError) as exc:
            raise ValueError("seq_token_counts must contain integer codec-token lengths") from exc
        if len(counts) != num_requests:
            raise ValueError(f"Expected {num_requests} codec-token lengths, got {len(counts)}")
        if any(count < 0 for count in counts):
            raise ValueError(f"Codec-token lengths must be non-negative, got {counts}")
        if sum(counts) != flat_ids.numel():
            raise ValueError(
                f"seq_token_counts sum to {sum(counts)}, but input_ids contain {flat_ids.numel()} codec tokens"
            )
        boundaries = [0]
        for count in counts:
            boundaries.append(boundaries[-1] + count)
        return [flat_ids[boundaries[i] : boundaries[i + 1]] for i in range(len(counts))]

    def _flow_input_size(self) -> int:
        flow = getattr(self.config, "flow", None)
        if not isinstance(flow, Mapping) or "input_size" not in flow:
            raise ValueError("FunAudioChat CosyVoice3 config is missing flow.input_size feature dimension")
        input_size = int(flow["input_size"])
        if input_size <= 0:
            raise ValueError(f"FunAudioChat flow.input_size must be positive, got {input_size}")
        return input_size

    @staticmethod
    def _audio_codes(payload: Any) -> torch.Tensor | None:
        codes = payload.codes if payload is not None else None
        audio = codes.audio if codes is not None else None
        if audio is None:
            return None
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio, dtype=torch.long)
        if audio.ndim > 1 and audio.shape[0] > 1:
            raise ValueError(f"Expected one codec-token sequence per request, got shape {tuple(audio.shape)}")
        return audio.reshape(-1).to(dtype=torch.long)

    @staticmethod
    def _request_metadata(raw: Any) -> tuple[str | None, int, bool, bool]:
        payload = to_struct(raw) if raw is not None else None
        meta = payload.meta if payload is not None else None
        request_id = None
        if meta is not None:
            request_id = _scalar_str(meta.req_id) or _scalar_str(meta.request_id)
        if request_id is None and payload is not None:
            request_id = _scalar_str(payload.request_id)
        left_context = _scalar_int(meta.left_context_size) if meta is not None else 0
        chunked = meta is not None and (meta.stream_finished is not None or meta.left_context_size is not None)
        finished = _scalar_bool(meta.stream_finished) if chunked and meta.stream_finished is not None else not chunked
        return request_id, left_context, finished, bool(chunked)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        runtime_info = kwargs.get("model_intermediate_buffer")
        if runtime_info is None:
            runtime_info = kwargs.get("runtime_additional_information", [])
        if not isinstance(runtime_info, list):
            runtime_info = []
        seq_token_counts = kwargs.get("seq_token_counts")
        num_requests = len(runtime_info) or (len(seq_token_counts) if seq_token_counts is not None else 1)
        segments = self._request_segments(input_ids, seq_token_counts, num_requests)
        if runtime_info and len(segments) != len(runtime_info):
            raise ValueError(
                f"FunAudioChat received {len(segments)} codec sequences but only {len(runtime_info)} request payloads"
            )
        if self._default_speaker_embedding.numel() == 0:
            raise RuntimeError(
                "FunAudioChat stage-1 decoder weights and official default speaker "
                "conditioning must be loaded before forward()."
            )

        input_size = self._flow_input_size()
        sample_rate_value = int(self.config.sample_rate)
        if sample_rate_value <= 0:
            raise ValueError(f"FunAudioChat CosyVoice3 sample_rate must be positive, got {sample_rate_value}")
        audio_chunks: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []
        empty_audio = _EMPTY_AUDIO.to(device=input_ids.device)
        sample_rate = torch.tensor(sample_rate_value, dtype=torch.int32, device=input_ids.device)

        for index, segment in enumerate(segments):
            raw = runtime_info[index] if index < len(runtime_info) else None
            payload = to_struct(raw) if raw is not None else None
            req_id, left_context, finished, chunked = self._request_metadata(raw)
            token = self._audio_codes(payload)
            if token is None:
                if payload is not None:
                    raise ValueError(f"FunAudioChat decoder payload for request index {index} is missing codes.audio")
                token = segment.to(dtype=torch.long)
            elif token.numel() != segment.numel():
                raise ValueError(
                    f"FunAudioChat decoder payload contains {token.numel()} codec tokens, "
                    f"but input_ids contain {segment.numel()} for request index {index}"
                )
            if left_context < 0 or left_context > token.numel():
                raise ValueError(
                    f"FunAudioChat left_context_size must be between 0 and the codec-token length "
                    f"({token.numel()}), got {left_context}"
                )
            if not req_id:
                if chunked:
                    raise ValueError("FunAudioChat streaming decoder payload is missing meta.req_id")
                req_id = f"sync-{index}"

            with self._stream_cache_lock:
                cache_state = self._stream_cache_by_req.get(req_id)
            if token.numel() == 0 and (not finished or cache_state is None):
                if finished:
                    with self._stream_cache_lock:
                        self._stream_cache_by_req.pop(req_id, None)
                audio_chunks.append(empty_audio)
                sample_rates.append(sample_rate)
                continue

            # Upstream FunAudioChat token2wav passes empty prompt-token/feature
            # sequences and conditions with its explicit default speaker vector.
            prompt_token = torch.empty((1, 0), dtype=torch.int32, device=token.device)
            prompt_feat = torch.empty(
                (1, 0, input_size),
                dtype=torch.float32,
                device=token.device,
            )
            try:
                speech, next_cache_state = self.code2wav.forward_streaming(
                    token=token.reshape(1, -1),
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    embedding=self._default_speaker_embedding,
                    cache_state=cache_state,
                    n_timesteps=10,
                    token_offset_tokens=left_context,
                    finalize=finished,
                )
            finally:
                if finished:
                    with self._stream_cache_lock:
                        self._stream_cache_by_req.pop(req_id, None)
            if not isinstance(speech, torch.Tensor):
                raise TypeError(f"CosyVoice3 decoder must return waveform tensor, got {type(speech).__name__}")
            if not finished:
                with self._stream_cache_lock:
                    self._stream_cache_by_req[req_id] = next_cache_state
            audio_chunks.append(speech.reshape(-1).to(dtype=torch.float32))
            sample_rates.append(sample_rate)

        return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audio_chunks, "sr": sample_rates})

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Release streaming HiFT state for completed or cancelled requests."""
        with self._stream_cache_lock:
            for request_id in finished_req_ids:
                self._stream_cache_by_req.pop(str(request_id), None)


__all__ = ["FunAudioChatCosyVoice3Code2Wav"]
