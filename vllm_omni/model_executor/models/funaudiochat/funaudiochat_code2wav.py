# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch.nn import functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.data_entry_keys import to_struct
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav
from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask
from vllm_omni.model_executor.models.funaudiochat.common import resolve_funaudiochat_root
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)
_OFFICIAL_TOKEN_HOP_LEN = 25 * 30
_OFFICIAL_MIN_SEGMENT_TOKENS = 50


class FunAudioChatCosyVoice3Code2Wav(nn.Module):
    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        del prefix
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = self._resolve_model_path(vllm_config.model_config.model)
        self.have_multimodal_outputs = True
        self.enable_update_additional_information = False
        self.requires_raw_input_tokens = True
        self.hf_config_path = getattr(vllm_config.model_config, "hf_config_path", None)

        from transformers import AutoConfig

        from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

        try:
            AutoConfig.register(CosyVoice3Config.model_type, CosyVoice3Config)
        except ValueError:
            pass

        config_source = self.hf_config_path or self.model_path
        self.config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
        self.code2wav = CosyVoice3Code2Wav(self.config)
        # Keep FunAudioChat's stage-1 flow stack in float32 to match the
        # official runtime without changing global CosyVoice3 behavior.
        self.code2wav.flow_model = self.code2wav.flow_model.float()
        device = vllm_config.device_config.device
        self.code2wav.load_weights(self.model_path, device=device)
        estimator = getattr(self.code2wav.decoder, "estimator", None)
        if estimator is not None and hasattr(estimator, "static_chunk_size"):
            estimator.static_chunk_size = 2 * _OFFICIAL_TOKEN_HOP_LEN
        self._speaker_embedding = self._load_default_speaker_embedding()
        self._max_codec_token_id = int(self.config.flow["vocab_size"]) - 1
        self._max_supported_token_len = self._compute_max_supported_token_len()
        self._dummy_profile_token_len = min(32, self._max_supported_token_len)
        self._logged_dummy_profile_cap = False
        samples_per_mel = int(self.config.hift["istft_params"]["hop_len"])
        for rate in self.config.hift["upsample_rates"]:
            samples_per_mel *= int(rate)
        self._funaudiochat_source_cache_len = (
            int(self.code2wav.mel_cache_len) * samples_per_mel
        )
        self._funaudiochat_hift_overlap_samples: dict[tuple[int, bool], int] = {}
        self._funaudiochat_speech_windows: dict[
            tuple[int, torch.device, torch.dtype], torch.Tensor
        ] = {}
        # Per-request state is fixed-size mel/source/speech continuity only.
        self._code2wav_sample_rate = int(self.config.hift["sampling_rate"])
        self._stream_vocoder_cache_by_req: dict[str, dict[str, Any] | None] = {}
        self._stream_audio_cache_lock = threading.Lock()
        logger.info(
            "FunAudioChat stage-1 stream mode: bounded_flow=true fixed_hift_cache=true "
            "original_noise=true original_attention=true mel_cache=%d source_cache=%d",
            int(self.code2wav.mel_cache_len),
            self._funaudiochat_source_cache_len,
        )

    def _resolve_model_path(self, model_path: str) -> str:
        local_path = Path(model_path)
        if local_path.exists():
            return str(local_path)

        logger.info("Resolving FunAudioChat CosyVoice3 weights to a local snapshot: %s", model_path)
        return snapshot_download(model_path)

    def _load_default_speaker_embedding(self) -> torch.Tensor:
        env_path = os.environ.get("FUN_AUDIO_CHAT_SPK_INFO")
        if env_path:
            spk_path = Path(env_path).expanduser()
        else:
            spk_path = resolve_funaudiochat_root() / "utils" / "new_spk2info.pt"
        if not spk_path.exists():
            raise FileNotFoundError(
                f"Default speaker embedding not found: {spk_path}. "
                "Set FUN_AUDIO_CHAT_SPK_INFO or install Fun-Audio-Chat from source."
            )
        spk_info = torch.load(spk_path, map_location="cpu")
        return spk_info["中文女"]["embedding"].reshape(1, -1).float()

    def _compute_max_supported_token_len(self) -> int:
        max_audio_samples = 300 * int(self.config.hift["sampling_rate"])
        sine_waves = getattr(self.code2wav.hift.m_source, "sine_waves", None)
        if isinstance(sine_waves, torch.Tensor) and sine_waves.ndim >= 2:
            max_audio_samples = int(sine_waves.shape[1])
        samples_per_mel = int(self.config.hift["istft_params"]["hop_len"])
        for rate in self.config.hift["upsample_rates"]:
            samples_per_mel *= int(rate)
        samples_per_token = int(self.config.flow["token_mel_ratio"]) * samples_per_mel
        return max_audio_samples // samples_per_token

    @staticmethod
    def _get_prompt_token_id_batches(sampling_metadata: Any) -> list[torch.Tensor] | None:
        prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        if prompt_token_ids is None:
            return None

        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = prompt_token_ids.detach().to(torch.long)
            if prompt_token_ids.ndim <= 1:
                return [prompt_token_ids.view(-1)]
            return [row.reshape(-1) for row in prompt_token_ids]

        if isinstance(prompt_token_ids, list):
            if len(prompt_token_ids) == 0:
                return None
            if isinstance(prompt_token_ids[0], (list, tuple, torch.Tensor)):
                batches = [torch.as_tensor(item, dtype=torch.long).reshape(-1) for item in prompt_token_ids]
            else:
                batches = [torch.tensor(prompt_token_ids, dtype=torch.long)]
            return batches or None

        return None

    @staticmethod
    def _split_request_ids(ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        return [ids]

    def _build_decode_tokens(
        self,
        input_ids: torch.Tensor,
        sampling_metadata: Any,
        seq_token_counts: list[int] | None = None,
    ) -> tuple[list[torch.Tensor], bool]:
        prompt_token_id_batches = self._get_prompt_token_id_batches(sampling_metadata)
        if prompt_token_id_batches is not None:
            raw_id_batches = prompt_token_id_batches
        elif input_ids is not None:
            raw_id_batches = self._split_request_ids(input_ids.reshape(-1), seq_token_counts)
        else:
            raw_id_batches = [torch.empty((0,), dtype=torch.long)]

        token_batches = [
            raw_ids.reshape(1, -1)
            .to(dtype=torch.long, device=self.vllm_config.device_config.device)
            .clamp_(
                min=0,
                max=self._max_codec_token_id,
            )
            for raw_ids in raw_id_batches
        ]

        is_dummy_profile = bool(
            sampling_metadata is None
            and prompt_token_id_batches is None
            and len(token_batches) == 1
            and (token_batches[0].numel() == 0 or torch.count_nonzero(token_batches[0]).item() == 0)
        )
        if is_dummy_profile and token_batches[0].shape[1] > self._dummy_profile_token_len:
            if not self._logged_dummy_profile_cap:
                logger.debug(
                    "FunAudioChat code2wav dummy/profile run detected. Capping decode length from %d to %d tokens.",
                    token_batches[0].shape[1],
                    self._dummy_profile_token_len,
                )
                self._logged_dummy_profile_cap = True
            token_batches[0] = token_batches[0][:, : self._dummy_profile_token_len]

        return token_batches, is_dummy_profile

    @staticmethod
    def _split_tokens_like_official(token: torch.Tensor) -> list[torch.Tensor]:
        flat = token.reshape(-1)
        if flat.numel() == 0:
            return [flat]

        segments: list[torch.Tensor] = []
        time_step = 0
        while time_step * 25 < flat.numel():
            start = time_step * 25
            end = min((time_step + 30) * 25, flat.numel())
            segments.append(flat[start:end])
            time_step += 30

        if len(segments) > 1 and segments[-1].numel() < _OFFICIAL_MIN_SEGMENT_TOKENS:
            merged = torch.cat([segments[-2], segments[-1]], dim=0)
            split_point = merged.numel() // 2
            segments = [*segments[:-2], merged[:split_point], merged[split_point:]]

        return segments

    @staticmethod
    def _fade_in_out(fade_in_tensor: torch.Tensor, fade_out_tensor: torch.Tensor, window: Any) -> torch.Tensor:
        if fade_in_tensor.numel() == 0 or fade_out_tensor.numel() == 0:
            return fade_in_tensor

        overlap = min(int(len(window) // 2), fade_in_tensor.shape[-1], fade_out_tensor.shape[-1])
        if overlap <= 0:
            return fade_in_tensor

        fade_window = torch.as_tensor(window, device=fade_in_tensor.device, dtype=fade_in_tensor.dtype)
        # The caller retains the unmodified chunk for cache bookkeeping, so
        # blend into a separate output tensor.
        mixed = fade_in_tensor.clone()
        mixed[..., :overlap] = (
            mixed[..., :overlap] * fade_window[:overlap] + fade_out_tensor[..., -overlap:] * fade_window[-overlap:]
        )
        return mixed

    def _run_flow_like_official(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        *,
        finalize: bool,
        flow_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flow_model = self.code2wav.flow_model
        device = token.device
        token = token.to(device=device, dtype=torch.long)
        prompt_token = prompt_token.to(device=device, dtype=torch.long)
        prompt_feat = prompt_feat.to(device=device, dtype=torch.float32)
        embedding = embedding.to(device=device, dtype=torch.float32)
        embedding = F.normalize(embedding, dim=1)
        embedding = flow_model.spk_embed_affine_layer(embedding)

        token_len = torch.tensor([token.shape[1]], dtype=torch.int32, device=device)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=device)
        full_token = torch.cat([prompt_token, token], dim=1)
        full_token_len = prompt_token_len + token_len
        mask = (~make_pad_mask(full_token_len)).unsqueeze(-1).to(embedding)
        token_emb = flow_model.input_embedding(torch.clamp(full_token, min=0)) * mask

        if finalize:
            h = flow_model.pre_lookahead_layer(token_emb)
        else:
            h = flow_model.pre_lookahead_layer(
                token_emb[:, : -flow_model.pre_lookahead_len],
                context=token_emb[:, -flow_model.pre_lookahead_len :],
            )
        h = h.repeat_interleave(flow_model.token_mel_ratio, dim=1)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1
        conds = torch.zeros(
            [1, mel_len1 + mel_len2, flow_model.output_size],
            device=device,
            dtype=h.dtype,
        )
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mel_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2], device=device))).to(h)
        decoder_kwargs = {
            "mu": h.transpose(1, 2).contiguous(),
            "mask": mel_mask.unsqueeze(1),
            "spks": embedding,
            "cond": conds,
            "n_timesteps": 10,
        }
        try:
            decoder_out = flow_model.decoder(cache=flow_cache, **decoder_kwargs)
        except TypeError as exc:
            if "cache" not in str(exc):
                raise
            decoder_out = flow_model.decoder(**decoder_kwargs)

        if isinstance(decoder_out, tuple):
            feat, next_flow_cache = decoder_out
        else:
            feat, next_flow_cache = decoder_out, flow_cache
        feat = feat[:, :, mel_len1:]
        return feat.float(), next_flow_cache

    def _run_hift_like_official(
        self,
        speech_feat: torch.Tensor,
        *,
        finalize: bool,
        cache_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hift = self.code2wav.hift
        speech_feat = speech_feat.to(dtype=torch.float32)
        hift.f0_predictor.to("cpu")
        f0 = hift.f0_predictor(speech_feat.cpu(), finalize=finalize).to(speech_feat)
        source = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _, _ = hift.m_source(source)
        source = source.transpose(1, 2)
        if cache_source.shape[2] != 0:
            source_overlap = min(source.shape[2], cache_source.shape[2])
            source[:, :, :source_overlap] = cache_source[:, :, :source_overlap]

        if finalize:
            speech = hift.decode(x=speech_feat, s=source, finalize=True)
        else:
            padding = hift.f0_predictor.condnet[0].causal_padding
            speech = hift.decode(x=speech_feat[:, :, :-padding], s=source, finalize=False)
        return speech, source

    def _run_streaming_flow_segment(
        self,
        token: torch.Tensor,
        *,
        token_offset: int,
        finalize: bool,
    ) -> torch.Tensor:
        """Generate only the new mel suffix from one bounded codec segment."""
        device = token.device
        return self.code2wav._forward_mel(
            token=token.unsqueeze(0),
            prompt_token=torch.zeros((1, 0), dtype=torch.long, device=device),
            prompt_feat=torch.zeros((1, 0, 80), dtype=torch.float32, device=device),
            embedding=self._speaker_embedding.to(device=device, dtype=torch.float32),
            n_timesteps=10,
            token_offset_tokens=token_offset,
            streaming=True,
            finalize=finalize,
        )

    def _get_streaming_hift_overlap_samples(
        self,
        cached_mel: torch.Tensor,
        *,
        finalize: bool,
    ) -> int:
        """Measure the waveform duration represented by the cached mel tail."""
        key = (int(cached_mel.shape[2]), bool(finalize))
        cached = self._funaudiochat_hift_overlap_samples.get(key)
        if cached is not None:
            return cached

        empty_source = torch.zeros(
            (cached_mel.shape[0], 1, 0),
            device=cached_mel.device,
            dtype=cached_mel.dtype,
        )
        probe_speech, _ = self._run_hift_like_official(
            cached_mel,
            finalize=finalize,
            cache_source=empty_source,
        )
        measured = int(probe_speech.reshape(probe_speech.shape[0], -1).shape[1])
        with self._stream_audio_cache_lock:
            measured = self._funaudiochat_hift_overlap_samples.setdefault(key, measured)
        logger.debug(
            "FunAudioChat HiFT measured overlap: mel=%d finalize=%s samples=%d",
            key[0],
            key[1],
            measured,
        )
        return measured

    def _get_streaming_speech_window(
        self,
        overlap_samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (int(overlap_samples), device, dtype)
        window = self._funaudiochat_speech_windows.get(key)
        if window is None:
            window = torch.hamming_window(
                2 * overlap_samples,
                periodic=False,
                device=device,
                dtype=dtype,
            )
            self._funaudiochat_speech_windows[key] = window
        return window

    def _run_streaming_hift_chunk(
        self,
        chunk_mel: torch.Tensor,
        cache_state: dict[str, Any] | None,
        *,
        finalize: bool,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Vocode new mel with fixed-size official-style continuity caches."""
        hift_weight = self.code2wav.hift.m_source.l_linear.weight
        chunk_mel = chunk_mel.to(device=hift_weight.device, dtype=hift_weight.dtype)
        state = cache_state or {}

        cached_mel = state.get("hift_mel")
        if isinstance(cached_mel, torch.Tensor) and cached_mel.numel() > 0:
            cached_mel = cached_mel.to(device=chunk_mel.device, dtype=chunk_mel.dtype)
            tts_mel = torch.cat([cached_mel, chunk_mel], dim=2)
        else:
            tts_mel = chunk_mel

        if tts_mel.shape[2] == 0:
            empty = torch.zeros(
                (tts_mel.shape[0], 0),
                device=tts_mel.device,
                dtype=tts_mel.dtype,
            )
            return empty, None if finalize else cache_state

        cached_source = state.get("hift_source")
        if isinstance(cached_source, torch.Tensor) and cached_source.numel() > 0:
            cached_source = cached_source.to(
                device=tts_mel.device,
                dtype=tts_mel.dtype,
            )
        else:
            cached_source = torch.zeros(
                (tts_mel.shape[0], 1, 0),
                device=tts_mel.device,
                dtype=tts_mel.dtype,
            )

        tts_speech, tts_source = self._run_hift_like_official(
            tts_mel,
            finalize=finalize,
            cache_source=cached_source,
        )
        tts_speech = tts_speech.reshape(tts_speech.shape[0], -1)

        cached_speech = state.get("hift_speech")
        if (
            isinstance(cached_mel, torch.Tensor)
            and cached_mel.numel() > 0
            and isinstance(cached_speech, torch.Tensor)
            and cached_speech.numel() > 0
        ):
            cached_speech = cached_speech.to(
                device=tts_speech.device,
                dtype=tts_speech.dtype,
            )
            prefix_samples = self._get_streaming_hift_overlap_samples(
                cached_mel,
                finalize=finalize,
            )
            held_samples = int(cached_speech.shape[1])
            if prefix_samples > held_samples:
                trim_samples = min(
                    prefix_samples - held_samples,
                    int(tts_speech.shape[1]),
                )
                tts_speech = tts_speech[:, trim_samples:]
            elif prefix_samples < held_samples:
                missing_samples = held_samples - prefix_samples
                tts_speech = torch.cat(
                    [cached_speech[:, :missing_samples], tts_speech],
                    dim=1,
                )
            speech_window = self._get_streaming_speech_window(
                held_samples,
                device=tts_speech.device,
                dtype=tts_speech.dtype,
            )
            tts_speech = self._fade_in_out(
                tts_speech,
                cached_speech,
                speech_window,
            )

        if finalize:
            return tts_speech, None

        mel_cache_len = int(self.code2wav.mel_cache_len)
        next_cached_mel = tts_mel[:, :, -mel_cache_len:].detach().contiguous()
        speech_overlap_len = self._get_streaming_hift_overlap_samples(
            next_cached_mel,
            finalize=False,
        )
        source_cache_len = int(self._funaudiochat_source_cache_len)
        new_state: dict[str, Any] = {
            "hift_mel": next_cached_mel,
            "hift_source": tts_source[
                :, :, -source_cache_len:
            ].detach().contiguous(),
            "hift_speech": (
                tts_speech[:, -speech_overlap_len:].detach().contiguous()
                if speech_overlap_len > 0
                else tts_speech[:, :0].detach().contiguous()
            ),
        }
        if speech_overlap_len <= 0:
            emitted_speech = tts_speech
        elif tts_speech.shape[1] > speech_overlap_len:
            emitted_speech = tts_speech[:, :-speech_overlap_len]
        else:
            emitted_speech = tts_speech[:, :0]
        return emitted_speech, new_state

    def _decode_segment_like_official(
        self,
        token_segment: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if token_segment.numel() == 0:
            return torch.zeros((0,), device=embedding.device, dtype=torch.float32)

        device = token_segment.device
        flow_cache = torch.zeros((1, 80, 0, 2), device=device, dtype=torch.float32)
        mel_overlap = torch.zeros((1, self.code2wav.output_size, 0), device=device, dtype=torch.float32)
        hift_cache: dict[str, torch.Tensor] | None = None
        pre_lookahead_len = int(self.config.flow["pre_lookahead_len"])
        token_offset = 0
        speech_chunks: list[torch.Tensor] = []

        while token_offset < token_segment.numel():
            chunk_len = min(token_offset + _OFFICIAL_TOKEN_HOP_LEN + pre_lookahead_len, token_segment.numel())
            chunk = token_segment[:chunk_len].reshape(1, -1)
            finalize = chunk.shape[1] == token_segment.numel()
            tts_mel, flow_cache = self._run_flow_like_official(
                chunk,
                prompt_token,
                prompt_feat,
                embedding,
                finalize=finalize,
                flow_cache=flow_cache,
            )
            if mel_overlap.shape[2] != 0:
                tts_mel = self._fade_in_out(tts_mel, mel_overlap, self.code2wav.mel_window)

            if hift_cache is not None:
                cache_source = hift_cache["source"]
                tts_mel = torch.cat([hift_cache["mel"], tts_mel], dim=2)
            else:
                cache_source = torch.zeros((1, 1, 0), device=device, dtype=tts_mel.dtype)

            if not finalize:
                mel_overlap = tts_mel[:, :, -self.code2wav.mel_overlap_len :]
                tts_mel = tts_mel[:, :, : -self.code2wav.mel_overlap_len]
                if tts_mel.shape[2] == 0:
                    token_offset += _OFFICIAL_TOKEN_HOP_LEN
                    continue
                tts_speech, tts_source = self._run_hift_like_official(
                    tts_mel,
                    finalize=False,
                    cache_source=cache_source,
                )
                if hift_cache is not None:
                    tts_speech = self._fade_in_out(tts_speech, hift_cache["speech"], self.code2wav.speech_window)
                hift_cache = {
                    "mel": tts_mel[:, :, -self.code2wav.mel_cache_len :],
                    "source": tts_source[:, :, -self.code2wav.source_cache_len :],
                    "speech": tts_speech[:, -self.code2wav.source_cache_len :],
                }
                if tts_speech.shape[1] > self.code2wav.source_cache_len:
                    tts_speech = tts_speech[:, : -self.code2wav.source_cache_len]
                else:
                    tts_speech = tts_speech[:, :0]
            else:
                tts_speech, _ = self._run_hift_like_official(
                    tts_mel,
                    finalize=True,
                    cache_source=cache_source,
                )
                if hift_cache is not None:
                    tts_speech = self._fade_in_out(tts_speech, hift_cache["speech"], self.code2wav.speech_window)

            if tts_speech.numel() > 0:
                speech_chunks.append(tts_speech.reshape(-1))

            token_offset += _OFFICIAL_TOKEN_HOP_LEN

        if not speech_chunks:
            return torch.zeros((0,), device=device, dtype=torch.float32)
        return torch.cat(speech_chunks, dim=0)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids is None or input_ids.numel() == 0:
            return torch.empty((0, 1), dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        del hidden_states, sampling_metadata
        return None

    def _decode_request_streaming(
        self,
        token: torch.Tensor,
        req_id: str | None,
        meta: Any,
        *,
        finalize: bool,
    ) -> torch.Tensor:
        """Decode one bounded Flow segment with fixed-size HiFT state."""
        device = token.device
        token_offset = max(0, int(getattr(meta, "left_context_size", 0) or 0))
        segment_start = max(
            0,
            int(getattr(meta, "num_processed_tokens", 0) or 0),
        )

        cache_state = None
        if req_id is not None:
            with self._stream_audio_cache_lock:
                cache_state = self._stream_vocoder_cache_by_req.get(req_id)
        else:
            logger.warning_once(
                "FunAudioChat streaming chunk has no req_id; per-request HiFT cache continuity is disabled"
            )
        logger.debug(
            "FAC stage1 bounded Flow: req=%s token_len=%d token_offset=%d "
            "segment_start=%d cache_hit=%s finalize=%s",
            req_id,
            int(token.numel()),
            token_offset,
            segment_start,
            cache_state is not None,
            finalize,
        )
        if token.numel() > 0:
            chunk_mel = self._run_streaming_flow_segment(
                token,
                token_offset=token_offset,
                finalize=finalize,
            )
        else:
            chunk_mel = torch.zeros(
                (1, int(self.code2wav.output_size), 0),
                device=device,
                dtype=torch.float32,
            )

        tts_speech, new_cache_state = self._run_streaming_hift_chunk(
            chunk_mel,
            cache_state,
            finalize=finalize,
        )

        if req_id is not None:
            with self._stream_audio_cache_lock:
                if new_cache_state is None or finalize:
                    self._stream_vocoder_cache_by_req.pop(req_id, None)
                else:
                    self._stream_vocoder_cache_by_req[req_id] = new_cache_state

        return tts_speech.reshape(-1).to(dtype=torch.float32)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Release streaming vocoder state for completed or aborted requests."""
        with self._stream_audio_cache_lock:
            for req_id in finished_req_ids:
                self._stream_vocoder_cache_by_req.pop(req_id, None)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        seq_token_counts: list[int] | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds

        token_batches, is_dummy_profile = self._build_decode_tokens(
            input_ids,
            sampling_metadata,
            seq_token_counts,
        )
        num_reqs = len(token_batches)
        empty = torch.zeros((0,), dtype=torch.float32)
        sr = torch.tensor(self._code2wav_sample_rate, dtype=torch.int32)
        if not token_batches:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [empty] * max(num_reqs, 1), "sr": [sr] * max(num_reqs, 1)},
            )

        # async_chunk path: the generation runner injects per-request chunk
        # metadata (OmniPayloadStruct with meta/codes) via model_intermediate_buffer
        # (gpu_model_runner.py:1338). When a chunk carries route fields
        # (stream_finished/left_context_size) we decode it incrementally with the
        # carried vocoder cache; otherwise we fall back to the full-segment path.
        runtime_info = model_intermediate_buffer
        if runtime_info is None:
            runtime_info = runtime_additional_information
        if runtime_info is not None and not isinstance(runtime_info, list):
            runtime_info = []

        stream_metas: list[Any | None] = []
        for idx in range(num_reqs):
            raw = (
                runtime_info[idx]
                if runtime_info is not None and idx < len(runtime_info) and isinstance(runtime_info[idx], dict)
                else {}
            )
            # Non-streaming calls may carry unrelated keys, so only validate
            # payloads that declare streaming route fields. Malformed streaming
            # payloads should fail validation instead of being silently ignored.
            raw_meta = raw.get("meta")
            uses_streaming = raw_meta is not None and (
                (
                    isinstance(raw_meta, dict)
                    and ("stream_finished" in raw_meta or "left_context_size" in raw_meta)
                )
                or getattr(raw_meta, "stream_finished", None) is not None
                or getattr(raw_meta, "left_context_size", None) is not None
            )
            stream_metas.append(to_struct(raw).meta if uses_streaming else None)

        has_streaming_metadata = any(meta is not None for meta in stream_metas)
        if is_dummy_profile and not has_streaming_metadata:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [empty.to(device=token_batches[0].device)], "sr": [sr]},
            )
        if all(token.numel() == 0 for token in token_batches) and not has_streaming_metadata:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [empty] * max(num_reqs, 1), "sr": [sr] * max(num_reqs, 1)},
            )

        audios: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        for idx, token in enumerate(token_batches):
            meta = stream_metas[idx]
            if meta is not None:
                stream_finished = meta.stream_finished
                finalize = bool(stream_finished is not None and bool(stream_finished.item()))
                req_id = meta.req_id[0] if (meta.req_id) else None
                token = token.reshape(-1).to(dtype=torch.long)
                audio = self._decode_request_streaming(token, req_id, meta, finalize=finalize)
                audios.append(audio.detach().cpu().reshape(-1))
                srs.append(sr)
                continue
            if token.numel() == 0:
                audios.append(empty)
                srs.append(sr)
                continue
            # Non-streaming fallback (sync full-payload / direct-call path): decode
            # the whole request in one shot, exactly as before.
            prompt_token = torch.zeros((1, 0), dtype=torch.long, device=token.device)
            prompt_feat = torch.zeros((1, 0, 80), dtype=torch.float32, device=token.device)
            embedding = self._speaker_embedding.to(device=token.device, dtype=torch.float32)
            audio_segments: list[torch.Tensor] = []
            for token_segment in self._split_tokens_like_official(token):
                if token_segment.numel() == 0:
                    continue
                segment_audio = self._decode_segment_like_official(
                    token_segment,
                    prompt_token,
                    prompt_feat,
                    embedding,
                )
                if segment_audio.numel() > 0:
                    audio_segments.append(segment_audio.reshape(-1))
            audio = torch.cat(audio_segments, dim=0) if audio_segments else torch.zeros((0,), device=token.device)
            audios.append(audio.reshape(-1).detach().cpu())
            srs.append(sr)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": audios, "sr": srs},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        # All parameters are loaded eagerly from the local snapshot in `__init__`.
        return {name for name, _ in self.named_parameters()}
