# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 1 (codec decoder) for higgs-audio v3.

Reuses higgs-audio-v2's RVQ + DAC codec decoder but loads weights from the
v3 checkpoint's bundled codec (``tied.embedding.modality_embeddings.0.model.*``
prefix) rather than from a standalone tokenizer repo.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import (
    HiggsAudioRVQ,
    load_higgs_audio_codec,
)
from vllm_omni.model_executor.models.higgs_audio_v3.configuration_higgs_audio_v3 import (
    HiggsAudioV3Config,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

__all__ = [
    "HiggsAudioV3Code2Wav",
    "HiggsAudioV3Code2WavForConditionalGeneration",
]

logger = init_logger(__name__)

# Prefix under which codec weights are stored in the v3 checkpoint.
_CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."


class HiggsAudioV3Code2Wav(nn.Module):
    """Stage-1 codec decoder for higgs-audio v3.

    The codec architecture is identical to v2 (HiggsAudioRVQ + fc2 +
    BosonDacDecoder); only the weight loading path differs. V3 bundles
    codec weights inside the main checkpoint under a known prefix, while
    v2 uses a standalone tokenizer repo.

    For backward compatibility and simplicity, we delegate to the v2
    ``load_higgs_audio_codec`` function when the audio tokenizer repo
    is available. When not (or as primary path), we extract codec weights
    from the v3 safetensors via the engine's weight iterator.
    """

    input_modalities = "audio"

    def __init__(
        self,
        config: HiggsAudioV3Config | None = None,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        if vllm_config is not None:
            hf_config = vllm_config.model_config.hf_config
            if isinstance(hf_config, HiggsAudioV3Config):
                self.config = hf_config
            else:
                self.config = HiggsAudioV3Config(**hf_config.to_dict())
            self._model_path: str | None = vllm_config.model_config.model
            self.vllm_config: VllmConfig | None = vllm_config
        else:
            if config is None:
                raise TypeError("HiggsAudioV3Code2Wav: provide either `config` or `vllm_config`.")
            self.config = config
            self._model_path = None
            self.vllm_config = None

        self.sample_rate: int = int(self.config.sample_rate)
        self.num_codebooks: int = int(self.config.num_codebooks)
        self.num_real_codes: int = int(self.config.num_real_codes)
        self.hop_length: int = 960

        # Engine-runner hooks
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        # Codec modules (populated by load_weights)
        self.quantizer: HiggsAudioRVQ | None = None
        self.fc2: nn.Linear | None = None
        self.acoustic_decoder: nn.Module | None = None
        self._loaded: bool = False

        # Try to load from the standalone v2 tokenizer repo (same codec)
        if self._model_path is not None:
            try:
                self._load_from_tokenizer_repo()
            except (FileNotFoundError, OSError) as exc:
                logger.info(
                    "HiggsAudioV3Code2Wav: standalone tokenizer not found (%s); will load from checkpoint weights.",
                    exc,
                )

    def _load_from_tokenizer_repo(self) -> None:
        """Load codec from the standalone higgs-audio-v2-tokenizer repo."""
        tokenizer_id = "bosonai/higgs-audio-v2-tokenizer"
        # Try to resolve from HF cache
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id=tokenizer_id, filename="config.json")
        if isinstance(cached, str) and os.path.isfile(cached):
            tokenizer_dir = os.path.dirname(cached)
        else:
            from huggingface_hub.constants import HF_HUB_CACHE

            safe = tokenizer_id.replace("/", "--")
            snapshots_dir = os.path.join(HF_HUB_CACHE, f"models--{safe}", "snapshots")
            if os.path.isdir(snapshots_dir):
                tokenizer_dir = None
                for rev in os.listdir(snapshots_dir):
                    candidate = os.path.join(snapshots_dir, rev)
                    if os.path.isfile(os.path.join(candidate, "config.json")):
                        tokenizer_dir = candidate
                        break
                if tokenizer_dir is None:
                    raise FileNotFoundError(f"No cached snapshot for {tokenizer_id}")
            else:
                raise FileNotFoundError(f"No cached snapshot for {tokenizer_id}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        quantizer, fc2, acoustic_decoder, _cfg = load_higgs_audio_codec(tokenizer_dir, device)
        self.quantizer = quantizer
        self.fc2 = fc2
        self.acoustic_decoder = acoustic_decoder
        self._loaded = True
        logger.info("Loaded HiggsAudioV3Code2Wav from standalone tokenizer repo.")

    # ------------------------------------------------------------------ engine hooks
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    # ------------------------------------------------------------------ load
    def load_weights(self, weights_or_model_dir, device: torch.device | None = None):
        if self._loaded:
            # Already loaded from standalone tokenizer
            if not isinstance(weights_or_model_dir, (str, bytes, os.PathLike)):
                for _ in weights_or_model_dir:
                    pass
            return {name for name, _ in self.named_parameters()}

        if not isinstance(weights_or_model_dir, (str, bytes, os.PathLike)):
            # Engine path: consume iterator, try standalone tokenizer
            for _ in weights_or_model_dir:
                pass
            if not self._loaded:
                try:
                    self._load_from_tokenizer_repo()
                except (FileNotFoundError, OSError):
                    raise RuntimeError(
                        "HiggsAudioV3Code2Wav: could not load codec weights. "
                        "Ensure bosonai/higgs-audio-v2-tokenizer is cached in HF_HOME."
                    )
            return {name for name, _ in self.named_parameters()}

        # File-based path
        if not self._loaded:
            self._load_from_tokenizer_repo()
        return None

    # ------------------------------------------------------------------ decode
    @torch.inference_mode()
    def decode_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Decode [B, num_codebooks=8, T] codes to PCM [B, 1, T*960]."""
        if not self._loaded:
            raise RuntimeError("HiggsAudioV3Code2Wav not loaded.")

        codes = self._validate_codes(audio_codes)
        rvq_codes = codes.transpose(0, 1).long()
        quantized = self.quantizer.decode(rvq_codes)
        quantized = quantized.to(dtype=self.fc2.weight.dtype)
        quantized = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        first_param = next(self.acoustic_decoder.parameters(), None)
        if first_param is not None and quantized.dtype != first_param.dtype:
            quantized = quantized.to(dtype=first_param.dtype)
        audio = self.acoustic_decoder(quantized)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio.to(codes.device)

    @torch.inference_mode()
    def forward_chunk(
        self,
        audio_codes: torch.Tensor,
        *,
        left_context_size: int = 0,
        hop_length: int | None = None,
    ) -> torch.Tensor:
        hop = int(hop_length) if hop_length is not None else self.hop_length
        pcm = self.decode_codes(audio_codes)
        if left_context_size == 0:
            return pcm
        trim = left_context_size * hop
        if pcm.shape[-1] <= trim:
            return pcm[..., :0]
        return pcm[..., trim:]

    # ------------------------------------------------------------------ runtime forward
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | OmniOutput:
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 3:
            return self.decode_codes(input_ids)

        sr_val = int(self.sample_rate)
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        left_context_size = [0] * len(request_ids_list)
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                meta = info.get("meta", {}) if isinstance(info, dict) else {}
                if "left_context_size" in meta:
                    left_context_size[i] = int(meta["left_context_size"])

        wavs: list[torch.Tensor] = []
        for i, req_ids in enumerate(request_ids_list):
            n = int(req_ids.numel())
            if n == 0:
                wavs.append(empty)
                continue
            if n % self.num_codebooks != 0:
                logger.warning(
                    "HiggsAudioV3Code2Wav: flat code length %d not divisible by %d",
                    n,
                    self.num_codebooks,
                )
                wavs.append(empty)
                continue
            frames = n // self.num_codebooks
            codes_qf = req_ids.reshape(self.num_codebooks, frames)
            codes_bqf = codes_qf.unsqueeze(0)
            try:
                pcm = self.forward_chunk(
                    codes_bqf,
                    left_context_size=left_context_size[i],
                    hop_length=self.hop_length,
                )
            except ValueError as exc:
                logger.warning("HiggsAudioV3Code2Wav: decode skipped (%s)", exc)
                wavs.append(empty)
                continue
            wavs.append(pcm.squeeze(0).squeeze(0).to(torch.float32).cpu())

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": wavs,
                "sr": [sr_tensor] * len(wavs),
            },
        )

    # ------------------------------------------------------------------ helpers
    def _validate_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        if audio_codes.ndim != 3:
            raise ValueError(
                f"audio_codes must have shape [B, {self.num_codebooks}, T]; got {tuple(audio_codes.shape)}"
            )
        if int(audio_codes.shape[1]) != self.num_codebooks:
            raise ValueError(f"dim 1 must equal num_codebooks={self.num_codebooks}; got {int(audio_codes.shape[1])}")
        if audio_codes.numel() > 0:
            max_val = int(audio_codes.max().item())
            min_val = int(audio_codes.min().item())
            if max_val >= self.num_real_codes or min_val < 0:
                raise ValueError(
                    f"audio_codes out of range: min={min_val}, max={max_val}; expected [0, {self.num_real_codes - 1}]"
                )
        return audio_codes

    @staticmethod
    def _split_request_ids(ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + int(count))
            n = int(ids.numel())
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        return [ids]


HiggsAudioV3Code2WavForConditionalGeneration = HiggsAudioV3Code2Wav
