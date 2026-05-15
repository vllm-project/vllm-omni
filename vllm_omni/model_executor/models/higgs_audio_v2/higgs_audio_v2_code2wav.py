# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 1 (codec decoder) for higgs-audio v2.

Two surfaces exposed:

1. **Direct decode API** (kept for offline / unit tests):
   ``decode_codes(audio_codes: [B, num_codebooks=8, T])`` -> ``[B, 1, T*hop]`` PCM.
   ``forward_chunk(audio_codes, *, left_context_size, hop_length=960)``
   trims overlap from streamed PCM. ``forward(audio_codes)`` retains the same
   single-tensor signature for backward compatibility with code that imports
   the class directly.

2. **vLLM stage runtime API** (used by the engine):
   ``__init__(*, vllm_config, prefix)``, ``embed_input_ids``,
   ``compute_logits=None``, plus a runtime ``forward(input_ids=..., positions=...,
   runtime_additional_information=...)`` that takes flat codebook-major
   ``input_ids`` (``[Q * num_frames]``) per request and returns an
   :class:`vllm_omni.model_executor.models.output_templates.OmniOutput`.

The kernel is the shared HiggsAudio codec helper in
``vllm_omni/model_executor/models/_shared/higgs_audio_decoder.py``; both surfaces
share the same RVQ + DAC weights.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models._shared.higgs_audio_decoder import (
    HiggsAudioRVQ,
    load_higgs_audio_codec,
)
from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
    HiggsAudioV2Config,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

__all__ = [
    "HiggsAudioV2Code2Wav",
    "HiggsAudioV2Code2WavForConditionalGeneration",
]

logger = init_logger(__name__)


class HiggsAudioV2Code2Wav(nn.Module):
    """Stage-1 codec decoder for higgs-audio v2.

    Constructor accepts either the vLLM-runtime signature (``*, vllm_config,
    prefix``) or the direct config-object form (``HiggsAudioV2Code2Wav(config)``)
    so both engine-side and unit-test callers work. When the runtime form is
    used we read the higgs_audio_v2 config out of ``vllm_config.model_config.hf_config``
    and the model_path off ``vllm_config.model_config.model`` so the codec can
    be loaded directly from disk.
    """

    input_modalities = "audio"

    def __init__(
        self,
        config: HiggsAudioV2Config | None = None,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        # Resolve config from either positional (unit-test) or kwarg (engine) form.
        if vllm_config is not None:
            hf_config = vllm_config.model_config.hf_config
            if isinstance(hf_config, HiggsAudioV2Config):
                self.config = hf_config
            else:
                self.config = HiggsAudioV2Config(**hf_config.to_dict())
            self._model_path: str | None = vllm_config.model_config.model
            self.vllm_config: VllmConfig | None = vllm_config
        else:
            if config is None:
                raise TypeError(
                    "HiggsAudioV2Code2Wav: provide either positional `config` "
                    "(HiggsAudioV2Config) or keyword `vllm_config` (VllmConfig)."
                )
            self.config = config
            self._model_path = None
            self.vllm_config = None

        self.sample_rate: int = int(self.config.sample_rate)
        self.num_codebooks: int = int(self.config.num_codebooks)
        self.num_real_codes: int = int(self.config.num_real_codes)
        # Each codec frame upsamples to 960 24 kHz samples (= 25 fps * 960 = 24000).
        self.hop_length: int = 960

        # Engine-runner hooks (Stage 1 has no token sampling).
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        # Populated by load_weights().
        self.quantizer: HiggsAudioRVQ | None = None
        self.fc2: nn.Linear | None = None
        self.acoustic_decoder: nn.Module | None = None
        self._loaded: bool = False

        # When constructed via the engine path, eagerly load weights from the
        # model directory so the runner sees a fully-initialized module after
        # construction (matches the eager-init pattern that
        # ``Qwen3TTSCode2Wav`` follows for its decoder).
        if self._model_path is not None:
            try:
                self.load_weights(self._model_path)
            except FileNotFoundError as exc:
                # Allow construction-time deferral when the checkpoint hasn't
                # been downloaded yet (the engine's loader will retry later).
                logger.warning(
                    "HiggsAudioV2Code2Wav: eager codec load deferred (%s)", exc
                )

    # ------------------------------------------------------------- engine hooks
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Stage 1 ignores embeddings; vLLM's runner still needs a stable shape."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    # ------------------------------------------------------------------ load
    def load_weights(self, model_dir: str, device: torch.device | None = None) -> None:
        """Load codec weights for Stage 1.

        ``model_dir`` may be either the standalone tokenizer repo (containing
        ``config.json`` + ``model.safetensors`` at the root) or the OmniVoice-
        style bundle that nests the codec under ``<model_dir>/audio_tokenizer/``.
        Controlled by :attr:`HiggsAudioV2Config.audio_tokenizer_subdir`.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        subdir = self.config.audio_tokenizer_subdir or ""
        audio_tokenizer_path = os.path.join(model_dir, subdir) if subdir else model_dir
        quantizer, fc2, acoustic_decoder, _tokenizer_config = load_higgs_audio_codec(
            audio_tokenizer_path, device
        )
        if len(quantizer.quantizers) != self.num_codebooks:
            raise ValueError(
                f"checkpoint has {len(quantizer.quantizers)} quantizers but config.num_codebooks={self.num_codebooks}"
            )
        self.quantizer = quantizer
        self.fc2 = fc2
        self.acoustic_decoder = acoustic_decoder
        self._loaded = True
        logger.info(
            "Loaded HiggsAudioV2Code2Wav: %d quantizers, fc2(%d->%d), sample_rate=%d",
            len(self.quantizer.quantizers),
            self.fc2.in_features,
            self.fc2.out_features,
            self.sample_rate,
        )

    # ------------------------------------------------------ direct decode API
    @torch.inference_mode()
    def decode_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Decode a ``[B, num_codebooks=8, T]`` code tensor to PCM ``[B, 1, T*hop]``."""
        if not self._loaded:
            raise RuntimeError("HiggsAudioV2Code2Wav not loaded. Call load_weights() first.")

        codes = self._validate_codes(audio_codes)
        device = codes.device

        rvq_codes = codes.transpose(0, 1).long()  # [num_codebooks, B, T]
        quantized = self.quantizer.decode(rvq_codes)  # [B, hidden, T]
        quantized = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        audio = self.acoustic_decoder(quantized)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio.to(device)

    @torch.inference_mode()
    def forward_chunk(
        self,
        audio_codes: torch.Tensor,
        *,
        left_context_size: int = 0,
        hop_length: int | None = None,
    ) -> torch.Tensor:
        """Chunked decode that trims ``left_context_size * hop_length`` samples
        off the leading edge of the upsampled PCM. Mirrors the qwen3_tts
        ``talker2code2wav_async_chunk`` overlap contract.
        """
        if left_context_size < 0:
            raise ValueError(f"left_context_size must be >= 0; got {left_context_size}")
        hop = int(hop_length) if hop_length is not None else self.hop_length
        if hop <= 0:
            raise ValueError(f"hop_length must be > 0; got {hop}")
        pcm = self.decode_codes(audio_codes)
        if left_context_size == 0:
            return pcm
        trim = left_context_size * hop
        if pcm.shape[-1] <= trim:
            return pcm[..., :0]
        return pcm[..., trim:]

    # ---------------------------------------------------- vLLM runtime forward
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
        """Dual-mode forward.

        - When ``input_ids`` is a 3-D ``[B, num_codebooks, T]`` tensor we treat
          this as a direct decode call (legacy API) and return raw PCM.
        - When ``input_ids`` is a flat 1-D / 2-D tensor we treat it as the
          engine runtime payload (codebook-major flat per request) and return
          an :class:`OmniOutput` carrying multimodal audio.
        """
        # Legacy direct-decode signature: caller passed a 3-D code tensor as input_ids.
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
                    "HiggsAudioV2Code2Wav: flat code length %d not divisible by num_codebooks=%d; dropping",
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
                logger.warning("HiggsAudioV2Code2Wav: decode skipped (%s)", exc)
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

    # --------------------------------------------------------------- helpers
    def _validate_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Ensure shape and value range; reject stream specials with ValueError."""
        if not isinstance(audio_codes, torch.Tensor):
            raise TypeError(f"audio_codes must be a torch.Tensor, got {type(audio_codes)!r}")
        if audio_codes.ndim != 3:
            raise ValueError(
                f"audio_codes must have shape [B, num_codebooks={self.num_codebooks}, T]; "
                f"got shape {tuple(audio_codes.shape)}"
            )
        if int(audio_codes.shape[1]) != self.num_codebooks:
            raise ValueError(
                f"audio_codes second dim must equal num_codebooks={self.num_codebooks}; "
                f"got {int(audio_codes.shape[1])}"
            )
        if audio_codes.numel() > 0:
            max_val = int(audio_codes.max().item())
            min_val = int(audio_codes.min().item())
            if max_val >= self.num_real_codes or min_val < 0:
                raise ValueError(
                    "audio_codes contains stream-special or out-of-range IDs: "
                    f"min={min_val}, max={max_val}; real code range is "
                    f"[0, {self.num_real_codes - 1}]. Filter audio_stream_bos_id="
                    f"{self.config.audio_stream_bos_id} and audio_stream_eos_id="
                    f"{self.config.audio_stream_eos_id} (and anything above) at "
                    "Stage 0 before sending codes to the codec decoder."
                )
        return audio_codes

    @staticmethod
    def _split_request_ids(
        ids: torch.Tensor, seq_token_counts: list[int] | None = None
    ) -> list[torch.Tensor]:
        """Split a concatenated flat-codes tensor into per-request segments."""
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + int(count))
            n = int(ids.numel())
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        return [ids]


# Engine-side architecture identifier alias (mirrors Qwen3TTSCode2Wav usage).
HiggsAudioV2Code2WavForConditionalGeneration = HiggsAudioV2Code2Wav
