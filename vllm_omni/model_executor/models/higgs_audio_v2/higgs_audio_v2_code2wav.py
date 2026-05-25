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

The kernel is the HiggsAudio codec helper in
``vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_decoder.py``; both
surfaces share the same RVQ + DAC weights.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
    HiggsAudioV2Config,
)
from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import (
    HiggsAudioRVQ,
    load_higgs_audio_codec,
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

        # When constructed via the engine path, eagerly load codec weights from
        # the model directory. The engine's subsequent ``load_weights(weights)``
        # call (with the Stage-0 safetensors iterator) is then a cheap no-op.
        if self._model_path is not None:
            try:
                self.load_codec_from_disk(self._model_path)
            except FileNotFoundError as exc:
                logger.warning("HiggsAudioV2Code2Wav: eager codec load deferred (%s)", exc)

    # ------------------------------------------------------------- engine hooks
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Stage 1 ignores embeddings; vLLM's runner still needs a stable shape."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    # ------------------------------------------------------------------ load
    def load_weights(self, weights_or_model_dir, device: torch.device | None = None):
        """Dual-signature load:

        - ``load_weights(weights_iterator)`` -- the vLLM engine path. The iterator
          yields ``(name, tensor)`` pairs from the Stage-0 safetensors which DO NOT
          contain codec weights, so we ignore them and eagerly load the codec from
          ``self._model_path``. Returns the set of all our params as "initialized"
          so the engine's loader passes its sanity check.
        - ``load_weights(model_dir: str)`` -- the legacy file-based call used by
          offline unit tests and ``end2end.py --mode stage1_only``. Returns ``None``.
        """
        if not isinstance(weights_or_model_dir, (str, bytes, os.PathLike)):
            # Engine call: consume the iterator (we don't need it) and eagerly
            # load the codec from disk.
            try:
                for _ in weights_or_model_dir:
                    pass
            except TypeError:
                pass
            if self._model_path is None:
                raise RuntimeError(
                    "HiggsAudioV2Code2Wav.load_weights called via the engine path but "
                    "no model directory is known. Construct with vllm_config so the "
                    "constructor caches vllm_config.model_config.model."
                )
            self.load_codec_from_disk(self._model_path, device=device)
            return {name for name, _ in self.named_parameters()}

        # Legacy file-based call.
        self.load_codec_from_disk(weights_or_model_dir, device=device)
        return None

    def load_codec_from_disk(self, model_dir: str, device: torch.device | None = None) -> None:
        """Load codec weights from disk (RVQ + fc2 + DAC decoder).

        Resolution order for the codec directory:
        1. If :attr:`HiggsAudioV2Config.audio_tokenizer_id` is set, treat it as
           either a local path (if it exists on disk) or an HF repo id that we
           resolve via ``huggingface_hub.snapshot_download``. This is the
           default for the boson-ai release (codec in a separate repo).
        2. Otherwise fall back to ``<model_dir>/<audio_tokenizer_subdir>``.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio_tokenizer_path = self._resolve_audio_tokenizer_path(model_dir)
        quantizer, fc2, acoustic_decoder, _tokenizer_config = load_higgs_audio_codec(audio_tokenizer_path, device)
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

    def _resolve_audio_tokenizer_path(self, model_dir: str) -> str:
        """Return a local filesystem path containing ``config.json`` + codec
        weights. See :meth:`load_codec_from_disk` for the resolution rules."""
        tokenizer_id = getattr(self.config, "audio_tokenizer_id", None)
        if tokenizer_id:
            if os.path.isdir(tokenizer_id):
                return tokenizer_id
            return self._resolve_hf_id_to_local(tokenizer_id)
        subdir = self.config.audio_tokenizer_subdir or ""
        if os.path.isdir(model_dir):
            return os.path.join(model_dir, subdir) if subdir else model_dir
        local = self._resolve_hf_id_to_local(model_dir)
        return os.path.join(local, subdir) if subdir else local

    @staticmethod
    def _resolve_hf_id_to_local(repo_id: str) -> str:
        """Resolve a HF repo id to a local snapshot directory.

        Prefers a read-only cache lookup (no write locks) so a quota-constrained
        Lustre cache can still serve already-downloaded weights. Falls back to
        ``snapshot_download`` only when nothing is cached locally.
        """
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
        if isinstance(cached, str) and os.path.isfile(cached):
            return os.path.dirname(cached)

        from huggingface_hub.constants import HF_HUB_CACHE

        safe = repo_id.replace("/", "--")
        snapshots_dir = os.path.join(HF_HUB_CACHE, f"models--{safe}", "snapshots")
        if os.path.isdir(snapshots_dir):
            for rev in os.listdir(snapshots_dir):
                candidate = os.path.join(snapshots_dir, rev)
                if os.path.isfile(os.path.join(candidate, "config.json")):
                    return candidate

        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=repo_id)

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
        # Match fc2's parameter dtype — vLLM may auto-cast the codec to the
        # stage's model dtype (e.g. bf16), while the quantizer's RVQ codebooks
        # often stay in float32 because they're embedded lookups. A defensive
        # cast keeps the matmul homogeneous.
        quantized = quantized.to(dtype=self.fc2.weight.dtype)
        quantized = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        # Acoustic decoder may have parameters in yet another dtype — match
        # its first parameter so the linear/conv ops stay consistent.
        first_param = next(self.acoustic_decoder.parameters(), None)
        if first_param is not None and quantized.dtype != first_param.dtype:
            quantized = quantized.to(dtype=first_param.dtype)
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
                f"audio_codes second dim must equal num_codebooks={self.num_codebooks}; got {int(audio_codes.shape[1])}"
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
    def _split_request_ids(ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
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
