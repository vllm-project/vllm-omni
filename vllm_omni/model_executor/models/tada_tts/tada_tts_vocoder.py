"""TADA TTS -- Stage 1: Vocoder (acoustic features → waveform @ 44.1 kHz).

Wraps HumeAI/tada-codec/decoder.  Receives acoustic_features [B, T, 512] and
text_token_mask [B, T] via runtime_additional_information.  input_ids carries
dummy token IDs (one per frame) for vLLM-Omni sequence-length bookkeeping.
Upsample factor: 4 × 4 × 5 × 6 = 480.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Codec sample rate (fixed by HumeAI/tada-codec)
TADA_CODEC_SAMPLE_RATE = 44_100
# Upsampling factor: strides [4, 4, 5, 6]
TADA_CODEC_UPSAMPLE = 4 * 4 * 5 * 6  # 480


class TadaVocoder(nn.Module):
    """Stage 1: TADA codec decoder (acoustic features → waveform)."""

    input_modalities = "audio"

    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self._decoder: nn.Module | None = None
        self._output_sample_rate = TADA_CODEC_SAMPLE_RATE
        self._upsample = TADA_CODEC_UPSAMPLE
        self._logged_stats = False

    def _ensure_decoder_loaded(self) -> None:
        if self._decoder is not None:
            return

        try:
            from tada.modules.decoder import Decoder
        except ImportError as e:
            raise ImportError(
                "hume-tada package is required for TadaVocoder. "
                "Install with: pip install hume-tada"
            ) from e

        logger.info("Loading TADA codec decoder from HumeAI/tada-codec …")
        decoder = Decoder.from_pretrained("HumeAI/tada-codec", subfolder="decoder")
        device = self.vllm_config.device_config.device
        decoder = decoder.to(device=device, dtype=torch.float32)
        decoder.eval()
        self._decoder = decoder
        logger.info("TADA codec decoder loaded (device=%s)", device)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        self._ensure_decoder_loaded()
        assert self._decoder is not None

        device = self.vllm_config.device_config.device
        sr_tensor = torch.tensor(self._output_sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if not runtime_additional_information:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        audios: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []

        for info in runtime_additional_information:
            if not isinstance(info, dict):
                audios.append(empty)
                srs.append(sr_tensor)
                continue

            af = info.get("acoustic_features")
            tm = info.get("text_token_mask")

            if af is None or not isinstance(af, torch.Tensor) or af.numel() == 0:
                audios.append(empty)
                srs.append(sr_tensor)
                continue

            try:
                af = af.to(device=device, dtype=torch.float32)
                T = af.shape[0]

                if isinstance(tm, torch.Tensor) and tm.numel() == T:
                    token_mask = tm.to(device=device, dtype=torch.long)
                else:
                    token_mask = torch.ones(T, device=device, dtype=torch.long)

                af_b = af.unsqueeze(0)           # [1, T, 512]
                tm_b = token_mask.unsqueeze(0)   # [1, T]

                if not self._logged_stats:
                    self._logged_stats = True
                    logger.info(
                        "TadaVocoder: T=%d frames, upsample=%d → ~%.2f s @%d Hz",
                        T, self._upsample,
                        T * self._upsample / self._output_sample_rate,
                        self._output_sample_rate,
                    )

                wav = self._decoder(af_b, tm_b)  # [1, 1, wav_len]
                wav = wav.squeeze(0).squeeze(0).to(dtype=torch.float32).reshape(-1)
                audios.append(wav.cpu())

            except Exception:
                logger.exception("TadaVocoder: error decoding request; returning empty audio")
                audios.append(empty)

            srs.append(sr_tensor)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(
        self, model_outputs: torch.Tensor | OmniOutput, **_: Any
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": [model_outputs], "sr": [
                torch.tensor(self._output_sample_rate, dtype=torch.int32)
            ]},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Weights loaded lazily from HumeAI/tada-codec; skip main checkpoint.
        return set()
