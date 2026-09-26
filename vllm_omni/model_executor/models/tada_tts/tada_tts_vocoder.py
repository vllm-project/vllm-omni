"""TADA TTS -- Stage 1: Vocoder (acoustic features → waveform @ 24 kHz).

Receives acoustic features [T, feat_dim], per-token durations time_before [T], and
text_token_mask [T] via runtime_additional_information. Before decoding it (1)
de-normalises the acoustic features (× acoustic_std + acoustic_mean) and (2) expands the
frame sequence by each token's duration, inserting silence frames. input_ids carries dummy
token IDs (one per frame) for sequence-length bookkeeping.
Codec upsample factor: 4 × 4 × 5 × 6 = 480 → 50 Hz frames × 480 = 24 000 Hz.
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

# Codec sample rate: 50 Hz acoustic frames × 480 upsample = 24 000 Hz.
TADA_CODEC_SAMPLE_RATE = 24_000
# Upsampling factor: strides [4, 4, 5, 6]
TADA_CODEC_UPSAMPLE = 4 * 4 * 5 * 6  # 480
# Frames-per-second of the acoustic token stream (used for leading-silence trim).
TADA_CODEC_FRAME_RATE = 50
# The codec decoder's local-attention RoPE / precomputed-mask buffers are sized for
# this many frames (LocalAttentionEncoder max_seq_len); longer inputs overflow them.
DECODER_MAX_SEQ_LEN = 8192


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

        cfg = vllm_config.model_config.hf_config
        # Acoustic features are generated in normalised space; de-normalise before decoding.
        self.acoustic_mean = float(getattr(cfg, "acoustic_mean", 0.0))
        self.acoustic_std = float(getattr(cfg, "acoustic_std", 1.5))
        self.acoustic_features_dim = int(getattr(cfg, "acoustic_dim", 512))

        self._decoder: nn.Module | None = None
        self._codec_path = self._resolve_codec_path()
        self._output_sample_rate = TADA_CODEC_SAMPLE_RATE
        self._upsample = TADA_CODEC_UPSAMPLE
        self._frame_rate = TADA_CODEC_FRAME_RATE
        self._logged_stats = False

    def _resolve_codec_path(self) -> str:
        """Resolve the tada-codec source. Priority: ``TADA_CODEC_PATH`` env →
        config attr → sibling ``tada-codec`` dir of the model path → HF hub id."""
        env = os.environ.get("TADA_CODEC_PATH")
        if env:
            return env
        cfg = self.vllm_config.model_config.hf_config
        cfg_path = getattr(cfg, "codec_model_name_or_path", None) or getattr(cfg, "codec_path", None)
        if cfg_path:
            return str(cfg_path)
        # Sibling of the AR model dir (e.g. models/tada-1b → models/tada-codec).
        try:
            parent = os.path.dirname(os.path.abspath(self.model_path))
            sibling = os.path.join(parent, "tada-codec")
            if os.path.isdir(os.path.join(sibling, "decoder")):
                return sibling
        except Exception:
            pass
        return "HumeAI/tada-codec"

    def _ensure_decoder_loaded(self) -> None:
        if self._decoder is not None:
            return

        from .codec import Decoder

        logger.info("Loading TADA codec decoder from %s (subfolder=decoder) …", self._codec_path)
        decoder = Decoder.from_pretrained(self._codec_path, subfolder="decoder")
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

            # Streaming path: ``codes.audio`` is a pre-expanded window of frames; decode it
            # and keep only the interior, dropping the left-context and right-lookahead frames
            # given by the meta drop counts. The presence of those counts selects this path.
            codes = info.get("codes")
            window = codes.get("audio") if isinstance(codes, dict) else None
            meta = info.get("meta") if isinstance(info.get("meta"), dict) else {}
            if (
                isinstance(window, torch.Tensor)
                and window.dim() == 2
                and window.shape[-1] == self.acoustic_features_dim
                and ("left_context_size" in meta or "right_holdback_size" in meta)
            ):
                try:
                    wav = self._decode_window(
                        window.to(device=device, dtype=torch.float32),
                        int(meta.get("left_context_size", 0) or 0),
                        int(meta.get("right_holdback_size", 0) or 0),
                    )
                    audios.append(wav.to(dtype=torch.float32).cpu())
                except Exception:
                    logger.exception("TadaVocoder: error decoding streaming window; empty audio")
                    audios.append(empty)
                srs.append(sr_tensor)
                continue

            af = info.get("acoustic_features")
            tb = info.get("time_before")

            if af is None or not isinstance(af, torch.Tensor) or af.numel() == 0:
                audios.append(empty)
                srs.append(sr_tensor)
                continue

            try:
                af = af.to(device=device, dtype=torch.float32)
                T = af.shape[0]

                # De-normalise: the diffusion head generates in normalised space.
                af = af * self.acoustic_std + self.acoustic_mean

                # Per-token durations (frames). Without them the timing cannot be
                # reconstructed; fall back to 1 frame/token (wrong, but avoids crash).
                if isinstance(tb, torch.Tensor) and tb.numel() >= T:
                    time_before = tb.to(device=device, dtype=torch.long).reshape(-1)
                else:
                    logger.warning_once(
                        "TadaVocoder: time_before missing/short; using 1 frame/token (audio timing will be wrong)"
                    )
                    time_before = torch.ones(T, device=device, dtype=torch.long)

                if not self._logged_stats:
                    self._logged_stats = True
                    total_frames = int(time_before[:T].clamp(min=1).sum().item())
                    tb_view = time_before[:T].float()
                    logger.info(
                        "TadaVocoder: %d tokens → ~%d expanded frames, upsample=%d → ~%.2f s @%d Hz "
                        "| time_before[min/mean/max]=%d/%.1f/%d | acoustic[mean/std/absmax]=%.3f/%.3f/%.3f",
                        T,
                        total_frames,
                        self._upsample,
                        total_frames * self._upsample / self._output_sample_rate,
                        self._output_sample_rate,
                        int(tb_view.min().item()),
                        float(tb_view.mean().item()),
                        int(tb_view.max().item()),
                        float(af.mean().item()),
                        float(af.std().item()),
                        float(af.abs().max().item()),
                    )

                wav = self._decode_wav(af, time_before)  # [wav_len]

                # Trim the leading silence: the first token's duration worth of samples.
                lead = int(self._output_sample_rate * int(time_before[0].item()) / self._frame_rate)
                if 0 < lead < wav.shape[0]:
                    wav = wav[lead:]

                audios.append(wav.to(dtype=torch.float32).cpu())

            except Exception:
                logger.exception("TadaVocoder: error decoding request; returning empty audio")
                audios.append(empty)

            srs.append(sr_tensor)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def _decode_window(self, window: torch.Tensor, left_drop: int, right_drop: int) -> torch.Tensor:
        """Decode one pre-expanded streaming window and return only its interior.

        The window holds duration-expanded frames with ``left_drop`` context frames and
        ``right_drop`` lookahead frames around the emitted region. Decode the whole window so
        the codec sees full context, then return only the emitted region's samples. The codec's
        constant convolutional edge offset is shared by every window, so fixed slicing stays
        continuous across chunks.
        """
        device = window.device
        W = window.shape[0]
        if W == 0:
            return torch.zeros(0, device=device)
        af = window * self.acoustic_std + self.acoustic_mean
        expanded = af.unsqueeze(0)
        if expanded.shape[1] > DECODER_MAX_SEQ_LEN:
            expanded = expanded[:, :DECODER_MAX_SEQ_LEN]
            W = DECODER_MAX_SEQ_LEN
        decoder_dtype = next(self._decoder.parameters()).dtype
        expanded = expanded.to(decoder_dtype)
        token_masks = (torch.norm(expanded, dim=-1) != 0).long()
        wav = self._decoder(expanded, token_masks).squeeze(0).squeeze(0).to(torch.float32).reshape(-1)
        up = self._upsample
        start = left_drop * up
        end = wav.shape[0] - right_drop * up
        if start >= end:
            return torch.zeros(0, device=device)
        return wav[start:end]

    def _decode_wav(self, encoded: torch.Tensor, time_before: torch.Tensor) -> torch.Tensor:
        """Expand acoustic frames by per-token duration, then run the codec decoder.

        Insert ``time_before[pos] - 1`` silence frames before each acoustic frame, append
        ``time_before[-1]`` trailing silence frames, then decode. ``token_masks`` marks the
        real (non-zero) frames and drives the decoder's block attention.
        """
        device = encoded.device
        T = encoded.shape[0]
        feat_dim = encoded.shape[-1]
        tb = time_before.reshape(-1).to(device=device, dtype=torch.long)
        tb = tb[: T + 1]
        if tb.numel() == 0 or T == 0:
            return torch.zeros(0, device=device)

        parts: list[torch.Tensor] = []
        for pos in range(T):
            n_zero = int((tb[pos] - 1).clamp(min=0).item())
            if n_zero > 0:
                parts.append(torch.zeros(n_zero, feat_dim, device=device, dtype=encoded.dtype))
            parts.append(encoded[pos].unsqueeze(0))
        n_tail = int(tb[-1].clamp(min=0).item())
        if n_tail > 0:
            parts.append(torch.zeros(n_tail, feat_dim, device=device, dtype=encoded.dtype))

        expanded = torch.cat(parts, dim=0).unsqueeze(0)  # [1, T_exp, 512]
        # The codec decoder's local-attention RoPE/precomputed-mask buffers are sized
        # for DECODER_MAX_SEQ_LEN frames; longer inputs overflow them. This only
        # happens with runaway AR generation (no EOS) — cap with a loud warning
        # rather than crash the whole pipeline.
        if expanded.shape[1] > DECODER_MAX_SEQ_LEN:
            logger.warning(
                "TadaVocoder: expanded sequence %d > decoder max %d frames; truncating. "
                "This indicates the AR stage did not stop (check EOS / max_tokens).",
                expanded.shape[1],
                DECODER_MAX_SEQ_LEN,
            )
            expanded = expanded[:, :DECODER_MAX_SEQ_LEN]
        decoder_dtype = next(self._decoder.parameters()).dtype
        expanded = expanded.to(decoder_dtype)
        token_masks = (torch.norm(expanded, dim=-1) != 0).long()  # [1, T_exp]
        wav = self._decoder(expanded, token_masks)  # [1, 1, wav_len]
        return wav.squeeze(0).squeeze(0).to(dtype=torch.float32).reshape(-1)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **_: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": [model_outputs],
                "sr": [torch.tensor(self._output_sample_rate, dtype=torch.int32)],
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Weights loaded lazily from HumeAI/tada-codec; skip main checkpoint.
        return set()
