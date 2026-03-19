"""Fish Speech S2 Pro -- DAC Decoder (Stage 1).

Loads the DAC codec from ``codec.pth`` and decodes codebook indices
[num_codebooks, T] → audio waveform at 44.1 kHz.

Analogous to ``Qwen3TTSCode2Wav`` in qwen3_tts.

Requires the ``fish-speech`` package for the DAC model architecture.
Install with: ``pip install fish-speech``
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.fish_speech.dac_utils import (
    DAC_HOP_LENGTH,
    DAC_NUM_CODEBOOKS,
    DAC_SAMPLE_RATE,
    build_dac_codec,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class FishSpeechDACDecoder(nn.Module):
    """Stage-1 DAC decoder for Fish Speech S2 Pro (GenerationModelRunner).

    Consumes frame-aligned codec tokens from input_ids and decodes waveform
    via the DAC codec decoder.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._codec: nn.Module | None = None
        self._num_codebooks: int = DAC_NUM_CODEBOOKS
        self._output_sample_rate: int = DAC_SAMPLE_RATE
        self._hop_length: int = DAC_HOP_LENGTH
        self._logged_codec_stats = False

    def _ensure_codec_loaded(self) -> None:
        if self._codec is not None:
            return

        codec_path = os.path.join(self.model_path, "codec.pth")
        if not os.path.exists(codec_path):
            # Try HuggingFace cache.
            try:
                from transformers.utils.hub import cached_file

                cached = cached_file(self.model_path, "codec.pth")
                if cached is not None:
                    codec_path = cached
            except Exception:
                pass

        if not os.path.exists(codec_path):
            raise FileNotFoundError(
                f"codec.pth not found at {codec_path}. Make sure the Fish Speech S2 Pro model includes codec.pth."
            )

        codec = build_dac_codec()

        # Load weights.
        state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
        # Some checkpoints wrap under "generator" key.
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        codec.load_state_dict(state_dict, strict=False)

        device = self.vllm_config.device_config.device
        codec = codec.to(device=device, dtype=torch.float32)
        codec.eval()
        self._codec = codec

        logger.info(
            "Fish Speech DAC codec loaded from %s (device=%s, sample_rate=%d)",
            codec_path,
            device,
            self._output_sample_rate,
        )

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

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
        """Decode codec codes into audio waveform.

        input_ids layout per request: flat codes [num_codebooks * num_frames].
        Codes are codebook-major: [cb0_f0, cb0_f1, ..., cb0_fN, cb1_f0, ...].
        """
        self._ensure_codec_loaded()
        assert self._codec is not None

        q = self._num_codebooks
        sr_val = self._output_sample_rate
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        parsed: list[tuple[int, int]] = []
        valid_codes_qf: list[torch.Tensor] = []
        valid_indices: list[int] = []
        left_context_size = [0] * len(request_ids_list)
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                if "left_context_size" in info:
                    left_context_size[i] = info["left_context_size"]

        for i, req_ids in enumerate(request_ids_list):
            if req_ids.numel() < 1:
                parsed.append((0, 0))
                continue
            ctx_frames = left_context_size[i]
            flat = req_ids
            n = flat.numel()
            if n == 0 or n % q != 0:
                if n > 0:
                    logger.warning(
                        "DAC decoder input_ids length %d not divisible by num_codebooks %d; returning empty audio.",
                        n,
                        q,
                    )
                parsed.append((0, 0))
                continue
            frames = n // q
            codes_qf = flat.reshape(q, frames)
            parsed.append((ctx_frames, frames))
            valid_codes_qf.append(codes_qf)
            valid_indices.append(i)

        num_req = len(request_ids_list)
        if not valid_codes_qf:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        if not self._logged_codec_stats:
            self._logged_codec_stats = True
            try:
                c = valid_codes_qf[0]
                logger.info(
                    "DAC decoder: frames=%d q=%d uniq=%d range=[%d,%d] batch=%d",
                    c.shape[1],
                    q,
                    int(torch.unique(c).numel()),
                    int(c.min().item()),
                    int(c.max().item()),
                    len(valid_codes_qf),
                )
            except Exception:
                pass

        # Decode each request individually.
        wav_tensors: list[torch.Tensor] = []
        for codes_qf in valid_codes_qf:
            codes_bqf = codes_qf.unsqueeze(0)  # [1, num_codebooks, num_frames]
            num_frames = codes_qf.shape[1]
            feature_lengths = torch.tensor([num_frames], device=codes_bqf.device)
            with torch.cuda.amp.autocast(enabled=False):
                wav, audio_lengths = self._codec.decode(codes_bqf, feature_lengths)
            # wav shape: [1, 1, wav_len]
            wav_tensors.append(wav.squeeze(0).squeeze(0))  # [wav_len]

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            ctx_frames, actual_frames = parsed[idx]
            wav = wav_tensors[j]
            # Trim context frames (left overlap for streaming).
            if ctx_frames > 0:
                cut = ctx_frames * self._hop_length
                if cut < wav.shape[0]:
                    wav = wav[cut:]
                else:
                    logger.warning(
                        "Context trim %d >= decoded length %d; returning empty audio.",
                        cut,
                        wav.shape[0],
                    )
                    continue
            if wav.shape[0] > 0:
                audios[idx] = wav.to(dtype=torch.float32).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"FishSpeechDACDecoder expected (audio_tensor, sr), got {type(model_outputs)}")
        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audio_tensor, "sr": sr},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # DAC codec weights are loaded lazily from codec.pth, not from the main checkpoint.
        return set()
