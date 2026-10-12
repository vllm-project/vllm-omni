# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview -- codec decoder (Stage 1).

Consumes frames of ``num_codebooks`` codec codes and decodes them to a 44.1 kHz
waveform. Streaming contract is delta: with ``async_chunk`` on, each chunk is
``[left_context | new_frames]``; the whole window is decoded so the causal
codec keeps continuous context, then only the new tail is returned.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.audio8_tts.codec_utils import load_arktts_codec
from vllm_omni.model_executor.models.audio8_tts.configuration_audio8_tts import (
    ARKTTS_CODEC_FRAME_SIZE,
    ARKTTS_CODEC_SAMPLE_RATE,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "": torch.float32,
}


def _connector_extra_config(vllm_config: VllmConfig) -> dict[str, Any]:
    model_config = getattr(vllm_config, "model_config", None)
    connector_cfg = getattr(model_config, "stage_connector_config", None)
    if isinstance(connector_cfg, dict):
        return connector_cfg.get("extra", connector_cfg)
    extra = getattr(connector_cfg, "extra", None)
    return extra if isinstance(extra, dict) else {}


def _resolve_dtype(extra_cfg: dict[str, Any]) -> torch.dtype:
    raw = str(extra_cfg.get("audio8_tts_codec_dtype", "float32")).strip().lower()
    if raw not in _DTYPES:
        raise ValueError(f"Invalid Audio8 TTS codec dtype: {raw!r}")
    return _DTYPES[raw]


class Audio8TTSCodecDecoder(nn.Module):
    """Stage 1: codec codes -> waveform (runs under the generation runner)."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        config = vllm_config.model_config.hf_config

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._num_codebooks = int(getattr(config, "num_codebooks", 10))
        self._sample_rate = int(getattr(config, "codec_sample_rate", ARKTTS_CODEC_SAMPLE_RATE))
        self._frame_size = int(getattr(config, "codec_frame_size", ARKTTS_CODEC_FRAME_SIZE))
        self._codec_kwargs = {
            "post_n_layer": int(getattr(config, "codec_post_n_layer", 8)),
            "post_n_head": int(getattr(config, "codec_post_n_head", 16)),
            "post_n_local_heads": int(getattr(config, "codec_post_n_local_heads", 8)),
            "post_intermediate_size": int(getattr(config, "codec_post_intermediate_size", 1216)),
        }
        # float32 by default: the codec's Snake activations and weight-normed
        # convs lose audible headroom in bf16.
        self._codec_dtype = _resolve_dtype(_connector_extra_config(vllm_config))
        self._codec: nn.Module | None = None
        self._codec_device: torch.device | None = None
        self._logged_stats = False

    # -------------------- codec --------------------

    def _ensure_codec_loaded(self) -> None:
        if self._codec is not None:
            return
        device = self.vllm_config.device_config.device
        self._codec = load_arktts_codec(
            self.model_path,
            device=device,
            dtype=self._codec_dtype,
            role="decode",
            **self._codec_kwargs,
        )
        self._codec_device = torch.device(device)

    # -------------------- vLLM hooks --------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """No-op: the codec comes from ``codec.pth``, not the main checkpoint."""
        del weights
        return set()

    # -------------------- request splitting --------------------

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            total = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], total)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for size in slices:
                    boundaries.append(boundaries[-1] + size)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    def _codes_from_runtime_info(
        self,
        info: dict[str, Any] | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Prefer the tensor payload the connector carried over input_ids."""
        if not isinstance(info, dict):
            return None
        codes = info.get("codes", {}).get("audio")
        if not isinstance(codes, torch.Tensor) or codes.numel() == 0:
            return None
        codes = codes.to(device=device, dtype=torch.long, non_blocking=True).contiguous()
        if codes.ndim != 2 or int(codes.shape[0]) != self._num_codebooks:
            logger.warning(
                "Audio8 TTS codec decoder expected codes of shape [%d, frames], got %s",
                self._num_codebooks,
                tuple(codes.shape),
            )
            return None
        return codes

    # -------------------- decode --------------------

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
        """Decode one chunk per request.

        ``input_ids`` is codebook-major flat codes
        (``[cb0_f0 ... cb0_fN, cb1_f0 ...]``); tensor codes on the connector
        win over ``input_ids``. Every return path must set ``model_outputs`` or
        the serving layer silently drops a chunk of audio.
        """
        del positions, intermediate_tensors, inputs_embeds
        sr_tensor = torch.tensor(self._sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        self._ensure_codec_loaded()
        codec = self._codec
        assert codec is not None

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        per_request_ids = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        num_req = len(per_request_ids)

        context_frames = [0] * num_req
        if runtime_additional_information is not None:
            for index, info in enumerate(runtime_additional_information[:num_req]):
                meta = info.get("meta", {}) if isinstance(info, dict) else {}
                context_frames[index] = int(meta.get("left_context_size", 0) or 0)

        audios: list[torch.Tensor] = [empty] * num_req
        for index, request_ids in enumerate(per_request_ids):
            codes_qf = (
                self._codes_from_runtime_info(runtime_additional_information[index], ids.device)
                if runtime_additional_information is not None and index < len(runtime_additional_information)
                else None
            )
            if codes_qf is None:
                if request_ids.numel() < self._num_codebooks:
                    continue
                if int(request_ids.numel()) % self._num_codebooks != 0:
                    logger.warning(
                        "Audio8 TTS codec decoder got %d ids, not divisible by num_codebooks=%d; skipping request",
                        int(request_ids.numel()),
                        self._num_codebooks,
                    )
                    continue
                codes_qf = request_ids.reshape(self._num_codebooks, -1)

            total_frames = int(codes_qf.shape[1])
            if total_frames <= 0:
                continue
            new_frames = total_frames - context_frames[index]
            if new_frames <= 0:
                logger.warning(
                    "Audio8 TTS codec decoder chunk has no new frames (total=%d, context=%d)",
                    total_frames,
                    context_frames[index],
                )
                continue

            if not self._logged_stats:
                self._logged_stats = True
                logger.info(
                    "Audio8 TTS codec decoder: frames=%d codebooks=%d dtype=%s sr=%d",
                    total_frames,
                    self._num_codebooks,
                    self._codec_dtype,
                    self._sample_rate,
                )

            with torch.amp.autocast("cuda", enabled=False):
                # Move codes to the codec's device explicitly: a mismatch here
                # silently falls back to CPU or raises deep inside the decoder.
                waveform = codec.decode(codes_qf.unsqueeze(0).to(device=self._codec_device))
            waveform = waveform.reshape(-1).to(dtype=torch.float32)

            # Trim the left-context prefix proportionally: the codec's padding /
            # rounding means the decoded length is not exactly
            # frames * frame_size.
            if context_frames[index] > 0:
                cut = int(round(context_frames[index] / total_frames * int(waveform.shape[0])))
                cut = max(0, min(cut, int(waveform.shape[0])))
                waveform = waveform[cut:]
            if waveform.numel() > 0:
                audios[index] = waveform.contiguous()

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": [sr_tensor] * num_req},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        del kwargs
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"Audio8TTSCodecDecoder expected (audio_tensor, sr), got {type(model_outputs)}")
        audio_tensor, sample_rate = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audio_tensor, "sr": sample_rate},
        )


__all__ = ["Audio8TTSCodecDecoder"]
