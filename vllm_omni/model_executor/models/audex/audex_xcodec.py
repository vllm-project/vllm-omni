# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex TTA stage 1: XCodec1 RVQ codes → waveform.

Unlike the streaming causal speech decoder used for TTS, XCodec1 is a CNN
codec that the official flow decodes over the full sequence, so this stage
runs sync full-payload only: each request delivers its complete
de-interleaved ``[frames, 4]`` codec-id payload once at finish and the
decode happens in a single call.

The checkpoint (``hf-audio/xcodec-hubert-general-balanced``) is external to
the Audex repo and loads through transformers remote code; the model
skeleton is built from config at startup for vLLM's memory profiler and the
weights arrive via the standard ``load_weights`` iteration over the
snapshot's safetensors.
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.audex.tta import (
    XCODEC1_CODEBOOK_SIZE,
    XCODEC1_NUM_CODEBOOKS,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Official cap on decoded frames (~10 s of audio at the codec frame rate).
_DEFAULT_MAX_TTA_FRAMES = 500


def _meta_str(value: Any) -> str | None:
    if isinstance(value, list | tuple):
        return _meta_str(value[0]) if value else None
    if isinstance(value, str):
        return value
    return None


def _codes_from_runtime_info(runtime_info: dict[str, Any] | None) -> torch.Tensor | None:
    """Extract the ``[frames, 4]`` codec-id payload from the connector info."""
    if not isinstance(runtime_info, dict):
        return None
    codes = runtime_info.get("codes")
    if not isinstance(codes, dict):
        return None
    audio = codes.get("audio")
    if isinstance(audio, torch.Tensor):
        return audio
    if isinstance(audio, list | tuple) and audio:
        return torch.as_tensor(audio, dtype=torch.long)
    return None


class AudexXCodec1(nn.Module):
    """Stage-1 model for Audex TTA (GenerationModelRunner)."""

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

        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.codec = AutoModel.from_config(config, trust_remote_code=True)
        self.codec.eval()
        self._sample_rate = int(getattr(config, "sampling_rate", 16000))
        self._max_frames = int(getattr(config, "max_tta_frames", _DEFAULT_MAX_TTA_FRAMES))

    # -------------------- vLLM model interface --------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings; keep a stable dummy for the runner.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        state_dict = dict(weights)
        self.codec.load_state_dict(state_dict, strict=True)
        device = self.vllm_config.device_config.device
        self.codec.to(device=device, dtype=torch.float32)
        return {f"codec.{name}" for name in state_dict}

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        # The runner may unpack the NamedTuple in transit; rebuild it.
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"AudexXCodec1 expected OmniOutput, got {type(model_outputs)}")

    # -------------------- decoding --------------------

    def _decode_codes(self, req_id: str | None, codes: torch.Tensor) -> torch.Tensor:
        """Decode a ``[frames, 4]`` codec-id payload to a mono waveform."""
        if codes.numel() == 0:
            # Zero codec tokens for the whole request. Do NOT raise: an
            # exception in forward() kills the engine core for every request.
            # Returning empty audio lets the serving layer fail this request.
            logger.error("Audex TTA: request %s finished with no codec tokens; no audio generated", req_id)
            return torch.empty(0, dtype=torch.float32)
        if codes.ndim != 2 or codes.shape[1] != XCODEC1_NUM_CODEBOOKS:
            logger.error(
                "Audex TTA: request %s payload has shape %s, expected [frames, %d]; no audio generated",
                req_id,
                tuple(codes.shape),
                XCODEC1_NUM_CODEBOOKS,
            )
            return torch.empty(0, dtype=torch.float32)

        device = self.vllm_config.device_config.device
        with torch.inference_mode():
            frames = codes[: self._max_frames].to(device=device, dtype=torch.long)
            # [frames, 4] codec ids -> [1, 4, frames] per-layer indices.
            layer_offsets = (
                torch.arange(XCODEC1_NUM_CODEBOOKS, device=frames.device, dtype=frames.dtype) * XCODEC1_CODEBOOK_SIZE
            )
            per_layer = (frames - layer_offsets.view(1, -1)).clamp(0, XCODEC1_CODEBOOK_SIZE - 1)
            stacked = per_layer.transpose(0, 1).unsqueeze(0)
            decoded = self.codec.decode(stacked)
            audio = decoded.audio_values if hasattr(decoded, "audio_values") else decoded[0]
            return audio.reshape(-1).to(dtype=torch.float32, device="cpu")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        runtime_infos = runtime_additional_information or []
        seq_token_counts = kwargs.get("seq_token_counts")
        if seq_token_counts is not None:
            num_reqs = len(seq_token_counts)
        elif input_ids is not None and input_ids.numel() > 0 and not runtime_infos:
            num_reqs = 1
        else:
            num_reqs = len(runtime_infos)

        audios: list[torch.Tensor] = []
        sr_tensor = torch.tensor(self._sample_rate, dtype=torch.int32)
        for i in range(num_reqs):
            info = runtime_infos[i] if i < len(runtime_infos) else {}
            info = info if isinstance(info, dict) else {}
            meta = info.get("meta", {})
            meta = meta if isinstance(meta, dict) else {}
            req_id = _meta_str(meta.get("req_id"))
            codes = _codes_from_runtime_info(info)
            if codes is None:
                audios.append(torch.empty(0, dtype=torch.float32))
                continue
            audios.append(self._decode_codes(req_id, codes))

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": [sr_tensor] * num_reqs},
        )
