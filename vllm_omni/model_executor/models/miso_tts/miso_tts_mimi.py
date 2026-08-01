# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Miso TTS Mimi stage: RVQ frames → waveform (upstream ``Generator`` codec path)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.miso_tts.modeling_miso_tts import (
    MISO_NUM_CODEBOOKS,
    load_mimi_codec,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _frames_from_runtime_info(info: dict[str, Any] | None, input_ids: torch.Tensor) -> torch.Tensor:
    """Return codec frames ``[T, Q]`` from connector payload or placeholder ids."""
    if isinstance(info, dict):
        codes = info.get("codes")
        if isinstance(codes, dict):
            audio = codes.get("audio")
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                t = audio.to(dtype=torch.long)
                if t.ndim == 2:
                    return t
                if t.ndim == 1:
                    n = t.numel()
                    if n % MISO_NUM_CODEBOOKS != 0:
                        return torch.zeros(0, MISO_NUM_CODEBOOKS, dtype=torch.long, device=input_ids.device)
                    # Codebook-major flat [Q*T] from stage_input_processors/miso_tts async chunks
                    q = MISO_NUM_CODEBOOKS
                    num_frames = n // q
                    return t.reshape(q, num_frames).transpose(0, 1)
    flat = input_ids.reshape(-1).to(dtype=torch.long)
    if flat.numel() % MISO_NUM_CODEBOOKS != 0:
        return torch.zeros(0, MISO_NUM_CODEBOOKS, dtype=torch.long, device=input_ids.device)
    return flat.reshape(-1, MISO_NUM_CODEBOOKS)


class MisoTTSMimiDecoder(nn.Module):
    """Stage-1: Mimi RVQ decode with sliding-window **delta** audio emission."""

    input_modalities = "audio"
    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_path = vllm_config.model_config.model

        self._mimi: Any | None = None
        self._device: torch.device | None = None
        self._sample_rate = int(getattr(self.config, "sample_rate", 24000))
        # Buffer all frames per request until finished, then decode all at once (like official implementation)
        self._frame_buffers: OrderedDict[str, list[torch.Tensor]] = OrderedDict()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self._mimi is not None:
            return None
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._mimi = load_mimi_codec(device, MISO_NUM_CODEBOOKS)
        self._sample_rate = int(self._mimi.sample_rate)
        for _ in weights:
            pass
        return None

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        if runtime_additional_information is None:
            runtime_additional_information = kwargs.get("model_intermediate_buffer") or kwargs.get(
                "runtime_additional_information"
            )
        if self._mimi is None:
            self.load_weights([])
        assert input_ids is not None
        mimi = self._mimi
        device = self._device or input_ids.device
        sr_tensor = torch.tensor(self._sample_rate, dtype=torch.int32)
        infos = runtime_additional_information or [{}]

        outputs: list[torch.Tensor] = []
        for idx, info in enumerate(infos):
            if info.get("_is_dummy"):
                outputs.append(torch.zeros(0, dtype=torch.float32))
                continue

            frames = _frames_from_runtime_info(info, input_ids)
            if frames.numel() == 0:
                outputs.append(torch.zeros(0, dtype=torch.float32))
                continue

            valid = (frames >= 0).all(dim=1) & frames.any(dim=1)
            frames = frames[valid]
            if frames.numel() == 0:
                outputs.append(torch.zeros(0, dtype=torch.float32))
                continue

            # In async_chunk mode, the connector sends cumulative frames (100, 101, 102... 150)
            # Just use the frames directly - they're already the full sequence up to this point
            finished = (
                bool(info.get("meta", {}).get("finished", info.get("meta", {}).get("is_segment_finished", False)))
                if isinstance(info.get("meta"), dict)
                else False
            )

            # Only decode when finished (like official implementation)
            if finished:
                # [1, Q, T]
                codes = frames.transpose(0, 1).unsqueeze(0).to(device=device)
                waveform = mimi.decode(codes).squeeze(0).squeeze(0).float()
                outputs.append(waveform.detach().cpu())
            else:
                # Return empty until finished
                outputs.append(torch.zeros(0, dtype=torch.float32))

        return OmniOutput(
            text_hidden_states=torch.zeros((len(outputs), 1), device=device, dtype=torch.float32),
            multimodal_outputs={"model_outputs": outputs, "sr": [sr_tensor] * len(outputs)},
        )
