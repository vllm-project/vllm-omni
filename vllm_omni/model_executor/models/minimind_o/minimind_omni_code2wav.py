# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn
from transformers import MimiModel
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.minimind_o.minimind_omni_config import MiniMindOmniCode2WavConfig
from vllm_omni.model_executor.models.minimind_o.resource_utils import resolve_model_dir
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _codec_ids_from_payload_or_input(
    input_ids: torch.Tensor,
    runtime_info: Mapping[str, Any] | None,
) -> torch.Tensor:
    """Prefer connector-delivered codec IDs over scheduler placeholders."""
    if isinstance(runtime_info, Mapping):
        codes = runtime_info.get("codes", {})
        if isinstance(codes, Mapping):
            audio = codes.get("audio")
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                return audio.reshape(-1).to(device=input_ids.device, dtype=torch.long)
            if isinstance(audio, (list, tuple)) and audio:
                return torch.as_tensor(audio, device=input_ids.device, dtype=torch.long).reshape(-1)
    return input_ids.reshape(-1).to(dtype=torch.long)


class MiniMindOmniCode2Wav(nn.Module):
    """MiniMind-O Mimi codec decoder for the Code2Wav stage."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        hf_config = vllm_config.model_config.hf_config
        self.code2wav_config: MiniMindOmniCode2WavConfig = hf_config.code2wav_config
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self.num_code_layers = int(self.code2wav_config.codec_num_code_layers)
        self.sample_rate = int(self.code2wav_config.codec_sample_rate)
        self.audio_pad_token = int(self.code2wav_config.codec_pad_token)
        self.mimi_path = self.code2wav_config.mimi_path
        self._mimi_model: MimiModel | None = None

    def get_language_model(self) -> nn.Module:
        return self

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
        return torch.device("cpu")

    def _ensure_mimi_loaded(self) -> MimiModel:
        if self._mimi_model is not None:
            return self._mimi_model

        if not self.mimi_path:
            raise ValueError(
                "MiniMind-O Code2Wav requires code2wav_config.mimi_path to point to a Mimi checkpoint directory."
            )
        mimi_dir = resolve_model_dir(os.fspath(self.mimi_path), "Mimi decoder")
        model = MimiModel.from_pretrained(mimi_dir, local_files_only=True)
        model.eval().to(self.vllm_config.device_config.device)
        self._mimi_model = model
        logger.info("Loaded MiniMind-O Mimi codec from %s", mimi_dir)
        return model

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(self, ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        if seq_token_counts is None or len(seq_token_counts) <= 1:
            return [ids]
        out: list[torch.Tensor] = []
        offset = 0
        for count in seq_token_counts:
            out.append(ids[offset : offset + int(count)])
            offset += int(count)
        return out

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
        mimi_model = self._ensure_mimi_loaded()
        device = self._module_device(mimi_model)
        sr_tensor = torch.tensor(self.sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)
        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        placeholder_ids = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        runtime_infos = runtime_additional_information or []
        request_ids = [
            _codec_ids_from_payload_or_input(
                req_ids,
                runtime_infos[idx] if idx < len(runtime_infos) else None,
            )
            for idx, req_ids in enumerate(placeholder_ids)
        ]
        audios: list[torch.Tensor] = []
        for req_ids in request_ids:
            if req_ids.numel() == 0 or req_ids.numel() % self.num_code_layers != 0:
                logger.warning(
                    "MiniMind Code2Wav input length %d is not divisible by %d; returning empty audio.",
                    int(req_ids.numel()),
                    self.num_code_layers,
                )
                audios.append(empty)
                continue
            frames = int(req_ids.numel()) // self.num_code_layers
            codes = req_ids.reshape(self.num_code_layers, frames).unsqueeze(0)
            codes = torch.where(codes >= self.audio_pad_token, torch.zeros_like(codes), codes)
            codes = codes.to(device=device, dtype=torch.long)
            audio = mimi_model.decode(codes).audio_values
            audios.append(audio.squeeze().detach().float().cpu().reshape(-1))

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": [sr_tensor] * len(audios)},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **_: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"MiniMindOmniCode2Wav expected OmniOutput, got {type(model_outputs)}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return set()
