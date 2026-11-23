# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape graph adapters for speaker and S3 reference-audio features."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalConfig

from ..encoder_cuda_graph import EncoderCudaGraph

if TYPE_CHECKING:
    from s3tokenizer.model_v2 import S3TokenizerV2

    from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import CampplusTorch


def _graph_config(vllm_config: VllmConfig) -> VllmConfig:
    # Code2Wav is a generation stage, so its model config need not carry
    # multimodal limits. Supply defaults without mutating the shared config.
    config = copy(vllm_config)
    config.model_config = copy(config.model_config)
    if config.model_config.multimodal_config is None:
        config.model_config.multimodal_config = MultiModalConfig()
    return config


def _precision_policy() -> tuple[bool, ...]:
    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )


class SpeakerCudaGraph:
    """Keep the shared eager model untouched and own this consumer's graphs.

    Shapes are not padded. A changed precision policy runs eagerly instead of
    replaying kernels captured with another policy. The existing encoder graph
    manager bounds shape admission, isolates streams, and returns owned outputs.
    """

    def __init__(self, speaker: CampplusTorch, vllm_config: VllmConfig, *, max_graphs: int = 4):
        self.speaker = speaker
        self.device = speaker.device
        config = _graph_config(vllm_config)
        self.graph = EncoderCudaGraph(speaker.model.forward, config, max_graphs=max_graphs)
        self._policy: tuple[bool, ...] | None = None

    @torch.inference_mode()
    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        if self.device.type != "cuda" or torch.is_autocast_enabled("cuda"):
            return self.speaker(feat)
        policy = _precision_policy()
        if self._policy is None:
            self._policy = policy
        elif policy != self._policy:
            return self.speaker(feat)
        feat = feat.to(self.device, non_blocking=True).unsqueeze(0).float()
        return self.graph(feat)


class S3CudaGraph:
    """Keep the original S3 modules, masks, arithmetic and quantizer.

    The caller supplies a single unpadded short reference. The pinned
    s3tokenizer 0.3.0 encoder uses tensor widths for its masks and is capturable
    once the caller has moved its rotary table to the device. The existing
    graph manager bounds shapes and owns replay outputs.
    """

    def __init__(self, tokenizer: S3TokenizerV2, vllm_config: VllmConfig, *, max_graphs: int = 2):
        from s3tokenizer.model_v2 import AudioEncoderV2

        if type(tokenizer.encoder) is not AudioEncoderV2:
            raise ValueError("S3 CUDA graphs require the S3 AudioEncoderV2")
        self.tokenizer = tokenizer
        config = _graph_config(vllm_config)
        self.graph = EncoderCudaGraph(self._forward, config, max_graphs=max_graphs)
        self._policy: tuple[bool, ...] | None = None

    def _forward(self, mels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden, lengths = self.tokenizer.encoder(mels, lengths)
        codes = self.tokenizer.quantizer.encode(hidden)
        # The graph manager returns one owned tensor. Keep token and length
        # outputs together so both survive subsequent graph replays.
        return torch.cat((codes, lengths.to(codes.dtype).unsqueeze(1)), dim=1)

    @torch.inference_mode()
    def __call__(self, mels: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy = _precision_policy()
        if self._policy is None:
            self._policy = policy
        if policy != self._policy or torch.is_autocast_enabled("cuda"):
            hidden, code_lengths = self.tokenizer.encoder(mels, lengths)
            return self.tokenizer.quantizer.encode(hidden), code_lengths
        packed = self.graph(mels, lengths)
        return packed[:, :-1], packed[:, -1].to(lengths.dtype)
