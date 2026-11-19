# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape CUDA replay for MiniCPM's short S3 reference tokenizer."""

from copy import copy

import torch
from s3tokenizer.model_v2 import AudioEncoderV2, S3TokenizerV2
from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalConfig

from .encoder_cuda_graph import EncoderCudaGraph
from .speaker_cuda_graph import _precision_policy


class S3CudaGraph:
    """Keep the original S3 modules, masks, arithmetic and quantizer.

    The caller supplies a single unpadded short reference. The pinned
    s3tokenizer 0.3.0 encoder uses tensor widths for its masks and is capturable
    once the caller has moved its rotary table to the device. The existing
    graph manager bounds shapes and owns replay outputs.
    """

    def __init__(self, tokenizer: S3TokenizerV2, vllm_config: VllmConfig, *, max_graphs: int = 2):
        if type(tokenizer.encoder) is not AudioEncoderV2:
            raise ValueError("S3 CUDA graphs require the S3 AudioEncoderV2")
        self.tokenizer = tokenizer
        config = copy(vllm_config)
        config.model_config = copy(config.model_config)
        if config.model_config.multimodal_config is None:
            config.model_config.multimodal_config = MultiModalConfig()
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
